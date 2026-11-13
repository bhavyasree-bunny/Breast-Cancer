"""Residual patient GAT: bounded validation search, then one test evaluation."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss
from torch_geometric.nn import GATConv

from multimodal_features import WORK, clinical_matrix, cosine_edges, transform_modalities, metrics, read, write


class CrossAttentionFusion(torch.nn.Module):
    """Bidirectional attention between image chunks and clinical feature tokens."""
    def __init__(self, clinical_dim, hidden=64, heads=4):
        super().__init__()
        encoder = torch.nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.image_embed = torch.nn.Linear(64, hidden)
        self.age_embed = torch.nn.Linear(2, hidden)
        self.demographic_embed = torch.nn.Linear(clinical_dim - 2, hidden)
        self.image_to_clinical = copy.deepcopy(encoder)
        self.clinical_to_image = copy.deepcopy(encoder)
        self.project = torch.nn.Linear(hidden, hidden)

    def forward(self, image, clinical):
        image_tokens = self.image_embed(image.reshape(-1, 32, 64))
        clinical_tokens = torch.stack([
            self.age_embed(clinical[:, :2]),
            self.demographic_embed(clinical[:, 2:]),
        ], dim=1)
        image_context, _ = self.image_to_clinical(image_tokens, clinical_tokens, clinical_tokens,
                                                   need_weights=False)
        clinical_context, _ = self.clinical_to_image(clinical_tokens, image_tokens, image_tokens,
                                                      need_weights=False)
        # Both modalities are represented in both contexts; sum them into one
        # shared vector rather than joining modality blocks by concatenation.
        pooled = image_context.mean(1) + clinical_context.mean(1)
        return torch.nn.functional.elu(self.project(pooled))


class PatientGraphGAT(torch.nn.Module):
    def __init__(self, clinical_dim, hidden=64, heads=4, dropout=.2, num_layers=2):
        super().__init__()
        if num_layers < 1:
            raise ValueError('num_layers must be at least one')
        self.fusion = CrossAttentionFusion(clinical_dim, hidden, heads)
        self.layers = torch.nn.ModuleList([GATConv(hidden, hidden // heads, heads=heads,
            edge_dim=1, fill_value=1., dropout=dropout) for _ in range(num_layers)])
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.classifier = torch.nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, image, clinical, edges, scores):
        x = self.fusion(image, clinical)
        for layer, norm in zip(self.layers, self.norms):
            message = layer(x, edges, scores)
            x = torch.nn.functional.elu(norm(x + message))
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


def filtered_edges(reference, queries=None, k=5, minimum=0.):
    edges, scores = cosine_edges(reference, queries, k)
    keep = scores[:, 0] >= minimum
    return edges[:, keep], scores[keep]


def make_graph(reference, queries, k, minimum):
    edge, scores = filtered_edges(reference, k=k, minimum=minimum)
    if queries is not None:
        extra, values = filtered_edges(reference, queries, k, minimum)
        edge, scores = np.concatenate([edge, extra], axis=1), np.concatenate([scores, values])
        reference = np.concatenate([reference, queries])
    return torch.tensor(reference, dtype=torch.float32), torch.tensor(edge), torch.tensor(scores)


def graph_similarity_features(image_features, clinical_features):
    """Build fixed, interpretable graph features from both normalized modalities.

    Each modality is L2-normalized by transform_modalities. Scaling both blocks
    by sqrt(2) makes cosine similarity the mean of their image and clinical
    cosine similarities, without an untrained fusion layer determining edges.
    """
    scale = np.sqrt(2.0)
    return np.concatenate((image_features / scale, clinical_features / scale), axis=1)


def load_data(work, include_test=False):
    rows = [r for r in read(work / 'patients.csv') if (include_test or r['split'] != 'test')
            and (work / 'embeddings' / (r['case_uuid'] + '.npz')).exists()]
    if len({r['case_uuid'] for r in rows}) != len(rows):
        raise ValueError('Duplicate patient IDs')
    images = []
    for r in rows:
        with np.load(work / 'embeddings' / (r['case_uuid'] + '.npz')) as artifact:
            if str(artifact['case_uuid']) != r['case_uuid']:
                raise ValueError('Embedding patient mismatch')
            vector = artifact['feature']
            if vector.shape != (2048,) or not np.isfinite(vector).all():
                raise ValueError('Invalid ResNet embedding')
            images.append(vector)
    labels = np.array([int(r['label']) for r in rows])
    indices = {s: np.array([i for i, r in enumerate(rows) if r['split'] == s], dtype=int)
               for s in ('train', 'validation', 'test')}
    for s in ('train', 'validation', 'test') if include_test else ('train', 'validation'):
        if set(labels[indices[s]]) != {0, 1}:
            raise ValueError(s + ' requires both classes')
    images = np.asarray(images)
    clinical = clinical_matrix(rows)
    image_features, clinical_features, transforms = transform_modalities(
        images[indices['train']], images, clinical[indices['train']], clinical
    )
    torch.manual_seed(314159)
    initial_fusion = CrossAttentionFusion(clinical_features.shape[1])
    initial_fusion.eval()
    graph_features = graph_similarity_features(image_features, clinical_features)
    fusion_initial_state = copy.deepcopy(initial_fusion.state_dict())
    return rows, labels, indices, image_features, clinical_features, graph_features, transforms, fusion_initial_state


def evaluate(labels, probability):
    result = metrics(labels, probability)
    predicted = (probability >= .5).astype(int)
    accuracy = float(accuracy_score(labels, predicted))
    n = len(labels)
    z = 1.96
    center = (accuracy + z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(accuracy*(1-accuracy)/n + z*z/(4*n*n))/(1+z*z/n)
    result.update(accuracy=accuracy, macro_f1=float(f1_score(labels, predicted, average='macro')),
                  log_loss=float(log_loss(labels, probability, labels=[0, 1])),
                  accuracy_95_wilson_interval=[center-half, center+half], patients=n,
                  correct=int(np.sum(predicted == labels)),
                  majority_class_accuracy=float(max(np.bincount(labels, minlength=2))/n))
    return result


def fit(image_features, clinical_features, graph_features, fusion_initial_state, labels, splits,
        config, seed, epochs, num_layers=2):
    torch.manual_seed(seed)
    train, validation = splits['train'], splits['validation']
    reference = graph_features[train]
    training_graph = make_graph(reference, None, config['k'], config['minimum'])
    validation_graph = make_graph(reference, graph_features[validation], config['k'], config['minimum'])
    model = PatientGraphGAT(clinical_features.shape[1], dropout=config['dropout'], num_layers=num_layers)
    model.fusion.load_state_dict(fusion_initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=.001)
    if config.get('class_weighted', True):
        counts = np.bincount(labels[train], minlength=2)
        weights = torch.tensor(len(train)/(2*counts), dtype=torch.float32)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    target = torch.tensor(labels[train], dtype=torch.long)
    best, rank, stale, history = None, None, 0, []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(torch.tensor(image_features[train]), torch.tensor(clinical_features[train]), *training_graph[1:]), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            probability = model(torch.tensor(image_features[np.r_[train, validation]]),
                                torch.tensor(clinical_features[np.r_[train, validation]]),
                                *validation_graph[1:])[len(train):].softmax(1)[:, 1].numpy()
        measured = evaluate(labels[validation], probability)
        # Predeclared criterion: balanced accuracy, macro-F1, then lower log loss.
        candidate = (measured['balanced_accuracy'], measured['macro_f1'], -measured['log_loss'])
        history.append({'epoch': epoch+1, 'train_loss': loss.item(), **measured})
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"      epoch {epoch + 1:03d} | train_loss={loss.item():.4f} | "
                  f"val_bal_acc={measured['balanced_accuracy']:.3f} | "
                  f"val_macro_f1={measured['macro_f1']:.3f} | val_log_loss={measured['log_loss']:.4f}",
                  flush=True)
        if rank is None or candidate > rank:
            best, rank, stale = copy.deepcopy(model.state_dict()), candidate, 0
            best_epoch = epoch + 1
        else:
            stale += 1
        if stale >= 30:
            break
    model.load_state_dict(best)
    model.eval()
    with torch.inference_mode():
        probability = model(torch.tensor(image_features[np.r_[train, validation]]),
                            torch.tensor(clinical_features[np.r_[train, validation]]),
                            *validation_graph[1:])[len(train):].softmax(1)[:, 1].numpy()
    return model, probability, best_epoch, history, training_graph


def run(args):
    torch.set_num_threads(4)
    if args.output.exists():
        raise ValueError('Output already exists; choose a new --output to preserve experiments')
    args.output.mkdir(parents=True)
    rows, labels, splits, image_features, clinical_features, graph_features, transforms, fusion_initial_state = load_data(args.work)
    torch.save(fusion_initial_state, args.output / 'initial_fusion_state.pt')
    seeds = [42, 7, 2026]
    # Exactly eight configurations fixed before observing new results.
    # Include a no-neighbour baseline to test whether graph edges add value.
    configs = [{'k': 0, 'minimum': 0., 'dropout': .2, 'lr': .001}]
    configs += [{'k': k, 'minimum': minimum, 'dropout': .2, 'lr': lr}
                for k in (3, 7) for minimum in (0., .25) for lr in (.0003, .001)]
    (args.output / 'protocol.json').write_text(json.dumps({'configs': configs, 'seeds': seeds,
        'criterion': 'mean seed validation balanced accuracy; mean macro-F1 then log-loss tie-break',
        'fusion': 'bidirectional multihead cross-attention between 32 image embedding chunks and age/demographic clinical tokens',
        'graph_features': 'equal-weight concatenation of training-fitted, L2-normalized image and clinical blocks; no untrained fusion layer determines graph edges',
        'graph_ablation': 'k=0 candidate uses self-loops only, with no inter-patient edges',
        'threshold': .5, 'epochs': args.epochs, 'patience': 30,
        'cohort_sha256': hashlib.sha256((args.work/'patients.csv').read_bytes()).hexdigest(),
        'test_note': 'Previously inspected holdout; no test features loaded during search.'}, indent=2))
    summaries, candidates = [], []
    for index, config in enumerate(configs):
        models, probabilities, seed_metrics = [], [], []
        for seed in seeds:
            print(f"  config {index}/7 | k={config['k']} min_cos={config['minimum']:.2f} "
                  f"lr={config['lr']:.4g} | seed={seed}", flush=True)
            model, probability, epoch, history, graph = fit(image_features, clinical_features, graph_features, fusion_initial_state,
                labels, splits, config, seed, args.epochs)
            models.append(model)
            probabilities.append(probability)
            measured = evaluate(labels[splits['validation']], probability)
            seed_metrics.append(measured)
            torch.save({'state_dict': model.state_dict(), 'clinical_dim': clinical_features.shape[1], 'num_layers': len(model.layers), 'config': config,
                        'seed': seed, 'best_epoch': epoch}, args.output / f'config{index}_seed{seed}.pt')
            (args.output / f'config{index}_seed{seed}_history.json').write_text(json.dumps(history))
        summary = {'config': index, **config,
            'mean_balanced_accuracy': float(np.mean([m['balanced_accuracy'] for m in seed_metrics])),
            'std_balanced_accuracy': float(np.std([m['balanced_accuracy'] for m in seed_metrics])),
            'mean_macro_f1': float(np.mean([m['macro_f1'] for m in seed_metrics])),
            'mean_log_loss': float(np.mean([m['log_loss'] for m in seed_metrics]))}
        summaries.append(summary)
        candidates.append((models, np.mean(probabilities, axis=0)))
        print(summary, flush=True)
    chosen = max(summaries, key=lambda s: (s['mean_balanced_accuracy'], s['mean_macro_f1'], -s['mean_log_loss']))
    selected = chosen['config']
    models, probability = candidates[selected]
    validation = evaluate(labels[splits['validation']], probability)
    report = {'selected': chosen, 'seeds': seeds, 'validation': validation, 'test_evaluated': False,
              'target_accuracy': .75, 'selection_metric': 'validation balanced accuracy',
              'architecture': 'bidirectional cross-attention fusion followed by two residual GAT layers with per-patient LayerNorm and linear classifier'}
    predictions = [{'patient_id': rows[i]['patient_id'], 'case_uuid': rows[i]['case_uuid'],
                    'split': 'validation', 'label': int(labels[i]), 'prediction': int(p >= .5),
                    'lobular_probability': float(p)} for i, p in zip(splits['validation'], probability)]
    reference = graph_features[splits['train']]
    export_rows = rows
    edge, scores = filtered_edges(reference, k=chosen['k'], minimum=chosen['minimum'])
    write(args.output / 'graph_edges.csv', [{'source': rows[splits['train'][s]]['patient_id'],
        'target': rows[splits['train'][t]]['patient_id'], 'cosine_similarity': float(v[0])}
        for (s, t), v in zip(edge.T, scores)], ['source', 'target', 'cosine_similarity'])
    write(args.output / 'graph_nodes.csv', [{'patient_id': rows[i]['patient_id'], 'label': int(labels[i])} for i in splits['train']])
    similarities = scores[:, 0]
    report['graph'] = {'training_nodes': len(reference), 'edges_without_self_loops': len(scores),
        'similarity_range': [float(similarities.min()), float(similarities.max())] if len(scores) else [],
        'self_loops': 'added by GATConv', 'isolated_before_self_loops': len(set(range(len(reference))) - set(edge[1]))}
    # Selection is saved before test embeddings are loaded.
    write(args.output / 'search.csv', summaries)
    (args.output / 'selection.json').write_text(json.dumps(report, indent=2))
    if not args.validation_only:
        all_rows, all_labels, all_splits, all_images, all_clinical, all_graph_features, _, all_fusion_initial_state = load_data(args.work, include_test=True)
        np.testing.assert_allclose(all_graph_features[all_splits['train']], reference)
        for key in fusion_initial_state:
            torch.testing.assert_close(fusion_initial_state[key], all_fusion_initial_state[key])
        export_rows = all_rows
        test = all_splits['test']
        graph = make_graph(reference, all_graph_features[test], chosen['k'], chosen['minimum'])
        model_input = (torch.tensor(all_images[np.r_[all_splits['train'], test]]),
                       torch.tensor(all_clinical[np.r_[all_splits['train'], test]]), *graph[1:])
        with torch.inference_mode():
            probability = np.mean([m(*model_input)[len(reference):].softmax(1)[:, 1].numpy() for m in models], axis=0)
        report['test'] = evaluate(all_labels[test], probability)
        report['test_evaluated'] = True
        predictions.extend({'patient_id': all_rows[i]['patient_id'], 'case_uuid': all_rows[i]['case_uuid'],
            'split': 'test', 'label': int(all_labels[i]), 'prediction': int(p >= .5),
            'lobular_probability': float(p)} for i, p in zip(test, probability))
    export_images, export_clinical = (all_images, all_clinical) if not args.validation_only else (image_features, clinical_features)
    fused_vectors = []
    with torch.inference_mode():
        for model in models:
            model.eval()
            fused_vectors.append(model.fusion(torch.tensor(export_images), torch.tensor(export_clinical)).numpy())
    export_features = np.mean(fused_vectors, axis=0)
    feature_columns = [f'fused_{index:04d}' for index in range(export_features.shape[1])]
    fused_rows = [
        {'patient_id': row['patient_id'], 'case_uuid': row['case_uuid'],
         'split': row['split'], 'label': int(row['label']),
         **{name: float(value) for name, value in zip(feature_columns, vector)}}
        for row, vector in zip(export_rows, export_features)
    ]
    fused_fields = ['patient_id', 'case_uuid', 'split', 'label', *feature_columns]
    write(args.output / 'fused_features.csv', fused_rows, fused_fields)
    if not args.validation_only:
        first_test = next(row for row in fused_rows if row['split'] == 'test')
        write(args.output / 'fused_test_vector.csv', [first_test], fused_fields)
    joblib.dump(transforms, args.output / 'transforms.joblib')
    np.savez_compressed(args.output / 'modalities.npz', case_uuid=np.asarray([row['case_uuid'] for row in export_rows]),
                        image_features=export_images, clinical_features=export_clinical,
                        graph_features=(all_graph_features if not args.validation_only else graph_features))
    np.savez(args.output / 'training_graph.npz', features=reference, edge_index=edge, edge_attr=scores)
    write(args.output / 'predictions.csv', predictions)
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=WORK)
    parser.add_argument('--output', type=Path, default=WORK.parent/'patient_graph_cross_attention')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--validation-only', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('epochs must be positive')
    run(args)


if __name__ == '__main__':
    main()
