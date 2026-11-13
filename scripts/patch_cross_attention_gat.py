"""Train a validation-only GAT using cross-attention over real patch tokens."""
import argparse
import copy
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from torch_geometric.nn import GATConv

from multimodal_features import ROOT, WORK, read, write
from patient_graph_model import evaluate, load_data, make_graph


class PatchCrossAttentionGAT(torch.nn.Module):
    def __init__(self, clinical_dim, hidden=64, heads=4, dropout=0.2, layers=2):
        super().__init__()
        self.fusion = PatchCrossAttention(clinical_dim, hidden, heads)
        self.layers = torch.nn.ModuleList([
            GATConv(hidden, hidden // heads, heads=heads, edge_dim=1,
                    fill_value=1.0, dropout=dropout)
            for _ in range(layers)
        ])
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden) for _ in range(layers)])
        self.classifier = torch.nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, patches, patch_padding, clinical, edges, scores):
        x = self.fusion(patches, clinical, patch_padding)
        for layer, norm in zip(self.layers, self.norms):
            message = layer(x, edges, scores)
            x = torch.nn.functional.elu(norm(x + message))
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


class PatchCrossAttention(torch.nn.Module):
    """Attend over real patch embeddings and clinical tokens to make one vector."""
    def __init__(self, clinical_dim, hidden=64, heads=4):
        super().__init__()
        encoder = torch.nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.image_embed = torch.nn.Linear(2048, hidden)
        self.age_embed = torch.nn.Linear(2, hidden)
        self.demographic_embed = torch.nn.Linear(clinical_dim - 2, hidden)
        self.image_to_clinical = copy.deepcopy(encoder)
        self.clinical_to_image = copy.deepcopy(encoder)
        self.project = torch.nn.Linear(hidden, hidden)

    def forward(self, patches, clinical, patch_padding=None):
        image_tokens = self.image_embed(patches)
        clinical_tokens = torch.stack([
            self.age_embed(clinical[:, :2]),
            self.demographic_embed(clinical[:, 2:]),
        ], dim=1)
        image_context, _ = self.image_to_clinical(
            image_tokens, clinical_tokens, clinical_tokens, need_weights=False)
        clinical_context, _ = self.clinical_to_image(
            clinical_tokens, image_tokens, image_tokens,
            key_padding_mask=patch_padding, need_weights=False)
        if patch_padding is None:
            image_pool = image_context.mean(1)
        else:
            valid = (~patch_padding).unsqueeze(-1).to(image_context.dtype)
            image_pool = (image_context * valid).sum(1) / valid.sum(1).clamp_min(1.0)
        pooled = image_pool + clinical_context.mean(1)
        return torch.nn.functional.elu(self.project(pooled))


def extract_patch_embeddings(rows, work, batch_size):
    """Embed saved normalized patches; never opens or downloads whole slides."""
    cache = work / 'patch_embeddings'
    cache.mkdir(parents=True, exist_ok=True)
    manifest = read(work / 'patch_manifest.csv')
    by_patient = defaultdict(list)
    wanted = {row['case_uuid'] for row in rows}
    for row in manifest:
        if row['case_uuid'] in wanted:
            path = ROOT / row['patch_file']
            if path.is_file():
                by_patient[row['case_uuid']].append(path)
    missing = [row['case_uuid'] for row in rows if not by_patient[row['case_uuid']]]
    if missing:
        raise ValueError(f'{len(missing)} train/validation patients have no saved patch PNGs')

    outputs = {case: cache / f'{case}.npz' for case in by_patient}
    todo = {case: paths for case, paths in by_patient.items() if not outputs[case].exists()}
    if not todo:
        print(f'Using cached patch embeddings for {len(outputs)} patients', flush=True)
        return

    from torchvision.models import ResNet50_Weights, resnet50

    torch.hub.set_dir(str(work / 'torch_cache'))
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    transform = weights.transforms()
    pending = []
    for case_uuid, paths in todo.items():
        for path in paths:
            pending.append((case_uuid, path))
    results = defaultdict(list)
    torch.set_num_threads(4)
    with torch.inference_mode():
        for start in range(0, len(pending), batch_size):
            batch_items = pending[start:start + batch_size]
            batch = torch.stack([transform(Image.open(path).convert('RGB'))
                                 for _, path in batch_items])
            features = model(batch).cpu().numpy().astype(np.float32)
            for (case_uuid, path), feature in zip(batch_items, features):
                results[case_uuid].append((str(path.relative_to(ROOT)), feature))
            print(f'Embedded patches {min(start + len(batch_items), len(pending))}/{len(pending)}', flush=True)
    for case_uuid, items in results.items():
        np.savez_compressed(outputs[case_uuid], case_uuid=case_uuid,
                            patch_files=np.asarray([name for name, _ in items]),
                            features=np.stack([feature for _, feature in items]))
        print(f'  saved {len(items)} patch tokens for {case_uuid}', flush=True)


def load_patch_bags(rows, work, train_indices):
    raw = []
    for row in rows:
        with np.load(work / 'patch_embeddings' / f"{row['case_uuid']}.npz") as artifact:
            if str(artifact['case_uuid']) != row['case_uuid']:
                raise ValueError('Patch embedding patient mismatch')
            values = artifact['features'].astype(np.float32)
            if values.ndim != 2 or values.shape[1] != 2048 or not np.isfinite(values).all():
                raise ValueError('Invalid patch embedding matrix')
            raw.append(values)
    scaler = make_pipeline(StandardScaler(), Normalizer())
    scaler.fit(np.concatenate([raw[i] for i in train_indices], axis=0))
    scaled = [scaler.transform(bag).astype(np.float32) for bag in raw]
    max_patches = max(map(len, scaled))
    padded = np.zeros((len(rows), max_patches, 2048), dtype=np.float32)
    mask = np.ones((len(rows), max_patches), dtype=bool)
    for index, bag in enumerate(scaled):
        padded[index, :len(bag)] = bag
        mask[index, :len(bag)] = False
    return padded, mask, scaler, [len(bag) for bag in scaled]


def make_initial_graph_features(patches, padding, clinical, clinical_dim):
    torch.manual_seed(314159)
    fusion = PatchCrossAttention(clinical_dim)
    fusion.eval()
    with torch.inference_mode():
        features = fusion(torch.tensor(patches), torch.tensor(clinical),
                          torch.tensor(padding)).numpy()
    return features, copy.deepcopy(fusion.state_dict())


def fit_seed(patches, padding, clinical, graph_features, fusion_state, labels,
             splits, config, seed, epochs):
    torch.manual_seed(seed)
    train, validation = splits['train'], splits['validation']
    ref = graph_features[train]
    train_graph = make_graph(ref, None, config['k'], config['minimum'])
    val_graph = make_graph(ref, graph_features[validation], config['k'], config['minimum'])
    model = PatchCrossAttentionGAT(clinical.shape[1], dropout=config['dropout'])
    model.fusion.load_state_dict(fusion_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.001)
    if config.get('class_weighted', True):
        counts = np.bincount(labels[train], minlength=2)
        weights = torch.tensor(
            len(train) / (2 * np.maximum(counts, 1)), dtype=torch.float32)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    target = torch.tensor(labels[train], dtype=torch.long)
    train_tensors = (torch.tensor(patches[train]), torch.tensor(padding[train]),
                     torch.tensor(clinical[train]))
    val_indices = np.r_[train, validation]
    val_tensors = (torch.tensor(patches[val_indices]), torch.tensor(padding[val_indices]),
                   torch.tensor(clinical[val_indices]))
    best, rank, stale, history = None, None, 0, []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(*train_tensors, *train_graph[1:]), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            prob = model(*val_tensors, *val_graph[1:])[len(train):].softmax(1)[:, 1].numpy()
        measured = evaluate(labels[validation], prob)
        candidate = (measured['balanced_accuracy'], measured['macro_f1'], -measured['log_loss'])
        history.append({'epoch': epoch + 1, 'train_loss': float(loss.item()), **measured})
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f'    epoch {epoch + 1:03d} train_loss={loss.item():.4f} '
                  f'val_bal_acc={measured["balanced_accuracy"]:.3f} '
                  f'val_log_loss={measured["log_loss"]:.4f}', flush=True)
        if rank is None or candidate > rank:
            best, rank, best_epoch, stale = copy.deepcopy(model.state_dict()), candidate, epoch + 1, 0
        else:
            stale += 1
        if stale >= 30:
            break
    model.load_state_dict(best)
    model.eval()
    with torch.inference_mode():
        prob = model(*val_tensors, *val_graph[1:])[len(train):].softmax(1)[:, 1].numpy()
    return model, prob, best_epoch, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=WORK)
    parser.add_argument('--output', type=Path,
                        default=WORK.parent / 'patch_cross_attention_gat')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output exists; choose a new --output to preserve prior runs')
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('--epochs and --batch-size must be positive')

    torch.set_num_threads(4)
    rows, labels, splits, _, clinical, _, transforms, _ = load_data(args.work)
    print(f'Loaded {len(rows)} patients: {len(splits["train"])} train, '
          f'{len(splits["validation"])} validation; test excluded.', flush=True)
    extract_patch_embeddings(rows, args.work, args.batch_size)
    patches, padding, patch_scaler, patch_counts = load_patch_bags(
        rows, args.work, splits['train'])
    graph_features, fusion_state = make_initial_graph_features(
        patches, padding, clinical, clinical.shape[1])

    seeds = (42, 7, 2026)
    config = {'k': 3, 'minimum': 0.0, 'dropout': 0.2, 'lr': 0.001}
    args.output.mkdir(parents=True)
    protocol = {
        'fusion': 'cross-attention over real ResNet patch embeddings and clinical tokens',
        'graph_features': 'initial attention-fused patient vectors',
        'layers': 2, 'fixed_config': config, 'seeds': seeds,
        'selection': 'validation balanced accuracy, macro-F1, then log loss',
        'test_evaluated': False,
        'patients': {'train': len(splits['train']), 'validation': len(splits['validation'])},
        'patch_tokens': {'minimum': min(patch_counts), 'maximum': max(patch_counts),
                         'total': int(sum(patch_counts))},
        'test_note': 'Test patients are excluded from patch extraction, training and validation.',
    }
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    torch.save(fusion_state, args.output / 'initial_fusion_state.pt')
    import joblib
    joblib.dump({'clinical': transforms['clinical_encoder'], 'patch_scaler': patch_scaler},
                args.output / 'transforms.joblib')

    models, probabilities, seed_metrics = [], [], []
    for seed in seeds:
        print(f'Fitting real-patch cross-attention GAT | seed={seed}', flush=True)
        model, probability, best_epoch, history = fit_seed(
            patches, padding, clinical, graph_features, fusion_state, labels,
            splits, config, seed, args.epochs)
        measured = evaluate(labels[splits['validation']], probability)
        models.append(model)
        probabilities.append(probability)
        seed_metrics.append(measured)
        torch.save({'state_dict': model.state_dict(), 'clinical_dim': clinical.shape[1],
                    'num_layers': len(model.layers), 'seed': seed,
                    'best_epoch': best_epoch, 'config': config},
                   args.output / f'seed{seed}.pt')
        (args.output / f'seed{seed}_history.json').write_text(json.dumps(history))
        print(f'  seed={seed} best_epoch={best_epoch} '
              f'val_bal_acc={measured["balanced_accuracy"]:.3f} '
              f'val_macro_f1={measured["macro_f1"]:.3f} '
              f'val_log_loss={measured["log_loss"]:.4f}', flush=True)

    ensemble = np.mean(probabilities, axis=0)
    summary = {
        'architecture': 'actual patch-token cross-attention -> two-layer residual GAT',
        'validation': evaluate(labels[splits['validation']], ensemble),
        'per_seed_validation': seed_metrics,
        'test_evaluated': False,
        'patch_tokens': protocol['patch_tokens'],
        'baseline_validation_balanced_accuracy': 0.6666666666666667,
        'comparison_note': 'Compare to the earlier fixed-split patient-vector model cautiously; repeated patient-level CV is needed.',
    }
    (args.output / 'metrics.json').write_text(json.dumps(summary, indent=2))
    predictions = []
    for index, probability in zip(splits['validation'], ensemble):
        predictions.append({'patient_id': rows[index]['patient_id'],
                            'case_uuid': rows[index]['case_uuid'],
                            'label': int(labels[index]), 'prediction': int(probability >= 0.5),
                            'lobular_probability': float(probability)})
    write(args.output / 'validation_predictions.csv', predictions)
    np.savez_compressed(args.output / 'validation_graph.npz',
                        features=graph_features[splits['train']])
    print('PATCH-LEVEL MODEL RESULT', json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
