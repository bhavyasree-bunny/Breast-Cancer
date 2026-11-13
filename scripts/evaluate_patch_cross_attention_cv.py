"""Patient-level five-fold evaluation of patch-token cross-attention GATs."""
import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, log_loss, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split

from multimodal_features import (clinical_matrix, read, transform_modalities,
                                 write)
from patient_graph_model import (make_graph, CrossAttentionFusion)
from patch_cross_attention_gat import (PatchCrossAttention, extract_patch_embeddings,
                                       fit_seed, load_patch_bags)


def graph_features_from_patch_bags(patches, padding, clinical):
    valid = (~padding).astype(np.float32)[..., None]
    pooled = (patches * valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1.0)
    norms = np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
    pooled = pooled / norms
    return np.concatenate((pooled / np.sqrt(2.0), clinical / np.sqrt(2.0)), axis=1)


def choose_threshold(labels, probabilities):
    values = np.unique(probabilities)
    candidates = np.unique(np.r_[0.0, 0.5, 1.0,
                                  (values[:-1] + values[1:]) / 2.0])
    scores = [(balanced_accuracy_score(labels, probabilities >= threshold),
               -abs(float(threshold) - 0.5), float(threshold))
              for threshold in candidates]
    return max(scores)[2]


def summarize(labels, probabilities, thresholds):
    predicted = probabilities >= thresholds
    return {
        'patients': int(len(labels)),
        'accuracy': float(accuracy_score(labels, predicted)),
        'balanced_accuracy': float(balanced_accuracy_score(labels, predicted)),
        'macro_f1': float(f1_score(labels, predicted, average='macro',
                                   zero_division=0)),
        'roc_auc': float(roc_auc_score(labels, probabilities)),
        'log_loss': float(log_loss(labels, probabilities, labels=[0, 1])),
        'confusion_matrix': confusion_matrix(labels, predicted,
                                              labels=[0, 1]).tolist(),
    }


def predict_queries(model, train_indices, query_indices, patches, padding,
                    clinical, graph_features, config):
    graph = make_graph(graph_features[train_indices],
                       graph_features[query_indices], config['k'], config['minimum'])
    joint = np.r_[train_indices, query_indices]
    model.eval()
    with torch.inference_mode():
        logits = model(torch.tensor(patches[joint]), torch.tensor(padding[joint]),
                       torch.tensor(clinical[joint]), *graph[1:])
        return logits[len(train_indices):].softmax(1)[:, 1].cpu().numpy()


def run(args):
    if args.output.exists():
        raise ValueError('Output exists; choose a fresh --output directory')
    args.output.mkdir(parents=True)
    rows = read(args.work / 'patients.csv')
    labels = np.asarray([int(row['label']) for row in rows], dtype=np.int64)
    cases = [row['case_uuid'] for row in rows]
    if len(set(cases)) != len(cases):
        raise ValueError('Cohort contains duplicate patients')
    image_vectors = []
    for case in cases:
        with np.load(args.work / 'embeddings' / (case + '.npz')) as artifact:
            if str(artifact['case_uuid']) != case:
                raise ValueError('Embedding patient mismatch for ' + case)
            vector = artifact['feature']
            if vector.shape != (2048,) or not np.isfinite(vector).all():
                raise ValueError('Invalid image embedding for ' + case)
            image_vectors.append(vector)
    image_vectors = np.asarray(image_vectors)
    raw_clinical = clinical_matrix(rows)
    # Frozen ResNet patch embeddings are label-independent; every bag is cached
    # once, while fold-specific patch and clinical scalers are fitted below.
    extract_patch_embeddings(rows, args.work, args.batch_size)
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True,
                               random_state=args.fold_seed)
    conditions = {
        'no_graph_weighted': {'k': 0, 'minimum': 0.0, 'dropout': 0.2,
                              'lr': 0.001, 'class_weighted': True},
        'k7_graph_weighted': {'k': 7, 'minimum': 0.0, 'dropout': 0.2,
                              'lr': 0.001, 'class_weighted': True},
        'no_graph_unweighted': {'k': 0, 'minimum': 0.0, 'dropout': 0.2,
                                'lr': 0.001, 'class_weighted': False},
        'k7_graph_unweighted': {'k': 7, 'minimum': 0.0, 'dropout': 0.2,
                                'lr': 0.001, 'class_weighted': False},
    }
    predictions = {name: [] for name in conditions}
    fold_reports = []
    for fold, (outer_train, outer_test) in enumerate(
            splitter.split(image_vectors, labels), start=1):
        fit_indices, stop_indices = train_test_split(
            outer_train, test_size=args.early_stop_fraction,
            random_state=args.fold_seed + fold,
            stratify=labels[outer_train])
        _, clinical, _ = transform_modalities(
            image_vectors[fit_indices], image_vectors,
            raw_clinical[fit_indices], raw_clinical)
        patches, padding, _, patch_counts = load_patch_bags(
            rows, args.work, fit_indices)
        graph_features = graph_features_from_patch_bags(patches, padding, clinical)
        torch.manual_seed(314159)
        fusion = PatchCrossAttention(clinical.shape[1])
        fusion_state = fusion.state_dict()
        for name, config in conditions.items():
            validation_probabilities = []
            test_probabilities = []
            for seed in args.model_seeds:
                log_dir = args.output / 'training_logs'
                log_dir.mkdir(exist_ok=True)
                with (log_dir / f'fold{fold}_{name}_seed{seed}.log').open('w') as log:
                    with contextlib.redirect_stdout(log):
                        model, _, _, _ = fit_seed(
                            patches, padding, clinical, graph_features,
                            fusion_state, labels,
                            {'train': fit_indices, 'validation': stop_indices},
                            config, seed, args.epochs)
                        validation_probabilities.append(predict_queries(
                            model, fit_indices, stop_indices, patches, padding,
                            clinical, graph_features, config))
                        test_probabilities.append(predict_queries(
                            model, fit_indices, outer_test, patches, padding,
                            clinical, graph_features, config))
            validation_probabilities = np.mean(validation_probabilities, axis=0)
            test_probabilities = np.mean(test_probabilities, axis=0)
            threshold = choose_threshold(labels[stop_indices],
                                         validation_probabilities)
            fold_reports.append({
                'fold': fold, 'model': name, 'k': config['k'],
                'class_weighted_loss': config['class_weighted'],
                'inner_train_patients': int(len(fit_indices)),
                'early_stop_patients': int(len(stop_indices)),
                'outer_patients': int(len(outer_test)),
                'min_patch_tokens': int(min(patch_counts)),
                'max_patch_tokens': int(max(patch_counts)),
                'selected_threshold': threshold,
                'inner_balanced_accuracy_at_threshold': float(
                    balanced_accuracy_score(labels[stop_indices],
                                            validation_probabilities >= threshold)),
            })
            for index, probability in zip(outer_test, test_probabilities):
                predictions[name].append({
                    'patient_id': rows[index]['patient_id'],
                    'case_uuid': rows[index]['case_uuid'],
                    'fold': fold, 'label': int(labels[index]),
                    'lobular_probability': float(probability),
                    'threshold': float(threshold),
                    'prediction': int(probability >= threshold),
                })
        print(f'Completed outer fold {fold}/{args.folds}', flush=True)

    report = {
        'cohort_patients': len(rows),
        'class_counts': {'ductal': int(np.sum(labels == 0)),
                         'lobular': int(np.sum(labels == 1))},
        'outer_folds': args.folds, 'fold_seed': args.fold_seed,
        'model_seeds': args.model_seeds, 'epochs': args.epochs,
        'protocol': ('Stratified patient-level outer folds; clinical and patch '
                     'scalers fit on inner training data only; early stopping and '
                     'threshold selection use inner validation only; outer-fold '
                     'predictions are out-of-fold. Fixed comparison: no graph '
                     'versus k=7 graph, each trained with balanced class weights '
                     'and unweighted loss. Frozen ImageNet ResNet-50 patch embeddings.'),
        'results': {}, 'folds': fold_reports,
    }
    for name, rows_out in predictions.items():
        write(args.output / (name + '_predictions.csv'), rows_out)
        prediction_labels = np.asarray([int(row['label']) for row in rows_out])
        probs = np.asarray([float(row['lobular_probability']) for row in rows_out])
        thresholds = np.asarray([float(row['threshold']) for row in rows_out])
        report['results'][name] = summarize(prediction_labels, probs, thresholds)
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['results'], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--fold-seed', type=int, default=20261101)
    parser.add_argument('--model-seeds', type=int, nargs='+', default=[42, 7, 2026])
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--early-stop-fraction', type=float, default=0.2)
    args = parser.parse_args()
    if (args.folds < 3 or args.epochs < 1 or args.batch_size < 1
            or not 0 < args.early_stop_fraction < 0.5):
        parser.error('invalid fold, epoch, batch, or early-stop settings')
    run(args)


if __name__ == '__main__':
    main()
