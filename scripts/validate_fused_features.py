"""Validate exported fused-feature CSVs and run one saved test vector."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from patient_graph_model import PatientGraphGAT, make_graph


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir

    rows = read_csv(run_dir / 'fused_features.csv')
    if not rows:
        raise ValueError('fused_features.csv has no patient rows')
    feature_columns = [name for name in rows[0] if name.startswith('fused_')]
    if not feature_columns:
        raise ValueError('No fused feature columns found')
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError('CSV rows have inconsistent columns')
    ids = [row['case_uuid'] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate patient case_uuid values')
    matrix = np.asarray([[float(row[column]) for column in feature_columns] for row in rows])
    if not np.isfinite(matrix).all():
        raise ValueError('Fused feature CSV contains nonfinite values')

    training = [row for row in rows if row['split'] == 'train']
    archive = np.load(run_dir / 'training_graph.npz')
    if len(training) != len(archive['features']):
        raise ValueError('Training row count does not match training_graph.npz')
    training_matrix = np.asarray([[float(row[column]) for column in feature_columns] for row in training])

    vector_rows = read_csv(run_dir / 'fused_test_vector.csv')
    if len(vector_rows) != 1 or vector_rows[0]['split'] != 'test':
        raise ValueError('fused_test_vector.csv must contain exactly one test patient')
    vector_row = vector_rows[0]
    selection = json.loads((run_dir / 'selection.json').read_text())
    chosen = selection['selected']
    reference = archive['features']
    modalities = np.load(run_dir / 'modalities.npz')
    ids = modalities['case_uuid'].astype(str).tolist()
    if ids != [row['case_uuid'] for row in rows]:
        raise ValueError('Modality artifact patient order does not match fused_features.csv')
    image_features = modalities['image_features']
    clinical_features = modalities['clinical_features']
    graph_features = modalities['graph_features']
    train_indices = [i for i, row in enumerate(rows) if row['split'] == 'train']
    test_index = ids.index(vector_row['case_uuid'])
    graph = make_graph(reference, graph_features[[test_index]], chosen['k'], chosen['minimum'])
    probabilities = []
    recomputed_training = []
    recomputed_queries = []
    for seed in selection['seeds']:
        checkpoint_path = run_dir / f"config{chosen['config']}_seed{seed}.pt"
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model = PatientGraphGAT(checkpoint['clinical_dim'], dropout=chosen['dropout'],
                                num_layers=checkpoint.get('num_layers', 2))
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        with torch.inference_mode():
            train_fused = model.fusion(torch.tensor(image_features[train_indices]),
                                       torch.tensor(clinical_features[train_indices]))
            query_fused = model.fusion(torch.tensor(image_features[[test_index]]),
                                       torch.tensor(clinical_features[[test_index]]))
            recomputed_training.append(train_fused.numpy())
            recomputed_queries.append(query_fused.numpy())
            model_input = (torch.tensor(np.concatenate([image_features[train_indices], image_features[[test_index]]])),
                           torch.tensor(np.concatenate([clinical_features[train_indices], clinical_features[[test_index]]])),
                           *graph[1:])
            probabilities.append(float(model(*model_input)[len(reference):].softmax(1)[0, 1]))
    np.testing.assert_allclose(training_matrix, np.mean(recomputed_training, axis=0), rtol=1e-5, atol=1e-6)
    vector = np.asarray([[float(vector_row[column]) for column in feature_columns]], dtype=np.float32)
    np.testing.assert_allclose(vector[0], np.mean(recomputed_queries, axis=0)[0], rtol=1e-5, atol=1e-6)
    probability = float(np.mean(probabilities))
    prediction = int(probability >= 0.5)
    recorded = next((row for row in read_csv(run_dir / 'predictions.csv')
                     if row['case_uuid'] == vector_row['case_uuid'] and row['split'] == 'test'), None)
    if recorded is None:
        raise ValueError('Test vector has no corresponding saved test prediction')
    if int(recorded['prediction']) != prediction or not np.isclose(
            float(recorded['lobular_probability']), probability, rtol=1e-5, atol=1e-6):
        raise ValueError('Recomputed vector prediction differs from predictions.csv')

    print(json.dumps({
        'status': 'passed',
        'patients_in_fused_csv': len(rows),
        'feature_dimensions': len(feature_columns),
        'training_vectors_match_saved_models': True,
        'test_vector_case_uuid': vector_row['case_uuid'],
        'test_vector_dimensions': int(vector.shape[1]),
        'recomputed_prediction': prediction,
        'recorded_prediction': int(recorded['prediction']),
        'lobular_probability': probability,
    }, indent=2))


if __name__ == '__main__':
    main()
