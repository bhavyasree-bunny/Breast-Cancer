"""Shared clinical preprocessing and patient feature/graph utilities."""
import csv
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/models/multimodal_gat"


def read(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cosine_edges(reference, queries=None, k=5):
    """Build directed cosine-neighbour edges; query nodes only receive messages."""
    reference = np.asarray(reference, dtype=float)
    query = reference if queries is None else np.asarray(queries, dtype=float)
    ref = reference / np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-12)
    qry = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    similarities = qry @ ref.T
    edges, values = [], []
    for target, scores in enumerate(similarities):
        order = np.argsort(-scores, kind="stable")
        if queries is None:
            order = order[order != target]
        for source in order[:min(k, len(order))]:
            edges.append((int(source), target if queries is None else len(reference) + target))
            values.append(float(scores[source]))
    return np.asarray(edges, dtype=np.int64).reshape(-1, 2).T, np.asarray(values, dtype=np.float32).reshape(-1, 1)


def clinical_matrix(rows):
    values = []
    for row in rows:
        age = float(row["age_at_diagnosis"]) / 365.25 if row["age_at_diagnosis"] else np.nan
        categories = [row[key] if row[key].strip().lower() not in {"", "not reported", "unknown"} else "missing" for key in ("race", "ethnicity")]
        values.append([age, *categories])
    return np.asarray(values, dtype=object)


def fuse(train_images, images, train_clinical, clinical):
    """Fit training-only modality transforms and concatenate image/clinical features."""
    image_features, clinical_features, transforms = transform_modalities(
        train_images, images, train_clinical, clinical
    )
    fused = np.concatenate([image_features, clinical_features], axis=1).astype(np.float32)
    return fused, transforms


def transform_modalities(train_images, images, train_clinical, clinical):
    """Fit training-only transforms and return separate image/clinical arrays."""
    image_scaler = make_pipeline(StandardScaler(), Normalizer())
    encoder = ColumnTransformer([
        ("age", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), [0]),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [1, 2]),
    ], sparse_threshold=0)
    image_scaler.fit(train_images)
    encoder.fit(train_clinical)
    image_features = image_scaler.transform(images).astype(np.float32)
    clinical_features = Normalizer().fit_transform(encoder.transform(clinical)).astype(np.float32)
    return image_features, clinical_features, {"image_scaler": image_scaler, "clinical_encoder": encoder}


def metrics(labels, probability):
    predicted = (probability >= .5).astype(int)
    return {"balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
            "roc_auc": float(roc_auc_score(labels, probability)),
            "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
            "classification_report": classification_report(labels, predicted, labels=[0, 1], output_dict=True, zero_division=0)}
