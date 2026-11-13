"""Train a clinical-only MLP on the frozen patient cohort.

The model uses age, race, and ethnicity from patients.csv. Preprocessing is fit
on training patients only. Validation patients select checkpoints; test patients
are excluded from feature and label arrays and are not evaluated.
"""
import argparse
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             log_loss, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from multimodal_features import ROOT, read, write


class ClinicalMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden=32, dropout=0.3):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden // 2, 2),
        )

    def forward(self, features):
        return self.layers(features)


def make_preprocessor():
    return ColumnTransformer([
        ("age", make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
        ), [0]),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [1, 2]),
    ], sparse_threshold=0)


def clinical_values(rows):
    result = []
    for row in rows:
        raw_age = row.get("age_at_diagnosis", "")
        try:
            age = float(raw_age) / 365.25 if raw_age else np.nan
        except (TypeError, ValueError):
            age = np.nan
        categories = []
        for key in ("race", "ethnicity"):
            value = (row.get(key) or "").strip()
            if value.lower() in {"", "not reported", "unknown", "not available"}:
                value = "missing"
            categories.append(value)
        result.append([age, *categories])
    return np.asarray(result, dtype=object)


def metrics(labels, probabilities):
    predictions = probabilities.argmax(axis=1)
    return {
        "patients": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities[:, 1])),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            labels, predictions, labels=[0, 1], output_dict=True, zero_division=0,
        ),
    }


def train_seed(features, labels, train_indices, validation_indices, seed,
               epochs, patience, learning_rate, weight_decay):
    torch.manual_seed(seed)
    model = ClinicalMLP(features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
    class_counts = np.bincount(labels[train_indices], minlength=2)
    class_weights = len(train_indices) / (2 * np.maximum(class_counts, 1))
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32))
    x_train = torch.tensor(features[train_indices], dtype=torch.float32)
    y_train = torch.tensor(labels[train_indices], dtype=torch.long)
    x_validation = torch.tensor(features[validation_indices], dtype=torch.float32)
    y_validation = torch.tensor(labels[validation_indices], dtype=torch.long)

    best_state, best_loss, best_epoch, stale = None, float("inf"), 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_loss = loss_fn(model(x_train), y_train)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            validation_logits = model(x_validation)
            validation_loss = float(loss_fn(validation_logits, y_validation).item())
            probabilities = validation_logits.softmax(dim=1).cpu().numpy()
        history.append({"epoch": epoch, "train_loss": float(train_loss.item()),
                        "validation_loss": validation_loss,
                        "validation_balanced_accuracy": float(
                            balanced_accuracy_score(labels[validation_indices],
                                                    probabilities.argmax(axis=1)))})
        if validation_loss < best_loss - 1e-6:
            best_loss, best_epoch = validation_loss, epoch
            best_state, stale = copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        probabilities = model(x_validation).softmax(dim=1).cpu().numpy()
    return model, probabilities, best_epoch, history


def run(args):
    if args.output.exists():
        raise ValueError("Output already exists; choose a new --output to preserve experiments")
    rows = read(args.work / "patients.csv")
    # Deliberately exclude test rows before constructing any feature or label arrays.
    rows = [row for row in rows if row["split"] in {"train", "validation"}]
    if not rows:
        raise ValueError("No train/validation patients found")
    if len({row["case_uuid"] for row in rows}) != len(rows):
        raise ValueError("Duplicate case_uuid values in patient cohort")
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    splits = np.asarray([row["split"] for row in rows])
    train_indices = np.flatnonzero(splits == "train")
    validation_indices = np.flatnonzero(splits == "validation")
    for name, indices in (("train", train_indices), ("validation", validation_indices)):
        if not len(indices) or set(labels[indices]) != {0, 1}:
            raise ValueError(name + " split must contain both classes")

    raw = clinical_values(rows)
    preprocessor = make_preprocessor()
    features = preprocessor.fit_transform(raw[train_indices], labels[train_indices])
    # Transform validation with training-fitted imputers, scaling, and categories.
    validation_features = preprocessor.transform(raw[validation_indices])
    all_features = np.empty((len(rows), features.shape[1]), dtype=np.float32)
    all_features[train_indices] = features.astype(np.float32)
    all_features[validation_indices] = validation_features.astype(np.float32)

    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True)
    seeds = [42, 7, 2026]
    probabilities, model_paths, histories = [], [], {}
    for seed in seeds:
        model, probability, best_epoch, history = train_seed(
            all_features, labels, train_indices, validation_indices, seed,
            args.epochs, args.patience, args.learning_rate, args.weight_decay,
        )
        model_path = args.output / ("seed_%d.pt" % seed)
        torch.save({"state_dict": model.state_dict(), "input_dim": all_features.shape[1],
                    "hidden": 32, "dropout": 0.3, "seed": seed,
                    "best_epoch": best_epoch}, model_path)
        probabilities.append(probability)
        model_paths.append(model_path.name)
        histories[str(seed)] = history
        print("seed=%d best_epoch=%d" % (seed, best_epoch), flush=True)

    mean_probabilities = np.mean(probabilities, axis=0)
    validation_labels = labels[validation_indices]
    report = {
        "architecture": "32 -> 16 -> 2 MLP with ReLU and dropout",
        "clinical_features": ["age_at_diagnosis in years", "race", "ethnicity"],
        "preprocessing_fit_on": "train only",
        "checkpoint_selection": "lowest validation cross-entropy with early stopping",
        "seeds": seeds,
        "ensemble": "mean of seed probabilities",
        "test_evaluated": False,
        "train_patients": int(len(train_indices)),
        "validation": metrics(validation_labels, mean_probabilities),
        "training_parameters": {"epochs": args.epochs, "patience": args.patience,
                                "learning_rate": args.learning_rate,
                                "weight_decay": args.weight_decay},
        "model_files": model_paths,
    }
    joblib.dump(preprocessor, args.output / "preprocessor.joblib")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "history.json").write_text(json.dumps(histories) + "\n")
    prediction_rows = []
    for i, probability in zip(validation_indices, mean_probabilities):
        prediction_rows.append({
            "patient_id": rows[i]["patient_id"], "case_uuid": rows[i]["case_uuid"],
            "label": int(labels[i]), "prediction": int(probability.argmax()),
            "ductal_probability": float(probability[0]),
            "lobular_probability": float(probability[1]),
        })
    write(args.output / "validation_predictions.csv", prediction_rows)
    print(json.dumps(report["validation"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=ROOT / "data/models/multimodal_gat")
    parser.add_argument("--output", type=Path, default=ROOT / "data/models/clinical_mlp")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
