"""Version 1: frozen ResNet-50 + clinical features + inductive patient GAT.

Commands: prepare, extract, train. Run from any working directory.
"""
import argparse
import copy
import csv
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data/models/multimodal_gat"
TARGETS = {"Infiltrating duct carcinoma, NOS": 0, "Lobular carcinoma, NOS": 1}


def read(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(work):
    work.mkdir(parents=True, exist_ok=True)
    destination = work / "patients.csv"
    if destination.exists():
        print(f"Keeping frozen cohort/splits: {destination}")
        return
    patients = [r for r in read(ROOT / "data/metadata/patients_250.csv") if r["primary_diagnosis"] in TARGETS]
    patients.sort(key=lambda r: r["case_uuid"])
    labels = np.array([TARGETS[r["primary_diagnosis"]] for r in patients])
    train, held = train_test_split(np.arange(len(patients)), test_size=.4, stratify=labels, random_state=42)
    validation, test = train_test_split(held, test_size=.5, stratify=labels[held], random_state=42)
    clinical = {r["case_uuid"]: r for r in read(ROOT / "data/metadata/clinical.csv")}
    split = {int(i): name for name, indices in [("train", train), ("validation", validation), ("test", test)] for i in indices}
    rows = []
    for i, patient in enumerate(patients):
        c = clinical[patient["case_uuid"]]
        if c["case_id"] != patient["case_id"]:
            raise ValueError("Clinical patient identifier mismatch")
        rows.append({"case_uuid": patient["case_uuid"], "patient_id": patient["case_id"],
                     "label": int(labels[i]), "split": split[i],
                     "age_at_diagnosis": c["age_at_diagnosis"], "race": c["race"], "ethnicity": c["ethnicity"]})
    write(destination, rows)
    print(json.dumps({name: {str(label): sum(r["split"] == name and r["label"] == label for r in rows) for label in (0, 1)} for name in ("train", "validation", "test")}, indent=2))


def cosine_edges(reference, queries=None, k=5):
    """Edges are source -> target; held-out queries never send messages to training."""
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
    for r in rows:
        age = float(r["age_at_diagnosis"]) / 365.25 if r["age_at_diagnosis"] else np.nan
        categories = [r[key] if r[key].strip().lower() not in {"", "not reported", "unknown"} else "missing" for key in ("race", "ethnicity")]
        values.append([age, *categories])
    return np.asarray(values, dtype=object)


def fuse(train_images, images, train_clinical, clinical):
    image_scaler = make_pipeline(StandardScaler(), Normalizer())
    encoder = ColumnTransformer([
        ("age", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), [0]),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [1, 2]),
    ], sparse_threshold=0)
    image_scaler.fit(train_images)
    encoder.fit(train_clinical)
    # Equal L2 weight for image and clinical blocks; fit transforms only on training.
    fused = np.concatenate([image_scaler.transform(images), Normalizer().fit_transform(encoder.transform(clinical))], axis=1).astype(np.float32)
    return fused, {"image_scaler": image_scaler, "clinical_encoder": encoder}


def extract(args):
    import torch
    import openslide
    from PIL import Image
    from torchvision.models import ResNet50_Weights, resnet50
    from image_preprocessing import modified_macenko, tissue_percentage

    patients = read(args.work / "patients.csv")
    status_path = getattr(args, "status_csv", ROOT / "data/metadata/download_status_500.csv")
    status = {r["slide_file_uuid"]: r for r in read(status_path)}
    mapped = read(ROOT / "data/metadata/patient_slide_mapping.csv")
    available = defaultdict(list)
    for r in mapped:
        path = ROOT / "data/images/tcga_brca_500" / r["patient_id"] / r["slide_file_name"]
        if status.get(r["slide_file_uuid"], {}).get("validation_result") == "passed" and path.is_file():
            available[r["case_uuid"]].append((r, path))
    # A balanced pilot selected independently of image content within frozen splits.
    groups = defaultdict(list)
    for p in patients:
        if available[p["case_uuid"]]:
            groups[(p["split"], p["label"])].append(p)
    selected = [p for key in sorted(groups) for p in groups[key][:args.patients_per_group]]
    weights = ResNet50_Weights.IMAGENET1K_V2
    torch.set_num_threads(4)
    torch.hub.set_dir(str(args.work / "torch_cache"))
    backbone = resnet50(weights=weights)
    backbone.fc = torch.nn.Identity()
    backbone.eval()
    transform = weights.transforms()
    cache = args.work / "embeddings"
    cache.mkdir(exist_ok=True)
    patches, failures = [], []
    for p in selected:
        output = cache / (p["case_uuid"] + ".npz")
        if output.exists():
            print("Cached", p["patient_id"], flush=True)
            continue
        slide_vectors, patient_patches = [], []
        for row, path in sorted(available[p["case_uuid"]], key=lambda pair: pair[0]["slide_file_uuid"]):
            vectors = []
            try:
                with openslide.OpenSlide(str(path)) as slide:
                    thumb = np.asarray(slide.get_thumbnail((1024, 1024)).convert("RGB"))
                    ys, xs = np.where((thumb.mean(axis=2) < 210) & (np.ptp(thumb, axis=2) > 30))
                    seen = set()
                    for index in np.linspace(0, max(0, len(xs)-1), min(len(xs), args.patches_per_slide * 30), dtype=int):
                        x = max(0, min(int(xs[index] * slide.dimensions[0] / thumb.shape[1]), slide.dimensions[0]-224))
                        y = max(0, min(int(ys[index] * slide.dimensions[1] / thumb.shape[0]), slide.dimensions[1]-224))
                        if (x, y) in seen:
                            continue
                        seen.add((x, y))
                        patch = np.asarray(slide.read_region((x, y), 0, (224, 224)).convert("RGB"))
                        if tissue_percentage(patch) < 50:
                            continue
                        normalized = Image.fromarray(modified_macenko(patch))
                        destination = args.work / "patches" / row["slide_file_uuid"] / f"x{x}_y{y}.png"
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        normalized.save(destination)
                        with torch.inference_mode():
                            vector = backbone(transform(normalized).unsqueeze(0)).squeeze(0).numpy()
                        if not np.isfinite(vector).all():
                            raise ValueError("Nonfinite embedding")
                        vectors.append(vector)
                        patient_patches.append({"case_uuid": p["case_uuid"], "slide_file_uuid": row["slide_file_uuid"],
                            "patch_file": str(destination.relative_to(ROOT)), "x": x, "y": y, "level": 0,
                            "mpp_x": slide.properties.get("openslide.mpp-x", ""), "normalization": "modified_macenko"})
                        if len(vectors) >= args.patches_per_slide:
                            break
                if vectors:
                    slide_vectors.append(np.mean(vectors, axis=0))
            except Exception as error:
                failures.append({"slide_file_uuid": row["slide_file_uuid"], "error": str(error)})
            print(p["patient_id"], row["slide_file_uuid"], len(vectors), "patches", flush=True)
        if slide_vectors:
            np.savez(output, feature=np.mean(slide_vectors, axis=0), case_uuid=p["case_uuid"],
                     weights="IMAGENET1K_V2", patches_per_slide=args.patches_per_slide)
            write(cache / (p["case_uuid"] + ".csv"), patient_patches)
            patches.extend(patient_patches)
    all_patches = [r for path in sorted(cache.glob("*.csv")) for r in read(path)]
    write(args.work / "patch_manifest.csv", all_patches, ["case_uuid", "slide_file_uuid", "patch_file", "x", "y", "level", "mpp_x", "normalization"])
    (args.work / "extraction_errors.json").write_text(json.dumps(failures, indent=2))
    print("Patient embeddings:", len(list(cache.glob("*.npz"))))


def metrics(labels, probability):
    predicted = (probability >= .5).astype(int)
    return {"balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
            "roc_auc": float(roc_auc_score(labels, probability)),
            "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
            "classification_report": classification_report(labels, predicted, labels=[0, 1], output_dict=True, zero_division=0)}


def compare(args):
    """Controlled modality ablation without retraining or overwriting the GAT."""
    rows = [r for r in read(args.work / "patients.csv")
            if (args.work / "embeddings" / (r["case_uuid"] + ".npz")).exists()]
    images = np.array([np.load(args.work / "embeddings" / (r["case_uuid"] + ".npz"))["feature"] for r in rows])
    labels = np.array([int(r["label"]) for r in rows])
    splits = {name: np.array([i for i, r in enumerate(rows) if r["split"] == name], dtype=int)
              for name in ("train", "validation", "test")}
    for name, indices in splits.items():
        if set(labels[indices]) != {0, 1}:
            raise ValueError(f"{name} needs both classes")
    training = splits["train"]
    clinical = clinical_matrix(rows)
    fused, transforms = fuse(images[training], images, clinical[training], clinical)
    blocks = {"image_only": fused[:, :images.shape[1]],
              "clinical_only": fused[:, images.shape[1]:], "fused": fused}
    destination = args.work / "comparison"
    destination.mkdir(exist_ok=True)
    report = {"patients": len(rows), "split_counts": {name: len(indices) for name, indices in splits.items()},
              "labels": {"0": "ductal", "1": "lobular"},
              "protocol": "Fixed logistic regression C=1, balanced class weights, threshold 0.5; training-only transforms. No test-based tuning.",
              "models": {}}
    predictions, summary = [], []
    for name, features in blocks.items():
        model = LogisticRegression(C=1., class_weight="balanced", max_iter=2000, random_state=42)
        model.fit(features[training], labels[training])
        report["models"][name] = {}
        for split in ("validation", "test"):
            indices = splits[split]
            probability = model.predict_proba(features[indices])[:, 1]
            result = metrics(labels[indices], probability)
            report["models"][name][split] = result
            summary.append({"model": name, "split": split, "patients": len(indices),
                            "balanced_accuracy": result["balanced_accuracy"], "roc_auc": result["roc_auc"]})
            predictions.extend({"model": name, "case_uuid": rows[i]["case_uuid"], "split": split,
                                "label": int(labels[i]), "lobular_probability": float(score)}
                               for i, score in zip(indices, probability))
        joblib.dump(model, destination / (name + ".joblib"))
    # Include existing GAT results only when held-out patients/labels match exactly.
    previous = args.work / "predictions.csv"
    if previous.exists():
        saved = read(previous)
        expected = {(r["case_uuid"], r["split"], r["label"]) for r in rows if r["split"] in {"validation", "test"}}
        actual = {(r["case_uuid"], r["split"], r["label"]) for r in saved}
        if actual == expected and len(saved) == len(expected):
            report["models"]["gat"] = {}
            for split in ("validation", "test"):
                subset = [r for r in saved if r["split"] == split]
                result = metrics(np.array([int(r["label"]) for r in subset]),
                                 np.array([float(r["lobular_probability"]) for r in subset]))
                report["models"]["gat"][split] = result
                summary.append({"model": "gat", "split": split, "patients": len(subset),
                                "balanced_accuracy": result["balanced_accuracy"], "roc_auc": result["roc_auc"]})
            predictions.extend({"model": "gat", **r} for r in saved)
        else:
            report["gat_note"] = "Omitted: saved GAT held-out cohort differs; retrain before comparing."
    joblib.dump(transforms, destination / "transforms.joblib")
    (destination / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    write(destination / "predictions.csv", predictions)
    write(destination / "summary.csv", summary)
    write(destination / "patients.csv", rows)
    for r in summary:
        print(f"{r['model']:14} {r['split']:10} balanced_accuracy={r['balanced_accuracy']:.3f} AUC={r['roc_auc']:.3f}")


def train(args):
    import torch
    from torch_geometric.nn import GATConv

    torch.manual_seed(42)
    torch.set_num_threads(4)
    validation_only = getattr(args, "validation_only", False)
    output = getattr(args, "run_dir", None) or args.work
    output.mkdir(parents=True, exist_ok=True)
    evaluation_splits = ("validation",) if validation_only else ("validation", "test")
    rows = [r for r in read(args.work / "patients.csv")
            if (not validation_only or r["split"] != "test")
            and (args.work / "embeddings" / (r["case_uuid"] + ".npz")).exists()]
    images = np.array([np.load(args.work / "embeddings" / (r["case_uuid"] + ".npz"))["feature"] for r in rows])
    labels = np.array([int(r["label"]) for r in rows])
    splits = {name: np.array([i for i, r in enumerate(rows) if r["split"] == name], dtype=int) for name in ("train", "validation", "test")}
    for name in ("train", *evaluation_splits):
        indices = splits[name]
        if set(labels[indices]) != {0, 1}:
            raise ValueError(f"{name} needs embeddings from both classes")
    training = splits["train"]
    clinical = clinical_matrix(rows)
    features, transforms = fuse(images[training], images, clinical[training], clinical)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite fused features")
    ref = features[training]
    edge, values = cosine_edges(ref, k=args.neighbours)
    x_train = torch.tensor(ref)
    y_train = torch.tensor(labels[training], dtype=torch.long)

    class PatientGAT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.project = torch.nn.Linear(features.shape[1], 64)
            self.first = GATConv(64, 16, heads=4, dropout=.2, edge_dim=1, fill_value=1.)
            self.second = GATConv(64, 2, heads=1, concat=False, edge_dim=1, fill_value=1.)

        def forward(self, x, edge_index, edge_attr):
            x = torch.nn.functional.elu(self.project(x))
            x = torch.nn.functional.elu(self.first(x, edge_index, edge_attr))
            x = torch.nn.functional.dropout(x, p=.3, training=self.training)
            return self.second(x, edge_index, edge_attr)

    model = PatientGAT()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.001)
    counts = np.bincount(labels[training], minlength=2)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(len(training)/(2*counts), dtype=torch.float32))

    def graph(indices):
        # Each query receives training neighbours and its own self-loop only.
        # No query-query edges and no query -> training messages, even across two layers.
        query_edge, query_values = cosine_edges(ref, features[indices], args.neighbours)
        return (torch.tensor(np.concatenate([ref, features[indices]])),
                torch.tensor(np.concatenate([edge, query_edge], axis=1)),
                torch.tensor(np.concatenate([values, query_values])))

    validation_graph = graph(splits["validation"])
    best_loss, best, stale, history = float("inf"), None, 0, []
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(x_train, torch.tensor(edge), torch.tensor(values)), y_train)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_logits = model(*validation_graph)[len(training):]
            val_loss = torch.nn.functional.cross_entropy(val_logits, torch.tensor(labels[splits["validation"]], dtype=torch.long)).item()
        history.append({"epoch": epoch+1, "train_loss": float(loss.item()), "validation_loss": val_loss})
        if val_loss < best_loss:
            best_loss, best, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= 20:
            break
    model.load_state_dict(best)
    model.eval()
    report = {"labels": {"0": "ductal", "1": "lobular"}, "patients": len(rows),
              "split_counts": {name: len(indices) for name, indices in splits.items()},
              "setting": "inductive: held-out nodes receive training messages only", "neighbours": args.neighbours,
              "normalization": "modified Macenko, external demo reference", "weights": "ResNet50 IMAGENET1K_V2",
              "clinical_features": ["age in years", "race", "ethnicity"], "evaluation": {}, "baselines": {},
              "validation_only": validation_only, "best_validation_loss": best_loss,
              "best_epoch": min(history, key=lambda r: r["validation_loss"])["epoch"]}
    predictions = []
    baseline = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42).fit(features[training], labels[training])
    for name in evaluation_splits:
        indices = splits[name]
        with torch.inference_mode():
            probability = model(*graph(indices))[len(training):].softmax(dim=1)[:, 1].numpy()
        report["evaluation"][name] = metrics(labels[indices], probability)
        report["baselines"][name] = metrics(labels[indices], baseline.predict_proba(features[indices])[:, 1])
        predictions.extend({"case_uuid": rows[i]["case_uuid"], "split": name, "label": int(labels[i]),
                            "lobular_probability": float(score)} for i, score in zip(indices, probability))
    torch.save({"state_dict": best, "input_dim": features.shape[1], "hidden_dim": 64, "heads": 4,
                "training_features": ref, "training_case_uuids": [rows[i]["case_uuid"] for i in training],
                "edge_index": edge, "edge_attr": values, "report": report}, output / "gat.pt")
    joblib.dump(transforms, output / "transforms.joblib")
    joblib.dump(baseline, output / "fusion_baseline.joblib")
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    (output / "history.json").write_text(json.dumps(history, indent=2))
    write(output / "predictions.csv", predictions)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "extract", "train", "compare"])
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--status-csv", type=Path, default=ROOT / "data/metadata/download_status_500.csv",
                        help="download validation log used by extract")
    parser.add_argument("--run-dir", type=Path, help="separate training output directory")
    parser.add_argument("--validation-only", action="store_true", help="exclude test patients entirely during training and evaluation")
    parser.add_argument("--patients-per-group", type=int, default=3, help="pilot cap per split and class")
    parser.add_argument("--patches-per-slide", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--neighbours", type=int, default=5)
    args = parser.parse_args()
    if min(args.patients_per_group, args.patches_per_slide, args.epochs, args.neighbours) < 1:
        parser.error("All numeric options must be positive")
    try:
        {"prepare": lambda a: prepare(a.work), "extract": extract, "train": train, "compare": compare}[args.command](args)
    except (ImportError, ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Cannot {args.command}: {error}\n")


if __name__ == "__main__":
    main()
