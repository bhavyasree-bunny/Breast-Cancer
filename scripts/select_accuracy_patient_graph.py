"""Select among saved patient graph candidates by validation accuracy, then evaluate test once."""
import argparse
import csv
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import torch

from multimodal_features import write
from patient_graph_model import PatientGraphGAT, evaluate, filtered_edges, load_data, make_graph


def ensemble_probability(output, config, seeds, features, indices):
    reference = features[indices["train"]]
    graph = make_graph(reference, features[indices["validation"]], config["k"], config["minimum"])
    probabilities = []
    for seed in seeds:
        checkpoint = torch.load(output / f"config{config['config']}_seed{seed}.pt", map_location="cpu")
        model = PatientGraphGAT(checkpoint["input_dim"], dropout=config["dropout"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.inference_mode():
            probabilities.append(model(*graph)[len(reference):].softmax(1)[:, 1].numpy())
    return np.mean(probabilities, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory is not empty; choose a fresh destination")
    args.output.mkdir(parents=True, exist_ok=True)

    # Candidate selection uses only train and validation features.
    rows, labels, splits, features, transforms = load_data(args.work)
    seed_info = json.loads((args.candidates / "metrics.json").read_text())
    seeds = seed_info["seeds"]
    search_rows = list(csv.DictReader((args.candidates / "search.csv").open(newline="")))
    configs, validation_rows = [], []
    for row in search_rows:
        config_id = int(row["config"])
        checkpoint = torch.load(args.candidates / f"config{config_id}_seed{seeds[0]}.pt", map_location="cpu")
        config = {"config": config_id, **checkpoint["config"]}
        probability = ensemble_probability(args.candidates, config, seeds, features, splits)
        measured = evaluate(labels[splits["validation"]], probability)
        configs.append((config, probability, measured))
        validation_rows.append({"config": config_id, "k": config["k"], "minimum": config["minimum"],
            "lr": config["lr"], "validation_accuracy": measured["accuracy"],
            "validation_balanced_accuracy": measured["balanced_accuracy"],
            "validation_macro_f1": measured["macro_f1"], "validation_log_loss": measured["log_loss"]})
    selected, validation_probability, validation_metrics = max(
        configs, key=lambda item: (item[2]["accuracy"], item[2]["macro_f1"], -item[2]["log_loss"]))
    write(args.output / "validation_search.csv", validation_rows)

    # Test data is loaded only after validation selection is frozen.
    all_rows, all_labels, all_splits, all_features, all_transforms = load_data(args.work, include_test=True)
    if [r["case_uuid"] for r in rows] != [r["case_uuid"] for r in all_rows if r["split"] != "test"]:
        raise ValueError("Patient ordering changed between validation and test loads")
    reference = all_features[all_splits["train"]]
    test_graph = make_graph(reference, all_features[all_splits["test"]], selected["k"], selected["minimum"])
    test_probabilities = []
    for seed in seeds:
        source = args.candidates / f"config{selected['config']}_seed{seed}.pt"
        checkpoint = torch.load(source, map_location="cpu")
        model = PatientGraphGAT(checkpoint["input_dim"], dropout=selected["dropout"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.inference_mode():
            test_probabilities.append(model(*test_graph)[len(reference):].softmax(1)[:, 1].numpy())
        shutil.copy2(source, args.output / source.name)
    test_probability = np.mean(test_probabilities, axis=0)
    test_metrics = evaluate(all_labels[all_splits["test"]], test_probability)

    edges, scores = filtered_edges(reference, k=selected["k"], minimum=selected["minimum"])
    write(args.output / "graph_edges.csv", [{"source": all_rows[all_splits["train"][s]]["patient_id"],
        "target": all_rows[all_splits["train"][t]]["patient_id"], "cosine_similarity": float(v[0])}
        for (s, t), v in zip(edges.T, scores)], ["source", "target", "cosine_similarity"])
    write(args.output / "graph_nodes.csv", [{"patient_id": all_rows[i]["patient_id"],
        "label": int(all_labels[i])} for i in all_splits["train"]])
    np.savez(args.output / "training_graph.npz", features=reference, edge_index=edges, edge_attr=scores)
    joblib.dump(all_transforms, args.output / "transforms.joblib")
    shutil.copy2(args.work / "patients.csv", args.output / "patients.csv")

    predictions = []
    for split, indices, probabilities in (("validation", all_splits["validation"], validation_probability),
                                          ("test", all_splits["test"], test_probability)):
        predictions.extend({"patient_id": all_rows[i]["patient_id"], "case_uuid": all_rows[i]["case_uuid"],
            "split": split, "label": int(all_labels[i]), "prediction": int(p >= .5),
            "lobular_probability": float(p)} for i, p in zip(indices, probabilities))
    write(args.output / "predictions.csv", predictions)
    selected_summary = {"config": selected["config"], **selected,
        "validation_accuracy": validation_metrics["accuracy"],
        "validation_macro_f1": validation_metrics["macro_f1"],
        "validation_balanced_accuracy": validation_metrics["balanced_accuracy"]}
    report = {"selected": selected_summary, "seeds": seeds, "validation": validation_metrics,
        "test": test_metrics, "test_evaluated": True,
        "selection_metric": "validation accuracy, then macro-F1, then lower log-loss",
        "test_note": "The accuracy-based follow-up was prompted after the original test result was inspected. Candidate choice used validation metrics, but this comparison is post-hoc, not confirmatory; this cohort also appeared in prior experiments."}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "protocol.json").write_text(json.dumps({
        "criterion": report["selection_metric"], "threshold": 0.5, "seeds": seeds,
        "cohort_sha256": __import__("hashlib").sha256((args.work / "patients.csv").read_bytes()).hexdigest(),
        "test_note": report["test_note"]}, indent=2) + "\n")
    print("Selected by validation accuracy:", selected_summary)
    print("Test accuracy:", test_metrics["accuracy"], "macro F1:", test_metrics["macro_f1"],
          "ROC-AUC:", test_metrics["roc_auc"], "confusion:", test_metrics["confusion_matrix"])
    print("Saved accuracy-selected experiment to", args.output)


if __name__ == "__main__":
    main()
