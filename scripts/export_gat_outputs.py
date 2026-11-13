"""Export learned edge-attention scores from a trained patient GAT."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from patient_graph_model import PatientGraphGAT, load_data, make_graph


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_attention(output, report, rows, splits, features):
    selected = report["selected"]
    reference = features[splits["train"]]
    graph = make_graph(reference, None, selected["k"], selected["minimum"])
    edge_index = graph[1]
    values = {}
    torch.set_num_threads(4)
    for seed in report["seeds"]:
        checkpoint = torch.load(
            output / f"config{selected['config']}_seed{seed}.pt", map_location="cpu")
        model = PatientGraphGAT(checkpoint["input_dim"], dropout=selected["dropout"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        x, edges, edge_scores = graph
        with torch.inference_mode():
            x = torch.nn.functional.elu(model.project(x))
            for layer_no, (layer, norm) in enumerate(zip(model.layers, model.norms), start=1):
                message, (attended_edges, alpha) = layer(
                    x, edges, edge_scores, return_attention_weights=True)
                for (source, target), heads in zip(attended_edges.T.tolist(), alpha.tolist()):
                    values.setdefault((layer_no, source, target), []).append(float(np.mean(heads)))
                x = torch.nn.functional.elu(norm(x + message))
                x = torch.nn.functional.dropout(x, p=model.dropout, training=False)

    node_ids = [rows[i]["patient_id"] for i in splits["train"]]
    cosine = {(int(source), int(target)): float(score[0])
              for (source, target), score in zip(edge_index.T.tolist(), graph[2].tolist())}
    records = []
    for (layer, source, target), scores in sorted(values.items()):
        records.append({"layer": layer, "source_patient": node_ids[source],
            "target_patient": node_ids[target], "cosine_similarity": cosine.get((source, target), ""),
            "mean_attention": float(np.mean(scores)), "std_across_seeds": float(np.std(scores)),
            "seeds": len(scores)})
    write_csv(output / "attention_scores.csv", records,
        ["layer", "source_patient", "target_patient", "cosine_similarity",
         "mean_attention", "std_across_seeds", "seeds"])
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.output / "metrics.json").read_text())
    if not report.get("test_evaluated"):
        raise ValueError("Model output has no test evaluation")
    rows, _, splits, features, _ = load_data(args.work, include_test=True)
    records = export_attention(args.output, report, rows, splits, features)
    print("Attention scores:", args.output / "attention_scores.csv")
    print("Attention edges:", len(records))
    for split in ("validation", "test"):
        metrics = report[split]
        macro = metrics["classification_report"]["macro avg"]
        print(split, "accuracy", round(metrics["accuracy"], 4),
              "precision_macro", round(macro["precision"], 4),
              "recall_macro", round(macro["recall"], 4),
              "f1_macro", round(metrics["macro_f1"], 4),
              "roc_auc", round(metrics["roc_auc"], 4),
              "confusion_matrix", metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
