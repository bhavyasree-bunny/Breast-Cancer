"""Create metric and patient graph figures from saved patient graph artifacts; no HTML report."""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/breast-cancer-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.output / "metrics.json").read_text())
    predictions = list(csv.DictReader((args.output / "predictions.csv").open(newline="")))
    figures = args.output / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, split in zip(axes, ("validation", "test")):
        matrix = np.asarray(report[split]["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center", fontsize=16)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Ductal", "Lobular"],
               yticklabels=["Ductal", "Lobular"], xlabel="Predicted", ylabel="Actual", title=split.title())
    fig.tight_layout(); fig.savefig(figures / "confusion_matrices.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    for split in ("validation", "test"):
        selected = [p for p in predictions if p["split"] == split]
        fpr, tpr, _ = roc_curve([int(p["label"]) for p in selected],
                                [float(p["lobular_probability"]) for p in selected])
        ax.plot(fpr, tpr, label=f"{split.title()} AUC={report[split]['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Ductal vs lobular ROC")
    ax.legend(); fig.tight_layout(); fig.savefig(figures / "roc_curve.png", dpi=180); plt.close(fig)

    graph = np.load(args.output / "training_graph.npz")
    centered = graph["features"] - graph["features"].mean(axis=0)
    u, singular, _ = np.linalg.svd(centered, full_matrices=False)
    position = u[:, :2] * singular[:2]
    nodes = list(csv.DictReader((args.output / "graph_nodes.csv").open(newline="")))
    edges = list(csv.DictReader((args.output / "graph_edges.csv").open(newline="")))
    attention = list(csv.DictReader((args.output / "attention_scores.csv").open(newline="")))
    index = {node["patient_id"]: i for i, node in enumerate(nodes)}
    layer2 = {(e["source_patient"], e["target_patient"]): float(e["mean_attention"])
              for e in attention if e["layer"] == "2"}
    fig, ax = plt.subplots(figsize=(12, 9))
    for edge in edges:
        source, target = edge["source"], edge["target"]
        score = layer2.get((source, target), 0.0)
        ax.annotate("", xy=position[index[target]], xytext=position[index[source]],
            arrowprops={"arrowstyle": "->", "color": "#64748b", "alpha": .30,
                        "linewidth": .35 + 6 * score})
    for label, color, name in ((0, "#2563eb", "Ductal"), (1, "#ea580c", "Lobular")):
        ids = [i for i, node in enumerate(nodes) if int(node["label"]) == label]
        ax.scatter(position[ids, 0], position[ids, 1], c=color, label=name, s=48, zorder=3)
    for xy, node in zip(position, nodes):
        ax.text(*xy, node["patient_id"], fontsize=5, alpha=.85)
    ax.set(title="Patient representation and GAT graph (line width = layer 2 attention)",
           xlabel="PCA component 1", ylabel="PCA component 2")
    ax.legend(); fig.tight_layout(); fig.savefig(figures / "patient_gat_graph.png", dpi=220); plt.close(fig)

    second = [e for e in attention if e["layer"] == "2"
              and e["source_patient"] != e["target_patient"]]
    incoming = {}
    for edge in second:
        incoming.setdefault(edge["target_patient"], []).append(edge)
    strongest = [e for group in incoming.values()
                 for e in sorted(group, key=lambda row: float(row["mean_attention"]), reverse=True)[:2]]
    fig, ax = plt.subplots(figsize=(12, 9))
    for edge in strongest:
        source, target = edge["source_patient"], edge["target_patient"]
        ax.annotate("", xy=position[index[target]], xytext=position[index[source]],
            arrowprops={"arrowstyle": "->", "color": "#7c3aed", "alpha": .48,
                        "linewidth": .5 + 10 * float(edge["mean_attention"])})
    for label, color, name in ((0, "#2563eb", "Ductal"), (1, "#ea580c", "Lobular")):
        ids = [i for i, node in enumerate(nodes) if int(node["label"]) == label]
        ax.scatter(position[ids, 0], position[ids, 1], c=color, label=name, s=48, zorder=3)
    for xy, node in zip(position, nodes):
        ax.text(*xy, node["patient_id"], fontsize=5, alpha=.85)
    ax.set(title="Top two incoming layer 2 attention links per patient",
           xlabel="PCA component 1", ylabel="PCA component 2")
    ax.legend(); fig.tight_layout(); fig.savefig(figures / "gat_attention_graph.png", dpi=220); plt.close(fig)
    print("Saved figures in", figures)


if __name__ == "__main__":
    main()
