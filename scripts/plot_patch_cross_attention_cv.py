"""Create report figures from patch cross-attention CV predictions."""
import argparse
import csv
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/breast_cancer_mplconfig')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def load_predictions(run_dir, name):
    rows = read_csv(run_dir / f'{name}_predictions.csv')
    labels = np.asarray([int(row['label']) for row in rows])
    probabilities = np.asarray([float(row['lobular_probability']) for row in rows])
    predictions = np.asarray([int(row['prediction']) for row in rows])
    if len(set(row['case_uuid'] for row in rows)) != len(rows):
        raise ValueError(f'Duplicate patients in {name} predictions')
    return rows, labels, probabilities, predictions


def plot_confusion(output, models, n):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), dpi=180)
    for ax, (name, title, labels, _, predictions) in zip(axes, models):
        cm = confusion_matrix(labels, predictions, labels=[0, 1])
        row_totals = cm.sum(axis=1, keepdims=True)
        rates = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float),
                          where=row_totals != 0)
        ax.imshow(cm, cmap='Blues')
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xticks([0, 1], ['Ductal', 'Lobular'])
        ax.set_yticks([0, 1], ['Ductal', 'Lobular'])
        ax.set_xlabel('Predicted diagnosis')
        ax.set_ylabel('True diagnosis')
        threshold = cm.max() / 2 if cm.size else 0
        for i in range(2):
            for j in range(2):
                color = 'white' if cm[i, j] > threshold else '#172033'
                ax.text(j, i, f'{cm[i, j]}\n{rates[i, j]:.1%}',
                        ha='center', va='center', color=color, fontsize=11)
    fig.suptitle(f'Out-of-fold confusion matrices (n={n})', fontsize=15, y=1.02)
    fig.text(.5, -.01, 'Cells show patient count and row-wise recall. '
             'Predictions use each fold’s inner-validation threshold.',
             ha='center', fontsize=9, color='#4b5563')
    fig.tight_layout()
    fig.savefig(output / 'confusion_matrices_no_graph_vs_k7.png',
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_roc(output, models, n):
    fig, ax = plt.subplots(figsize=(7.2, 6), dpi=180)
    colors = ['#2878B5', '#D9534F']
    for color, (name, title, labels, probabilities, _) in zip(colors, models):
        fpr, tpr, _ = roc_curve(labels, probabilities)
        auc = roc_auc_score(labels, probabilities)
        ax.plot(fpr, tpr, lw=2.5, color=color, label=f'{title} (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], linestyle='--', color='#6b7280', lw=1.2,
            label='Chance (AUC=0.500)')
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel='False positive rate',
           ylabel='True positive rate', title=f'ROC curves from out-of-fold predictions (n={n})')
    ax.grid(alpha=.2)
    ax.legend(loc='lower right', frameon=True)
    fig.tight_layout()
    fig.savefig(output / 'roc_curves_no_graph_vs_k7.png',
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_precision_recall(output, models, n):
    fig, ax = plt.subplots(figsize=(7.2, 6), dpi=180)
    colors = ['#2878B5', '#D9534F']
    prevalence = float(models[0][2].mean())
    for color, (name, title, labels, probabilities, _) in zip(colors, models):
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        ap = average_precision_score(labels, probabilities)
        ax.plot(recall, precision, lw=2.5, color=color,
                label=f'{title} (AP={ap:.3f})')
    ax.axhline(prevalence, linestyle='--', color='#6b7280', lw=1.2,
               label=f'Positive prevalence ({prevalence:.3f})')
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel='Recall (lobular sensitivity)',
           ylabel='Precision (lobular positive predictive value)',
           title=f'Precision–recall curves from out-of-fold predictions (n={n})')
    ax.grid(alpha=.2)
    ax.legend(loc='lower left', frameon=True)
    fig.tight_layout()
    fig.savefig(output / 'precision_recall_curves_no_graph_vs_k7.png',
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_metrics(output, metrics, models, n):
    names = [item[0] for item in models]
    display = ['No graph', 'k=7 graph']
    keys = ['accuracy', 'balanced_accuracy', 'macro_f1', 'roc_auc']
    labels = ['Accuracy', 'Balanced accuracy', 'Macro-F1', 'ROC-AUC']
    colors = ['#2878B5', '#D9534F']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), dpi=180,
                             gridspec_kw={'width_ratios': [2.5, 1]})
    ax = axes[0]
    x = np.arange(len(keys)); width = .34
    for index, (name, label, color) in enumerate(zip(names, display, colors)):
        values = [metrics['results'][name][key] for key in keys]
        bars = ax.bar(x + (index - .5) * width, values, width,
                      label=label, color=color, alpha=.9)
        ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=2)
    ax.set_xticks(x, labels, rotation=15, ha='right')
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Score')
    ax.set_title('Classification metrics')
    ax.grid(axis='y', alpha=.2)
    ax.legend()
    loss_ax = axes[1]
    losses = [metrics['results'][name]['log_loss'] for name in names]
    bars = loss_ax.bar(display, losses, color=colors, alpha=.9, width=.58)
    loss_ax.bar_label(bars, fmt='%.3f', fontsize=9, padding=3)
    loss_ax.set_ylim(0, max(losses) * 1.2)
    loss_ax.set_ylabel('Log loss (lower is better)')
    loss_ax.set_title('Probability error')
    loss_ax.grid(axis='y', alpha=.2)
    fig.suptitle(f'No-graph vs k=7 patch-attention model (n={n})', fontsize=15, y=1.02)
    fig.text(.5, -.02, 'Metrics are computed from pooled out-of-fold predictions; '
             'thresholded metrics use fold-specific thresholds selected on inner validation.',
             ha='center', fontsize=9, color='#4b5563')
    fig.tight_layout()
    fig.savefig(output / 'performance_metrics_no_graph_vs_k7.png',
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    metrics = json.loads((args.run_dir / 'metrics.json').read_text())
    output = args.run_dir / 'figures'
    output.mkdir(exist_ok=True)
    available = metrics.get('results', {})
    if {'no_graph', 'k7_graph'} <= set(available):
        model_specs = [('no_graph', 'No graph'), ('k7_graph', 'k=7 graph')]
    elif {'no_graph_weighted', 'k7_graph_weighted'} <= set(available):
        model_specs = [('no_graph_weighted', 'No graph, weighted loss'),
                       ('k7_graph_weighted', 'k=7 graph, weighted loss')]
    else:
        raise ValueError('Run metrics must contain no-graph and k=7 graph results')
    models = []
    for name, title in model_specs:
        _, labels, probabilities, predictions = load_predictions(args.run_dir, name)
        models.append((name, title, labels, probabilities, predictions))
    n = len(models[0][2])
    if any(len(item[2]) != n for item in models):
        raise ValueError('Model prediction files have different patient counts')
    plot_confusion(output, models, n)
    plot_roc(output, models, n)
    plot_precision_recall(output, models, n)
    plot_metrics(output, metrics, models, n)
    graph = args.run_dir / 'fold1_k7_graph.png'
    if graph.exists():
        shutil.copy2(graph, output / 'patient_level_graph_fold1_k7.png')
    print('Saved report figures to', output)


if __name__ == '__main__':
    main()
