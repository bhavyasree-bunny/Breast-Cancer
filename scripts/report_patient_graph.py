"""Export figures and an HTML validation report from saved patient graph results (no fitting)."""
import csv
import argparse
import html
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/breast-cancer-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT/'data/models/patient_graph_cross_attention_run2'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=WORK)
    work = parser.parse_args().run_dir
    report = json.loads((work/'metrics.json').read_text())
    predictions = list(csv.DictReader((work/'predictions.csv').open()))
    figures = work/'figures'
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, split in zip(axes, ['validation', 'test']):
        matrix = np.array(report[split]['confusion_matrix'])
        ax.imshow(matrix, cmap='Blues', vmin=0, vmax=max(matrix.max(), 1))
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha='center', va='center', fontsize=18)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Ductal', 'Lobular'],
               yticklabels=['Ductal', 'Lobular'], xlabel='Predicted', ylabel='Actual', title=split.title())
    fig.tight_layout(); fig.savefig(figures/'confusion_matrices.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    for split in ['validation', 'test']:
        rows = [r for r in predictions if r['split'] == split]
        fpr, tpr, _ = roc_curve([int(r['label']) for r in rows], [float(r['lobular_probability']) for r in rows])
        ax.plot(fpr, tpr, label=f"{split.title()} (AUC {report[split]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title='Lobular versus ductal ROC')
    ax.legend(); fig.tight_layout(); fig.savefig(figures/'roc.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    index = report['selected']['config']
    for seed in report['seeds']:
        history = json.loads((work/f'config{index}_seed{seed}_history.json').read_text())
        axes[0].plot([r['epoch'] for r in history], [r['train_loss'] for r in history], label=f'Seed {seed}')
        axes[1].plot([r['epoch'] for r in history], [r['balanced_accuracy'] for r in history], label=f'Seed {seed}')
    axes[0].set(title='Weighted training loss', xlabel='Epoch', ylabel='Cross entropy')
    axes[1].set(title='Validation balanced accuracy', xlabel='Epoch', ylabel='Balanced accuracy', ylim=(0, 1))
    for ax in axes: ax.legend()
    fig.tight_layout(); fig.savefig(figures/'learning_curves.png', dpi=180); plt.close(fig)
    graph = np.load(work/'training_graph.npz')
    edges, scores, features = graph['edge_index'], graph['edge_attr'], graph['features']
    nodes = list(csv.DictReader((work/'graph_nodes.csv').open()))
    centered = features-features.mean(axis=0)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    position = u[:, :2]*s[:2]
    fig, ax = plt.subplots(figsize=(12, 9))
    for (source, target), score in zip(edges.T, scores):
        ax.annotate('', xy=position[target], xytext=position[source],
            arrowprops={'arrowstyle': '->', 'color': '#64748b', 'alpha': .16,
                        'linewidth': max(.2, float(score[0])*1.5)})
    for label, color, name in [(0, '#2563eb', 'Ductal'), (1, '#ea580c', 'Lobular')]:
        indices = [i for i, n in enumerate(nodes) if int(n['label']) == label]
        ax.scatter(position[indices, 0], position[indices, 1], c=color, label=name, s=45, zorder=3)
    for xy, node in zip(position, nodes): ax.text(*xy, node['patient_id'], fontsize=5, alpha=.85)
    ax.set(title='Actual selected GAT training edges; arrows: neighbour → recipient',
           xlabel='Training feature PCA axis 1', ylabel='Training feature PCA axis 2')
    ax.legend(); fig.tight_layout(); fig.savefig(figures/'patient_graph.png', dpi=220); plt.close(fig)
    table = ''.join(f"<tr><td>{split.title()}</td><td>{report[split]['patients']}</td><td>{report[split]['accuracy']:.1%}</td><td>{report[split]['balanced_accuracy']:.1%}</td><td>{report[split]['macro_f1']:.3f}</td><td>{report[split]['roc_auc']:.3f}</td></tr>" for split in ['validation', 'test'])
    test = report.get('test')
    test_summary = (f"Test accuracy {test['accuracy']:.1%} ({test['correct']}/{test['patients']}), "
                    f"balanced accuracy {test['balanced_accuracy']:.1%}, ROC-AUC {test['roc_auc']:.3f}. "
                    "This test split was inspected and is not fresh external validation.") if test else "Test split was not evaluated."
    fusion = report.get('architecture', 'Cross-attention fusion followed by residual GAT.')
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Patient graph validation report</title>
    <style>body{{font:16px system-ui;max-width:1050px;margin:40px auto;padding:20px;color:#172033}}table{{border-collapse:collapse}}td,th{{padding:12px;border:1px solid #ccd5df}}img{{max-width:100%}}p{{line-height:1.6}}</style>
    <h1>Patient graph model: validation report</h1><p>{html.escape(fusion)} → ductal/lobular prediction.</p>
    <p>Eight predefined configurations, three seeds, fixed patient splits. Selection uses mean seed validation balanced accuracy, then macro-F1 and loss. Fixed probability threshold: 0.5. Test features were loaded only after selection. {html.escape(test_summary)}</p>
    <table><tr><th>Split</th><th>Patients</th><th>Accuracy</th><th>Balanced accuracy</th><th>Macro-F1</th><th>ROC-AUC</th></tr>{table}</table>
    <p>These are small-sample exploratory results. Validation metrics are optimistic after selection; ROC-AUC does not establish thresholded accuracy or calibration.</p>
    <p>Selected configuration: {html.escape(str(report['selected']))}. Similarity filtering at 0.25 did not change the candidate graphs in this search. Residual pathways preserve patient information; improvements cannot be attributed solely to graph reasoning without a graph-ablation comparison.</p>
    <h2>Confusion matrices</h2><img src="figures/confusion_matrices.png" alt="Actual and predicted class counts">
    <h2>ROC curves</h2><img src="figures/roc.png" alt="Validation and test ROC curves">
    <h2>Learning curves</h2><img src="figures/learning_curves.png" alt="Per-seed training and validation curves">
    <h2>Patient graph</h2><p>Actual saved edges and cosine scores, with PCA used only for visual positions. Self-loops are omitted. Labels colour the visualization and were not used to choose edges. Similarity edges are not explanations or clinical relationships.</p><img src="figures/patient_graph.png" alt="Actual patient training graph">
    <p><a href="metrics.json">Metrics</a> · <a href="predictions.csv">Per-patient predictions</a> · <a href="search.csv">Validation search</a> · <a href="protocol.json">Protocol</a></p></html>'''
    (work/'report.html').write_text(text)
    print(work/'report.html')


if __name__ == '__main__': main()
