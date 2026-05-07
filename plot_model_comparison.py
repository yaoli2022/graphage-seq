"""
plot_model_comparison.py
------------------------
Reads pre-saved test_predictions.csv from each model checkpoint
and generates a publication-quality 3-panel scatter plot.

Titles are deliberately model-descriptive rather than named after
any specific prior work, in line with the paper's framing that
positions this as a field-level contribution.

Panel titles:
  (a) GNN Baseline (No Sequence)
  (b) GNN + CNN Sequence Encoding
  (c) GNN + Statistical Features (Ours)

Run on Spartan:
  cd /data/gpfs/projects/punim2698/yao14/graphage_seq
  source /data/projects/punim2698/yao14/venvs/ham_env/bin/activate
  python plot_model_comparison.py

Output:
  /data/gpfs/projects/punim2698/yao14/graphage_seq/model_comparison_scatter.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# Paths to pre-saved predictions
# ============================================================
BASE_DIR = '/data/gpfs/projects/punim2698/yao14/graphage_seq'

MODELS = [
    {
        'csv':   os.path.join(BASE_DIR, 'checkpoint/test_predictions_original.csv'),
        'title': 'GNN Baseline\n(No Sequence)',
        'color': '#e07b54',   # warm orange for baseline
    },
    {
        'csv':   os.path.join(BASE_DIR, 'graphagewithCNN_3.26_checkpoint/test_predictions_cnn.csv'),
        'title': 'GNN + CNN\nSequence Encoding',
        'color': '#5b8db8',   # blue for CNN
    },
    {
        'csv':   os.path.join(BASE_DIR, 'checkpoint_seq_no_CNN/test_predictions.csv'),
        'title': 'GNN + Statistical\nFeatures (Ours)',
        'color': '#4bae6e',   # green for ours
    },
]

OUTPUT_PATH = os.path.join(BASE_DIR, 'model_comparison_scatter.png')

# ============================================================
# Plot
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('white')

for ax, model in zip(axes, MODELS):
    # ── Load predictions ──────────────────────────────────
    df = pd.read_csv(model['csv'])

    # Support both column naming conventions
    if 'true_age' in df.columns and 'predicted_age' in df.columns:
        truths = df['true_age'].values
        preds  = df['predicted_age'].values
    elif 'True Age' in df.columns and 'Predicted Age' in df.columns:
        truths = df['True Age'].values
        preds  = df['Predicted Age'].values
    else:
        # Fallback: assume first col = true, second col = pred
        truths = df.iloc[:, 0].values
        preds  = df.iloc[:, 1].values

    # ── Metrics ───────────────────────────────────────────
    mae  = mean_absolute_error(truths, preds)
    mse  = mean_squared_error(truths, preds)
    r2   = r2_score(truths, preds)
    errs = np.abs(preds - truths)

    # ── Fit line slope ────────────────────────────────────
    slope = np.polyfit(truths, preds, 1)[0]

    # ── Scatter (colour = absolute error) ─────────────────
    ax.set_facecolor('white')
    sc = ax.scatter(
        truths, preds,
        c=errs,
        cmap='RdYlGn_r',
        vmin=0, vmax=14,
        s=12, alpha=0.75, linewidths=0,
        zorder=3
    )

    # ── y = x reference line ──────────────────────────────
    lo = min(truths.min(), preds.min()) - 2
    hi = max(truths.max(), preds.max()) + 2
    ax.plot([lo, hi], [lo, hi],
            color='black', linestyle='--', linewidth=1.2,
            label='$y = x$', zorder=4)

    # ── Fit line ──────────────────────────────────────────
    fit_y = np.poly1d(np.polyfit(truths, preds, 1))(np.array([lo, hi]))
    ax.plot([lo, hi], fit_y,
            color=model['color'], linewidth=1.8,
            label=f'Slope = {slope:.3f}', zorder=5)

    # ── Colourbar ─────────────────────────────────────────
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Abs. Error (years)', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # ── Metrics text box ──────────────────────────────────
    textstr = (f'MAE = {mae:.3f}\n'
               f'MSE = {mse:.3f}\n'
               f'$R^2$ = {r2:.4f}')
    ax.text(0.04, 0.96, textstr,
            transform=ax.transAxes,
            fontsize=9.5,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4',
                      facecolor='white', alpha=0.85,
                      edgecolor='#cccccc'))

    # ── Axes formatting ───────────────────────────────────
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('True Chronological Age (years)', fontsize=11)
    ax.set_ylabel('Predicted Epigenetic Age (years)', fontsize=11)
    ax.set_title(model['title'], fontsize=12, fontweight='bold', pad=10)
    ax.legend(fontsize=9, loc='lower right', framealpha=0.85)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout(w_pad=2.5)
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {OUTPUT_PATH}')
