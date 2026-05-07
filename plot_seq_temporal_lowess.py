"""
plot_seq_temporal_lowess.py
---------------------------
Generates a temporal analysis plot of sequence feature importance
scores vs chronological age using LOWESS regression curves.

Unlike linear regression, LOWESS (Locally Weighted Scatterplot
Smoothing) makes no assumption about the functional form of the
relationship, allowing non-linear trends to be revealed.

Linear r values are still computed and shown in the legend to
indicate the direction and approximate strength of each trend,
for consistency with the paper text.

Run on Spartan:
  cd /data/gpfs/projects/punim2698/yao14/graphage_seq
  source /data/projects/punim2698/yao14/venvs/ham_env/bin/activate
  python plot_seq_temporal_lowess.py

Output:
  checkpoint_seq_no_CNN/explainer/seq_temporal_analysis_lowess.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

# ============================================================
# Paths
# ============================================================
BASE = '/data/gpfs/projects/punim2698/yao14/graphage_seq/checkpoint_seq_no_CNN/explainer'
OUT  = BASE + '/seq_temporal_analysis_lowess.png'

# ============================================================
# Load data
# ============================================================
seq_arr  = np.load(f'{BASE}/seq_all_per_sample.npy')
ages_arr = np.load(f'{BASE}/all_ages.npy')

# ============================================================
# Feature names and colours
# ============================================================
NAMES = [
    'GC content',
    'CpG density',
    'Upstream GC',
    'Downstream GC',
    'Local A freq.',
    'Local T freq.',
    'Local C freq.',
    'Local G freq.',
]
COLORS = [
    '#2980b9',  # GC content       — blue
    '#e74c3c',  # CpG density      — red
    '#1abc9c',  # Upstream GC      — teal
    '#8e44ad',  # Downstream GC    — purple
    '#f39c12',  # Local A freq.    — orange
    '#27ae60',  # Local T freq.    — green
    '#d35400',  # Local C freq.    — dark orange
    '#95a5a6',  # Local G freq.    — grey
]

# ============================================================
# LOWESS smoothing parameter
# frac: fraction of data used in each local regression
#   lower  = more wiggly curve (captures fine structure)
#   higher = smoother curve (emphasises overall trend)
# Suggested range: 0.2 – 0.4
# ============================================================
LOWESS_FRAC = 0.3

# ============================================================
# Sort samples by age
# ============================================================
sorted_idx  = np.argsort(ages_arr)
ages_sorted = ages_arr[sorted_idx]
seq_sorted  = seq_arr[sorted_idx]

# ============================================================
# Plot
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

for i, (name, color) in enumerate(zip(NAMES, COLORS)):
    y_vals  = seq_sorted[:, i]
    nonzero = y_vals > 0.01
    x_plot  = ages_sorted[nonzero]
    y_plot  = y_vals[nonzero]

    if len(x_plot) < 5:
        continue

    # ── Scatter points ────────────────────────────────────
    ax.scatter(x_plot, y_plot, s=5, alpha=0.25, color=color)

    # ── LOWESS curve ──────────────────────────────────────
    smoothed = lowess(y_plot, x_plot,
                      frac=LOWESS_FRAC,
                      return_sorted=True)
    x_smooth = smoothed[:, 0]
    y_smooth = smoothed[:, 1]

    # ── Linear r for legend label ─────────────────────────
    slope, _, r, p, _ = stats.linregress(x_plot, y_plot)
    direction = '↑' if slope > 0 else '↓'

    # Line style based on |r| from linear fit
    linestyle = '-' if abs(r) > 0.5 else '--'

    if p < 0.01:
        label = f'{name} {direction} (r={r:.2f})'
    else:
        label = f'{name} (no trend)'
        linestyle = ':'

    ax.plot(x_smooth, y_smooth,
            color=color,
            linewidth=2.2,
            alpha=0.95,
            linestyle=linestyle,
            label=label,
            zorder=5)

# ============================================================
# Formatting
# ============================================================
ax.set_xlabel('Chronological Age (years)', fontsize=12)
ax.set_ylabel('Importance Score', fontsize=12)
ax.set_title(
    'Temporal Analysis: Sequence Feature Importance vs Chronological Age\n'
    f'LOWESS curves (frac={LOWESS_FRAC}); '
    'solid: |r| > 0.5, dashed: |r| ≤ 0.5; '
    '↑ increasing, ↓ decreasing with age',
    fontsize=11, fontweight='bold'
)
ax.legend(fontsize=8.5, ncol=2, loc='upper right',
          framealpha=0.9, edgecolor='#cccccc')
ax.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {OUT}')
