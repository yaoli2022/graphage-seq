"""
plot_mae_by_age_group.py
------------------------
Generates a grouped bar chart of MAE by age group for 4 models:
  1. Horvath
  2. ResnetAge
  3. PNA-GNN
  4. PNA-GNN + Stat. Features (Ours)

Run on Spartan:
  cd /data/gpfs/projects/punim2698/yao14/graphage_seq
  source /data/projects/punim2698/yao14/venvs/ham_env/bin/activate
  python plot_mae_by_age_group.py

Output:
  figures/mae_by_age_group.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Paths — adjust BASE_DIR if needed
# ============================================================
BASE_DIR   = '/data/gpfs/projects/punim2698/yao14/graphage_seq'
OUTPUT_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = [
    {
        'name':  'Horvath',
        'csv':   os.path.join(BASE_DIR, 'baselines/horvath_predictions.csv'),
        'color': '#E53935',    # Google Red 600
        'ls':    '-',
    },
    {
        'name':  'ResnetAge',
        'csv':   os.path.join(BASE_DIR, 'baselines/resnetage_predictions.csv'),
        'color': '#FB8C00',    # Google Orange 600
        'ls':    '--',
    },
    {
        'name':  'PNA-GNN',
        'csv':   os.path.join(BASE_DIR, 'checkpoint/test_predictions_original.csv'),
        'color': '#1E88E5',    # Google Blue 600
        'ls':    ':',
    },
    {
        'name':  'PNA-GNN + Stat. Features (Ours)',
        'csv':   os.path.join(BASE_DIR, 'checkpoint_seq_no_CNN/test_predictions.csv'),
        'color': '#43A047',    # Google Green 600
        'ls':    '-.',
    },
]

# ============================================================
# Age group helper
# ============================================================
def age_group_label(age):
    if age <= 0:   return '0'
    if age <= 20:  return '0--20'
    if age <= 45:  return '20--45'
    if age <= 55:  return '45--55'
    if age <= 65:  return '55--65'
    if age <= 75:  return '65--75'
    if age <= 80:  return '75--80'
    return '80+'

AGE_LABELS = ['0', '0--20', '20--45', '45--55',
              '55--65', '65--75', '75--80', '80+']

# ============================================================
# Load data and compute per-group MAE
# ============================================================
results = []
for m in MODELS:
    df  = pd.read_csv(m['csv'])
    mae = np.mean(np.abs(df['true_age'] - df['predicted_age']))

    group_errors = {lbl: [] for lbl in AGE_LABELS}
    for _, row in df.iterrows():
        lbl = age_group_label(row['true_age'])
        group_errors[lbl].append(abs(row['predicted_age'] - row['true_age']))

    mae_vals = [np.mean(group_errors[lbl]) if group_errors[lbl] else 0
                for lbl in AGE_LABELS]
    counts   = [len(group_errors[lbl]) for lbl in AGE_LABELS]

    results.append({
        'name':     m['name'],
        'color':    m['color'],
        'ls':       m['ls'],
        'mae':      mae,
        'mae_vals': mae_vals,
        'counts':   counts,
    })

# ============================================================
# Plot
# ============================================================
n_models = len(results)
n_groups = len(AGE_LABELS)
bar_width = 0.18
x = np.arange(n_groups)

fig, ax = plt.subplots(figsize=(13, 5.5))

for i, r in enumerate(results):
    offset = (i - n_models / 2 + 0.5) * bar_width
    bars = ax.bar(
        x + offset,
        r['mae_vals'],
        width=bar_width,
        color=r['color'],
        alpha=0.88,
        edgecolor='white',
        linewidth=0.5,
        label=f"{r['name']}  (MAE={r['mae']:.3f})",
    )

# Overall MAE horizontal lines
for r in results:
    ax.axhline(
        r['mae'],
        color=r['color'],
        linestyle=r['ls'],
        linewidth=1.4,
        alpha=0.85,
    )



ax.set_xticks(x)
ax.set_xticklabels(AGE_LABELS, fontsize=10)
ax.set_xlabel('Age Group (years)', fontsize=12)
ax.set_ylabel('Mean Absolute Error (years)', fontsize=12)
ax.set_title('Epigenetic Age Prediction: MAE by Age Group (Test Set)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9.5, loc='upper left', framealpha=0.9)
ax.set_ylim(0, max(max(r['mae_vals']) for r in results) * 1.2)
ax.grid(axis='y', alpha=0.25, linestyle='--')
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'mae_by_age_group.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {out_path}')
