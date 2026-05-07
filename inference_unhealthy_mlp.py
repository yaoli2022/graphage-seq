"""
inference_unhealthy_mlp.py
--------------------------
Runs DeepMAge and ResnetAge on the unhealthy dataset and
merges results into existing unhealthy CSVs.

Run on Spartan:
  cd /data/gpfs/projects/punim2698/yao14/graphage_seq
  source /data/projects/punim2698/yao14/venvs/ham_env/bin/activate
  python inference_unhealthy_mlp.py

Output:
  baselines/unhealthy_predictions.csv  -- updated
  baselines/unhealthy_acceleration.csv -- updated
"""

import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# Paths
# ============================================================
BASE_DIR      = '/data/gpfs/projects/punim2698/yao14/graphage_seq'
DATA_DIR      = os.path.join(BASE_DIR, 'all-organs4/all_organs')
UNHEALTHY_DIR = os.path.join(BASE_DIR, 'unhelathy-dataset/Unhealthy Normalized')
ALTUMAGE_CPGS = os.path.join(BASE_DIR, 'graph-age/example_dependencies/multi_platform_cpgs.pkl')
BASELINES_DIR = os.path.join(BASE_DIR, 'baselines')
DEEPMAGE_CKPT = os.path.join(BASELINES_DIR, 'deepmage_best.pth')
RESNET_CKPT   = os.path.join(BASELINES_DIR, 'resnetage_best.pth')
PRED_CSV      = os.path.join(BASELINES_DIR, 'unhealthy_predictions.csv')
ACCEL_CSV     = os.path.join(BASELINES_DIR, 'unhealthy_acceleration.csv')

# ============================================================
# Reproducibility
# ============================================================
SEED         = 0
K_FOLDS      = 5
DESIRED_FOLD = 2
TOP_K_CPGS   = 1000

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Disease map
DISEASE_MAP = {
    'GSE19711':    'Ovarian Cancer',
    'GSE41037':    'Schizophrenia',
    'GSE27044':    'Osteoporosis',
    'GSE99624':    'Osteoporosis',
    'E-GEOD-44763':'Osteoporosis',
    'GSE77241':    'Osteoporosis',
}

# ============================================================
# Load AltumAge CpG list
# ============================================================
print('Loading CpG list...')
AltumAge_cpgs = np.array(pd.read_pickle(ALTUMAGE_CPGS)).tolist()
print(f'  {len(AltumAge_cpgs)} CpG sites')

# ============================================================
# Load healthy training data — needed for DeepMAge feature selection
# ============================================================
print('\nLoading healthy training data for feature selection...')

def select_healthy(d):
    a = d[d.columns[d.columns.isin(AltumAge_cpgs)].tolist() +
          ['age', 'gender', 'dataset', 'tissue_type']]
    return a[a['tissue_type'].str.lower().str.contains('blood')].dropna()

train_frames = []
for filename in os.listdir(DATA_DIR):
    if filename.endswith('.pkl'):
        df = select_healthy(pd.read_pickle(os.path.join(DATA_DIR, filename)))
        if len(df) > 0:
            tr, _ = train_test_split(df, test_size=0.2, random_state=42)
            train_frames.append(tr)

train_combined = pd.concat(train_frames)
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
for fold, (train_idx, _) in enumerate(kf.split(train_combined)):
    if fold != DESIRED_FOLD:
        continue
    fold_train = train_combined.iloc[train_idx].sample(frac=1, random_state=42)
    break

X_train = fold_train.drop(
    columns=['age','gender','dataset','tissue_type']).astype('float')
y_train = fold_train['age'].values.astype('float32')
print(f'  Training samples: {len(X_train)}')

# ============================================================
# DeepMAge feature selection — top 1000 CpGs by |correlation|
# (must be identical to training)
# ============================================================
print(f'\nSelecting top {TOP_K_CPGS} CpGs by |correlation| with age...')
corr     = X_train.corrwith(
    pd.Series(y_train, index=X_train.index)).abs()
top_cpgs = corr.nlargest(TOP_K_CPGS).index.tolist()
print(f'  Selected {len(top_cpgs)} CpGs')

# Save for future use
np.save(os.path.join(BASELINES_DIR, 'deepmage_top_cpgs.npy'),
        np.array(top_cpgs))
print('  Saved: baselines/deepmage_top_cpgs.npy')

# ============================================================
# Load unhealthy dataset
# ============================================================
print('\nLoading unhealthy dataset...')

unhealthy_frames = []
for filename in os.listdir(UNHEALTHY_DIR):
    if filename.endswith('.pkl'):
        df = pd.read_pickle(os.path.join(UNHEALTHY_DIR, filename))
        cols = [c for c in df.columns if c in AltumAge_cpgs]
        cols += ['age', 'gender', 'dataset', 'tissue_type']
        df = df[[c for c in cols if c in df.columns]]
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df = df.dropna(subset=['age'])
        if len(df) > 0:
            unhealthy_frames.append(df)

unhealthy_data = pd.concat(unhealthy_frames)
unhealthy_data['disease'] = unhealthy_data['dataset'].map(
    DISEASE_MAP).fillna('Unknown')

X_unhealthy = unhealthy_data.drop(
    columns=['age','gender','dataset','tissue_type','disease'],
    errors='ignore').astype('float')
y_unhealthy = unhealthy_data['age'].astype('float')
print(f'  Unhealthy samples: {len(unhealthy_data)}')

# ============================================================
# Model definitions
# ============================================================
class DeepMAge(nn.Module):
    def __init__(self, input_dim, hidden_size=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.ELU(), nn.BatchNorm1d(channels),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.ELU(), nn.BatchNorm1d(channels),
        )
    def forward(self, x):
        return x + self.block(x)


class ResnetAge(nn.Module):
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ELU(), nn.BatchNorm1d(32),
        )
        self.res_blocks = nn.Sequential(
            ResBlock(32),
            nn.Conv1d(32, 64, kernel_size=1),
            ResBlock(64),
            nn.Conv1d(64, 128, kernel_size=1),
            ResBlock(128),
            nn.Conv1d(128, 128, kernel_size=1),
            ResBlock(128),
            nn.Conv1d(128, 64, kernel_size=1),
            ResBlock(64),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.initial(x)
        x = self.res_blocks(x)
        x = self.pool(x)
        return self.head(x).squeeze(-1)


def predict_mlp(model, X_np, batch_size=64):
    model.eval()
    ds     = TensorDataset(torch.tensor(X_np, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds  = []
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(preds).flatten()

# ============================================================
# Model 1: DeepMAge
# ============================================================
print('\n--- DeepMAge (reimplemented) ---')

X_unhealthy_deepmage = X_unhealthy[top_cpgs].values.astype('float32')

deepmage_model = DeepMAge(input_dim=TOP_K_CPGS).to(device)
deepmage_model.load_state_dict(
    torch.load(DEEPMAGE_CKPT, map_location=device))

pred_deepmage = predict_mlp(deepmage_model, X_unhealthy_deepmage)
mae_dm = mean_absolute_error(y_unhealthy, pred_deepmage)
print(f'  MAE={mae_dm:.3f}')
print(f'  Mean age acceleration: {(pred_deepmage - y_unhealthy.values).mean():.3f}')

# ============================================================
# Model 2: ResnetAge
# ============================================================
print('\n--- ResnetAge (reimplemented) ---')

X_unhealthy_resnet = X_unhealthy[AltumAge_cpgs].values.astype('float32')

resnet_model = ResnetAge(input_dim=len(AltumAge_cpgs)).to(device)
resnet_model.load_state_dict(
    torch.load(RESNET_CKPT, map_location=device))

pred_resnet = predict_mlp(resnet_model, X_unhealthy_resnet)
mae_rn = mean_absolute_error(y_unhealthy, pred_resnet)
print(f'  MAE={mae_rn:.3f}')
print(f'  Mean age acceleration: {(pred_resnet - y_unhealthy.values).mean():.3f}')

# ============================================================
# Merge into existing predictions CSV
# ============================================================
print('\nMerging into existing predictions CSV...')

if os.path.exists(PRED_CSV):
    existing = pd.read_csv(PRED_CSV)
    existing['DeepMAge']       = pred_deepmage
    existing['DeepMAge_accel'] = pred_deepmage - existing['true_age']
    existing['ResnetAge']       = pred_resnet
    existing['ResnetAge_accel'] = pred_resnet - existing['true_age']
    existing.to_csv(PRED_CSV, index=False)
    print(f'  Updated: {PRED_CSV}')

# ============================================================
# Update acceleration summary CSV
# ============================================================
print('\nUpdating acceleration summary...')

if os.path.exists(ACCEL_CSV):
    summary = pd.read_csv(ACCEL_CSV)
    for disease in unhealthy_data['disease'].unique():
        mask = (unhealthy_data['disease'] == disease).values
        idx  = summary[summary['disease'] == disease].index
        if len(idx) > 0:
            for col, pred in [('DeepMAge', pred_deepmage),
                               ('ResnetAge', pred_resnet)]:
                accel = pred[mask] - y_unhealthy.values[mask]
                summary.loc[idx, f'{col}_mean_accel'] = round(float(accel.mean()), 3)
                summary.loc[idx, f'{col}_std_accel']  = round(float(accel.std()),  3)
    summary.to_csv(ACCEL_CSV, index=False)
    print(f'  Updated: {ACCEL_CSV}')

# ============================================================
# Print final summary
# ============================================================
print('\n' + '='*60)
print('AGE ACCELERATION BY DISEASE')
print('='*60)
print(f'{"Disease":20s}  {"n":>3}  {"DeepMAge":>10}  {"ResnetAge":>10}')
print('-'*55)
for disease in ['Ovarian Cancer', 'Schizophrenia', 'Osteoporosis']:
    mask = (unhealthy_data['disease'] == disease).values
    n    = mask.sum()
    dm   = (pred_deepmage[mask] - y_unhealthy.values[mask]).mean()
    rn   = (pred_resnet[mask]   - y_unhealthy.values[mask]).mean()
    print(f'{disease:20s}  {n:3d}  {dm:+10.3f}  {rn:+10.3f}')
