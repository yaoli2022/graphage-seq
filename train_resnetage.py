"""
train_resnetage.py
------------------
Re-implements the ResnetAge architecture (Shi et al., Bioengineering 2024)
and retrains it on our dataset under identical experimental conditions.

Architecture adapted for our dataset scale (2360 training samples):
  - Input: 20,318 CpG beta values treated as 1D sequence
  - 1 initial conv layer (32 channels)
  - 5 residual blocks: 32 -> 64 -> 128 -> 128 -> 64 channels
  - Tail: global avg pool -> FC -> age prediction
  - Activation: ELU, BatchNorm throughout

Key difference from original paper: channel sizes scaled down to match
our training set size (2360 vs ~10,000+ in original).

Training:
  - 500 epochs, no early stopping
  - Saves best val MAE checkpoint
  - Evaluates best checkpoint on test set

Data split: identical to inference_and_plot.py
  - blood samples only, AltumAge CpG list (20,318 sites)
  - test_size=0.2, random_state=42 per dataset
  - 5-fold CV, fold 2: 2360 train / 591 val / 756 test

Run on Spartan:
  cd /data/gpfs/projects/punim2698/yao14/graphage_seq
  source /data/projects/punim2698/yao14/venvs/ham_env/bin/activate
  python train_resnetage.py

Output:
  baselines/resnetage_best.pth
  baselines/resnetage_predictions.csv
  baselines/resnetage_results.csv
"""

import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# Paths
# ============================================================
BASE_DIR      = '/data/gpfs/projects/punim2698/yao14/graphage_seq'
DATA_DIR      = os.path.join(BASE_DIR, 'all-organs4/all_organs')
ALTUMAGE_CPGS = os.path.join(BASE_DIR, 'graph-age/example_dependencies/multi_platform_cpgs.pkl')
BASELINES_DIR = os.path.join(BASE_DIR, 'baselines')
OUTPUT_PATH   = os.path.join(BASELINES_DIR, 'resnetage_results.csv')
CKPT_PATH     = os.path.join(BASELINES_DIR, 'resnetage_best.pth')

os.makedirs(BASELINES_DIR, exist_ok=True)

# ============================================================
# Reproducibility
# ============================================================
SEED         = 0
K_FOLDS      = 5
DESIRED_FOLD = 2

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ============================================================
# Hyperparameters
# ============================================================
BATCH_SIZE    = 64
LEARNING_RATE = 5e-4
WEIGHT_DECAY  = 1e-4
MAX_EPOCHS    = 500

# ============================================================
# Load CpG list
# ============================================================
print('Loading CpG list...')
AltumAge_cpgs = np.array(pd.read_pickle(ALTUMAGE_CPGS)).tolist()
INPUT_DIM = len(AltumAge_cpgs)
print(f'  {INPUT_DIM} CpG sites')

# ============================================================
# Load data — identical split to inference_and_plot.py
# ============================================================
print('\nLoading methylation data...')

def select(d):
    a = d[d.columns[d.columns.isin(AltumAge_cpgs)].tolist() +
          ['age', 'gender', 'dataset', 'tissue_type']]
    return a[a['tissue_type'].str.lower().str.contains('blood')].dropna()

train_frames, test_frames = [], []
for filename in os.listdir(DATA_DIR):
    if filename.endswith('.pkl'):
        df = select(pd.read_pickle(os.path.join(DATA_DIR, filename)))
        if len(df) <= 0:
            continue
        tr, te = train_test_split(df, test_size=0.2, random_state=42)
        train_frames.append(tr)
        test_frames.append(te)

train_combined = pd.concat(train_frames)
test_combined  = pd.concat(test_frames)
print(f'  Total test samples: {len(test_combined)}')

kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(train_combined)):
    if fold != DESIRED_FOLD:
        continue
    fold_train = train_combined.iloc[train_idx].sample(frac=1, random_state=42)
    fold_val   = train_combined.iloc[val_idx]
    break

X_train = fold_train.drop(columns=['age','gender','dataset','tissue_type']).astype('float32').values
y_train = fold_train['age'].values.astype('float32')
X_val   = fold_val.drop(columns=['age','gender','dataset','tissue_type']).astype('float32').values
y_val   = fold_val['age'].values.astype('float32')
X_test  = test_combined.drop(columns=['age','gender','dataset','tissue_type']).astype('float32').values
y_test  = test_combined['age'].values.astype('float32')

print(f'  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')

# ============================================================
# ResnetAge Model
# Treats CpG beta values as a 1D sequence for residual CNN.
# Architecture scaled to match our dataset size.
# ============================================================
class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.ELU(),
            nn.BatchNorm1d(channels),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.ELU(),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetAge(nn.Module):
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()

        # Initial conv: 1 -> 32 channels
        self.initial = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ELU(),
            nn.BatchNorm1d(32),
        )

        # 5 residual blocks: 32 -> 64 -> 128 -> 128 -> 64
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

        # Global average pooling then FC layers
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, n_cpgs) -> (batch, 1, n_cpgs)
        x = x.unsqueeze(1)
        x = self.initial(x)
        x = self.res_blocks(x)
        x = self.pool(x)       # (batch, 64, 1)
        x = self.head(x)       # (batch, 1)
        return x.squeeze(-1)


# ============================================================
# DataLoaders
# ============================================================
train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_ds   = TensorDataset(torch.tensor(X_val),   torch.tensor(y_val))
test_ds  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# Training — 500 epochs, save best val MAE checkpoint
# ============================================================
model     = ResnetAge(input_dim=INPUT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

# Print model parameter count
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'\nModel parameters: {n_params:,}')

best_val_mae = float('inf')
best_epoch   = 0

print(f'Training ResnetAge for {MAX_EPOCHS} epochs...')
for epoch in range(MAX_EPOCHS):
    # Train
    model.train()
    train_losses = []
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # Validate
    model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            val_preds.extend(model(X_batch.to(device)).cpu().numpy())
            val_true.extend(y_batch.numpy())

    val_mae = mean_absolute_error(val_true, val_preds)
    scheduler.step(val_mae)

    # Save best checkpoint
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_epoch   = epoch + 1
        torch.save(model.state_dict(), CKPT_PATH)

    if (epoch + 1) % 10 == 0:
        print(f'  Epoch {epoch+1:3d}/{MAX_EPOCHS}: '
              f'train_loss={np.mean(train_losses):.3f}, '
              f'val_mae={val_mae:.3f} | '
              f'best_val_mae={best_val_mae:.3f} (epoch {best_epoch})')

print(f'\nTraining complete. Best val MAE={best_val_mae:.3f} at epoch {best_epoch}.')

# ============================================================
# Test Evaluation — best val MAE checkpoint
# ============================================================
print('\nEvaluating best checkpoint on test set...')
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model.eval()

test_preds, test_true = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        test_preds.extend(model(X_batch.to(device)).cpu().numpy())
        test_true.extend(y_batch.numpy())

test_preds = np.array(test_preds)
test_true  = np.array(test_true)

mae = mean_absolute_error(test_true, test_preds)
mse = mean_squared_error(test_true, test_preds)
r2  = r2_score(test_true, test_preds)

print(f'  MAE={mae:.3f}, MSE={mse:.3f}, R2={r2:.4f}')

# ============================================================
# Save results
# ============================================================
pd.DataFrame({
    'true_age': test_true,
    'predicted_age': test_preds
}, index=test_combined.index).to_csv(
    os.path.join(BASELINES_DIR, 'resnetage_predictions.csv'))

pd.DataFrame([{
    'Model': 'ResnetAge (reimplemented)',
    'MAE': round(mae, 4),
    'MSE': round(mse, 4),
    'R2':  round(r2, 4),
    'Best_epoch': best_epoch
}]).set_index('Model').to_csv(OUTPUT_PATH)

print('\n' + '='*50)
print('FINAL RESULT — ResnetAge (reimplemented)')
print('='*50)
print(f'  MAE : {mae:.3f}')
print(f'  MSE : {mse:.3f}')
print(f'  R2  : {r2:.4f}')
print(f'  Best checkpoint from epoch {best_epoch}/{MAX_EPOCHS}')
print(f'\nFiles saved to: {BASELINES_DIR}/')
