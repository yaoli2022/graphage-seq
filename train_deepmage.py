"""
train_deepmage.py
-----------------
Re-implements the DeepMAge architecture (Galkin et al., 2021) and retrains
it on our dataset under identical experimental conditions to all other models.

Architecture (from the paper):
  - Correlation-based feature selection: top 1000 CpGs by |Pearson r| with age
  - 4 hidden layers x 512 neurons, ReLU activation, Dropout
  - Output: single neuron (age regression)

Training:
  - Runs for all 300 epochs (no early stopping)
  - Saves the checkpoint with the best validation MAE during training
  - Evaluates the best checkpoint on the test set

Data split: identical to inference_and_plot.py
  - blood samples only, AltumAge CpG list (20,318 sites)
  - test_size=0.2, random_state=42 per dataset
  - 5-fold CV, fold 2: 2360 train / 591 val / 756 test

Run on Spartan:
  cd /data/gpfs/projects/punim2698/yao14/graphage_seq
  source /data/projects/punim2698/yao14/venvs/ham_env/bin/activate
  python train_deepmage.py

Output:
  baselines/deepmage_best.pth
  baselines/deepmage_predictions.csv
  baselines/deepmage_results.csv
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
OUTPUT_PATH   = os.path.join(BASELINES_DIR, 'deepmage_results.csv')
CKPT_PATH     = os.path.join(BASELINES_DIR, 'deepmage_best.pth')

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
TOP_K_CPGS    = 1000
HIDDEN_SIZE   = 512
DROPOUT       = 0.3
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-5
BATCH_SIZE    = 64
MAX_EPOCHS    = 500

# ============================================================
# Load CpG list
# ============================================================
print('Loading CpG list...')
AltumAge_cpgs = np.array(pd.read_pickle(ALTUMAGE_CPGS)).tolist()
print(f'  {len(AltumAge_cpgs)} CpG sites')

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

X_train = fold_train.drop(columns=['age','gender','dataset','tissue_type']).astype('float')
y_train = fold_train['age'].values.astype('float32')
X_val   = fold_val.drop(columns=['age','gender','dataset','tissue_type']).astype('float')
y_val   = fold_val['age'].values.astype('float32')
X_test  = test_combined.drop(columns=['age','gender','dataset','tissue_type']).astype('float')
y_test  = test_combined['age'].values.astype('float32')

print(f'  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')

# ============================================================
# Feature selection: top-k CpGs by |Pearson correlation| with age
# ============================================================
print(f'\nSelecting top {TOP_K_CPGS} CpGs by |correlation| with age...')
corr    = X_train.corrwith(pd.Series(y_train, index=X_train.index)).abs()
top_cpgs = corr.nlargest(TOP_K_CPGS).index.tolist()
print(f'  Selected {len(top_cpgs)} CpGs')

X_train_sel = X_train[top_cpgs].values.astype('float32')
X_val_sel   = X_val[top_cpgs].values.astype('float32')
X_test_sel  = X_test[top_cpgs].values.astype('float32')

# ============================================================
# DeepMAge Model
# Input -> 512 -> 512 -> 512 -> 512 -> 1
# ============================================================
class DeepMAge(nn.Module):
    def __init__(self, input_dim, hidden_size=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ============================================================
# DataLoaders
# ============================================================
train_ds = TensorDataset(torch.tensor(X_train_sel), torch.tensor(y_train))
val_ds   = TensorDataset(torch.tensor(X_val_sel),   torch.tensor(y_val))
test_ds  = TensorDataset(torch.tensor(X_test_sel),  torch.tensor(y_test))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# Training — full 300 epochs, save best val MAE checkpoint
# ============================================================
model     = DeepMAge(input_dim=TOP_K_CPGS, hidden_size=HIDDEN_SIZE, dropout=DROPOUT).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

best_val_mae  = float('inf')
best_epoch    = 0

print(f'\nTraining DeepMAge for {MAX_EPOCHS} epochs...')
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
# Test Evaluation — using best val MAE checkpoint
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
}, index=X_test.index).to_csv(
    os.path.join(BASELINES_DIR, 'deepmage_predictions.csv'))

pd.DataFrame([{
    'Model': 'DeepMAge (reimplemented)',
    'MAE': round(mae, 4),
    'MSE': round(mse, 4),
    'R2':  round(r2, 4),
    'Best_epoch': best_epoch
}]).set_index('Model').to_csv(OUTPUT_PATH)

print('\n' + '='*50)
print('FINAL RESULT — DeepMAge (reimplemented)')
print('='*50)
print(f'  MAE : {mae:.3f}')
print(f'  MSE : {mse:.3f}')
print(f'  R2  : {r2:.4f}')
print(f'  Best checkpoint from epoch {best_epoch}/{MAX_EPOCHS}')
print(f'\nFiles saved to: {BASELINES_DIR}/')
