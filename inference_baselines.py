"""
inference_baselines.py
----------------------
Runs Horvath 2013 and AltumAge on the SAME 756-sample test set
used by the main GNN models, enabling a fair controlled comparison.

Models:
  1. Horvath (2013) -- linear regression, 353 CpGs
  2. AltumAge (2022) -- TensorFlow MLP loaded from AltumAge.h5

Required files in baselines/:
  coefficients.csv     -- Horvath CpG coefficients
  scaler.pkl           -- AltumAge RobustScaler
  AltumAge.h5          -- AltumAge trained model

Run on Spartan:
  cd /data/gpfs/projects/punim2698/yao14/graphage_seq
  source /data/projects/punim2698/yao14/venvs/ham_env/bin/activate
  python inference_baselines.py

Output:
  baselines/baseline_results.csv
  baselines/horvath_predictions.csv
  baselines/altumage_predictions.csv
"""

import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import linear_model

# ============================================================
# Paths
# ============================================================
BASE_DIR      = '/data/gpfs/projects/punim2698/yao14/graphage_seq'
DATA_DIR      = os.path.join(BASE_DIR, 'all-organs4/all_organs')
ALTUMAGE_CPGS = os.path.join(BASE_DIR, 'graph-age/example_dependencies/multi_platform_cpgs.pkl')
BASELINES_DIR = os.path.join(BASE_DIR, 'baselines')
COEF_PATH     = os.path.join(BASELINES_DIR, 'coefficients.csv')
SCALER_PATH   = os.path.join(BASELINES_DIR, 'scaler.pkl')
H5_PATH       = os.path.join(BASELINES_DIR, 'AltumAge.h5')
OUTPUT_PATH   = os.path.join(BASELINES_DIR, 'baseline_results.csv')

os.makedirs(BASELINES_DIR, exist_ok=True)

# ============================================================
# Reproducibility — must match inference_and_plot.py
# ============================================================
SEED         = 0
K_FOLDS      = 5
DESIRED_FOLD = 2

random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# Horvath age transformation
# ============================================================
ADULT_AGE = 20

def anti_transform_age(x):
    x = np.array(x, dtype=float)
    return np.where(
        x < 0,
        np.exp(x + np.log(ADULT_AGE + 1)) - 1,
        x * (ADULT_AGE + 1) + ADULT_AGE
    )

# ============================================================
# Load AltumAge CpG list
# ============================================================
print('Loading CpG list...')
AltumAge_cpgs = np.array(pd.read_pickle(ALTUMAGE_CPGS)).tolist()
print(f'  {len(AltumAge_cpgs)} CpG sites')

# ============================================================
# Load and split data — identical to inference_and_plot.py
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
    break

X_test = test_combined.drop(
    columns=['age', 'gender', 'dataset', 'tissue_type']).astype('float')
y_test = test_combined['age'].values
print(f'  Test: {len(X_test)}')

results = {}

# ============================================================
# Model 1: Horvath (2013)
# ============================================================
print('\n--- Horvath (2013) ---')
coef_data    = pd.read_csv(COEF_PATH)
intercept    = coef_data[
    coef_data.CpGmarker == '(Intercept)']['CoefficientTraining'].values[0]
coef_df      = coef_data[coef_data.CpGmarker != '(Intercept)']
horvath_cpgs = np.array(coef_df['CpGmarker'])
coefs        = np.array(coef_df['CoefficientTraining'])

available = [c for c in horvath_cpgs if c in X_test.columns]
missing   = [c for c in horvath_cpgs if c not in X_test.columns]
print(f'  CpGs available: {len(available)}/{len(horvath_cpgs)}, '
      f'missing: {len(missing)} (set to 0)')

X_test_horvath = pd.DataFrame(0.0, index=X_test.index, columns=horvath_cpgs)
for cpg in available:
    X_test_horvath[cpg] = X_test[cpg].values

horvath_model             = linear_model.LinearRegression()
horvath_model.coef_       = coefs
horvath_model.intercept_  = intercept

pred_horvath = anti_transform_age(horvath_model.predict(X_test_horvath))
pred_horvath = np.clip(pred_horvath, 0, 120)

mae_h = mean_absolute_error(y_test, pred_horvath)
mse_h = mean_squared_error(y_test, pred_horvath)
r2_h  = r2_score(y_test, pred_horvath)
print(f'  MAE={mae_h:.3f}, MSE={mse_h:.3f}, R2={r2_h:.4f}')
results['Horvath (2013)'] = {'MAE': mae_h, 'MSE': mse_h, 'R2': r2_h}

pd.DataFrame({
    'true_age': y_test,
    'predicted_age': pred_horvath
}, index=X_test.index).to_csv(
    os.path.join(BASELINES_DIR, 'horvath_predictions.csv'))

# ============================================================
# Model 2: AltumAge (2022)
# Load directly from .h5 file using TensorFlow
# Input: RobustScaler-transformed beta values (20318 features)
# Output: predicted age (direct, no anti_transform needed)
# ============================================================
print('\n--- AltumAge (2022) ---')

import tensorflow as tf

model = tf.keras.models.load_model(H5_PATH, compile=False)
print(f'  Model loaded: input={model.input_shape}, output={model.output_shape}')

# Load scaler fitted on AltumAge training data
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# Scale test data using AltumAge's RobustScaler
X_test_scaled = scaler.transform(X_test[AltumAge_cpgs]).astype(np.float32)
print(f'  Scaled data shape: {X_test_scaled.shape}')

# Predict (training=False disables GaussianDropout)
pred_altumage = model(X_test_scaled, training=False).numpy().flatten()
print(f'  Predictions: mean={pred_altumage.mean():.2f}, '
      f'min={pred_altumage.min():.2f}, max={pred_altumage.max():.2f}')

mae_a = mean_absolute_error(y_test, pred_altumage)
mse_a = mean_squared_error(y_test, pred_altumage)
r2_a  = r2_score(y_test, pred_altumage)
print(f'  MAE={mae_a:.3f}, MSE={mse_a:.3f}, R2={r2_a:.4f}')
results['AltumAge (2022)'] = {'MAE': mae_a, 'MSE': mse_a, 'R2': r2_a}

pd.DataFrame({
    'true_age': y_test,
    'predicted_age': pred_altumage
}, index=X_test.index).to_csv(
    os.path.join(BASELINES_DIR, 'altumage_predictions.csv'))

# ============================================================
# Save summary
# ============================================================
results_df = pd.DataFrame(results).T
results_df.index.name = 'Model'
results_df = results_df.round(4)
results_df.to_csv(OUTPUT_PATH)

print('\n' + '=' * 50)
print('SUMMARY')
print('=' * 50)
print(results_df.to_string())
print(f'\nFiles saved to: {BASELINES_DIR}/')
print('  baseline_results.csv')
print('  horvath_predictions.csv')
print('  altumage_predictions.csv')
