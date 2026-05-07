"""
extract_dataset_stats.py
------------------------
Extract detailed statistics from all blood methylation PKL files.
Run on Spartan:
  cd /data/gpfs/projects/punim2698/yao14/graphage_seq
  source /data/projects/punim2698/yao14/venvs/ham_env/bin/activate
  python extract_dataset_stats.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR      = '/data/gpfs/projects/punim2698/yao14/graphage_seq'
DATA_DIR      = os.path.join(BASE_DIR, 'all-organs4/all_organs')
ALTUMAGE_CPGS = os.path.join(BASE_DIR, 'graph-age/example_dependencies/multi_platform_cpgs.pkl')

AltumAge_cpgs = np.array(pd.read_pickle(ALTUMAGE_CPGS)).tolist()

def select(d):
    a = d[d.columns[d.columns.isin(AltumAge_cpgs)].tolist() +
          ['age', 'gender', 'dataset', 'tissue_type']]
    return a[a['tissue_type'].str.lower().str.contains('blood')].dropna()

# ── Collect per-dataset stats ──────────────────────────────
records = []
total_count = 0

for filename in sorted(os.listdir(DATA_DIR)):
    if not filename.endswith('.pkl'):
        continue
    df = select(pd.read_pickle(os.path.join(DATA_DIR, filename)))
    if len(df) == 0:
        continue

    total_count += len(df)
    ages    = df['age'].astype(float)
    genders = df['gender'].values if 'gender' in df.columns else []

    # Gender counts
    n_male   = sum(1 for g in genders if str(g).strip().upper() in ['M','MALE'])
    n_female = sum(1 for g in genders if str(g).strip().upper() in ['F','FEMALE'])

    # Dataset name (GEO accession)
    dataset_name = df['dataset'].iloc[0] if 'dataset' in df.columns else filename

    records.append({
        'Dataset'       : dataset_name,
        'N'             : len(df),
        'Age_min'       : ages.min(),
        'Age_max'       : ages.max(),
        'Age_mean'      : ages.mean(),
        'Age_std'       : ages.std(),
        'N_male'        : n_male,
        'N_female'      : n_female,
    })

# ── Print per-dataset table ────────────────────────────────
print("\n" + "="*80)
print("PER-DATASET SUMMARY")
print("="*80)
print(f"{'Dataset':<20} {'N':>5} {'Age range':>12} {'Mean±Std':>14} {'M':>5} {'F':>5}")
print("-"*80)
for r in records:
    age_range = f"{r['Age_min']:.0f}--{r['Age_max']:.0f}"
    mean_std  = f"{r['Age_mean']:.1f}±{r['Age_std']:.1f}"
    print(f"{r['Dataset']:<20} {r['N']:>5} {age_range:>12} {mean_std:>14} "
          f"{r['N_male']:>5} {r['N_female']:>5}")

# ── Overall summary ────────────────────────────────────────
all_frames = []
train_frames, test_frames = [], []
for filename in sorted(os.listdir(DATA_DIR)):
    if not filename.endswith('.pkl'):
        continue
    df = select(pd.read_pickle(os.path.join(DATA_DIR, filename)))
    if len(df) == 0:
        continue
    all_frames.append(df)
    tr, te = train_test_split(df, test_size=0.2, random_state=42)
    train_frames.append(tr)
    test_frames.append(te)

all_data       = pd.concat(all_frames)
train_combined = pd.concat(train_frames)
test_combined  = pd.concat(test_frames)

all_ages  = all_data['age'].astype(float)
test_ages = test_combined['age'].astype(float)

print("\n" + "="*80)
print("OVERALL SUMMARY")
print("="*80)
print(f"Number of datasets:      {len(records)}")
print(f"Total samples:           {len(all_data)}")
print(f"  Train+Val:             {len(train_combined)}")
print(f"  Test:                  {len(test_combined)}")
print(f"\nFull dataset age stats:")
print(f"  Range:   {all_ages.min():.1f} -- {all_ages.max():.1f} years")
print(f"  Mean:    {all_ages.mean():.1f} ± {all_ages.std():.1f} years")
print(f"  Median:  {all_ages.median():.1f} years")

# Gender overall
all_genders = all_data['gender'].values
n_m = sum(1 for g in all_genders if str(g).strip().upper() in ['M','MALE'])
n_f = sum(1 for g in all_genders if str(g).strip().upper() in ['F','FEMALE'])
print(f"\nGender distribution (full dataset):")
print(f"  Male:    {n_m} ({n_m/len(all_data)*100:.1f}%)")
print(f"  Female:  {n_f} ({n_f/len(all_data)*100:.1f}%)")
print(f"  Unknown: {len(all_data)-n_m-n_f} ({(len(all_data)-n_m-n_f)/len(all_data)*100:.1f}%)")

# Age group distribution (full dataset)
def age_group(age):
    if age <= 0:   return '0'
    if age <= 20:  return '0-20'
    if age <= 45:  return '20-45'
    if age <= 55:  return '45-55'
    if age <= 65:  return '55-65'
    if age <= 75:  return '65-75'
    if age <= 80:  return '75-80'
    return '80+'

groups = [age_group(a) for a in all_ages]
print(f"\nAge group distribution (full dataset):")
for g in ['0','0-20','20-45','45-55','55-65','65-75','75-80','80+']:
    cnt = groups.count(g)
    print(f"  {g:<8}: {cnt:>4} ({cnt/len(all_ages)*100:.1f}%)")
