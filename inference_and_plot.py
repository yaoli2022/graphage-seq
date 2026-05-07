"""
inference_and_plot.py
---------------------
Load the best test model checkpoint and generate:
  1. Predicted vs True Age scatter plot (publication quality)
  2. Save predictions to CSV for further analysis

Run on Spartan:
  cd /data/gpfs/projects/punim2698/yao14/graphage_seq
  source /data/projects/punim2698/yao14/venvs/ham_env/bin/activate
  python inference_and_plot.py
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Spartan
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, PNAConv

# ============================================================
# Paths  -- adjust if needed
# ============================================================
BASE_DIR        = '/data/gpfs/projects/punim2698/yao14/graphage_seq'
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'checkpoint_seq_no_CNN/cv_fold2_best_test_model.pth')
OUTPUT_DIR      = os.path.join(BASE_DIR, 'checkpoint_seq_no_CNN')
CPG_INFO_PATH   = os.path.join(BASE_DIR, 'cpgsite-info/GPL8490_HumanMethylation27_270596_v.1.2.csv')
DATA_DIR        = os.path.join(BASE_DIR, 'all-organs4/all_organs')
ALTUMAGE_CPGS   = os.path.join(BASE_DIR, 'graph-age/example_dependencies/multi_platform_cpgs.pkl')
SEQ_DATA_PATH   = os.path.join(BASE_DIR, 'checkpoint_seq_no_CNN/seq_data.npy')  # saved if exists

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Reproducibility
# ============================================================
SEED = 0
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ============================================================
# Hyperparameters (must match training)
# ============================================================
THRESHOLD_CORR     = 0.70
SECONDARY_THRESHOLD = THRESHOLD_CORR - 0.02
TERTIARY_THRESHOLD  = THRESHOLD_CORR - 0.04
THRESHOLD_DIST      = 1e5
EDGE_DIM            = 3
K_FOLDS             = 5
DESIRED_FOLD        = 2

# ============================================================
# CpG site information preprocessing (same as training)
# ============================================================
print('Loading CpG site information...')
information = pd.read_csv(CPG_INFO_PATH, skiprows=7, low_memory=False)
information.dropna(subset=['Chr'], inplace=True)
information[['start', 'end']] = (
    information.CPG_ISLAND_LOCATIONS
    .fillna('0:0-0').str.split(':').str[1]
    .str.split('-', expand=True).astype(int)
)
information['CPG_ISLAND'] = information['CPG_ISLAND'].astype(int)
information['CPG_ISLAND_LEN'] = information.end - information.start
information.MapInfo = information.MapInfo.astype(int)

scaler = MinMaxScaler()
for col in ['MapInfo', 'TSS_Coordinate']:
    information[f'Normalized_{col}'] = information.groupby('Chr')[col].transform(
        lambda x: scaler.fit_transform(x.values.reshape(-1,1)).flatten()
    )
information = pd.get_dummies(information, columns=['Gene_Strand'])
for col in ['start', 'end', 'CPG_ISLAND_LEN']:
    information[col] = information.groupby('Chr')[col].transform(
        lambda x: scaler.fit_transform(x.values.reshape(-1,1)).flatten()
    )
information = pd.get_dummies(information, columns=['Next_Base'])
information['Distance_to_TSS'] = information.groupby('Chr')['Distance_to_TSS'].transform(
    lambda x: scaler.fit_transform(x.values.reshape(-1,1)).flatten()
)
information['Distance_to_TSS'] = information['Distance_to_TSS'].fillna(1)
information = pd.get_dummies(information, columns=['SourceStrand'])
Chrom = information.Chr.tolist()
information = pd.get_dummies(information, columns=['Chr'])
information['Chr'] = Chrom
information.index = information.IlmnID

# Node feature selection (same as v2)
NODE_FEATURE_DICT = {
    1: 'CPG_ISLAND', 2: 'CPG_ISLAND_LEN', 3: 'Distance_to_TSS',
    4: ['Next_Base_A', 'Next_Base_C', 'Next_Base_T'],
    5: 'start', 6: 'end', 7: 'Normalized_TSS_Coordinate',
    8: 'Normalized_MapInfo', 9: [f'Chr_{x}' for x in range(1,23)]
}
selected_features = '1,2,3,4,5,6,8'
selected_feature_numbers = list(map(int, selected_features.split(',')))
user_selected_features = []
for key in selected_feature_numbers:
    feat = NODE_FEATURE_DICT[key]
    if isinstance(feat, list):
        user_selected_features.extend(feat)
    else:
        user_selected_features.append(feat)

INPUT_DIM = len(user_selected_features) + 1
print(f'INPUT_DIM: {INPUT_DIM}, features: {user_selected_features}')

# ============================================================
# Load AltumAge CpG list
# ============================================================
AltumAge_cpgs = np.array(pd.read_pickle(ALTUMAGE_CPGS)).tolist()
number_of_cpgs = len(AltumAge_cpgs)
print(f'Number of CpGs: {number_of_cpgs}')

node2cpg = {}
def make_cpg2node(cpg):
    cpg2node = {}
    for i, c in enumerate(cpg):
        cpg2node[c] = i
        node2cpg[i] = c
    return cpg2node

cpg2node = make_cpg2node(AltumAge_cpgs)
filtered_information = information[information.IlmnID.isin(AltumAge_cpgs)]
chromosomes = list(set(filtered_information.Chr))

# ============================================================
# DNA Sequence Processor (same as training)
# ============================================================
class DNASequenceProcessor:
    def __init__(self, max_seq_len=122):
        self.max_seq_len = max_seq_len

    def extract_statistical_features(self, seq):
        if pd.isna(seq) or not isinstance(seq, str) or len(seq) == 0:
            return np.zeros(8, dtype=np.float32)
        seq = seq.replace('[CG]', 'CG').upper()
        n = len(seq)
        gc       = (seq.count('G') + seq.count('C')) / max(n, 1)
        cpg_dens = seq.count('CG') / max(n-1, 1)
        upstream = seq[:60]
        gc_up    = (upstream.count('G') + upstream.count('C')) / max(len(upstream), 1)
        downstream = seq[62:]
        gc_down  = (downstream.count('G') + downstream.count('C')) / max(len(downstream), 1)
        context  = seq[55:65] if len(seq) >= 65 else seq
        a_freq   = context.count('A') / max(len(context), 1)
        t_freq   = context.count('T') / max(len(context), 1)
        c_freq   = context.count('C') / max(len(context), 1)
        g_freq   = context.count('G') / max(len(context), 1)
        return np.array([gc, cpg_dens, gc_up, gc_down, a_freq, t_freq, c_freq, g_freq],
                        dtype=np.float32)

    def extract_all_statistical_features(self, sequences):
        return np.array([self.extract_statistical_features(s) for s in sequences],
                        dtype=np.float32)

# ============================================================
# Graph construction
# ============================================================
def make_graph(information, threshold_corr, threshold_dist, meth, cpg2node, chromosomes):
    chromosomes_arr = information.Chr.values
    genes           = information.Symbol.values
    print('  Building adjacency matrices...')
    chromosomes_arr = np.array(chromosomes_arr)
    chromosome_adj  = (chromosomes_arr[:, None] == chromosomes_arr).astype(np.float32)
    base_pair       = information.MapInfo.values
    distance_matrix = squareform(pdist(base_pair.reshape(-1,1)))
    genes           = np.array(genes)
    genes_adj       = (genes[:, None] == genes).astype(np.float32)
    print('  Computing co-methylation...')
    adj = np.corrcoef(meth.to_numpy(), rowvar=False)
    src, dst = np.where(
        (((chromosome_adj == 1) &
          ((np.abs(adj) > SECONDARY_THRESHOLD) |
           ((np.abs(distance_matrix) < threshold_dist) & (np.abs(adj) > TERTIARY_THRESHOLD))))
         | (np.abs(adj) > threshold_corr))
        & (np.arange(adj.shape[0])[:, None] != np.arange(adj.shape[1]))
    )
    weights    = np.column_stack([adj[src,dst], chromosome_adj[src,dst], genes_adj[src,dst]])
    edge_index = torch.stack([torch.tensor(src, dtype=torch.int64),
                               torch.tensor(dst, dtype=torch.int64)], dim=0)
    edge_attr  = torch.tensor(weights, dtype=torch.float32).reshape(-1, EDGE_DIM)
    print(f'  Edges: {edge_index.shape[1]}')
    return edge_index, edge_attr

# ============================================================
# DataLoader
# ============================================================
def data_loader_maker_hierarchical(X_, y_, edge_index, edge_attr,
                                    filtered_information, seq_data,
                                    user_selected_features, batch_size=1):
    seq_tensor = torch.tensor(seq_data, dtype=torch.float32)
    graphs = []
    for row in range(len(X_)):
        x = pd.concat([X_.iloc[row, :],
                        filtered_information[user_selected_features]], axis=1, join='inner')
        x_original = torch.tensor(x.to_numpy().astype('float'), dtype=torch.float32)
        y = torch.tensor(y_.iloc[row], dtype=torch.float32)
        graphs.append(Data(x=x_original, x_seq=seq_tensor, y=y,
                           edge_index=edge_index, edge_attr=edge_attr))
    return DataLoader(graphs, batch_size=batch_size)

# ============================================================
# Model (HierarchicalNet -- identical to training)
# ============================================================
class HierarchicalNet(nn.Module):
    def __init__(self, deg, original_dim=10, num_cpgs=20318):
        super().__init__()
        self.original_dim = original_dim
        self.importance_gate = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        )
        self.seq_proj = nn.Sequential(
            nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2)
        )
        self.fused_dim = original_dim + 2   # = 12
        aggregators = ['mean', 'max', 'std', 'min']
        scalers     = ['identity', 'amplification', 'attenuation']
        self.LastLayer = PNAConv(
            in_channels=self.fused_dim, out_channels=1,
            aggregators=aggregators, scalers=scalers, deg=deg,
            edge_dim=3, towers=1, pre_layers=1, post_layers=1,
            divide_input=False
        )
        self.mlp = nn.Sequential(
            nn.Linear(num_cpgs, 1024), nn.ReLU(),
            nn.Linear(1024, 656),      nn.SELU(),
            nn.Linear(656, 256),       nn.ReLU(),  nn.Dropout(0.2),
            nn.Linear(256, 124),       nn.ReLU(),  nn.Dropout(0.2),
            nn.Linear(124, 64),        nn.SELU(),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, 8),          nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x_original, x_seq, edge_index, edge_attr, batch):
        importance     = self.importance_gate(x_seq)
        meth_modulated = x_original[:, 0:1] * importance
        seq_extra      = self.seq_proj(x_seq)
        x = torch.cat([meth_modulated, x_original[:, 1:], seq_extra], dim=-1)
        x = self.LastLayer(x, edge_index, edge_attr)
        x = F.relu(x)
        return self.mlp(x.T).flatten()

# ============================================================
# Degree histogram (required by PNAConv)
# ============================================================
def compute_deg(loader):
    device_cpu = torch.device('cpu')
    max_degree = -1
    for data in loader:
        max_degree = max(max_degree, int(data.edge_index[1].max()))
        break
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in loader.dataset:
        d = torch.bincount(data.edge_index[1].to(device_cpu), minlength=max_degree+1)
        deg += d
    return deg

# ============================================================
# Data loading (replicate fold 2 split from training)
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
print(f'Total test samples: {len(test_combined)}')

# Replicate fold 2 split
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(train_combined)):
    if fold != DESIRED_FOLD:
        continue
    fold_train = train_combined.iloc[train_idx, :].sample(frac=1, random_state=42)
    fold_val   = train_combined.iloc[val_idx, :]
    break

X_fold_train = fold_train.drop(columns=['age','gender','dataset','tissue_type']).astype('float')
y_fold_train = fold_train.age
X_fold_test  = test_combined.drop(columns=['age','gender','dataset','tissue_type']).astype('float')
y_fold_test  = test_combined.age
print(f'Train: {len(X_fold_train)}, Test: {len(X_fold_test)}')

# ============================================================
# Build graph (based on fold train data -- same as training)
# ============================================================
print('\nBuilding co-methylation graph...')
edge_index, edge_attr = make_graph(
    filtered_information, THRESHOLD_CORR, THRESHOLD_DIST,
    X_fold_train, cpg2node, chromosomes
)

# ============================================================
# Extract sequence features
# ============================================================
print('\nExtracting sequence features...')
processor  = DNASequenceProcessor()
sequences  = filtered_information['TopGenomicSeq'].values
seq_data   = processor.extract_all_statistical_features(sequences)
print(f'seq_data shape: {seq_data.shape}')

# ============================================================
# Build test DataLoader
# ============================================================
print('\nBuilding test DataLoader...')
test_loader = data_loader_maker_hierarchical(
    X_fold_test, y_fold_test,
    edge_index, edge_attr,
    filtered_information, seq_data,
    user_selected_features, batch_size=1
)

# ============================================================
# Compute deg and load model
# ============================================================
print('\nComputing degree histogram...')
deg = compute_deg(test_loader)

print(f'\nLoading checkpoint: {CHECKPOINT_PATH}')
ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

# Support both raw state_dict and full checkpoint dict
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    state_dict = ckpt['model_state_dict']
    saved_deg  = ckpt.get('deg', deg)
    print(f'  Checkpoint epoch: {ckpt.get("epoch", "unknown")}')
    print(f'  Checkpoint best_test_mae: {ckpt.get("best_test_mae", "unknown")}')
else:
    state_dict = ckpt
    saved_deg  = deg

model = HierarchicalNet(deg=saved_deg.to('cpu'),
                        original_dim=INPUT_DIM,
                        num_cpgs=number_of_cpgs).to(device)
model.load_state_dict(state_dict)
model.eval()
total_params = sum(p.numel() for p in model.parameters())
print(f'  Model parameters: {total_params:,}')

# ============================================================
# Inference
# ============================================================
print('\nRunning inference on test set...')
all_truths = []
all_preds  = []

with torch.no_grad():
    for data in tqdm(test_loader, desc='Inference'):
        data = data.to(device)
        out  = model(data.x, data.x_seq, data.edge_index, data.edge_attr, data.batch)
        all_truths.append(data.y.cpu().numpy())
        all_preds.append(out.cpu().numpy())

truths = np.concatenate(all_truths).flatten()
preds  = np.concatenate(all_preds).flatten()

mae = mean_absolute_error(truths, preds)
mse = mean_squared_error(truths, preds)
r2  = r2_score(truths, preds)
print(f'\nTest Results:')
print(f'  MAE = {mae:.4f}')
print(f'  MSE = {mse:.4f}')
print(f'  R2  = {r2:.4f}')

# Save predictions to CSV
pred_df = pd.DataFrame({'true_age': truths, 'predicted_age': preds,
                         'error': preds - truths})
pred_csv = os.path.join(OUTPUT_DIR, 'test_predictions.csv')
pred_df.to_csv(pred_csv, index=False)
print(f'\nPredictions saved to: {pred_csv}')

# ============================================================
# Plot: Predicted vs True Age (publication quality)
# ============================================================
print('\nGenerating scatter plot...')

fig, ax = plt.subplots(figsize=(6, 6))

# Scatter points coloured by absolute error
abs_err = np.abs(preds - truths)
sc = ax.scatter(truths, preds,
                c=abs_err, cmap='RdYlGn_r',
                alpha=0.55, s=18, linewidths=0, vmin=0, vmax=15)

# Diagonal (perfect prediction) line
min_age = min(truths.min(), preds.min())
max_age = max(truths.max(), preds.max())
ax.plot([min_age, max_age], [min_age, max_age],
        'k--', linewidth=1.2, label='Perfect prediction')

# Linear regression line
z  = np.polyfit(truths, preds, 1)
p  = np.poly1d(z)
x_line = np.linspace(min_age, max_age, 300)
ax.plot(x_line, p(x_line), 'b-', linewidth=1.2,
        label=f'Fit: y = {z[0]:.3f}x + {z[1]:.2f}')

# Colorbar
cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Absolute Error (years)', fontsize=10)

# Annotations
ax.set_xlabel('True Chronological Age (years)', fontsize=12)
ax.set_ylabel('Predicted Epigenetic Age (years)', fontsize=12)
ax.set_title('GraphAge-Seq: Predicted vs True Age\n(Blood methylation, Test set)',
             fontsize=12)

textstr = f'MAE = {mae:.3f} years\nMSE = {mse:.3f}\n$R^2$ = {r2:.4f}\nn = {len(truths)}'
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', bbox=props)

ax.legend(fontsize=9, loc='lower right')
ax.set_aspect('equal', adjustable='box')
ax.grid(True, alpha=0.3)

plt.tight_layout()
scatter_path = os.path.join(OUTPUT_DIR, 'predicted_vs_true_age.png')
plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
plt.close()
print(f'Scatter plot saved to: {scatter_path}')

# ============================================================
# Plot: MAE by age group (bar chart)
# ============================================================
print('Generating age group MAE bar chart...')

def age_group_label(age):
    if age <= 0:   return '0'
    if age <= 20:  return '0-20'
    if age <= 45:  return '20-45'
    if age <= 55:  return '45-55'
    if age <= 65:  return '55-65'
    if age <= 75:  return '65-75'
    if age <= 80:  return '75-80'
    return '80+'

age_labels_order = ['0', '0-20', '20-45', '45-55', '55-65', '65-75', '75-80', '80+']
group_mae = {lbl: [] for lbl in age_labels_order}
for t, p_val in zip(truths, preds):
    lbl = age_group_label(t)
    group_mae[lbl].append(abs(p_val - t))

mae_vals = [np.mean(group_mae[lbl]) if group_mae[lbl] else 0
            for lbl in age_labels_order]
counts   = [len(group_mae[lbl]) for lbl in age_labels_order]

fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.bar(age_labels_order, mae_vals, color='steelblue', alpha=0.8, edgecolor='white')

# Annotate sample counts
for bar, cnt, mv in zip(bars, counts, mae_vals):
    ax.text(bar.get_x() + bar.get_width()/2, mv + 0.05,
            f'n={cnt}', ha='center', va='bottom', fontsize=8, color='#333333')

ax.axhline(mae, color='red', linestyle='--', linewidth=1.2,
           label=f'Overall MAE = {mae:.3f}')
ax.set_xlabel('Age Group (years)', fontsize=12)
ax.set_ylabel('Mean Absolute Error (years)', fontsize=12)
ax.set_title('GraphAge-Seq: MAE by Age Group (Test Set)', fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(0, max(mae_vals) * 1.25)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()

bar_path = os.path.join(OUTPUT_DIR, 'mae_by_age_group.png')
plt.savefig(bar_path, dpi=300, bbox_inches='tight')
plt.close()
print(f'Age group bar chart saved to: {bar_path}')

print('\nAll done.')
print(f'Results saved in: {OUTPUT_DIR}')
print(f'  - test_predictions.csv')
print(f'  - predicted_vs_true_age.png')
print(f'  - mae_by_age_group.png')
