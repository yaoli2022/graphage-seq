"""
explainer_node_attr_no_cnn.py
==============================
GNN Explainer — Option A: Node attribute importance
Model: GraphAge + Statistical Features (no CNN)

Generates:
  1. x_original importance (10-dim): which positional/methylation features matter most
  2. x_seq importance (8-dim): which sequence statistical features matter most
  3. Bar charts for both (publication quality)
  4. Temporal analysis plots (importance vs chronological age)
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, PNAConv
from torch_geometric.explain import Explainer, GNNExplainer

# ============================================================
# Paths
# ============================================================
BASE_DIR        = '/data/gpfs/projects/punim2698/yao14/graphage_seq'
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'checkpoint_seq_no_CNN/cv_fold2_best_test_model.pth')
OUTPUT_DIR      = os.path.join(BASE_DIR, 'checkpoint_seq_no_CNN/explainer')
CPG_INFO_PATH   = os.path.join(BASE_DIR, 'cpgsite-info/GPL8490_HumanMethylation27_270596_v.1.2.csv')
DATA_DIR        = os.path.join(BASE_DIR, 'all-organs4/all_organs')
ALTUMAGE_CPGS   = os.path.join(BASE_DIR, 'graph-age/example_dependencies/multi_platform_cpgs.pkl')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Reproducibility
# ============================================================
SEED = 0
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ============================================================
# Hyperparameters
# ============================================================
THRESHOLD_CORR      = 0.70
SECONDARY_THRESHOLD = THRESHOLD_CORR - 0.02
TERTIARY_THRESHOLD  = THRESHOLD_CORR - 0.04
THRESHOLD_DIST      = 1e5
EDGE_DIM            = 3
K_FOLDS             = 5
DESIRED_FOLD        = 2

# ============================================================
# Feature names for plotting
# ============================================================
# x_original: 10-dim [methylation, CPG_ISLAND, CPG_ISLAND_LEN,
#   Distance_to_TSS, Next_Base_A, Next_Base_C, Next_Base_T,
#   start, end, Normalized_MapInfo]
X_ORIGINAL_FEATURE_NAMES = [
    'Methylation',
    'CpG Island',
    'CpG Island Length',
    'Distance to TSS',
    'Next Base A',
    'Next Base C',
    'Next Base T',
    'Island Start',
    'Island End',
    'Normalised MapInfo'
]

# x_seq: 8-dim statistical features
X_SEQ_FEATURE_NAMES = [
    'GC content',
    'CpG density',
    'Upstream GC',
    'Downstream GC',
    'Local A freq.',
    'Local T freq.',
    'Local C freq.',
    'Local G freq.'
]

# ============================================================
# Age group function (same as training)
# ============================================================
def age_group(age):
    if age <= 0:   return 0
    if age <= 20:  return 1
    if age <= 45:  return 2
    if age <= 55:  return 3
    if age <= 65:  return 4
    if age <= 75:  return 5
    if age <= 80:  return 6
    return 7

AGE_GROUP_LABELS = ['0', '0-20', '20-45', '45-55', '55-65', '65-75', '75-80', '80+']

# ============================================================
# CpG site information preprocessing
# ============================================================
print('Loading CpG site information...')
information = pd.read_csv(CPG_INFO_PATH, skiprows=7, low_memory=False)
information.dropna(subset=['Chr'], inplace=True)
information[['start', 'end']] = (
    information.CPG_ISLAND_LOCATIONS
    .fillna('0:0-0').str.split(':').str[1]
    .str.split('-', expand=True).astype(int)
)
information['CPG_ISLAND']     = information['CPG_ISLAND'].astype(int)
information['CPG_ISLAND_LEN'] = information.end - information.start
information.MapInfo            = information.MapInfo.astype(int)

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
Chrom        = information.Chr.tolist()
information  = pd.get_dummies(information, columns=['Chr'])
information['Chr'] = Chrom
information.index  = information.IlmnID

NODE_FEATURE_DICT = {
    1: 'CPG_ISLAND', 2: 'CPG_ISLAND_LEN', 3: 'Distance_to_TSS',
    4: ['Next_Base_A', 'Next_Base_C', 'Next_Base_T'],
    5: 'start', 6: 'end', 7: 'Normalized_TSS_Coordinate',
    8: 'Normalized_MapInfo', 9: [f'Chr_{x}' for x in range(1,23)]
}
selected_features        = '1,2,3,4,5,6,8'
selected_feature_numbers = list(map(int, selected_features.split(',')))
user_selected_features   = []
for key in selected_feature_numbers:
    feat = NODE_FEATURE_DICT[key]
    if isinstance(feat, list):
        user_selected_features.extend(feat)
    else:
        user_selected_features.append(feat)

INPUT_DIM = len(user_selected_features) + 1
print(f'INPUT_DIM: {INPUT_DIM}')

# ============================================================
# Load AltumAge CpG list
# ============================================================
AltumAge_cpgs  = np.array(pd.read_pickle(ALTUMAGE_CPGS)).tolist()
number_of_cpgs = len(AltumAge_cpgs)
print(f'Number of CpGs: {number_of_cpgs}')

node2cpg = {}
def make_cpg2node(cpg):
    cpg2node = {}
    for i, c in enumerate(cpg):
        cpg2node[c] = i
        node2cpg[i] = c
    return cpg2node

cpg2node             = make_cpg2node(AltumAge_cpgs)
filtered_information = information[information.IlmnID.isin(AltumAge_cpgs)]

# ============================================================
# DNA Sequence Processor (statistical features)
# ============================================================
class DNASequenceProcessor:
    def __init__(self, max_seq_len=122):
        self.max_seq_len = max_seq_len

    def extract_statistical_features(self, seq):
        if pd.isna(seq) or not isinstance(seq, str) or len(seq) == 0:
            return np.zeros(8, dtype=np.float32)
        seq      = seq.replace('[CG]', 'CG').upper()
        n        = len(seq)
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
def make_graph(information, threshold_corr, threshold_dist, meth):
    chromosomes_arr = np.array(information.Chr.values)
    genes           = np.array(information.Symbol.values)
    chromosome_adj  = (chromosomes_arr[:, None] == chromosomes_arr).astype(np.float32)
    base_pair       = information.MapInfo.values
    distance_matrix = squareform(pdist(base_pair.reshape(-1,1)))
    genes_adj       = (genes[:, None] == genes).astype(np.float32)
    print('  Computing co-methylation...')
    adj             = np.corrcoef(meth.to_numpy(), rowvar=False)
    src, dst        = np.where(
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
# HierarchicalNet (no CNN) — identical to training
# ============================================================
class HierarchicalNet(nn.Module):
    def __init__(self, deg, original_dim=10, num_cpgs=20318):
        super().__init__()
        self.original_dim    = original_dim
        self.importance_gate = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        )
        self.seq_proj = nn.Sequential(
            nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2)
        )
        self.fused_dim = original_dim + 2  # = 12

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
# Wrapper A: expose x_original to Explainer, fix x_seq
# Used for analysing 10-dim x_original feature importance
# ============================================================
class WrapperOriginal(nn.Module):
    """Fix x_seq; let Explainer vary x (= x_original)."""
    def __init__(self, model, x_seq_fixed):
        super().__init__()
        self.model = model
        self.register_buffer('x_seq_fixed', x_seq_fixed)

    def forward(self, x, edge_index, edge_attr, batch=None):
        return self.model(x, self.x_seq_fixed, edge_index, edge_attr, batch)

# ============================================================
# Wrapper B: expose x_seq to Explainer, fix x_original
# Used for analysing 8-dim sequence feature importance
# ============================================================
class WrapperSeq(nn.Module):
    """Fix x_original per sample; let Explainer vary x (= x_seq)."""
    def __init__(self, model, x_original_fixed):
        super().__init__()
        self.model = model
        self.register_buffer('x_original_fixed', x_original_fixed)

    def forward(self, x, edge_index, edge_attr, batch=None):
        # here x is x_seq [num_nodes, 8]
        return self.model(self.x_original_fixed, x, edge_index, edge_attr, batch)

# ============================================================
# Degree histogram
# ============================================================
def compute_deg(loader):
    max_degree = -1
    for data in loader:
        max_degree = max(max_degree, int(data.edge_index[1].max()))
        break
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in loader.dataset:
        d    = torch.bincount(data.edge_index[1], minlength=max_degree+1)
        deg += d
    return deg

# ============================================================
# Data loading — replicate fold 2 split
# ============================================================
print('\nLoading methylation data...')

def select(d):
    a = d[d.columns[d.columns.isin(AltumAge_cpgs)].tolist() +
          ['age', 'gender', 'dataset', 'tissue_type']]
    return a[a['tissue_type'].str.lower().str.contains('blood')].dropna()

train_frames, test_frames = [], []
for filename in os.listdir(DATA_DIR):
    if not filename.endswith('.pkl'):
        continue
    df = select(pd.read_pickle(os.path.join(DATA_DIR, filename)))
    if len(df) == 0:
        continue
    tr, te = train_test_split(df, test_size=0.2, random_state=42)
    train_frames.append(tr)
    test_frames.append(te)

train_combined = pd.concat(train_frames)
test_combined  = pd.concat(test_frames)

kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(train_combined)):
    if fold != DESIRED_FOLD:
        continue
    fold_train = train_combined.iloc[train_idx, :].sample(frac=1, random_state=42)
    break

X_fold_train = fold_train.drop(columns=['age','gender','dataset','tissue_type']).astype('float')
X_fold_test  = test_combined.drop(columns=['age','gender','dataset','tissue_type']).astype('float')
y_fold_test  = test_combined.age
print(f'Train: {len(X_fold_train)}, Test: {len(X_fold_test)}')

# ============================================================
# Build graph
# ============================================================
print('\nBuilding co-methylation graph...')
edge_index, edge_attr = make_graph(
    filtered_information, THRESHOLD_CORR, THRESHOLD_DIST, X_fold_train
)

# ============================================================
# Extract sequence features
# ============================================================
print('\nExtracting sequence features...')
processor = DNASequenceProcessor()
sequences = filtered_information['TopGenomicSeq'].values
seq_data  = processor.extract_all_statistical_features(sequences)  # [num_cpgs, 8]
seq_tensor_global = torch.tensor(seq_data, dtype=torch.float32)
print(f'seq_data shape: {seq_data.shape}')

# ============================================================
# Build test DataLoader
# ============================================================
def make_test_loader(X_, y_, seq_data):
    seq_tensor = torch.tensor(seq_data, dtype=torch.float32)
    graphs = []
    for row in range(len(X_)):
        x = pd.concat([X_.iloc[row, :],
                        filtered_information[user_selected_features]],
                       axis=1, join='inner')
        x_original = torch.tensor(x.to_numpy().astype('float'), dtype=torch.float32)
        y          = torch.tensor(y_.iloc[row], dtype=torch.float32)
        graphs.append(Data(x=x_original, x_seq=seq_tensor, y=y,
                           edge_index=edge_index, edge_attr=edge_attr))
    return DataLoader(graphs, batch_size=1)

print('\nBuilding test DataLoader...')
test_loader = make_test_loader(X_fold_test, y_fold_test, seq_data)

# ============================================================
# Load model
# ============================================================
print('\nComputing degree histogram...')
deg = compute_deg(test_loader)

print(f'Loading checkpoint: {CHECKPOINT_PATH}')
ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    state_dict = ckpt['model_state_dict']
    saved_deg  = ckpt.get('deg', deg)
else:
    state_dict = ckpt
    saved_deg  = deg

model = HierarchicalNet(deg=saved_deg.to('cpu'),
                        original_dim=INPUT_DIM,
                        num_cpgs=number_of_cpgs).to(device)
model.load_state_dict(state_dict)
model.eval()
print(f'Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}')

# ============================================================
# PART A1: x_original importance (10-dim)
# common_attributes → one shared importance vector across all nodes
# ============================================================
print('\n' + '='*60)
print('PART A1: x_original feature importance (10-dim)')
print('='*60)

wrapper_orig = WrapperOriginal(model, seq_tensor_global.to(device)).to(device)
wrapper_orig.eval()

explainer_orig = Explainer(
    model=wrapper_orig,
    algorithm=GNNExplainer(epochs=120),
    explanation_type='phenomenon',
    node_mask_type='common_attributes',
    edge_mask_type='object',
    model_config=dict(mode='regression', task_level='graph', return_type='raw'),
)

# Accumulators
n_age_groups        = 8
age_group_sum_orig  = np.zeros((n_age_groups, INPUT_DIM))  # [8, 10]
age_group_count     = np.zeros(n_age_groups)
all_orig_importance = []   # per-sample, for temporal analysis
all_ages            = []

print(f'Running explainer on {len(test_loader)} test samples...')
for data in tqdm(test_loader, desc='x_original explainer'):
    data = data.to(device)
    age  = data.y.cpu().item()
    ag   = age_group(age)

    explanation = explainer_orig(
        x          = data.x,
        edge_index = data.edge_index,
        edge_attr  = data.edge_attr,
        batch      = data.batch_size if hasattr(data, 'batch_size') else None,
        target     = data.y
    )
    imp = explanation.node_mask.cpu().detach().flatten().numpy()  # [10]
    age_group_sum_orig[ag] += imp
    age_group_count[ag]    += 1
    all_orig_importance.append(imp)
    all_ages.append(age)

# Average by group
age_group_mean_orig = age_group_sum_orig / np.maximum(age_group_count[:, None], 1)
mean_orig           = np.array(all_orig_importance).mean(axis=0)  # [10]

# Save
np.save(os.path.join(OUTPUT_DIR, 'orig_age_group_mean.npy'), age_group_mean_orig)
np.save(os.path.join(OUTPUT_DIR, 'orig_mean_overall.npy'),   mean_orig)
np.save(os.path.join(OUTPUT_DIR, 'orig_all_per_sample.npy'), np.array(all_orig_importance))
np.save(os.path.join(OUTPUT_DIR, 'all_ages.npy'),            np.array(all_ages))
print('x_original importance saved.')

# ============================================================
# PART A2: x_seq importance (8-dim)
# ============================================================
print('\n' + '='*60)
print('PART A2: x_seq (sequence statistical) feature importance (8-dim)')
print('='*60)

age_group_sum_seq  = np.zeros((n_age_groups, 8))
all_seq_importance = []

for data in tqdm(test_loader, desc='x_seq explainer'):
    data = data.to(device)
    age  = data.y.cpu().item()
    ag   = age_group(age)

    # Fix x_original for this sample, expose x_seq
    wrapper_seq = WrapperSeq(model, data.x).to(device)
    wrapper_seq.eval()

    explainer_seq = Explainer(
        model=wrapper_seq,
        algorithm=GNNExplainer(epochs=120),
        explanation_type='phenomenon',
        node_mask_type='common_attributes',
        edge_mask_type='object',
        model_config=dict(mode='regression', task_level='graph', return_type='raw'),
    )

    explanation = explainer_seq(
        x          = data.x_seq,
        edge_index = data.edge_index,
        edge_attr  = data.edge_attr,
        batch      = data.batch_size if hasattr(data, 'batch_size') else None,
        target     = data.y
    )
    imp = explanation.node_mask.cpu().detach().flatten().numpy()  # [8]
    age_group_sum_seq[ag] += imp
    all_seq_importance.append(imp)

age_group_mean_seq = age_group_sum_seq / np.maximum(age_group_count[:, None], 1)
mean_seq           = np.array(all_seq_importance).mean(axis=0)

np.save(os.path.join(OUTPUT_DIR, 'seq_age_group_mean.npy'), age_group_mean_seq)
np.save(os.path.join(OUTPUT_DIR, 'seq_mean_overall.npy'),   mean_seq)
np.save(os.path.join(OUTPUT_DIR, 'seq_all_per_sample.npy'), np.array(all_seq_importance))
print('x_seq importance saved.')

# ============================================================
# PLOTTING
# ============================================================
print('\nGenerating plots...')

# ── Plot 1: Mean importance of x_original features (bar chart)
fig, ax = plt.subplots(figsize=(9, 5))
sorted_idx = np.argsort(mean_orig)
colors     = ['#5B8DB8'] * INPUT_DIM
ax.barh([X_ORIGINAL_FEATURE_NAMES[i] for i in sorted_idx],
        mean_orig[sorted_idx], color=colors, edgecolor='white', height=0.65)
ax.set_xlabel('Mean Importance Score', fontsize=12)
ax.set_title('Mean Importance of Node Attributes (x_original, 10-dim)', fontsize=12)
ax.grid(True, axis='x', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'orig_feature_importance.png'), dpi=300, bbox_inches='tight')
plt.close()
print('  Saved: orig_feature_importance.png')

# ── Plot 2: Mean importance of x_seq features (bar chart)
fig, ax = plt.subplots(figsize=(9, 4))
sorted_idx = np.argsort(mean_seq)
colors     = ['#4BAE6E'] * 8
ax.barh([X_SEQ_FEATURE_NAMES[i] for i in sorted_idx],
        mean_seq[sorted_idx], color=colors, edgecolor='white', height=0.65)
ax.set_xlabel('Mean Importance Score', fontsize=12)
ax.set_title('Mean Importance of Sequence Statistical Features (x_seq, 8-dim)', fontsize=12)
ax.grid(True, axis='x', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'seq_feature_importance.png'), dpi=300, bbox_inches='tight')
plt.close()
print('  Saved: seq_feature_importance.png')

# ── Plot 3: Combined bar chart (both together, normalized)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: x_original
sorted_idx = np.argsort(mean_orig)
axes[0].barh([X_ORIGINAL_FEATURE_NAMES[i] for i in sorted_idx],
             mean_orig[sorted_idx], color='#5B8DB8', edgecolor='white', height=0.65)
axes[0].set_xlabel('Mean Importance Score', fontsize=11)
axes[0].set_title('(a) Positional & methylation features', fontsize=11, fontweight='bold')
axes[0].grid(True, axis='x', alpha=0.3)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Right: x_seq
sorted_idx = np.argsort(mean_seq)
axes[1].barh([X_SEQ_FEATURE_NAMES[i] for i in sorted_idx],
             mean_seq[sorted_idx], color='#4BAE6E', edgecolor='white', height=0.65)
axes[1].set_xlabel('Mean Importance Score', fontsize=11)
axes[1].set_title('(b) DNA sequence statistical features', fontsize=11, fontweight='bold')
axes[1].grid(True, axis='x', alpha=0.3)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.suptitle('Mean Node Attribute Importance (GNN Explainer, Test Set)', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'combined_feature_importance.png'), dpi=300, bbox_inches='tight')
plt.close()
print('  Saved: combined_feature_importance.png')

# ── Plot 4: Temporal analysis — x_original importance vs age
ages_arr      = np.array(all_ages)
orig_arr      = np.array(all_orig_importance)  # [n_samples, 10]
sorted_by_age = np.argsort(ages_arr)
ages_sorted   = ages_arr[sorted_by_age]
orig_sorted   = orig_arr[sorted_by_age]

fig, ax = plt.subplots(figsize=(10, 5))
colors_temp = plt.cm.tab10(np.linspace(0, 1, INPUT_DIM))
for i, (name, color) in enumerate(zip(X_ORIGINAL_FEATURE_NAMES, colors_temp)):
    ax.scatter(ages_sorted, orig_sorted[:, i], s=6, alpha=0.4, color=color, label=name)
ax.set_xlabel('Chronological Age (years)', fontsize=12)
ax.set_ylabel('Importance Score', fontsize=12)
ax.set_title('Temporal Analysis: x_original Feature Importance vs Age', fontsize=12)
ax.legend(fontsize=7, ncol=2, loc='upper right')
ax.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'orig_temporal_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()
print('  Saved: orig_temporal_analysis.png')

# ── Plot 5: Temporal analysis — x_seq importance vs age
seq_arr     = np.array(all_seq_importance)
seq_sorted  = seq_arr[sorted_by_age]

fig, ax = plt.subplots(figsize=(10, 5))
colors_temp = plt.cm.Set2(np.linspace(0, 1, 8))
for i, (name, color) in enumerate(zip(X_SEQ_FEATURE_NAMES, colors_temp)):
    ax.scatter(ages_sorted, seq_sorted[:, i], s=6, alpha=0.4, color=color, label=name)
ax.set_xlabel('Chronological Age (years)', fontsize=12)
ax.set_ylabel('Importance Score', fontsize=12)
ax.set_title('Temporal Analysis: Sequence Feature Importance vs Age', fontsize=12)
ax.legend(fontsize=8, ncol=2, loc='upper right')
ax.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'seq_temporal_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()
print('  Saved: seq_temporal_analysis.png')

# ── Plot 6: Age group heatmap — x_seq importance across age groups
fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(age_group_mean_seq, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(8))
ax.set_xticklabels(X_SEQ_FEATURE_NAMES, rotation=30, ha='right', fontsize=9)
ax.set_yticks(range(8))
ax.set_yticklabels(AGE_GROUP_LABELS, fontsize=9)
ax.set_xlabel('Sequence Feature', fontsize=11)
ax.set_ylabel('Age Group', fontsize=11)
ax.set_title('Sequence Feature Importance by Age Group', fontsize=12)
plt.colorbar(im, ax=ax, label='Mean Importance')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'seq_age_group_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()
print('  Saved: seq_age_group_heatmap.png')

print('\n' + '='*60)
print('All done. Results saved to:')
print(f'  {OUTPUT_DIR}')
print('Files:')
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f'  - {f}')
