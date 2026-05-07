"""
explainer_node_attr_cnn.py
===========================
GNN Explainer — Option A: Node attribute importance
Model: GraphAge + CNN Sequence Encoder (CNN version)

Mirrors explainer_node_attr_no_cnn.py exactly, with three differences:
  1. x_seq shape is [num_nodes, 122, 4] (one-hot) instead of [num_nodes, 8]
  2. HierarchicalNet uses a CNN encoder (seq_embed_dim=32)
  3. Checkpoint path points to graphagewithCNN_3.26_checkpoint/

What this script generates:
  Part A1 — x_original (10-dim) feature importance
    - orig_feature_importance.png
    - orig_temporal_analysis.png
    - orig_age_group_mean.npy / orig_mean_overall.npy / orig_all_per_sample.npy

  Part A2 — x_seq (122×4 one-hot → CNN → 32-dim) feature importance
    NOTE: node_mask_type='common_attributes' on x_seq gives a 488-dim mask
    (122 positions × 4 channels). We aggregate across channels and positions
    to produce per-position (122) and per-channel (4) importance profiles.
    - seq_position_importance.png   (122 positions)
    - seq_channel_importance.png    (4 channels: A/T/C/G)
    - seq_combined.png              (both side by side)
    - seq_age_group_heatmap.png     (positions × age groups)

  Combined:
    - combined_feature_importance.png
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
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import PNAConv
from torch_geometric.explain import Explainer, GNNExplainer

# ============================================================
# Paths
# ============================================================
BASE_DIR        = '/data/gpfs/projects/punim2698/yao14/graphage_seq'
CHECKPOINT_PATH = os.path.join(BASE_DIR,
    'graphagewithCNN_3.26_checkpoint/cv_fold2_best_model.pth')
OUTPUT_DIR      = os.path.join(BASE_DIR,
    'graphagewithCNN_3.26_checkpoint/explainer')
CPG_INFO_PATH   = os.path.join(BASE_DIR,
    'cpgsite-info/GPL8490_HumanMethylation27_270596_v.1.2.csv')
DATA_DIR        = os.path.join(BASE_DIR, 'all-organs4/all_organs')
ALTUMAGE_CPGS   = os.path.join(BASE_DIR,
    'graph-age/example_dependencies/multi_platform_cpgs.pkl')

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
# Hyperparameters (identical to training)
# ============================================================
THRESHOLD_CORR      = 0.70
SECONDARY_THRESHOLD = THRESHOLD_CORR - 0.02
TERTIARY_THRESHOLD  = THRESHOLD_CORR - 0.04
THRESHOLD_DIST      = 1e5
EDGE_DIM            = 3
K_FOLDS             = 5
DESIRED_FOLD        = 2
SEQ_LEN             = 122
SEQ_EMBED_DIM       = 32   # CNN version uses 32-dim embedding

# ============================================================
# Feature names
# ============================================================
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

# One-hot channel names
CHANNEL_NAMES   = ['A', 'T', 'C', 'G']
AGE_GROUP_LABELS = ['0', '0-20', '20-45', '45-55', '55-65', '65-75', '75-80', '80+']

# ============================================================
# Age group function
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

# ============================================================
# CpG site information preprocessing (identical to training)
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
        lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
    )
information = pd.get_dummies(information, columns=['Gene_Strand'])
for col in ['start', 'end', 'CPG_ISLAND_LEN']:
    information[col] = information.groupby('Chr')[col].transform(
        lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
    )
information = pd.get_dummies(information, columns=['Next_Base'])
information['Distance_to_TSS'] = information.groupby('Chr')['Distance_to_TSS'].transform(
    lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
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
    8: 'Normalized_MapInfo', 9: [f'Chr_{x}' for x in range(1, 23)]
}
user_selected_features = []
for key in [1, 2, 3, 4, 5, 6, 8]:
    feat = NODE_FEATURE_DICT[key]
    if isinstance(feat, list):
        user_selected_features.extend(feat)
    else:
        user_selected_features.append(feat)

INPUT_DIM = len(user_selected_features) + 1   # = 10
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
# DNA Sequence Processor — ONE-HOT encoding (CNN version)
# Output shape: [num_cpgs, 122, 4]
# ============================================================
class DNASequenceProcessor:
    def __init__(self, max_seq_len=122):
        self.max_seq_len     = max_seq_len
        self.nucleotide_idx  = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

    def process_single_sequence(self, seq):
        if pd.isna(seq) or not isinstance(seq, str) or len(seq) == 0:
            return np.zeros((self.max_seq_len, 4), dtype=np.float32)
        seq = seq.replace('[CG]', 'CG').upper()
        if len(seq) > self.max_seq_len:
            seq = seq[:self.max_seq_len]
        one_hot = np.zeros((self.max_seq_len, 4), dtype=np.float32)
        for i, nuc in enumerate(seq):
            if nuc in self.nucleotide_idx:
                one_hot[i, self.nucleotide_idx[nuc]] = 1.0
            else:
                one_hot[i, :] = 0.25   # ambiguous base
        return one_hot

    def process_all_sequences(self, sequences):
        return np.array(
            [self.process_single_sequence(s)
             for s in tqdm(sequences, desc='One-hot encoding')],
            dtype=np.float32
        )   # [num_cpgs, 122, 4]

# ============================================================
# Graph construction (identical to training)
# ============================================================
def make_graph(information, threshold_corr, threshold_dist, meth):
    chromosomes_arr = np.array(information.Chr.values)
    genes           = np.array(information.Symbol.values)
    chromosome_adj  = (chromosomes_arr[:, None] == chromosomes_arr).astype(np.float32)
    base_pair       = information.MapInfo.values
    distance_matrix = squareform(pdist(base_pair.reshape(-1, 1)))
    genes_adj       = (genes[:, None] == genes).astype(np.float32)
    print('  Computing co-methylation...')
    adj             = np.corrcoef(meth.to_numpy(), rowvar=False)
    src, dst        = np.where(
        (((chromosome_adj == 1) &
          ((np.abs(adj) > SECONDARY_THRESHOLD) |
           ((np.abs(distance_matrix) < threshold_dist) &
            (np.abs(adj) > TERTIARY_THRESHOLD))))
         | (np.abs(adj) > threshold_corr))
        & (np.arange(adj.shape[0])[:, None] != np.arange(adj.shape[1]))
    )
    weights    = np.column_stack([adj[src, dst],
                                   chromosome_adj[src, dst],
                                   genes_adj[src, dst]])
    edge_index = torch.stack([torch.tensor(src, dtype=torch.int64),
                               torch.tensor(dst, dtype=torch.int64)], dim=0)
    edge_attr  = torch.tensor(weights, dtype=torch.float32).reshape(-1, EDGE_DIM)
    print(f'  Edges: {edge_index.shape[1]}')
    return edge_index, edge_attr

# ============================================================
# HierarchicalNet — CNN version (identical to training)
# ============================================================
class HierarchicalNet(nn.Module):
    def __init__(self, deg, seq_len=122, original_dim=10,
                 seq_embed_dim=32, num_cpgs=20318):
        super().__init__()
        self.original_dim  = original_dim
        self.seq_embed_dim = seq_embed_dim

        # CNN sequence encoder
        self.seq_cnn = nn.Sequential(
            nn.Conv1d(4,  16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, seq_embed_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(seq_embed_dim), nn.ReLU(),
        )
        self.seq_pool = nn.AdaptiveMaxPool1d(1)

        # Gate and projection (same as no-CNN version but input dim=seq_embed_dim)
        self.importance_gate = nn.Sequential(
            nn.Linear(seq_embed_dim, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )
        self.seq_proj = nn.Sequential(
            nn.Linear(seq_embed_dim, 16), nn.ReLU(),
            nn.Linear(16, 2)
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
        """
        x_original : [num_nodes, 10]
        x_seq      : [num_nodes, 122, 4]
        """
        # CNN: [num_nodes, 122, 4] → permute → [num_nodes, 4, 122] → CNN → pool
        cnn_in  = x_seq.permute(0, 2, 1)              # [N, 4, 122]
        cnn_out = self.seq_cnn(cnn_in)                 # [N, 32, 122]
        cnn_out = self.seq_pool(cnn_out).squeeze(-1)   # [N, 32]

        importance     = self.importance_gate(cnn_out)       # [N, 1]
        meth_modulated = x_original[:, 0:1] * importance     # [N, 1]
        seq_extra      = self.seq_proj(cnn_out)               # [N, 2]

        x = torch.cat([meth_modulated,
                        x_original[:, 1:],
                        seq_extra], dim=-1)                   # [N, 12]
        x   = self.LastLayer(x, edge_index, edge_attr)
        x   = F.relu(x)
        return self.mlp(x.T).flatten()

# ============================================================
# Wrapper A1: fix x_seq (one-hot), expose x_original
# ============================================================
class WrapperOriginal(nn.Module):
    """Fix x_seq globally; Explainer optimises over x_original."""
    def __init__(self, model, x_seq_fixed):
        super().__init__()
        self.model = model
        self.register_buffer('x_seq_fixed', x_seq_fixed)

    def forward(self, x, edge_index, edge_attr, batch=None):
        return self.model(x, self.x_seq_fixed, edge_index, edge_attr, batch)

# ============================================================
# Wrapper A2: fix x_original per sample, expose x_seq (flattened)
# GNNExplainer sees x_seq as [num_nodes, 488] (122*4 flattened).
# We reshape back inside forward.
# ============================================================
class WrapperSeq(nn.Module):
    """Fix x_original for one sample; Explainer optimises over x_seq (flattened)."""
    def __init__(self, model, x_original_fixed):
        super().__init__()
        self.model = model
        self.register_buffer('x_original_fixed', x_original_fixed)

    def forward(self, x_flat, edge_index, edge_attr, batch=None):
        # x_flat: [num_nodes, 122*4=488]  → reshape to [num_nodes, 122, 4]
        x_seq = x_flat.view(x_flat.shape[0], SEQ_LEN, 4)
        return self.model(self.x_original_fixed, x_seq,
                          edge_index, edge_attr, batch)

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
        d    = torch.bincount(data.edge_index[1], minlength=max_degree + 1)
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

X_fold_train = fold_train.drop(
    columns=['age', 'gender', 'dataset', 'tissue_type']).astype('float')
X_fold_test  = test_combined.drop(
    columns=['age', 'gender', 'dataset', 'tissue_type']).astype('float')
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
# Extract ONE-HOT sequence features  [num_cpgs, 122, 4]
# ============================================================
print('\nExtracting one-hot sequence features...')
processor  = DNASequenceProcessor(max_seq_len=SEQ_LEN)
sequences  = filtered_information['TopGenomicSeq'].values
seq_data   = processor.process_all_sequences(sequences)   # [20318, 122, 4]
seq_tensor_global = torch.tensor(seq_data, dtype=torch.float32)
print(f'seq_data shape: {seq_data.shape}')   # (20318, 122, 4)

# ============================================================
# Build test DataLoader
# seq_data stored as [num_nodes, 122, 4] in Data.x_seq
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
print('\nLoading model checkpoint...')
deg  = compute_deg(test_loader)
ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    state_dict = ckpt['model_state_dict']
    saved_deg  = ckpt.get('deg', deg)
else:
    state_dict = ckpt
    saved_deg  = deg

model = HierarchicalNet(
    deg=saved_deg.to('cpu'),
    seq_len=SEQ_LEN,
    original_dim=INPUT_DIM,
    seq_embed_dim=SEQ_EMBED_DIM,
    num_cpgs=number_of_cpgs
).to(device)
model.load_state_dict(state_dict)
model.eval()
print(f'Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}')

# ============================================================
# PART A1: x_original importance (10-dim)
# common_attributes → one shared vector across all nodes
# ============================================================
print('\n' + '='*60)
print('PART A1: x_original feature importance (10-dim)')
print('='*60)

wrapper_orig = WrapperOriginal(
    model, seq_tensor_global.to(device)
).to(device)
wrapper_orig.eval()

explainer_orig = Explainer(
    model=wrapper_orig,
    algorithm=GNNExplainer(epochs=120),
    explanation_type='phenomenon',
    node_mask_type='common_attributes',
    edge_mask_type='object',
    model_config=dict(mode='regression', task_level='graph', return_type='raw'),
)

n_age_groups        = 8
age_group_sum_orig  = np.zeros((n_age_groups, INPUT_DIM))
age_group_count     = np.zeros(n_age_groups)
all_orig_importance = []
all_ages            = []

print(f'Running on {len(test_loader)} test samples...')
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
    imp = explanation.node_mask.cpu().detach().flatten().numpy()   # [10]
    age_group_sum_orig[ag] += imp
    age_group_count[ag]    += 1
    all_orig_importance.append(imp)
    all_ages.append(age)

age_group_mean_orig = age_group_sum_orig / np.maximum(age_group_count[:, None], 1)
mean_orig           = np.array(all_orig_importance).mean(axis=0)   # [10]

np.save(os.path.join(OUTPUT_DIR, 'orig_age_group_mean.npy'),  age_group_mean_orig)
np.save(os.path.join(OUTPUT_DIR, 'orig_mean_overall.npy'),    mean_orig)
np.save(os.path.join(OUTPUT_DIR, 'orig_all_per_sample.npy'),  np.array(all_orig_importance))
np.save(os.path.join(OUTPUT_DIR, 'all_ages.npy'),             np.array(all_ages))
print('x_original importance saved.')

# ============================================================
# PART A2: x_seq importance (one-hot, 122×4 = 488-dim flattened)
# We present results both per-position (122) and per-channel (4)
# ============================================================
print('\n' + '='*60)
print('PART A2: x_seq (one-hot) feature importance (122×4 flattened)')
print('='*60)

# Re-use age_group_count from A1
age_group_sum_seq  = np.zeros((n_age_groups, SEQ_LEN * 4))   # [8, 488]
all_seq_importance = []   # per-sample [num_samples, 488]

for data in tqdm(test_loader, desc='x_seq explainer'):
    data = data.to(device)
    age  = data.y.cpu().item()
    ag   = age_group(age)

    # Fix x_original for this sample, expose x_seq (flattened)
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

    # Pass x_seq flattened: [num_nodes, 488]
    x_seq_flat = data.x_seq.view(data.x_seq.shape[0], SEQ_LEN * 4)

    explanation = explainer_seq(
        x          = x_seq_flat,
        edge_index = data.edge_index,
        edge_attr  = data.edge_attr,
        batch      = data.batch_size if hasattr(data, 'batch_size') else None,
        target     = data.y
    )
    imp = explanation.node_mask.cpu().detach().flatten().numpy()   # [488]
    age_group_sum_seq[ag] += imp
    all_seq_importance.append(imp)

age_group_mean_seq = age_group_sum_seq / np.maximum(age_group_count[:, None], 1)
mean_seq           = np.array(all_seq_importance).mean(axis=0)   # [488]

# Reshape for analysis: [488] → [122, 4]
mean_seq_2d           = mean_seq.reshape(SEQ_LEN, 4)             # [122, 4]
age_group_mean_seq_2d = age_group_mean_seq.reshape(
    n_age_groups, SEQ_LEN, 4)                                     # [8, 122, 4]

# Per-position importance: average across 4 channels
mean_seq_pos    = mean_seq_2d.mean(axis=1)      # [122]
# Per-channel importance: average across 122 positions
mean_seq_ch     = mean_seq_2d.mean(axis=0)      # [4]

np.save(os.path.join(OUTPUT_DIR, 'seq_age_group_mean.npy'),    age_group_mean_seq)
np.save(os.path.join(OUTPUT_DIR, 'seq_mean_overall.npy'),      mean_seq)
np.save(os.path.join(OUTPUT_DIR, 'seq_all_per_sample.npy'),    np.array(all_seq_importance))
np.save(os.path.join(OUTPUT_DIR, 'seq_mean_2d.npy'),           mean_seq_2d)
np.save(os.path.join(OUTPUT_DIR, 'seq_age_group_mean_2d.npy'), age_group_mean_seq_2d)
print('x_seq importance saved.')

# ============================================================
# PLOTTING
# ============================================================
print('\nGenerating plots...')

# ── Plot 1: x_original mean importance (bar chart)
fig, ax = plt.subplots(figsize=(9, 5))
sorted_idx = np.argsort(mean_orig)
ax.barh([X_ORIGINAL_FEATURE_NAMES[i] for i in sorted_idx],
        mean_orig[sorted_idx],
        color='#5B8DB8', edgecolor='white', height=0.65)
ax.set_xlabel('Mean Importance Score', fontsize=12)
ax.set_title('Mean Importance of Node Attributes\n(x_original, 10-dim, CNN model)',
             fontsize=12)
ax.grid(True, axis='x', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'orig_feature_importance.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print('  Saved: orig_feature_importance.png')

# ── Plot 2: x_seq per-position importance (line chart — 122 positions)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(range(SEQ_LEN), mean_seq_pos,
        color='#4BAE6E', linewidth=1.5, alpha=0.9)
ax.fill_between(range(SEQ_LEN), mean_seq_pos,
                alpha=0.2, color='#4BAE6E')
ax.axvline(x=60, color='red', linestyle='--', linewidth=1.2,
           label='CpG site position (bp 60-61)')
ax.axvline(x=61, color='red', linestyle='--', linewidth=1.2)
ax.set_xlabel('Sequence Position (bp)', fontsize=12)
ax.set_ylabel('Mean Importance Score', fontsize=12)
ax.set_title('Sequence Position Importance (averaged across 4 channels)\n'
             'CNN model — 122bp window around each CpG site', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'seq_position_importance.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print('  Saved: seq_position_importance.png')

# ── Plot 3: x_seq per-channel importance (bar chart — A/T/C/G)
fig, ax = plt.subplots(figsize=(6, 4))
colors_ch = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12']
ax.bar(CHANNEL_NAMES, mean_seq_ch, color=colors_ch,
       edgecolor='white', width=0.6)
ax.set_xlabel('Nucleotide Channel', fontsize=12)
ax.set_ylabel('Mean Importance Score', fontsize=12)
ax.set_title('Per-Channel Importance (A/T/C/G)\nCNN model', fontsize=12)
ax.grid(True, axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'seq_channel_importance.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print('  Saved: seq_channel_importance.png')

# ── Plot 4: Combined — x_original + seq_position + seq_channel
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) x_original
sorted_idx = np.argsort(mean_orig)
axes[0].barh([X_ORIGINAL_FEATURE_NAMES[i] for i in sorted_idx],
             mean_orig[sorted_idx],
             color='#5B8DB8', edgecolor='white', height=0.65)
axes[0].set_xlabel('Mean Importance Score', fontsize=10)
axes[0].set_title('(a) Node attribute features', fontsize=11, fontweight='bold')
axes[0].grid(True, axis='x', alpha=0.3)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# (b) sequence position
axes[1].plot(range(SEQ_LEN), mean_seq_pos,
             color='#4BAE6E', linewidth=1.5)
axes[1].fill_between(range(SEQ_LEN), mean_seq_pos,
                     alpha=0.2, color='#4BAE6E')
axes[1].axvline(x=60, color='red', linestyle='--', linewidth=1.2)
axes[1].axvline(x=61, color='red', linestyle='--', linewidth=1.2)
axes[1].set_xlabel('Sequence Position (bp)', fontsize=10)
axes[1].set_ylabel('Mean Importance', fontsize=10)
axes[1].set_title('(b) Sequence position importance', fontsize=11, fontweight='bold')
axes[1].grid(True, alpha=0.2)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# (c) channel
axes[2].bar(CHANNEL_NAMES, mean_seq_ch,
            color=colors_ch, edgecolor='white', width=0.6)
axes[2].set_xlabel('Nucleotide', fontsize=10)
axes[2].set_ylabel('Mean Importance', fontsize=10)
axes[2].set_title('(c) Nucleotide channel importance', fontsize=11, fontweight='bold')
axes[2].grid(True, axis='y', alpha=0.3)
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)

plt.suptitle('GNN Explainer — Node Attribute Importance (CNN model)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'combined_feature_importance.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print('  Saved: combined_feature_importance.png')

# ── Plot 5: x_original temporal analysis
ages_arr      = np.array(all_ages)
orig_arr      = np.array(all_orig_importance)
sorted_by_age = np.argsort(ages_arr)
ages_sorted   = ages_arr[sorted_by_age]
orig_sorted   = orig_arr[sorted_by_age]

fig, ax = plt.subplots(figsize=(10, 5))
colors_t = plt.cm.tab10(np.linspace(0, 1, INPUT_DIM))
for i, (name, color) in enumerate(zip(X_ORIGINAL_FEATURE_NAMES, colors_t)):
    ax.scatter(ages_sorted, orig_sorted[:, i],
               s=6, alpha=0.4, color=color, label=name)
ax.set_xlabel('Chronological Age (years)', fontsize=12)
ax.set_ylabel('Importance Score', fontsize=12)
ax.set_title('Temporal Analysis: Node Attribute Importance vs Age (CNN model)',
             fontsize=12)
ax.legend(fontsize=7, ncol=2, loc='upper right')
ax.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'orig_temporal_analysis.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print('  Saved: orig_temporal_analysis.png')

# ── Plot 6: sequence position importance by age group (heatmap)
# age_group_mean_seq_2d: [8, 122, 4] → average across channels → [8, 122]
heatmap_data = age_group_mean_seq_2d.mean(axis=2)   # [8, 122]

fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
ax.set_xticks(np.arange(0, SEQ_LEN, 10))
ax.set_xticklabels(np.arange(0, SEQ_LEN, 10), fontsize=8)
ax.axvline(x=60, color='blue', linestyle='--', linewidth=1.5,
           label='CpG site')
ax.set_yticks(range(n_age_groups))
ax.set_yticklabels(AGE_GROUP_LABELS, fontsize=9)
ax.set_xlabel('Sequence Position (bp)', fontsize=11)
ax.set_ylabel('Age Group', fontsize=11)
ax.set_title('Sequence Position Importance by Age Group (CNN model)',
             fontsize=12)
plt.colorbar(im, ax=ax, label='Mean Importance')
ax.legend(fontsize=9, loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'seq_age_group_heatmap.png'),
            dpi=300, bbox_inches='tight')
plt.close()
print('  Saved: seq_age_group_heatmap.png')

print('\n' + '='*60)
print('All done. Results saved to:')
print(f'  {OUTPUT_DIR}')
print('Files:')
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f'  - {f}')
print('='*60)
