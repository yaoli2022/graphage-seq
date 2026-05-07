"""
explainer_node_object_no_cnn.py
================================
GNN Explainer — Option B: Node importance (which CpG SITES are most important)
Model: GraphAge + Statistical Features (no CNN)

Corresponds to original paper Figure 5 (MRN subnetwork) and Figure 6 (temporal analysis).

What this script does:
  1. Run explainer3 (node_mask_type='object') on all 756 test samples
     → each sample gets a scalar importance per CpG site [num_nodes=20318]
     → each sample also gets an edge importance [num_edges]
  2. Average node + edge importance by age group (8 groups)
  3. Top-10 upward-trending / downward-trending CpG sites (Figure 6A/B)
  4. MRN subnetwork visualisation via Graphviz (Figure 5A equivalent)
     - Remove zero-importance nodes
     - Remove low-importance edges (< 0.1)
     - Colour nodes red (hypomethylating) / blue (hypermethylating)
     - Circle nodes green (importance increasing with age) / yellow (decreasing)
     - Annotate edges with co-methylation value and edge importance

Run on Spartan:
  cd /data/gpfs/projects/punim2698/yao14/graphage_seq
  source /data/projects/punim2698/yao14/venvs/ham_env/bin/activate
  pip install graphviz   # if not yet installed
  python explainer_node_object_no_cnn.py

Outputs saved to:
  /data/gpfs/projects/punim2698/yao14/graphage_seq/checkpoint_seq_no_CNN/explainer_node/
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
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
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'checkpoint_seq_no_CNN/cv_fold2_best_test_model.pth')
OUTPUT_DIR      = os.path.join(BASE_DIR, 'checkpoint_seq_no_CNN/explainer_node')
CPG_INFO_PATH   = os.path.join(BASE_DIR, 'cpgsite-info/GPL8490_HumanMethylation27_270596_v.1.2.csv')
DATA_DIR        = os.path.join(BASE_DIR, 'all-organs4/all_organs')
ALTUMAGE_CPGS   = os.path.join(BASE_DIR, 'graph-age/example_dependencies/multi_platform_cpgs.pkl')
STRING_PPI_PATH = os.path.join(BASE_DIR, 'string_ppi')  # local STRING files if available

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
SECONDARY_THRESHOLD = THRESHOLD_CORR - 0.02   # 0.68
TERTIARY_THRESHOLD  = THRESHOLD_CORR - 0.04   # 0.66
THRESHOLD_DIST      = 1e5
EDGE_DIM            = 3
K_FOLDS             = 5
DESIRED_FOLD        = 2

# MRN filtering thresholds (same as original paper)
EDGE_IMPORTANCE_THRESHOLD = 0.1   # edges below this are removed in MRN
MIN_SUBNETWORK_SIZE       = 10    # only visualise subnetworks with >= this many nodes

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

# Feature selection (identical to training: "1,2,3,4,5,6,8")
NODE_FEATURE_DICT = {
    1: 'CPG_ISLAND', 2: 'CPG_ISLAND_LEN', 3: 'Distance_to_TSS',
    4: ['Next_Base_A', 'Next_Base_C', 'Next_Base_T'],
    5: 'start', 6: 'end', 7: 'Normalized_TSS_Coordinate',
    8: 'Normalized_MapInfo', 9: [f'Chr_{x}' for x in range(1,23)]
}
user_selected_features = []
for key in [1,2,3,4,5,6,8]:
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
filtered_information = information[information.IlmnID.isin(AltumAge_cpgs)].copy()

# ============================================================
# DNA Sequence Processor (statistical features, 8-dim)
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
        return np.array([self.extract_statistical_features(s) for s in
                         tqdm(sequences, desc='Extracting seq features')],
                        dtype=np.float32)

# ============================================================
# Graph construction (identical to training)
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
    return edge_index, edge_attr, adj  # also return adj for MRN co-methylation annotation

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
# Wrapper: fix x_seq, expose x_original to explainer3
# node_mask_type='object' needs the full x as its input
# ============================================================
class WrapperOriginal(nn.Module):
    """Fix x_seq globally; let GNNExplainer optimise node masks over x_original."""
    def __init__(self, model, x_seq_fixed):
        super().__init__()
        self.model = model
        self.register_buffer('x_seq_fixed', x_seq_fixed)

    def forward(self, x, edge_index, edge_attr, batch=None):
        return self.model(x, self.x_seq_fixed, edge_index, edge_attr, batch)

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
# Data loading — replicate fold 2 split (identical to training)
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
y_fold_test  = test_combined.age.reset_index(drop=True)
X_fold_test  = X_fold_test.reset_index(drop=True)
print(f'Train: {len(X_fold_train)}, Test: {len(X_fold_test)}')

# Keep a copy of test methylation values for hypo/hypermethylation classification
# We need each CpG's mean methylation value trend across samples vs age
meth_test_values = X_fold_test.copy()   # [756, 20318]

# ============================================================
# Build graph
# ============================================================
print('\nBuilding co-methylation graph...')
edge_index, edge_attr, adj_matrix = make_graph(
    filtered_information, THRESHOLD_CORR, THRESHOLD_DIST, X_fold_train
)
# Save edge_index for MRN later
np.save(os.path.join(OUTPUT_DIR, 'edge_index.npy'), edge_index.numpy())
np.save(os.path.join(OUTPUT_DIR, 'edge_attr.npy'),  edge_attr.numpy())

# ============================================================
# Extract sequence features
# ============================================================
print('\nExtracting sequence features...')
processor         = DNASequenceProcessor()
sequences         = filtered_information['TopGenomicSeq'].values
seq_data          = processor.extract_all_statistical_features(sequences)   # [20318, 8]
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
print('\nLoading model...')
deg  = compute_deg(test_loader)
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
# Build explainer3 wrapper (fix x_seq, object mask on x_original)
# ============================================================
wrapper = WrapperOriginal(model, seq_tensor_global.to(device)).to(device)
wrapper.eval()

explainer3 = Explainer(
    model=wrapper,
    algorithm=GNNExplainer(epochs=120),
    explanation_type='phenomenon',
    node_mask_type='object',        # ← scalar per node (which CpG site matters)
    edge_mask_type='object',        # ← scalar per edge (for MRN)
    model_config=dict(
        mode='regression',
        task_level='graph',
        return_type='raw',
    ),
)

# ============================================================
# Run explainer3 on all test samples
# ============================================================
print('\n' + '='*60)
print('PART B: Node importance (object mask) — which CpG sites matter')
print(f'Running on {len(test_loader)} test samples...')
print('='*60)

n_age_groups          = 8
age_group_sum_node    = np.zeros((n_age_groups, number_of_cpgs))   # [8, 20318]
age_group_sum_edge    = np.zeros((n_age_groups, edge_index.shape[1]))  # [8, num_edges]
age_group_count       = np.zeros(n_age_groups)
all_node_importance   = []    # [n_samples, 20318]  per-sample node importance
all_edge_importance   = []    # [n_samples, num_edges]
all_ages              = []

SAVE_EVERY = 10   # checkpoint every 10 samples

for cnt, data in enumerate(tqdm(test_loader, desc='Node explainer')):
    data = data.to(device)
    age  = data.y.cpu().item()
    ag   = age_group(age)

    explanation = explainer3(
        x          = data.x,
        edge_index = data.edge_index,
        edge_attr  = data.edge_attr,
        batch      = data.batch_size if hasattr(data, 'batch_size') else None,
        target     = data.y
    )

    node_imp = explanation.node_mask.cpu().detach().flatten().numpy()  # [20318]
    edge_imp = explanation.edge_mask.cpu().detach().flatten().numpy()  # [num_edges]

    age_group_sum_node[ag] += node_imp
    age_group_sum_edge[ag] += edge_imp
    age_group_count[ag]    += 1

    all_node_importance.append(node_imp)
    all_edge_importance.append(edge_imp)
    all_ages.append(age)

    # Periodic checkpoint save
    if (cnt + 1) % SAVE_EVERY == 0 or cnt == len(test_loader) - 1:
        np.save(os.path.join(OUTPUT_DIR, 'age_group_sum_node.npy'),  age_group_sum_node)
        np.save(os.path.join(OUTPUT_DIR, 'age_group_sum_edge.npy'),  age_group_sum_edge)
        np.save(os.path.join(OUTPUT_DIR, 'age_group_count.npy'),     age_group_count)
        np.save(os.path.join(OUTPUT_DIR, 'node_importance.npy'),     np.array(all_node_importance))
        np.save(os.path.join(OUTPUT_DIR, 'edge_importance.npy'),     np.array(all_edge_importance))
        np.save(os.path.join(OUTPUT_DIR, 'all_ages.npy'),            np.array(all_ages))
        print(f'  [Checkpoint] {cnt+1}/{len(test_loader)} samples saved.')

print('Explainer done. Computing averages...')

# ============================================================
# Average by age group
# ============================================================
safe_count = np.maximum(age_group_count[:, None], 1)
age_group_mean_node = age_group_sum_node / safe_count             # [8, 20318]
age_group_mean_edge = age_group_sum_edge / safe_count             # [8, num_edges]
mean_node_overall   = np.array(all_node_importance).mean(axis=0) # [20318]

np.save(os.path.join(OUTPUT_DIR, 'age_group_mean_node.npy'), age_group_mean_node)
np.save(os.path.join(OUTPUT_DIR, 'age_group_mean_edge.npy'), age_group_mean_edge)
np.save(os.path.join(OUTPUT_DIR, 'mean_node_overall.npy'),   mean_node_overall)
print('Averages saved.')

# ============================================================
# Temporal slope analysis (linear regression: importance ~ age)
# correspondence with paper Figure 6
# ============================================================
print('\nComputing temporal slopes (linear regression per CpG)...')

ages_arr      = np.array(all_ages)                      # [756]
node_arr      = np.array(all_node_importance)           # [756, 20318]

slopes = np.zeros(number_of_cpgs)
for i in range(number_of_cpgs):
    slope, _, _, _, _ = stats.linregress(ages_arr, node_arr[:, i])
    slopes[i] = slope

np.save(os.path.join(OUTPUT_DIR, 'node_importance_slopes.npy'), slopes)

# Top 10 upward / downward trending CpG sites
top10_up_idx   = np.argsort(slopes)[::-1][:10]
top10_down_idx = np.argsort(slopes)[:10]

# Build a human-readable table with gene names
def node_to_gene(idx):
    cpg_name = node2cpg.get(idx, f'node_{idx}')
    if cpg_name in filtered_information.index:
        gene = filtered_information.loc[cpg_name, 'Symbol'] if 'Symbol' in filtered_information.columns else 'N/A'
    else:
        gene = 'N/A'
    return cpg_name, gene

up_records   = [(node_to_gene(i)[0], node_to_gene(i)[1], slopes[i], mean_node_overall[i])
                for i in top10_up_idx]
down_records = [(node_to_gene(i)[0], node_to_gene(i)[1], slopes[i], mean_node_overall[i])
                for i in top10_down_idx]

df_up   = pd.DataFrame(up_records,   columns=['CpG_site', 'Gene', 'Slope', 'Mean_Importance'])
df_down = pd.DataFrame(down_records, columns=['CpG_site', 'Gene', 'Slope', 'Mean_Importance'])

df_up.to_csv(os.path.join(OUTPUT_DIR,   'top10_upward_cpg.csv'),   index=False)
df_down.to_csv(os.path.join(OUTPUT_DIR, 'top10_downward_cpg.csv'), index=False)

print('Top 10 upward-trending CpG sites:')
print(df_up.to_string(index=False))
print('\nTop 10 downward-trending CpG sites:')
print(df_down.to_string(index=False))

# ============================================================
# Hypo / Hypermethylation classification
# For each CpG site: compute slope of methylation value vs age
# across test samples. Positive slope = hypermethylating with age.
# ============================================================
print('\nClassifying hypo/hypermethylation per CpG site...')

meth_slopes = np.zeros(number_of_cpgs)
for i in range(number_of_cpgs):
    slope, _, _, _, _ = stats.linregress(ages_arr, meth_test_values.iloc[:, i].values)
    meth_slopes[i] = slope

# Positive meth_slope → hypermethylating (methylation increases with age)
# Negative meth_slope → hypomethylating  (methylation decreases with age)
is_hyper = meth_slopes > 0   # bool array [20318]
np.save(os.path.join(OUTPUT_DIR, 'meth_slopes.npy'), meth_slopes)
np.save(os.path.join(OUTPUT_DIR, 'is_hyper.npy'),    is_hyper.astype(int))
print(f'  Hypermethylating: {is_hyper.sum():,}  Hypomethylating: {(~is_hyper).sum():,}')

# ============================================================
# Plotting: Figure 6A/B equivalent
# ============================================================
print('\nGenerating plots...')

# ── Plot 1: Top-10 upward trending CpG sites importance vs age
fig, ax = plt.subplots(figsize=(11, 5))
colors_up = plt.cm.tab10(np.linspace(0, 1, 10))
sorted_by_age = np.argsort(ages_arr)
ages_sorted   = ages_arr[sorted_by_age]
node_sorted   = node_arr[sorted_by_age]
for rank, idx in enumerate(top10_up_idx):
    cpg_name, gene = node_to_gene(idx)
    label = f'{cpg_name} ({gene})'
    ax.scatter(ages_sorted, node_sorted[:, idx],
               s=5, alpha=0.4, color=colors_up[rank], label=label)
ax.set_xlabel('Chronological Age (years)', fontsize=12)
ax.set_ylabel('Node Importance Score', fontsize=12)
ax.set_title('Top 10 Upward-Trending CpG Sites (Importance Increases with Age)', fontsize=12)
ax.legend(fontsize=6.5, ncol=2, loc='upper left',
          markerscale=2, framealpha=0.7)
ax.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'top10_upward_trend.png'), dpi=300, bbox_inches='tight')
plt.close()
print('  Saved: top10_upward_trend.png')

# ── Plot 2: Top-10 downward trending CpG sites importance vs age
fig, ax = plt.subplots(figsize=(11, 5))
colors_down = plt.cm.Set2(np.linspace(0, 1, 10))
for rank, idx in enumerate(top10_down_idx):
    cpg_name, gene = node_to_gene(idx)
    label = f'{cpg_name} ({gene})'
    ax.scatter(ages_sorted, node_sorted[:, idx],
               s=5, alpha=0.4, color=colors_down[rank], label=label)
ax.set_xlabel('Chronological Age (years)', fontsize=12)
ax.set_ylabel('Node Importance Score', fontsize=12)
ax.set_title('Top 10 Downward-Trending CpG Sites (Importance Decreases with Age)', fontsize=12)
ax.legend(fontsize=6.5, ncol=2, loc='upper right',
          markerscale=2, framealpha=0.7)
ax.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'top10_downward_trend.png'), dpi=300, bbox_inches='tight')
plt.close()
print('  Saved: top10_downward_trend.png')

# ── Plot 3: Mean node importance distribution (histogram)
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(mean_node_overall, bins=100, color='#5B8DB8', edgecolor='none', alpha=0.8)
ax.set_xlabel('Mean Node Importance Score', fontsize=12)
ax.set_ylabel('Number of CpG Sites', fontsize=12)
ax.set_title('Distribution of Mean Node Importance across 20,318 CpG Sites', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'node_importance_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print('  Saved: node_importance_distribution.png')

# ── Plot 4: Top-50 most important CpG sites (bar chart)
top50_idx   = np.argsort(mean_node_overall)[::-1][:50]
top50_names = [f"{node_to_gene(i)[0]}\n({node_to_gene(i)[1]})" for i in top50_idx]
top50_vals  = mean_node_overall[top50_idx]

fig, ax = plt.subplots(figsize=(18, 6))
colors_bar = ['#D95F5F' if is_hyper[i] else '#5B8DB8' for i in top50_idx]
ax.bar(range(50), top50_vals, color=colors_bar, edgecolor='none')
ax.set_xticks(range(50))
ax.set_xticklabels(top50_names, rotation=90, fontsize=6)
ax.set_ylabel('Mean Importance Score', fontsize=11)
ax.set_title('Top 50 CpG Sites by Mean Importance\n(Red = Hypermethylating, Blue = Hypomethylating)',
             fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'top50_node_importance.png'), dpi=300, bbox_inches='tight')
plt.close()
print('  Saved: top50_node_importance.png')

# ============================================================
# MRN Subnetwork Visualisation using Graphviz
# Corresponds to original paper Figure 5A
# ============================================================
print('\n' + '='*60)
print('Generating MRN subnetwork visualisations...')
print('='*60)

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
    print('Graphviz available.')
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print('WARNING: graphviz Python package not found. Skipping MRN visualisation.')
    print('Install with: pip install graphviz')
    print('Also ensure system graphviz is installed: apt-get install graphviz')

if GRAPHVIZ_AVAILABLE:
    edge_index_np = edge_index.numpy()   # [2, num_edges]
    edge_attr_np  = edge_attr.numpy()    # [num_edges, 3]

    # Pick a representative age group to visualise (e.g., age group 2: 20-45)
    # You can change this to visualise different age groups
    VIS_AGE_GROUPS = [2, 3, 4]   # 20-45, 45-55, 55-65

    for ag in VIS_AGE_GROUPS:
        ag_label = AGE_GROUP_LABELS[ag]
        print(f'\n  Building MRN for age group {ag_label}...')

        mean_node_ag = age_group_mean_node[ag]   # [20318]
        mean_edge_ag = age_group_mean_edge[ag]   # [num_edges]

        # Step 1: find non-zero importance nodes
        nonzero_node_mask = mean_node_ag > 0
        nonzero_node_ids  = np.where(nonzero_node_mask)[0]
        print(f'  Non-zero importance nodes: {len(nonzero_node_ids)}')

        # Step 2: find edges where BOTH endpoints have non-zero importance
        #         AND edge importance > threshold
        src_arr = edge_index_np[0]
        dst_arr = edge_index_np[1]

        edge_mask = (
            nonzero_node_mask[src_arr] &
            nonzero_node_mask[dst_arr] &
            (mean_edge_ag > EDGE_IMPORTANCE_THRESHOLD)
        )
        kept_src  = src_arr[edge_mask]
        kept_dst  = dst_arr[edge_mask]
        kept_edge_imp  = mean_edge_ag[edge_mask]
        kept_edge_attr = edge_attr_np[edge_mask]   # co-meth, same_chrom, same_gene

        print(f'  Edges after filtering: {edge_mask.sum()}')

        # Step 3: find connected components (subnetworks)
        # Build adjacency as dict
        from collections import defaultdict, deque

        adj_dict = defaultdict(set)
        for s, d in zip(kept_src, kept_dst):
            adj_dict[int(s)].add(int(d))
            adj_dict[int(d)].add(int(s))

        visited    = set()
        components = []

        all_involved_nodes = set(kept_src.tolist() + kept_dst.tolist())
        for start_node in all_involved_nodes:
            if start_node in visited:
                continue
            # BFS
            component = []
            queue = deque([start_node])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                for nb in adj_dict[node]:
                    if nb not in visited:
                        queue.append(nb)
            components.append(component)

        # Keep only components with >= MIN_SUBNETWORK_SIZE nodes
        large_components = [c for c in components if len(c) >= MIN_SUBNETWORK_SIZE]
        print(f'  Subnetworks with >= {MIN_SUBNETWORK_SIZE} nodes: {len(large_components)}')

        if len(large_components) == 0:
            print(f'  No large subnetworks found for age group {ag_label}. Skipping.')
            continue

        # Step 4: for each large subnetwork, draw Graphviz graph
        # (same style as original paper Figure 5A)
        mrn_dir = os.path.join(OUTPUT_DIR, f'mrn_age_{ag_label.replace("-", "_")}')
        os.makedirs(mrn_dir, exist_ok=True)

        for sub_idx, component in enumerate(large_components):
            component_set = set(component)

            # Build edge list for this component
            comp_edges = [
                (int(s), int(d), float(ei), e_a)
                for s, d, ei, e_a in zip(kept_src, kept_dst, kept_edge_imp, kept_edge_attr)
                if int(s) in component_set and int(d) in component_set
            ]
            # Deduplicate undirected edges (keep one direction)
            seen_edges = set()
            comp_edges_dedup = []
            for s, d, ei, e_a in comp_edges:
                pair = (min(s, d), max(s, d))
                if pair not in seen_edges:
                    seen_edges.add(pair)
                    comp_edges_dedup.append((s, d, ei, e_a))

            print(f'    Subnetwork {sub_idx+1}: {len(component)} nodes, '
                  f'{len(comp_edges_dedup)} edges')

            # Create Graphviz graph
            dot = graphviz.Graph(
                name=f'MRN_age{ag_label}_sub{sub_idx+1}',
                engine='neato',
                graph_attr={
                    'overlap': 'false',
                    'splines': 'true',
                    'sep':     '+5',
                    'fontsize': '10',
                }
            )

            # Add nodes
            for node_id in component:
                cpg_name, gene = node_to_gene(node_id)
                node_imp_val   = mean_node_ag[node_id]
                hyper_val      = meth_slopes[node_id]    # positive = hyper, negative = hypo
                slope_val      = slopes[node_id]         # importance slope

                # Node colour: red = hypomethylating, blue = hypermethylating
                fill_color = '#D95F5F' if is_hyper[node_id] else '#5B8DB8'

                # TSS distance for annotation
                if cpg_name in filtered_information.index:
                    dist_tss = filtered_information.loc[cpg_name, 'Distance_to_TSS'] \
                               if 'Distance_to_TSS' in filtered_information.columns else 0.0
                    chrom    = filtered_information.loc[cpg_name, 'Chr'] \
                               if 'Chr' in filtered_information.columns else 'N/A'
                else:
                    dist_tss = 0.0
                    chrom    = 'N/A'

                # Build node label (same style as original paper)
                label = (f'{cpg_name}\n'
                         f'{gene}\n'
                         f'dist_tss:{dist_tss:.0f}\n'
                         f'imprtnce:{node_imp_val:.2f}\n'
                         f'{"hyper" if is_hyper[node_id] else "hypo"}:{abs(hyper_val):.2f}\n'
                         f'Chrom:{chrom}')

                # Green/yellow circle: importance trend
                if abs(slope_val) < 1e-6:
                    pen_color = 'gray'
                    pen_width = '1'
                elif slope_val > 0:
                    pen_color = 'green'
                    pen_width = '3'
                else:
                    pen_color = 'gold'
                    pen_width = '3'

                # Node size proportional to importance
                node_size = max(0.4, node_imp_val * 3.0)

                dot.node(
                    str(node_id),
                    label=label,
                    shape='circle',
                    style='filled',
                    fillcolor=fill_color,
                    color=pen_color,
                    penwidth=pen_width,
                    fontsize='7',
                    fontcolor='white',
                    width=str(node_size),
                    fixedsize='false',
                )

            # Add edges
            for s, d, ei, e_a in comp_edges_dedup:
                cometh_val = float(e_a[0])
                edge_label = f'cometh:{cometh_val:.2f}\nimprtnce:{ei:.2f}'
                # Edge width proportional to importance
                pw = max(0.5, ei * 5.0)
                dot.edge(
                    str(s), str(d),
                    label=edge_label,
                    fontsize='6',
                    penwidth=str(pw),
                    color='#555555',
                )

            # Render
            out_name = f'mrn_sub{sub_idx+1}_n{len(component)}'
            out_path = os.path.join(mrn_dir, out_name)
            try:
                dot.render(out_path, format='png', cleanup=True)
                print(f'    Saved: {out_path}.png')
            except Exception as e:
                # Also try saving the .gv source for manual rendering
                dot.save(out_path + '.gv')
                print(f'    Graphviz render failed ({e}). Saved .gv source: {out_path}.gv')
                print(f'    You can render manually with: dot -Tpng {out_path}.gv -o {out_path}.png')

        print(f'  MRN visualisation complete for age group {ag_label}.')

# ============================================================
# Summary CSV — top 100 most important CpG sites with gene info
# ============================================================
print('\nGenerating summary CSV...')
top100_idx = np.argsort(mean_node_overall)[::-1][:100]
summary_records = []
for rank, idx in enumerate(top100_idx):
    cpg_name, gene = node_to_gene(idx)
    summary_records.append({
        'Rank':              rank + 1,
        'CpG_site':          cpg_name,
        'Gene':              gene,
        'Mean_Importance':   mean_node_overall[idx],
        'Slope':             slopes[idx],
        'Trend':             'up' if slopes[idx] > 0 else 'down',
        'Methylation':       'hyper' if is_hyper[idx] else 'hypo',
        'Meth_Slope':        meth_slopes[idx],
        'Age_Group_0_Imp':   age_group_mean_node[0, idx],
        'Age_Group_20_45':   age_group_mean_node[2, idx],
        'Age_Group_65_75':   age_group_mean_node[5, idx],
        'Age_Group_80plus':  age_group_mean_node[7, idx],
    })

df_summary = pd.DataFrame(summary_records)
df_summary.to_csv(os.path.join(OUTPUT_DIR, 'top100_node_summary.csv'), index=False)
print('  Saved: top100_node_summary.csv')

# ============================================================
# Final summary
# ============================================================
print('\n' + '='*60)
print('All done. Results saved to:')
print(f'  {OUTPUT_DIR}')
print('Files:')
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f'  - {f}')
print('='*60)
