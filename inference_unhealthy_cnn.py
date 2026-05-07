"""
inference_unhealthy_cnn.py
--------------------------
Runs PNA-GNN + CNN Sequence Encoding on the unhealthy dataset
and merges results into existing unhealthy CSVs.

Run on Spartan:
  cd /data/gpfs/projects/punim2698/yao14/graphage_seq
  source /data/projects/punim2698/yao14/venvs/ham_env/bin/activate
  python inference_unhealthy_cnn.py
"""

import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import PNAConv

# ============================================================
# Paths
# ============================================================
BASE_DIR      = '/data/gpfs/projects/punim2698/yao14/graphage_seq'
DATA_DIR      = os.path.join(BASE_DIR, 'all-organs4/all_organs')
UNHEALTHY_DIR = os.path.join(BASE_DIR, 'unhelathy-dataset/Unhealthy Normalized')
ALTUMAGE_CPGS = os.path.join(BASE_DIR, 'graph-age/example_dependencies/multi_platform_cpgs.pkl')
CPG_INFO_PATH = os.path.join(BASE_DIR, 'cpgsite-info/GPL8490_HumanMethylation27_270596_v.1.2.csv')
CKPT_CNN      = os.path.join(BASE_DIR, 'graphagewithCNN_3.26_checkpoint/cv_fold2_best_model.pth')
BASELINES_DIR = os.path.join(BASE_DIR, 'baselines')
PRED_CSV      = os.path.join(BASELINES_DIR, 'unhealthy_predictions.csv')
ACCEL_CSV     = os.path.join(BASELINES_DIR, 'unhealthy_acceleration.csv')

# ============================================================
# Reproducibility
# ============================================================
SEED         = 0
K_FOLDS      = 5
DESIRED_FOLD = 2

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

THRESHOLD_CORR      = 0.70
SECONDARY_THRESHOLD = THRESHOLD_CORR - 0.02
TERTIARY_THRESHOLD  = THRESHOLD_CORR - 0.04
THRESHOLD_DIST      = 1e5
EDGE_DIM            = 3

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
# Load CpG site information
# ============================================================
print('Loading CpG site information...')
information = pd.read_csv(CPG_INFO_PATH, skiprows=7, low_memory=False)
information.dropna(subset=['Chr'], inplace=True)
information[['start','end']] = (
    information.CPG_ISLAND_LOCATIONS
    .fillna('0:0-0').str.split(':').str[1]
    .str.split('-', expand=True).astype(int))
information['CPG_ISLAND']     = information['CPG_ISLAND'].astype(int)
information['CPG_ISLAND_LEN'] = information.end - information.start
information.MapInfo            = information.MapInfo.astype(int)

sc = MinMaxScaler()
for col in ['MapInfo','TSS_Coordinate']:
    information[f'Normalized_{col}'] = information.groupby('Chr')[col].transform(
        lambda x: sc.fit_transform(x.values.reshape(-1,1)).flatten())
information = pd.get_dummies(information, columns=['Gene_Strand'])
for col in ['start','end','CPG_ISLAND_LEN']:
    information[col] = information.groupby('Chr')[col].transform(
        lambda x: sc.fit_transform(x.values.reshape(-1,1)).flatten())
information = pd.get_dummies(information, columns=['Next_Base'])
information['Distance_to_TSS'] = information.groupby('Chr')['Distance_to_TSS'].transform(
    lambda x: sc.fit_transform(x.values.reshape(-1,1)).flatten())
information['Distance_to_TSS'] = information['Distance_to_TSS'].fillna(1)
information = pd.get_dummies(information, columns=['SourceStrand'])
Chrom = information.Chr.tolist()
information = pd.get_dummies(information, columns=['Chr'])
information['Chr'] = Chrom
information.index  = information.IlmnID

NODE_FEATURE_DICT = {
    1:'CPG_ISLAND', 2:'CPG_ISLAND_LEN', 3:'Distance_to_TSS',
    4:['Next_Base_A','Next_Base_C','Next_Base_T'],
    5:'start', 6:'end', 7:'Normalized_TSS_Coordinate',
    8:'Normalized_MapInfo', 9:[f'Chr_{x}' for x in range(1,23)]
}
selected_feature_numbers = list(map(int, '1,2,3,4,5,6,8'.split(',')))
user_selected_features   = []
for key in selected_feature_numbers:
    feat = NODE_FEATURE_DICT[key]
    if isinstance(feat, list):
        user_selected_features.extend(feat)
    else:
        user_selected_features.append(feat)

INPUT_DIM            = len(user_selected_features) + 1
filtered_information = information[information.IlmnID.isin(AltumAge_cpgs)]

# ============================================================
# One-hot encode DNA sequences
# ============================================================
print('Encoding DNA sequences (one-hot)...')
BASE_TO_IDX = {'A':0, 'C':1, 'G':2, 'T':3}
SEQ_LEN     = 122

def encode_sequence(seq):
    if pd.isna(seq) or not isinstance(seq, str):
        return np.zeros((SEQ_LEN, 4), dtype=np.float32)
    seq = seq.replace('[CG]','CG').upper()
    arr = np.zeros((SEQ_LEN, 4), dtype=np.float32)
    for i, base in enumerate(seq[:SEQ_LEN]):
        if base in BASE_TO_IDX:
            arr[i, BASE_TO_IDX[base]] = 1.0
    return arr

sequences   = filtered_information['TopGenomicSeq'].values
seq_onehot  = np.stack([encode_sequence(s) for s in sequences])  # (20318, 122, 4)
seq_tensor  = torch.tensor(seq_onehot, dtype=torch.float32)
print(f'  seq_onehot shape: {seq_onehot.shape}')

# ============================================================
# Load healthy training data for graph construction
# ============================================================
print('\nLoading healthy training data...')

def select_healthy(d):
    a = d[d.columns[d.columns.isin(AltumAge_cpgs)].tolist() +
          ['age','gender','dataset','tissue_type']]
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

X_fold_train = fold_train.drop(
    columns=['age','gender','dataset','tissue_type']).astype('float')

# ============================================================
# Build co-methylation graph
# ============================================================
print('\nBuilding co-methylation graph...')
chromosomes_arr = filtered_information.Chr.values
genes_arr       = filtered_information.Symbol.values
chromosome_adj  = (chromosomes_arr[:,None] == chromosomes_arr).astype(np.float32)
base_pair       = filtered_information.MapInfo.values
distance_matrix = squareform(pdist(base_pair.reshape(-1,1)))
genes_adj       = (genes_arr[:,None] == genes_arr).astype(np.float32)
adj             = np.corrcoef(X_fold_train.to_numpy(), rowvar=False)

src, dst = np.where(
    (((chromosome_adj==1) &
      ((np.abs(adj)>SECONDARY_THRESHOLD) |
       ((np.abs(distance_matrix)<THRESHOLD_DIST) &
        (np.abs(adj)>TERTIARY_THRESHOLD))))
     | (np.abs(adj)>THRESHOLD_CORR))
    & (np.arange(adj.shape[0])[:,None] != np.arange(adj.shape[1]))
)
weights    = np.column_stack([adj[src,dst], chromosome_adj[src,dst], genes_adj[src,dst]])
edge_index = torch.stack([torch.tensor(src,dtype=torch.int64),
                           torch.tensor(dst,dtype=torch.int64)], dim=0)
edge_attr  = torch.tensor(weights, dtype=torch.float32).reshape(-1, EDGE_DIM)
print(f'  Edges: {edge_index.shape[1]}')

# ============================================================
# Load unhealthy dataset
# ============================================================
print('\nLoading unhealthy dataset...')

unhealthy_frames = []
for filename in os.listdir(UNHEALTHY_DIR):
    if filename.endswith('.pkl'):
        df = pd.read_pickle(os.path.join(UNHEALTHY_DIR, filename))
        cols = [c for c in df.columns if c in AltumAge_cpgs]
        cols += ['age','gender','dataset','tissue_type']
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
# CNN Model definition
# ============================================================
class HierarchicalNetCNN(nn.Module):
    def __init__(self, deg, seq_len=122, original_dim=10,
                 seq_embed_dim=32, num_cpgs=20318):
        super().__init__()
        self.seq_cnn = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, seq_embed_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(seq_embed_dim), nn.ReLU(),
        )
        self.seq_pool = nn.AdaptiveMaxPool1d(1)
        self.importance_gate = nn.Sequential(
            nn.Linear(seq_embed_dim, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )
        self.seq_proj = nn.Sequential(
            nn.Linear(seq_embed_dim, 16), nn.ReLU(),
            nn.Linear(16, 2)
        )
        aggregators = ['mean','max','std','min']
        scalers     = ['identity','amplification','attenuation']
        self.LastLayer = PNAConv(
            in_channels=original_dim+2, out_channels=1,
            aggregators=aggregators, scalers=scalers, deg=deg,
            edge_dim=3, towers=1, pre_layers=1, post_layers=1,
            divide_input=False
        )
        self.mlp = nn.Sequential(
            nn.Linear(num_cpgs,1024), nn.ReLU(),
            nn.Linear(1024,656),      nn.SELU(),
            nn.Linear(656,256),       nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256,124),       nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(124,64),        nn.SELU(),
            nn.Linear(64,32),         nn.ReLU(),
            nn.Linear(32,8),          nn.ReLU(),
            nn.Linear(8,1)
        )

    def forward(self, x_original, x_seq, edge_index, edge_attr, batch):
        cnn_in  = x_seq.permute(0,2,1)          # (N,122,4) -> (N,4,122)
        cnn_out = self.seq_cnn(cnn_in)           # (N,32,122)
        cnn_out = self.seq_pool(cnn_out).squeeze(-1)  # (N,32)
        gate    = self.importance_gate(cnn_out)  # (N,1)
        meth    = x_original[:,0:1] * gate
        proj    = self.seq_proj(cnn_out)         # (N,2)
        x       = torch.cat([meth, x_original[:,1:], proj], dim=-1)
        x       = self.LastLayer(x, edge_index, edge_attr)
        x       = F.relu(x)
        return self.mlp(x.T).flatten()

# ============================================================
# Build DataLoader
# ============================================================
def make_loader_cnn(X_, y_, batch_size=1):
    graphs = []
    for row in range(len(X_)):
        x = pd.concat([X_.iloc[row,:],
                        filtered_information[user_selected_features]],
                       axis=1, join='inner')
        x_t = torch.tensor(x.to_numpy().astype('float'), dtype=torch.float32)
        y_t = torch.tensor(float(y_.iloc[row]), dtype=torch.float32)
        graphs.append(Data(x=x_t, x_seq=seq_tensor, y=y_t,
                           edge_index=edge_index, edge_attr=edge_attr))
    return DataLoader(graphs, batch_size=batch_size)

def compute_deg(loader):
    max_degree = -1
    for data in loader:
        max_degree = max(max_degree, int(data.edge_index[1].max()))
        break
    deg = torch.zeros(max_degree+1, dtype=torch.long)
    for data in loader.dataset:
        d = torch.bincount(data.edge_index[1].cpu(), minlength=max_degree+1)
        deg += d
    return deg

print('\nBuilding unhealthy DataLoader...')
unhealthy_loader = make_loader_cnn(X_unhealthy, y_unhealthy)
deg_ref          = compute_deg(unhealthy_loader)

# ============================================================
# Load and run CNN model
# ============================================================
print('\n--- PNA-GNN + CNN Sequence Encoding ---')
ckpt = torch.load(CKPT_CNN, map_location=device)
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    state_dict = ckpt['model_state_dict']
    saved_deg  = ckpt.get('deg', deg_ref)
else:
    state_dict = ckpt
    saved_deg  = deg_ref

model = HierarchicalNetCNN(
    deg=saved_deg.to('cpu'),
    original_dim=INPUT_DIM,
    num_cpgs=len(AltumAge_cpgs)).to(device)
model.load_state_dict(state_dict)
model.eval()

preds = []
with torch.no_grad():
    for data in tqdm(unhealthy_loader, desc='  CNN inference'):
        data = data.to(device)
        out  = model(data.x, data.x_seq,
                     data.edge_index, data.edge_attr, data.batch)
        preds.append(out.cpu().numpy())

pred_cnn = np.concatenate(preds).flatten()
mae_cnn  = mean_absolute_error(y_unhealthy, pred_cnn)
print(f'  MAE={mae_cnn:.3f}')
print(f'  Mean age acceleration: {(pred_cnn - y_unhealthy.values).mean():.3f}')

# ============================================================
# Merge into existing predictions CSV
# ============================================================
print('\nMerging into existing predictions CSV...')
if os.path.exists(PRED_CSV):
    existing = pd.read_csv(PRED_CSV)
    existing['PNA-GNN+CNN']       = pred_cnn
    existing['PNA-GNN+CNN_accel'] = pred_cnn - existing['true_age']
    existing.to_csv(PRED_CSV, index=False)
    print(f'  Updated: {PRED_CSV}')

# ============================================================
# Update acceleration summary CSV
# ============================================================
if os.path.exists(ACCEL_CSV):
    summary = pd.read_csv(ACCEL_CSV)
    for disease in unhealthy_data['disease'].unique():
        mask  = (unhealthy_data['disease'] == disease).values
        accel = pred_cnn[mask] - y_unhealthy.values[mask]
        idx   = summary[summary['disease'] == disease].index
        if len(idx) > 0:
            summary.loc[idx, 'PNA-GNN+CNN_mean_accel'] = round(float(accel.mean()), 3)
            summary.loc[idx, 'PNA-GNN+CNN_std_accel']  = round(float(accel.std()),  3)
    summary.to_csv(ACCEL_CSV, index=False)
    print(f'  Updated: {ACCEL_CSV}')

# ============================================================
# Print summary
# ============================================================
print('\n' + '='*60)
print('PNA-GNN + CNN AGE ACCELERATION BY DISEASE')
print('='*60)
for disease in ['Ovarian Cancer', 'Schizophrenia', 'Osteoporosis']:
    mask  = (unhealthy_data['disease'] == disease).values
    accel = pred_cnn[mask] - y_unhealthy.values[mask]
    print(f'{disease:20s}  n={mask.sum():3d}  '
          f'mean={accel.mean():+.3f}  std={accel.std():.3f}')
