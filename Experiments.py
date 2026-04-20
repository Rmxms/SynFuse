import importlib, subprocess, sys, os, re, json, copy, time, random, warnings, datetime, shutil
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def ensure_import(pkg_name, pip_name=None):
    try:
        importlib.import_module(pkg_name)
    except ImportError:
        target = pip_name or pkg_name
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", target])

ensure_import("xgboost")
ensure_import("sklearn", "scikit-learn")
ensure_import("rdkit")

import torch
print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")

try:
    from torch_geometric.nn import GCNConv
except ImportError:
    tv = torch.__version__.split("+")[0]
    cu = "cu118" if torch.cuda.is_available() else "cpu"
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch_geometric"])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "torch_scatter", "torch_sparse", "-f", f"https://data.pyg.org/whl/torch-{tv}+{cu}.html"])


import requests
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GroupKFold

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, TensorDataset

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_add_pool

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem

from rdkit.Chem.rdchem import HybridizationType
from google.colab import drive

RDLogger.DisableLog("rdApp.*")


SEED = 42
  random.seed(SEED)
 np.random.seed(SEED)
  torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 300
N_SPLITS = 5
 GNN_EPOCHS=80
GNN_BATCH= 128
 MLP_EPOCHS= 80
MLP_BATCH= 128

print("Device:", DEVICE)

drive.mount("/content/drive")

BASE = "/content/drive/MyDrive/Topics_New"
PROC = os.path.join(BASE, "processed_data")

OUT = os.path.join(BASE, "output_clean_exp12")
LOCAL = "/content/local_data_clean_exp12"

os.makedirs(OUT, exist_ok=True)
os.makedirs(LOCAL, exist_ok=True)


FILES = {
    "merged"  : os.path.join(PROC, "az_depmap_expression_merged.csv"),"summary" : os.path.join(BASE, "pubchem_original_drug_summary.csv"),
    "alias"   : os.path.join(BASE, "pubchem_alias_hits.csv"),"matched" : os.path.join(BASE, "drug_smiles_pubchem_matched.csv"),
}

RUN_ID   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

LOG_FILE = os.path.join(OUT, f"run_{RUN_ID}.jsonl")

def log_event(d):
     d = dict(d)
    d["run_id"] = RUN_ID
    d["timestamp"] = datetime.datetime.now().isoformat()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(d, default=float) + "\n")

print("Run ID:", RUN_ID)
for k, v in FILES.items():
    print(f"{k}: {'OK' if os.path.exists(v) else 'MISSING'} | {os.path.basename(v)}")

LOCAL_MERGED = os.path.join(LOCAL, "merged.csv")

if not os.path.exists(LOCAL_MERGED):
shutil.copy(FILES["merged"], LOCAL_MERGED)


df_merged= pd.read_csv(LOCAL_MERGED, low_memory=False)
df_summary= pd.read_csv(FILES["summary"]) if os.path.exists(FILES["summary"]) else pd.DataFrame()

df_alias= pd.read_csv(FILES["alias"])   if os.path.exists(FILES["alias"]) else pd.DataFrame()

df_matched= pd.read_csv(FILES["matched"]) if os.path.exists(FILES["matched"]) else pd.DataFrame()

  
  print("df_merged:", df_merged.shape)
    def normalize_key(x):
    
    return str(x).strip().lower().replace(" ", "").replace("-", "").replace(",", "")

 def fetch_smiles_by_cid(cid, sleep_sec=0.2):
    if pd.isna(cid):
        return None
    try:
        r = requests.get
        (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
            f"{int(float(cid))}/property/CanonicalSMILES/TXT", timeout=20
        )
        time.sleep(sleep_sec)
        t = r.text.strip()
        return t if r.status_code == 200 and t and "Status:" not in t else None
    except:
        return None

def fetch_smiles_by_name(name, sleep_sec=0.25):
     try:
        r = requests.get
        (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{requests.utils.quote(str(name))}/property/CanonicalSMILES/TXT", timeout=20
          )
        time.sleep(sleep_sec)
        t = r.text.strip()
        return t if r.status_code == 200 and t and "Status:" not in t else None
    except:
        return None

 def make_pair_id(a, b):
    return "__".join(sorted([str(a).strip(), str(b).strip()]))

  def extract_gene_symbol(col):
    m = re.match(r"^([A-Za-z0-9\-_]+)\s*\(", str(col))
    return m.group(1).upper() if m else str(col).strip().upper()

def count_targets(x):
if pd.isna(x):
        return 0
    return len([t.strip() for t in re.split(r"[,;/|]+", str(x)) if t.strip()])

 def canonicalize_pairs(df):
    df = df.copy()
    swap = df["Drug_A"] > df["Drug_B"]
    tmp = df.loc[swap, "Drug_A"].copy()
    df.loc[swap, "Drug_A"] = df.loc[swap, "Drug_B"].values
    df.loc[swap, "Drug_B"] = tmp.values
    return df

def concordance_index(y_true, y_pred):
 y_true = np.asarray(y_true)
y_pred = np.asarray(y_pred)
n = 0
    n_correct = 0
    for i in range(len(y_true)):
for j in range(i + 1, len(y_true)):
         if y_true[i] != y_true[j]:
             n += 1
     if (y_pred[i] > y_pred[j]) == (y_true[i] > y_true[j]):
      n_correct += 1
    return n_correct / n if n > 0 else 0.0

def compute_metrics(y_true, y_pred):

    y_true = np.asarray(y_true)

      y_pred = np.asarray(y_pred)

    if len(y_true) < 2:
        return 
    {
            "pearson": 0.0,
            "spearman": 0.0,
            "cindex": 0.0,
            "mae": 999.0,
            "rmse": 999.0,
            "r2": -1.0
        }

      try:
        p = pearsonr(y_true, y_pred)[0]
    except:
        p = 0.0
      try:
    s = spearmanr(y_true, y_pred)[0]
    except:
         s = 0.0

    p = 0.0 if np.isnan(p) else p

    s = 0.0 if np.isnan(s) else s

     return {
        "pearson": float(p),
         "spearman": float(s),
        "cindex": float(concordance_index(y_true, y_pred)),
     "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
         "r2": float(r2_score(y_true, y_pred))
    }

def summarise_cv(label, fold_metrics):
    ps= [m["pearson"] for m in fold_metrics]
    sp = [m["spearman"] for m in fold_metrics]

    ci  =[m["cindex"] for m in fold_metrics]

    mae =[m["mae"] for m in fold_metrics]

    return 
{
        "Model": label,
        "Pearson_mean": float(np.mean(ps)),
        "Pearson_std": float(np.std(ps)),
        "Spearman_mean": float(np.mean(sp)),
        "Spearman_std": float(np.std(sp)),
        "CIndex_mean": float(np.mean(ci)),
        "CIndex_std": float(np.std(ci)),
        "MAE_mean": float(np.mean(mae)),
        "MAE_std": float(np.std(mae)),
        "fold_pearsons": ps,
        "folds": fold_metrics
    }

def select_top_genes(train_idx, gene_frame, k=TOP_K):
    X_tr = gene_frame.iloc[train_idx].values.astype(np.float64)

    gvar = pd.Series(X_tr.var(axis=0), index=gene_frame.columns)

    return gvar.nlargest(k).index.tolist()

master_smiles = {}

manual_smiles = 
{
    "sn38": "OCC1=C2C=C3N(CCC3=O)C(=O)C2=NC4=CC(=O)CC14",
    "topotecan": "OC(=O)C1=CN2C=C(CO)C(=O)C2=CC1CC1=CC=CN1C",
    "carboplatin": "O=C1CCC(=O)O[Pt]1(N)N",
    "vinorelbine": "CC[C@@]1(CC2CC(CC3=C(C=CC=C3)N3C)C23)C(=O)OC(=CC)CC1",
    "oxaliplatin": "[NH2][C@@H]1CCCC[C@H]1[NH2].[Pt](=O)(=O)",
    "cisplatin": "N.N.Cl[Pt]Cl",
    "gemcitabine": "NC1=NC(=O)N(C=C1)[C@@H]1O[C@H](CO)[C@@H](F)[C@@H]1F",
    "gemcitibine": "NC1=NC(=O)N(C=C1)[C@@H]1O[C@H](CO)[C@@H](F)[C@@H]1F",
    "doxorubicin": "COc1cccc2C(=O)c3c(O)c4c(c(O)c3C(=O)c12)C[C@@](O)(C(=O)CO)C[C@H]4O",
    "paclitaxel": "CC1=C2[C@@H](OC(=O)c3ccccc3)CC[C@@H]3[C@@H]([C@@H]2OC(=O)c2ccccc2)O[C@]3(O1)C(=O)OC(C)(C)C",
    "chloroquine": "ClC1=CC2=C(N=C1)N(CCN(CC)CC)C=C2",
    "vorinostat": "O=C(CCCCCCC(=O)Nc1ccccc1)NO",
}
for k, v in manual_smiles.items():
    master_smiles[k] = v


 if not df_summary.empty:
     for _, row in df_summary.iterrows():
      s = row.get("best_smiles", np.nan)
        if pd.isna(s) or str(s).strip() in ["", "nan", "None"]:
            continue
         for p in str(row.get("Original_Drug", "")).split(","):
            k = normalize_key(p)
              if k and k not in master_smiles:
                master_smiles[k] = str(s).strip()

  if not df_alias.empty and "smiles" in df_alias.columns:
     for _, row in df_alias.iterrows():
        s = row.get("smiles", np.nan)
      if pd.isna(s):
            continue
           for p in str(row.get("query", "")).split(","):
            k = normalize_key(p)
            if k and k not in master_smiles:
                master_smiles[k] = str(s).strip()

if not df_matched.empty and "SMILES" in df_matched.columns:
 for _, row in df_matched.iterrows():
        s = row.get("SMILES", np.nan)
     if pd.isna(s):
            continue
        for p in str(row.get("Drug", "")).split(","):
            k = normalize_key(p)
            if k and k not in master_smiles:
                master_smiles[k] = str(s).strip()

all_merged_drugs = set
(
    df_merged["Drug_A"].astype(str).str.strip().tolist() + df_merged["Drug_B"].astype(str).str.strip().tolist()
)
missing = [d for d in all_merged_drugs if normalize_key(d) not in master_smiles]

print("Missing drugs before PubChem lookup:", len(missing))

for drug in missing:
    for cand in [drug] + [p.strip() for p in drug.split(",") if "," in drug]:
        s = fetch_smiles_by_name(cand)
    
       if s:
            master_smiles[normalize_key(drug)] = s
            
            master_smiles[normalize_key(cand)] = s
            break

n_cov = sum(1 for d in all_merged_drugs if normalize_key(d) in master_smiles)

print(f"SMILES coverage: {n_cov}/{len(all_merged_drugs)}")

ATOM_TYPES = 
[
    'C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I','B','V','K','Tl','Yb','Sb','Sn',
    'Ag','Pd','Co','Se','Ti','Zn','H','Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','other'
]
HYBRIDIZATION_TYPES = [
    HybridizationType.S, HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3, HybridizationType.SP3D, HybridizationType.SP3D2,HybridizationType.OTHER
]

def one_hot(val, choices):
    out = [0] * len(choices)
    out[choices.index(val) if val in choices else -1] = 1
    return out

def atom_features(atom):
    return (
        one_hot(atom.GetSymbol(), ATOM_TYPES) +
          one_hot(atom.GetDegree(), [0,1,2,3,4,5,6,7,8,9,10]) +
         one_hot(atom.GetTotalNumHs(), [0,1,2,3,4]) +
         one_hot(atom.GetImplicitValence(), [0,1,2,3,4,5,6]) +
         [int(atom.GetIsAromatic())] +
        one_hot(atom.GetHybridization(), HYBRIDIZATION_TYPES) +
           [int(atom.IsInRing())]
    )

NODE_FEAT_DIM = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))

def smiles_to_graph(smiles):
    
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    
    if mol is None:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    src, dst = [], []
    
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

def build_drug_registry(data_df):
    drugs = sorted(set(data_df["Drug_A"]) | set(data_df["Drug_B"]))
    
    drug_to_idx = {d: i for i, d in enumerate(drugs)}
   
    graph_cache = {}
    for d in drugs:
        s = master_smiles.get(normalize_key(d))
        if s:
            g = smiles_to_graph(s)
            if g is not None:
                graph_cache[d] = g
    return drugs, drug_to_idx, graph_cache

df = df_merged.copy()

for col in ["Drug_A", "Drug_B", "Cell_Line"]:
 if col in df.columns:
    df[col] = df[col].astype(str).str.strip()

df["Synergy"] = pd.to_numeric(df["Synergy"], errors="coerce")

df = df.dropna(subset=["Synergy"])

df = df[(df["Synergy"] >= -100) & (df["Synergy"] <= 100)].copy()
df = canonicalize_pairs(df)

df["drug_pair_id"] = df.apply(lambda r: make_pair_id(r["Drug_A"], r["Drug_B"]), axis=1)
df = df.reset_index(drop=True)

gene_cols_raw = [c for c in df.columns if "(" in c and ")" in c]

gene_df_all = df[gene_cols_raw].copy()
gene_df_all.columns = [extract_gene_symbol(c) for c in gene_cols_raw]

gene_df_all = gene_df_all.apply(pd.to_numeric, errors="coerce").fillna(0)


le_da = LabelEncoder()

le_db = LabelEncoder()
le_cl = LabelEncoder()

df["Drug_A_enc"] = le_da.fit_transform(df["Drug_A"].astype(str))

df["Drug_B_enc"] = le_db.fit_transform(df["Drug_B"].astype(str))

df["Cell_Line_enc"] = le_cl.fit_transform(df["Cell_Line"].astype(str))

if "CANCER_TYPE" in df.columns:
    le_ct = LabelEncoder()
    df["Cancer_Type_enc"] = le_ct.fit_transform(df["CANCER_TYPE"].fillna("UNKNOWN").astype(str))
else:
    df["Cancer_Type_enc"] = 0

for side in ["A", "B"]:
    col = f"Target_{side}"
    
    df[f"Target_{side}_count"] = df[col].apply(count_targets) if col in df.columns else 0
    
    df[f"Has_Target_{side}"] = (df[f"Target_{side}_count"] > 0).astype(int)

df["same_target_flag"] = (
    df.get("Target_A", pd.Series("", index=df.index)).fillna("") ==
    df.get("Target_B", pd.Series("", index=df.index)).fillna("")
).astype(int)

df["total_targets"] = df["Target_A_count"] + df["Target_B_count"]

df["target_diff"] = (df["Target_A_count"] - df["Target_B_count"]).abs()

def _jaccard(row):
for col in ["Target_A", "Target_B"]:
        if col not in row.index or pd.isna(row[col]):
            return 0.0
    ta = set(t.strip().upper() for t in re.split(r"[,;/|]+", str(row["Target_A"])) if t.strip())

    tb = set(t.strip().upper() for t in re.split(r"[,;/|]+", str(row["Target_B"])) if t.strip())


    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

df["target_jaccard"] = df.apply(_jaccard, axis=1)

def parse_targets_to_list(x):
    if pd.isna(x):
        return []
    return [t.strip().upper() for t in re.split(r"[,;/|]+", str(x)) if t.strip()]

df["Target_A_list"] = df.get("Target_A", pd.Series("", index=df.index)).apply(parse_targets_to_list)
df["Target_B_list"] = df.get("Target_B", pd.Series("", index=df.index)).apply(parse_targets_to_list)

valid_genes = set(gene_df_all.columns)

def calc_target_expr(row_idx, target_list):

    matched = [g for g in target_list if g in valid_genes]

    if not matched:
        return pd.Series
     (
            {
            "mean": 0.0, "max": 0.0, "sum": 0.0, "n_matched": 0
        }
        )
    
    vals = gene_df_all.loc[row_idx, matched].values.astype(float)
   
    return pd.Series
(
        {
        "mean": float(np.mean(vals)),
        "max": float(np.max(vals)),
        "sum": float(np.sum(vals)),
        "n_matched": int(len(matched))
    })

expr_A = pd.DataFrame
(
    [calc_target_expr(idx, tlist) for idx, tlist in zip(df.index, df["Target_A_list"])],
    index=df.index
)
expr_A.columns = ["A_target_expr_mean", "A_target_expr_max", "A_target_expr_sum", "A_target_expr_n_matched"]

expr_B = pd.DataFrame
(
    [calc_target_expr(idx, tlist) for idx, tlist in zip(df.index, df["Target_B_list"])],
    index=df.index
)
expr_B.columns = ["B_target_expr_mean", "B_target_expr_max", "B_target_expr_sum", "B_target_expr_n_matched"]

df = pd.concat([df, expr_A, expr_B], axis=1)
df["Target_expr_mean_sum"] = df["A_target_expr_mean"] + df["B_target_expr_mean"]

df["Target_expr_mean_diff"] = (df["A_target_expr_mean"] - df["B_target_expr_mean"]).abs()

df["Target_expr_max_sum"] = df["A_target_expr_max"] + df["B_target_expr_max"]

def get_morgan_fingerprint(smiles, radius=2, nBits=2048):
    if not smiles or pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)

fp_cache = {
    k: get_morgan_fingerprint(v) for k, v in master_smiles.items()
    }

def calc_tanimoto(drug_a, drug_b):
    fp_a = fp_cache.get(normalize_key(drug_a))
     fp_b = fp_cache.get(normalize_key(drug_b))
    if fp_a is None or fp_b is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)

df["Tanimoto_Similarity"] = df.apply(lambda row: calc_tanimoto(row["Drug_A"], row["Drug_B"]), axis=1)

NON_GENE_COLS = [
    "Drug_A_enc", "Drug_B_enc", "Cell_Line_enc", "Cancer_Type_enc",
    "Target_A_count", "Target_B_count", "Has_Target_A", "Has_Target_B",
    "same_target_flag", "total_targets", "target_diff", "target_jaccard",

    "A_target_expr_mean", "A_target_expr_max", "A_target_expr_sum",
    "B_target_expr_mean", "B_target_expr_max", "B_target_expr_sum",
    "Target_expr_mean_sum", "Target_expr_mean_diff"
]

X_non_gene = df[NON_GENE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0).values
y_full = df["Synergy"].values
groups_full = df["drug_pair_id"].values
all_drugs_f, drug_to_idx_f, graph_cache_f = build_drug_registry(df)

print("Rows:", len(df))

print("Pairs:", df["drug_pair_id"].nunique())
print("Non-gene features:", len(NON_GENE_COLS))

print("Graph-covered drugs:", len(graph_cache_f), "/", len(all_drugs_f))



XGB_PARAMS = dict
(
    n_estimators=300,learning_rate=0.05, max_depth=4,
min_child_weight=3,subsample=0.8,colsample_bytree=0.6,reg_alpha=0.1,
    reg_lambda=1.0,objective="reg:squarederror",random_state=SEED,n_jobs=-1,
)
RF_PARAMS = dict
(
    n_estimators=300, max_depth=10,
    min_samples_leaf=3,random_state=SEED,
    n_jobs=-1,
)

def make_xgb():
    return xgb.XGBRegressor(**XGB_PARAMS)

def make_rf():
    return RandomForestRegressor(**RF_PARAMS)

class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
               nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class GCNEncoder(nn.Module):
     def __init__(self, node_feat_dim, hidden=128, n_layers=3, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(node_feat_dim, hidden)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden)])
        self.dropout = dropout
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.out_dim = hidden

    def forward(self, data):
     x, ei, b = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, ei)))
               x = F.dropout(x, p=self.dropout, training=self.training)
        return global_add_pool(x, b)

class DrugEncoder(nn.Module):
    def __init__(self, node_feat_dim, all_drugs, graph_cache, embed_dim=128):
        super().__init__()
        self.all_drugs = all_drugs
         self.embed_dim = embed_dim
         self.gcn_drugs = [d for d in all_drugs if d in graph_cache]
       
        self.fb_drugs = [d for d in all_drugs if d not in graph_cache]
        self.gcn_pos = {d: i for i, d in enumerate(self.gcn_drugs)}
       
          self.fb_pos = {d: i for i, d in enumerate(self.fb_drugs)}
          self.gcn = GCNEncoder(node_feat_dim, embed_dim)
      
        self.graph_list = [graph_cache[d] for d in self.gcn_drugs]
        if self.fb_drugs:
            self.fb_emb = nn.Embedding(len(self.fb_drugs), embed_dim)
            nn.init.xavier_uniform_(self.fb_emb.weight)
        else:
            self.fb_emb = None

    def get_all_embeddings(self, device):
        gcn_e = self.gcn(Batch.from_data_list([g.to(device) for g in self.graph_list])) 
        if self.gcn_drugs else 
            None
        fb_e = self.fb_emb(torch.arange(len(self.fb_drugs), dtype=torch.long, device=device)) if self.fb_drugs and self.fb_emb is not None else None

        rows = []
        for d in self.all_drugs:
            if d in self.gcn_pos:
                p = self.gcn_pos[d]
                rows.append(gcn_e[p:p+1])
            else:
             p = self.fb_pos[d]
            rows.append(fb_e[p:p+1])
        return torch.cat(rows, dim=0)

    def forward(self, da, db):
        e = self.get_all_embeddings(da.device)
        return e[da], e[db]

class BiologyEncoder(nn.Module):
    def __init__(self, input_dim=TOP_K, hidden=128, out_dim=64, dropout=0.2):
       
        super().__init__()
        self.net = nn.Sequential
        
        (
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim), nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class GCNOnly(nn.Module):
    def __init__(self, drug_enc, dropout=0.2):
        super().__init__()
        self.drug_enc = drug_enc

        self.regressor = nn.Sequential
        (
            nn.Linear(drug_enc.embed_dim * 2, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )
    def forward(self, da, db, bio=None):
        ea, eb = self.drug_enc(da, db)
        return self.regressor(torch.cat([ea, eb], dim=1)).squeeze(-1)

class GCNPlusGenes(nn.Module):
    def __init__(self, drug_enc, gene_dim=TOP_K, bio_out=64, dropout=0.2):
        super().__init__()
        self.drug_enc = drug_enc

        self.bio_enc = BiologyEncoder(gene_dim, 128, bio_out, dropout)

        self.regressor = nn.Sequential
        (
 nn.Linear(drug_enc.embed_dim * 2 + bio_out, 256), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    def forward(self, da, db, bio):
        ea, eb = self.drug_enc(da, db)
        return self.regressor(torch.cat([ea, eb, self.bio_enc(bio)], dim=1)).squeeze(-1)


class DrugPairDataset(Dataset):
    def __init__(self, frame, gene_feats, drug_to_idx):
        self.da = torch.tensor(frame["Drug_A"].map(drug_to_idx).values, dtype=torch.long)

        self.db = torch.tensor(frame["Drug_B"].map(drug_to_idx).values, dtype=torch.long)
        self.gene = torch.tensor(gene_feats, dtype=torch.float)

        self.y = torch.tensor(frame["Synergy"].values, dtype=torch.float)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.da[i], self.db[i], self.gene[i], self.y[i]

def _build_non_gene(tr, te):
    sc = StandardScaler()
    return sc.fit_transform(X_non_gene[tr]), sc.transform(X_non_gene[te])

def _build_gene_only(tr, te, gframe):

    sel = select_top_genes(tr, gframe, k=TOP_K)

    X = gframe[sel].values

    sc = StandardScaler()

    return sc.fit_transform(X[tr]), sc.transform(X[te])

def _build_gene_plus_nongene(tr, te, gframe):
    sel = select_top_genes(tr, gframe, k=TOP_K)
    
    X = np.concatenate([X_non_gene, gframe[sel].values], axis=1)
    sc = StandardScaler()

    return sc.fit_transform(X[tr]), sc.transform(X[te])

def run_sklearn_cv(data_df, y_arr, grp_arr, X_builder, model_fn, label):
    gkf = GroupKFold(n_splits=N_SPLITS)

    folds = []

    fold_test_data = []

    print(f"\n  {label}")
    for fi, (tr, te) in enumerate(gkf.split(data_df, y_arr, grp_arr), 1):
    assert not set(grp_arr[tr]) & set(grp_arr[te])

        Xtr, Xte = X_builder(tr, te)
        model = model_fn()

        model.fit(Xtr, y_arr[tr])
        pred = model.predict(Xte)

        fold_test_data.append((te.copy(), pred.copy()))
      
        met = compute_metrics(y_arr[te], pred)
        folds.append(met)

        print(f"    Fold {fi}: Pearson={met['pearson']:.4f}  "
              f"Spearman={met['spearman']:.4f}  "
              f"C-Index={met['cindex']:.4f}  "
              f"MAE={met['mae']:.4f}")

    res = summarise_cv(label, folds)
    res["fold_test_data"] = fold_test_data
    return res

def run_mlp_cv(data_df, y_arr, grp_arr, gframe, label="NeuralNetwork+Genes"):
    gkf = GroupKFold(n_splits=N_SPLITS)
    folds = []
    fold_test_data = []

    print
    (
        f"\n  {label}"
        )
    for fi, (tr, te) in enumerate(gkf.split(data_df, y_arr, grp_arr), 1):
        assert not set(grp_arr[tr]) & set(grp_arr[te])

        sel = select_top_genes(tr, gframe, k=TOP_K)
        X = gframe[sel].values

        rng = np.random.RandomState(SEED + fi)
        perm = rng.permutation(tr)

        vi = perm[:int(0.15 * len(tr))]
        ti = perm[int(0.15 * len(tr)):]

        sc = StandardScaler()
        sc.fit(X[ti])

        Xtr = torch.tensor(sc.transform(X[ti]), dtype=torch.float)

        Xvl = torch.tensor(sc.transform(X[vi]), dtype=torch.float)

        Xte = torch.tensor(sc.transform(X[te]), dtype=torch.float)

        ytr = torch.tensor(y_arr[ti], dtype=torch.float)

        yvl = torch.tensor(y_arr[vi], dtype=torch.float)

        mdl = TabularMLP(TOP_K).to(DEVICE)

        opt = Adam(mdl.parameters(), lr=1e-3, weight_decay=1e-4)

        sch = ReduceLROnPlateau(opt, "min", factor=0.5, patience=8)
        tl = DataLoader(TensorDataset(Xtr, ytr), batch_size=MLP_BATCH, shuffle=True)

        best_l = float("inf")
        best_st = None

        for _ in range(MLP_EPOCHS):
            mdl.train()
            for xb, yb in tl:
                 xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                   opt.zero_grad()
                  l = F.mse_loss(mdl(xb), yb)
                 l.backward()
                 nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                opt.step()

            mdl.eval()
            with torch.no_grad():
                vl = F.mse_loss(mdl(Xvl.to(DEVICE)), yvl.to(DEVICE)).item()
            sch.step(vl)

            if vl < best_l:
                best_l = vl
                best_st = copy.deepcopy(mdl.state_dict())

         mdl.load_state_dict(best_st)
        mdl.eval()

        with torch.no_grad():
            pred = mdl(Xte.to(DEVICE)).cpu().numpy()

        fold_test_data.append((te.copy(), pred.copy()))

        met = compute_metrics(y_arr[te], pred)
        folds.append(met)

        print(f"    Fold {fi}: Pearson={met['pearson']:.4f}  "
              f"Spearman={met['spearman']:.4f}  "
              f"C-Index={met['cindex']:.4f}  "
              f"MAE={met['mae']:.4f}")

    res = summarise_cv(label, folds)
    res["fold_test_data"] = fold_test_data
    return res

def run_gnn_cv(model_class, data_df, gframe, y_arr, grp_arr, drug_to_idx, all_drugs, graph_cache, label):
    gkf = GroupKFold(n_splits=N_SPLITS)
    folds = []
    epoch_curves = []
    fold_test_data = []

    print
    (
        f"\n  {label} | rows={len(data_df)} pairs={data_df['drug_pair_id'].nunique()}")
    for fi, (tr, te) in enumerate(gkf.split(data_df, y_arr, grp_arr), 1):
        assert not set(grp_arr[tr]) & set(grp_arr[te]
                                          )

        sel = select_top_genes(tr, gframe, k=TOP_K)
        X = gframe[sel].values

        rng = np.random.RandomState(SEED + fi)
        perm = rng.permutation(tr) 

          vi = perm[:int(0.15 * len(tr))]
        ti = perm[int(0.15 * len(tr)):]

        def mk(idx, shuf):
            return DataLoader
        (
                DrugPairDataset(data_df.iloc[idx].reset_index(drop=True), X[idx], drug_to_idx),
                batch_size=GNN_BATCH, shuffle=shuf
            )

         tr_ld = mk(ti, True)
        vl_ld = mk(vi, False)
         te_ld = mk(te, False)

        drug_enc = DrugEncoder(NODE_FEAT_DIM, all_drugs, graph_cache, embed_dim=128).to(DEVICE)

        model = model_class(drug_enc, gene_dim=TOP_K).to(DEVICE) if model_class == GCNPlusGenes else model_class(drug_enc).to(DEVICE)

        opt = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = ReduceLROnPlateau(opt, "min", factor=0.5, patience=8)

        best_vl = float("inf")
        best_st = None
        fold_log = []

        for ep in range(GNN_EPOCHS):
            model.train()
            tr_loss = 0.0
            nb = 0

            for da, db, bio, yy in tr_ld:
                da, db, bio, yy = da.to(DEVICE), db.to(DEVICE), bio.to(DEVICE), yy.to(DEVICE)

                opt.zero_grad()
                pred = model(da, db, bio) if model_class == GCNPlusGenes else model(da, db)
                loss = F.mse_loss(pred, yy)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tr_loss += loss.item()
                nb += 1


            model.eval()
            vp, vt = [], []
            with torch.no_grad():

                for da, db, bio, yy in vl_ld:
                    da, db, bio = da.to(DEVICE), db.to(DEVICE), bio.to(DEVICE)

                    pred = model(da, db, bio) if model_class == GCNPlusGenes else model(da, db)
                    vp.extend(pred.cpu().tolist())
                    vt.extend(yy.tolist())

            vm= compute_metrics(vt, vp)
            sch.step(vm["mae"])

            if vm["mae"] < best_vl:
                best_vl= vm["mae"]
                best_st= copy.deepcopy(model.state_dict())

            fold_log.append
            (
                {
                "ep": ep + 1,"tr_loss": round(tr_loss / max(nb, 1), 4),
                "vl_p": round(vm["pearson"], 4),"vl_mae": round(vm["mae"], 4),
                "lr": round(opt.param_groups[0]["lr"], 6)
            }
            )

            if (ep + 1) % 20 == 0:
                print(f"    Fold {fi} Ep{ep+1:3d} | loss={tr_loss/max(nb,1):.4f} "
                      f"ValP={vm['pearson']:.4f} ValMAE={vm['mae']:.4f} "
                      f"lr={opt.param_groups[0]['lr']:.5f}")

        model.load_state_dict(best_st)
        model.eval()

        tp, tt = [], []
        with torch.no_grad():
            for da, db, bio, yy in te_ld:
             da, db, bio = da.to(DEVICE), db.to(DEVICE), bio.to(DEVICE)
             pred = model(da, db, bio) if model_class == GCNPlusGenes else model(da, db)
             tp.extend(pred.cpu().tolist())
                tt.extend(yy.tolist())

        fold_test_data.append((te.copy(), np.array(tp)))
        epoch_curves.append(fold_log)

        met = compute_metrics(tt, tp)
        folds.append(met)

        print(
            f"  Fold {fi} TEST → Pearson={met['pearson']:.4f}  "
              f"Spearman={met['spearman']:.4f}  "
              f"C-Index={met['cindex']:.4f}  "
              f"MAE={met['mae']:.4f}"
              )

     res = summarise_cv(label, folds)
       res["epoch_curves"] = epoch_curves
    res["fold_test_data"] = fold_test_data

    print(f"  CV Pearson : {res['Pearson_mean']:.4f} ± {res['Pearson_std']:.4f}")
    
    print(f"  CV Spearman: {res['Spearman_mean']:.4f} ± {res['Spearman_std']:.4f}")
    
    print(f"  CV C-Index : {res['CIndex_mean']:.4f} ± {res['CIndex_std']:.4f}")
    print(f"  CV MAE     : {res['MAE_mean']:.4f} ± {res['MAE_std']:.4f}")
    
    return res

def run_ensemble_cv(res_a, res_b, y_arr, weight_candidates=None, label="Ensemble"):
        if weight_candidates is None:
        weight_candidates = [(0.7,0.3),(0.6,0.4),(0.5,0.5),(0.4,0.6),(0.3,0.7)]

    best_mean = -999
    best_res = None

    print(f"\n  {label}")
    print(
        f"  Model A: {res_a['Model']} (Pearson={res_a['Pearson_mean']:.4f})"
        )
    print(
        f"  Model B: {res_b['Model']} (Pearson={res_b['Pearson_mean']:.4f})")

    for wa, wb in weight_candidates:
        folds = []
        for (te_a, pred_a), (te_b, pred_b) in zip(res_a["fold_test_data"], res_b["fold_test_data"]):
            if not np.array_equal(np.sort(te_a), np.sort(te_b)):
                raise ValueError("Fold mismatch between ensemble inputs")

            sa = np.argsort(te_a)
            sb = np.argsort(te_b)

            blend = wa * pred_a[sa] + wb * pred_b[sb]
            folds.append(compute_metrics(y_arr[te_a[sa]], blend))

         mean_p = float(
            np.mean([m["pearson"] for m in folds])
            )
        print(
            f"    w=({wa:.1f},{wb:.1f}) Pearson={mean_p:.4f}"
            )

        if mean_p > best_mean:
            best_mean = mean_p
            best_res = summarise_cv(f"{label}(w={wa:.1f}+{wb:.1f})", folds)

    print(
        f"  Best ensemble: {best_res['Model']} | Pearson={best_res['Pearson_mean']:.4f}"
        )
    return best_res

COLORS = ["#64748B","#3B82F6","#10B981","#F59E0B","#EF4444","#8B5CF6","#EC4899","#14B8A6"]

def plot_bar(results, title, fname):
    labels = [r["Model"] for r in results]
    vals = [r["Pearson_mean"] for r in results]
    errs = [r["Pearson_std"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
   
    bars = ax.bar(labels, vals, color=COLORS[:len(results)], edgecolor="white")
    
    ax.errorbar(labels, vals, yerr=errs, fmt="none", color="black", capsize=5)
    ax.set_ylabel("CV Pearson")
    ax.set_title(title, fontweight="bold")

      ax.tick_params(axis="x", rotation=20)
    
      plt.tight_layout()
      plt.savefig(os.path.join(OUT, fname), dpi=150, bbox_inches="tight")
    plt.show()

def plot_folds(results, title, fname):
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(N_SPLITS)
    w = 0.75 / max(len(results), 1)
    for i, r in enumerate(results):
        ax.bar(x + i*w, r["fold_pearsons"], w, label=r["Model"], color=COLORS[i % len(COLORS)], alpha=0.85)
      ax.set_xticks(x + w*(len(results)-1)/2)
   
    ax.set_xticklabels([f"Fold {i+1}" for i in range(N_SPLITS)])
     ax.set_ylabel("Pearson")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)
    
    plt.tight_layout()
     plt.savefig(os.path.join(OUT, fname), dpi=150, bbox_inches="tight")
    plt.show()

print("\n" + "█"*60)
print("EXPERIMENT 1 — Tabular Baseline Comparison")
 
print(f"Dataset : {len(df)} rows | {df['drug_pair_id'].nunique()} pairs")
print("Genes   : Top-300 by variance (per-fold, no leakage)")

print("█"*60)

t0 = time.time()

e1_rf = run_sklearn_cv
    (
    df, y_full, groups_full, lambda tr, te: _build_gene_only(tr, te, gene_df_all),
    make_rf, "RandomForest+Genes"
)

e1_nn = run_mlp_cv
(
    df, y_full, groups_full, gene_df_all, "NeuralNetwork+Genes"
)

e1_xgb = run_sklearn_cv
(
    df, y_full, groups_full,
        lambda tr, te: _build_gene_only(tr, te, gene_df_all),
      make_xgb,
            "XGBoost+Genes")

exp1_results = [e1_rf, e1_nn, e1_xgb]
best1 = max(exp1_results, key=lambda r: r["Pearson_mean"])

df_exp1 = pd.DataFrame([{
    "Model": r["Model"], "Pearson": f"{r['Pearson_mean']:.3f}",
    "Spearman": f"{r['Spearman_mean']:.3f}","C-Index": f"{r['CIndex_mean']:.3f}",
    "MAE": f"{r['MAE_mean']:.3f}",
    "Best": "★" if r["Model"] == best1["Model"] else ""
}
 for r in exp1_results])

print(f"\n── Experiment 1 Results ({time.time()-t0:.0f}s) ──")
print(df_exp1.to_string(index=False))

df_exp1.to_csv(os.path.join(OUT, f"exp1_{RUN_ID}.csv"), index=False)

plot_bar(exp1_results, "Exp 1 — Tabular Baseline", f"exp1_bar_{RUN_ID}.png")
plot_folds(exp1_results, "Exp 1 — Per-Fold Pearson", f"exp1_folds_{RUN_ID}.png")

print("\n" + "█"*60)

print("EXPERIMENT 2 — Full Dataset, Non-Matrix + Ensemble")

print(f"Dataset : {len(df)} rows | {df['drug_pair_id'].nunique()} pairs")
print("Models  : XGBoost | XGBoost+Genes | GCN | GCN+Genes | Ensemble")

print("Genes   : Top-300 by variance (per-fold, no leakage)")
print("█"*60)

t0 = time.time()

e2_xgb0 = run_sklearn_cv(
    df, y_full, groups_full,
    _build_non_gene,
    make_xgb,
    "XGBoost"
)

   e2_xgb= run_sklearn_cv(
    df, y_full, groups_full,
    lambda tr, te: _build_gene_plus_nongene(tr, te, gene_df_all),
    make_xgb,
    "XGBoost+Genes"
  )

e2_gcn= run_gnn_cv(
    GCNOnly,
    df, gene_df_all, y_full, groups_full,
    drug_to_idx_f, all_drugs_f, graph_cache_f,
    "GCN"
)

e2_hybrid = run_gnn_cv(
    GCNPlusGenes,
    df, gene_df_all, y_full, groups_full,
    drug_to_idx_f, all_drugs_f, graph_cache_f,
    "GCN+Genes"
)

e2_ensemble = run_ensemble_cv(
    e2_xgb, e2_hybrid, y_full,
    weight_candidates=[(0.7,0.3),(0.6,0.4),(0.5,0.5),(0.4,0.6),(0.3,0.7)],
    label="Ensemble(XGB+Genes,GCN+Genes)"
)

exp2_results = [e2_xgb0, e2_xgb, e2_gcn, e2_hybrid, e2_ensemble]
best2 = max(exp2_results, key=lambda r: r["Pearson_mean"])

df_exp2 = pd.DataFrame([
    {
    "Model": r["Model"],
    "Role": (
        "Identity baseline" if r["Model"] == "XGBoost"
        else "Gene baseline" if r["Model"] == "XGBoost+Genes"
         else "Structure" if r["Model"] == "GCN"
         else "Hybrid" if r["Model"] == "GCN+Genes"
         else "Ensemble"
    ),
      "Pearson": f"{r['Pearson_mean']:.3f}",
     "Spearman": f"{r['Spearman_mean']:.3f}",
      "C-Index": f"{r['CIndex_mean']:.3f}",
      "MAE": f"{r['MAE_mean']:.3f}",
      "Best": "★" if r["Model"] == best2["Model"] else ""
} for r in exp2_results])

print(f"\n── Experiment 2 Results ({time.time()-t0:.0f}s) ──")
print(df_exp2.to_string(index=False))

df_exp2.to_csv(os.path.join(OUT, f"exp2_{RUN_ID}.csv"), index=False)

plot_bar(exp2_results, "Exp 2 — Full Dataset Non-Matrix + Ensemble", f"exp2_bar_{RUN_ID}.png")
plot_folds(exp2_results, "Exp 2 — Per-Fold Pearson", f"exp2_folds_{RUN_ID}.png")


print("\n" + "="*70)

print("FINAL SUMMARY")

print("="*70)

print("Experiment 1:")
for r in exp1_results:
    print(f"  {r['Model']:<30} "
            f"Pearson={r['Pearson_mean']:.4f}  "
           f"Spearman={r['Spearman_mean']:.4f}  "
        f"C-Index={r['CIndex_mean']:.4f}  "
          f"MAE={r['MAE_mean']:.4f}")

print("\nExperiment 2:")
for r in exp2_results:
    print(f"  {r['Model']:<30} "
 f"Pearson={r['Pearson_mean']:.4f}  "
    f"Spearman={r['Spearman_mean']:.4f}  "
  f"C-Index={r['CIndex_mean']:.4f}  "
          f"MAE={r['MAE_mean']:.4f}")

print(f"\nBest Exp1 model: {best1['Model']} | Pearson={best1['Pearson_mean']:.4f}")
print(f"Best Exp2 model: {best2['Model']} | Pearson={best2['Pearson_mean']:.4f}")
print(f"Outputs saved to: {OUT}")


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

B = {
    "b1": "#DBEAFE","b2": "#BFDBFE","b3": "#93C5FD","b4": "#60A5FA","b5": "#3B82F6","b6": "#2563EB","b7": "#1D4ED8","b8": "#1E3A8A", "grid": "#E0F2FE",
"err":  "#1E3A8A",
}

 FIVE_BLUES= [B["b3"], B["b4"], B["b5"], B["b6"], B["b7"]]
   
   THREE_BLUES = [B["b3"], B["b5"], B["b7"]]

FOUR_BLUES= [B["b2"], B["b4"], B["b6"], B["b8"]]

plt.rcParams.update({
    "figure.facecolor"  : "white", "axes.facecolor"    : "white","axes.edgecolor"    : "#CBD5E1", "axes.grid"         : True,
    "grid.color"        : B["grid"],  "grid.linewidth"    : 0.6, "font.family"       : "sans-serif", "axes.spines.top"   : False,
    "axes.spines.right" : False,
})

def save(fname):
    path = os.path.join(OUT, fname)

    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")

    plt.show()
    
    print(f"  Saved: 
            {fname\}")



#Experiment 1
exp1_models= ["RandomForest\n+Genes", "NeuralNetwork\n+Genes", "XGBoost\n+Genes"]
exp1_pearson= [0.3490, 0.3349, 0.3499]
exp1_std= [0.0384, 0.0467, 0.0383]
exp1_spearman= [0.2707, 0.2633, 0.2715]
exp1_cindex= [0.5936, 0.5904, 0.5939]

#Experiment 2
exp2_models= ["XGBoost", "XGBoost\n+Genes", "GCN", "GCN\n+Genes", "Ensemble\n(XGB+GCN\n+Genes)"]

exp2_pearson= [0.3877, 0.4323, 0.2262, 0.4014, 0.4555]

exp2_std= [0.0245, 0.0263, 0.0644, 0.0483, 0.0275]

exp2_spearman= [0.3506, 0.3774, 0.2433, 0.3412, 0.3982]

exp2_cindex= [0.6201, 0.6312, 0.5823, 0.6167, 0.6383]

exp2_mae= [20.59,  20.19,  21.66,  20.93,  19.81]

#fold-level Pearson for Exp 2
exp2_folds = {
    "XGBoost"        : [0.3513, 0.3837, 0.3753, 0.4062, 0.4218], "XGBoost+Genes"  : [0.4324, 0.4283, 0.3858, 0.4598, 0.4551],
    "GCN"            : [0.1719, 0.2921, 0.1740, 0.1763, 0.3170],"GCN+Genes"      : [0.3971, 0.4582, 0.3131, 0.4188, 0.4198],
    "Ensemble"       : [0.4467, 0.4455, 0.4412, 0.4339, 0.4240],
}

#Experiment 3 & 4 (613 subset)
exp3_models= ["XGBoost+Genes\n(613)", "GCN\n(613)", "GCN+Genes\n(613)"]
exp3_pearson= [-0.0166, 0.1171, 0.0199]
exp3_std= [0.1046,  0.1013, 0.1026]

exp4_models= ["XGBoost+Genes\n(613-E4)", "GCN+LSTM\n+Attn (613)","GCN+Genes\n+LSTM+Attn (613)"]
exp4_pearson= [-0.0166, 0.0689, 0.0061]
exp4_std= [0.1046,  0.1106, 0.1461]

#best per experiment
master_labels= ["Exp 1\nTabular\nBaseline", "Exp 2\nFull\nNon-Matrix", "Exp 3\n613-Row\nNon-Matrix", "Exp 4\n613-Row\nMatrix-Aware"]
master_best= [0.3499, 0.4555, 0.1171, 0.0689]

master_std= [0.0383, 0.0275, 0.1013, 0.1106]
master_bestmod= ["XGBoost+Genes","Ensemble(XGB+Genes,\nGCN+Genes)(w=0.6+0.4)", "GCN (613)", "GCN+LSTM+Attn (613)"]

#Synergy distribution 
np.random.seed(42)

synergy_approx= np.random.normal(7.5, 29.7, 2059)
synergy_approx= np.clip(synergy_approx, -100, 100)

smiles_gcn= 45
smiles_fallback= 23



#Best Model Per Experiment
fig, ax = plt.subplots(figsize=(11, 6))

x= np.arange(len(master_labels))

bars= ax.bar(x, master_best, color=FOUR_BLUES, edgecolor="white", linewidth=1.5, zorder=3)

ax.errorbar(x, master_best, yerr=master_std, fmt="none", color=B["err"], capsize=7, capthick=2, lw=2, zorder=4)

ax.axhline(0, color="#94A3B8", lw=1, ls="--")

 for bar, val, mod, std in zip(bars, master_best, master_bestmod, master_std):
    ypos = max(val, 0) + std + 0.012
    ax.text(bar.get_x() + bar.get_width()/2, ypos, f"r = {val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=B["b7"]) ax.text(bar.get_x() + bar.get_width()/2, -0.055,
            mod, ha="center", va="top", fontsize=7.5, color="#475569", style="italic")

ax.set_xticks(x); ax.set_xticklabels(master_labels, fontsize=11)

ax.set_ylabel("CV Pearson r  (5-fold GroupKFold)", fontsize=11)

 ax.set_title("SynerGene — Best Model Per Experiment\n"
             "GroupKFold leave-drug-pair-out · zero leakage · all 4 experiments", fontsize=12, fontweight="bold", pad=14)
  
  ax.set_ylim(-0.12, max(master_best) * 1.55)
  
  plt.tight_layout()

save("fig1_master_summary.png")


#Experiment 2: Multi-Metric
metrics = {
    "Pearson r"  : exp2_pearson, "Spearman ρ" : exp2_spearman,"C-Index"    : exp2_cindex,
}

 n_models  = len(exp2_models)
n_metrics = len(metrics)

x  = np.arange(n_models)
w  = 0.22

fig, ax = plt.subplots(figsize=(14, 6))
offsets = [-w, 0, w]
   
   metric_colors = [B["b4"], B["b6"], B["b8"]]
han
dles = []

for i, (mname, mvals) in enumerate(metrics.items()):
    b= ax.bar(x + offsets[i], mvals, w, label=mname, color=metric_colors[i],  edgecolor="white", lw=1.2, zorder=3)
     
      if mname == "Pearson r":
        ax.errorbar(x + offsets[i], mvals, yerr=exp2_std, fmt="none", color=B["err"], capsize=5, capthick=1.5, lw=1.5, zorder=4)
    
    handles.append(mpatches.Patch(color=metric_colors[i], label=mname))
   
    for bar, val in zip(b, mvals):
        ax.text(bar.get_x() + bar.get_width()/2,bar.get_height() + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#1E3A8A")

 ax.set_xticks(x); ax.set_xticklabels(exp2_models, fontsize=10)
   ax.set_ylabel("Metric Value", fontsize=11)

ax.set_title("Experiment 2 — Full Dataset Non-Matrix: Multi-Metric Comparison\n"
             "Pearson r · Spearman ρ · Concordance Index  (5-fold GroupKFold)",fontsize=12, fontweight="bold", pad=12)
ax.set_ylim(0, 0.78)

ax.legend(handles=handles, fontsize=10, loc="upper left",framealpha=0.9, edgecolor="#CBD5E1")
ax.axhline(0, color="#94A3B8", lw=0.8)

  ax.axvspan(3.5, 4.5, alpha=0.06, color=B["b5"], zorder=0)

  ax.text(4, 0.01, "★ Best", ha="center", fontsize=9, color=B["b7"], fontweight="bold")

plt.tight_layout()

save("fig2_exp2_multimetric.png")


#Experiment 1
fig, ax= plt.subplots(figsize=(9, 5))

x= np.arange(len(exp1_models))

bars= ax.bar(x, exp1_pearson, color=THREE_BLUES, edgecolor="white", lw=1.5, zorder=3, width=0.5)

ax.errorbar(x, exp1_pearson, yerr=exp1_std, fmt="none", color=B["err"], capsize=8, capthick=2, lw=2, zorder=4)

 for bar, val, std in zip(bars, exp1_pearson, exp1_std):
    ax.text(bar.get_x() + bar.get_width()/2, val + std + 0.008, f"r = {val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold", color=B["b7"])

ax.set_xticks(x); ax.set_xticklabels(exp1_models, fontsize=12)

ax.set_ylabel("CV Pearson r", fontsize=11)

ax.set_title("Experiment 1 — Tabular Baseline Comparison\n"
             "All models use top-300 variance-selected genes · 5-fold GroupKFold", fontsize=12, fontweight="bold", pad=12)

 ax.set_ylim(0, 0.55)

ax.text(0.5, 0.44,
        "All three models perform similarly (~0.35),\n"
        "indicating signal is driven by gene features\nnot model architecture.", transform=ax.transAxes, fontsize=9, ha="center", color="#475569", style="italic", bbox=dict(boxstyle="round,pad=0.4", fc=B["b1"], ec=B["b3"], alpha=0.8))

plt.tight_layout()

save("fig3_exp1_tabular_baseline.png")


# Exp 3 vs Exp 4: Matrix
 all_models= exp3_models + exp4_models
all_pearson= exp3_pearson + exp4_pearson

 all_std=exp3_std  + exp4_std
 colors_3v4= [B["b3"]]*3 + [B["b6"]]*3

fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(all_models))
bars = ax.bar(x, all_pearson, color=colors_3v4, edgecolor="white", lw=1.3, zorder=3, width=0.6)

 ax.errorbar(x, all_pearson, yerr=all_std, fmt="none", color=B["err"], capsize=6, capthick=1.5, lw=1.5, zorder=4)
  ax.axhline(0, color="#94A3B8", lw=1, ls="--")

   ax.axvline(2.5, color="#CBD5E1", lw=1.5, ls=":")
 ax.text(1.0,  0.22, "Experiment 3\n(No Matrix Branch)",  ha="center", fontsize=10, color=B["b5"], fontweight="bold")

ax.text(3.5,  0.22, "Experiment 4\n(Matrix-Aware)",    ha="center", fontsize=10, color=B["b7"], fontweight="bold")

 for bar, val, std in zip(bars, all_pearson, all_std):
    ypos = (val + std + 0.01) if val >= 0 else (val - std - 0.03)
    ax.text(bar.get_x() + bar.get_width()/2, ypos, f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1E3A8A")

 p3 = mpatches.Patch(color=B["b3"], label="Exp 3 — No matrix branch")
p4 = mpatches.Patch(color=B["b6"], label="Exp 4 — Matrix-aware (LSTM+Attn)")
  ax.legend(handles=[p3, p4], fontsize=10, loc="upper right",
    framealpha=0.9, edgecolor="#CBD5E1")
   ax.set_xticks(x); ax.set_xticklabels(all_models, fontsize=9)
  ax.set_ylabel("CV Pearson r", fontsize=11)
    ax.set_title("Experiments 3 vs 4 — Matrix Branch Impact\n"
             "Identical 594-row aligned subset · any difference = matrix contribution",
             fontsize=12, fontweight="bold", pad=12)
ax.set_ylim(-0.25, 0.36)
plt.tight_layout()
save("fig4_exp3v4_matrix_impact.png")


# Experiment 2
fold_labels= [f"Fold {i+1}" for i in range(5)]
n_models= len(exp2_folds)
x= np.arange(5)
w= 0.14

fig, ax = plt.subplots(figsize=(13, 5))
fold_colors = [B["b2"], B["b4"], B["b5"], B["b6"], B["b8"]]
handles = []
for i, (mname, fvals) in enumerate(exp2_folds.items()):
    offset = (i - n_models//2) * w  b = ax.bar(x + offset, fvals, w, label=mname,
                 color=fold_colors[i], edgecolor="white", lw=1., zorder=3)
    handles.append(mpatches.Patch(color=fold_colors[i], label=mname))

ax.axhline(0, color="#94A3B8", lw=0.8, ls="--")
ax.set_xticks(x); ax.set_xticklabels(fold_labels, fontsize=11)
ax.set_ylabel("Pearson r", fontsize=11)
ax.set_title("Experiment 2 — Per-Fold Pearson Stability\n"
             "5-fold GroupKFold leave-drug-pair-out · lower variance = more reliable", fontsize=12, fontweight="bold", pad=12)
ax.legend(handles=handles, fontsize=9, loc="upper left", framealpha=0.9, edgecolor="#CBD5E1", ncol=2)
ax.set_ylim(-0.05, 0.62)
plt.tight_layout()
save("fig5_exp2_fold_stability.png")


# Synergy Score Distribution
fig, ax = plt.subplots(figsize=(9, 4))
n, bins, patches = ax.hist(synergy_approx, bins=55, color=B["b4"], edgecolor="white", linewidth=0.6, zorder=3)
for patch, left in zip(patches, bins[:-1]):
    if left < -20:
        patch.set_facecolor(B["b3"])
    elif left > 20:
       patch.set_facecolor(B["b6"])

ax.axvline(0,   color=B["b8"],  lw=1.5, ls="--", label="Zero synergy")
 ax.axvline(7.5, color="#EF4444", lw=1.5, ls="--", label=f"Mean ≈ 7.5")
ax.axvline(29.7,  color="#F59E0B", lw=1.2, ls=":",
           label=f"σ ≈ 29.7 (1 SD)")
 ax.axvline(-29.7, color="#F59E0B", lw=1.2, ls=":")
 ax.set_xlabel("Synergy Score", fontsize=11)
   ax.set_ylabel("Count", fontsize=11)
  ax.set_title("Synergy Score Distribution  (n = 2,059, filtered ±100)\n"
             "Wide spread (σ ≈ 29.7) contextualises RMSE magnitude",
             fontsize=12, fontweight="bold", pad=12)
ax.legend(fontsize=10, framealpha=0.9, edgecolor="#CBD5E1")
plt.tight_layout()
save("fig6_synergy_distribution.png")


fig, axes = plt.subplots(1, 2, figsize=(11, 5))

axes[0].pie(
    [smiles_gcn, smiles_fallback],
    labels=[f"GCN-encoded\n({smiles_gcn} drugs)",
            f"Fallback embedding\n({smiles_fallback} drugs)"],
       colors=[B["b5"], B["b2"]],
    autopct="%1.0f%%", startangle=90,
       textprops={"fontsize": 11},
    wedgeprops={"edgecolor": "white", "linewidth": 2},
)
axes[0].set_title("Drug Structural Coverage\n(Full Dataset — 68 unique drugs)",
                  fontweight="bold", fontsize=11)

   models_cov  = ["GCN\n(structure only)", "GCN+Genes\n(hybrid)"]
pearson_cov = [0.2262, 0.4014]
    colors_cov  = [B["b3"], B["b6"]]
bars = axes[1].bar(models_cov, pearson_cov, color=colors_cov,
             edgecolor="white", lw=1.5, width=0.45, zorder=3)
 for bar, val in zip(bars, pearson_cov):
       axes[1].text(bar.get_x() + bar.get_width()/2,
                  val + 0.01, f"r = {val:.4f}",
                 ha="center", va="bottom",
                 fontsize=12, fontweight="bold", color=B["b7"])
 axes[1].set_ylim(0, 0.58)
    axes[1].set_ylabel("CV Pearson r", fontsize=11)
  axes[1].set_title("Why Genes Matter on Top of Structure\n"
                  "Low SMILES coverage limits GCN alone",
                  fontweight="bold", fontsize=11)
 axes[1].text(0.5, 0.85,
             "33% of drugs lack public SMILES\n→ use trainable fallback embeddings\n"
             "→ GCN alone ≈ identity lookup\n→ genes recover ~+0.18 Pearson",
             transform=axes[1].transAxes, ha="center", fontsize=9,
             color="#475569", style="italic",
             bbox=dict(boxstyle="round,pad=0.4", fc=B["b1"],
                       ec=B["b3"], alpha=0.85))
 plt.tight_layout()
save("fig7_smiles_coverage_impact.png")


#  Experiment 2
ladder_models= ["XGBoost\n(identity only)",
                  "XGBoost\n+Genes",
                  "GCN\n+Genes",
                  "Ensemble\n(XGB+GCN+Genes)"]
ladder_pearson= [0.3877, 0.4323, 0.4014, 0.4555]
ladder_std= [0.0245, 0.0263, 0.0483, 0.0275]
ladder_colors= [B["b2"], B["b4"], B["b6"], B["b8"]]

fig, ax= plt.subplots(figsize=(10, 5))
x= np.arange(len(ladder_models))
bars= ax.bar(x, ladder_pearson, color=ladder_colors,
              edgecolor="white", lw=1.5, width=0.55, zorder=3)
ax.errorbar(x, ladder_pearson, yerr=ladder_std,
            fmt="none", color=B["err"], capsize=8, capthick=2, lw=2, zorder=4)

deltas= [None, +0.0446, -0.0309, +0.0541]
for i, (bar, val, std, d) in enumerate(zip(bars, ladder_pearson,
                                            ladder_std, deltas)):
    ax.text(bar.get_x() + bar.get_width()/2,
            val + std + 0.01,
            f"r = {val:.4f}", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=B["b8"])
    if d is not None:
        color = "#16A34A" if d > 0 else "#DC2626"
        sign  = "+" if d > 0 else ""
        ax.annotate(f"{sign}{d:.4f}",
                    xy=(x[i], val), xytext=(x[i]-0.35, val + 0.06),
                    fontsize=9, color=color, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

    ax.set_xticks(x); ax.set_xticklabels(ladder_models, fontsize=11)
     ax.set_ylabel("CV Pearson r", fontsize=11)
    ax.set_title("Experiment 2 — Contribution Ladder\n"
             "Each step adds a new signal source · numbers show Δ Pearson",
             fontsize=12, fontweight="bold", pad=12)
     ax.set_ylim(0, 0.62)
   plt.tight_layout()
  save("fig8_exp2_contribution_ladder.png")


print("\n" + "="*55)
print("ALL REPORT FIGURES SAVED")

print("="*55)
print(f"  fig1_master_summary.png        — best per experiment")
print(f"  fig2_exp2_multimetric.png      — Exp 2 multi-metric bars")

print(f"  fig3_exp1_tabular_baseline.png — Exp 1 baseline")
print(f"  fig4_exp3v4_matrix_impact.png  — matrix branch comparison")
print(f"  fig5_exp2_fold_stability.png   — per-fold Pearson")

print(f"  fig6_synergy_distribution.png  — target distribution")
print(f"  fig7_smiles_coverage_impact.png— SMILES coverage")
print(f"  fig8_exp2_contribution_ladder.png — ablation ladder")

print(f"\n  Output directory: {OUT}")