import importlib, subprocess, sys, os, re, json, copy, time, random, warnings, datetime, shutil
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore") #suppress irrelevant warnings so output stays clean

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

SEED = 42 #fixed seed for reproducibility across all runs
  random.seed(SEED)
 np.random.seed(SEED)
  torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #use gpu if available
TOP_K = 300 #number of top variance genes to select per fold
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
os.makedirs(OUT, exist_ok=True) #create output dirs if they dont exist yet
os.makedirs(LOCAL, exist_ok=True)


FILES = {
    "merged" : os.path.join(PROC, "az_depmap_expression_merged.csv"),"summary" : os.path.join(BASE, "pubchem_original_drug_summary.csv"),
    "alias" : os.path.join(BASE, "pubchem_alias_hits.csv"),"matched" : os.path.join(BASE, "drug_smiles_pubchem_matched.csv"),
}

RUN_ID   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") #unique id for each run based on timestamp

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
df_alias= pd.read_csv(FILES["alias"]) if os.path.exists(FILES["alias"]) else pd.DataFrame()
df_matched= pd.read_csv(FILES["matched"]) if os.path.exists(FILES["matched"]) else pd.DataFrame()

  
  print("df_merged:", df_merged.shape)
    def normalize_key(x): #strip and lowercase so drug names match regardless of formatting to not miss any or cause errors
    
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
        time.sleep(sleep_sec) #small delay to avoid hitting rate limits
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
    return "__".join(sorted([str(a).strip(), str(b).strip()])) #sort so (A,B) and (B,A) give the same id

  def extract_gene_symbol(col):
    m = re.match(r"^([A-Za-z0-9\-_]+)\s*\(", str(col))
    return m.group(1).upper() if m else str(col).strip().upper()

def count_targets(x):
if pd.isna(x):
        return 0
    return len([t.strip() for t in re.split(r"[,;/|]+", str(x)) if t.strip()])

 def canonicalize_pairs(df):
    df = df.copy()
    swap = df["Drug_A"] > df["Drug_B"] #ensure drug A always comes first alphabetically
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
            "pearson": 0.0, "spearman": 0.0, "cindex": 0.0, "mae": 999.0, "rmse": 999.0, "r2": -1.0
        }

      try:
        p = pearsonr(y_true, y_pred)[0]
    except:
        p = 0.0
      try:
    s = spearmanr(y_true, y_pred)[0]
    except:
         s = 0.0

    p = 0.0 if np.isnan(p) else p #replace nan with 0 to avoid downstream errors
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
        "Model": label, "Pearson_mean": float(np.mean(ps)), "Pearson_std": float(np.std(ps)), "Spearman_mean": float(np.mean(sp)), "Spearman_std": float(np.std(sp)), "CIndex_mean": float(np.mean(ci)), "CIndex_std": float(np.std(ci)), "MAE_mean": float(np.mean(mae)), "MAE_std": float(np.std(mae)),"fold_pearsons": ps,"folds": fold_metrics
    }

def select_top_genes(train_idx, gene_frame, k=TOP_K): #select genes using only training data to avoid leakage
    X_tr = gene_frame.iloc[train_idx].values.astype(np.float64)
    gvar = pd.Series(X_tr.var(axis=0), index=gene_frame.columns)
    return gvar.nlargest(k).index.tolist()

master_smiles = {} #central dict mapping drug name to its SMILES string

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
} #hardcoded smiles for drugs that are hard to find via pubchem
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

for drug in missing: #fetch any remaining smiles from pubchem api
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
    out[choices.index(val) if val in choices else -1] = 1 #unknown values map to the last slot
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

NODE_FEAT_DIM = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0))) #compute feature dim from a dummy atom

def smiles_to_graph(smiles):
    
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    
    if mol is None:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    src, dst = [], []
    
    for bond in mol.GetBonds(): #add edges in both directions since graph is undirected
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

def build_drug_registry(data_df):
    drugs = sorted(set(data_df["Drug_A"]) | set(data_df["Drug_B"]))
    
    drug_to_idx = {d: i for i, d in enumerate(drugs)}
   
    graph_cache = {} #only cache drugs that have valid smiles and parse correctly
    for d in drugs:
        s = master_smiles.get(normalize_key(d))
        if s:
            g = smiles_to_graph(s)
            if g is not None:
                graph_cache[d] = g
    return drugs, drug_to_idx, graph_cache

df = df_merged.copy() #work on a copy so original stays intact

for col in ["Drug_A", "Drug_B", "Cell_Line"]:
 if col in df.columns:
    df[col] = df[col].astype(str).str.strip()

df["Synergy"] = pd.to_numeric(df["Synergy"], errors="coerce")
df = df.dropna(subset=["Synergy"])
df = df[(df["Synergy"] >= -100) & (df["Synergy"] <= 100)].copy() #filter out extreme outliers
df = canonicalize_pairs(df)
df["drug_pair_id"] = df.apply(lambda r: make_pair_id(r["Drug_A"], r["Drug_B"]), axis=1)
df = df.reset_index(drop=True)
gene_cols_raw = [c for c in df.columns if "(" in c and ")" in c] #gene columns follow the format SYMBOL(ID)
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
    return len(ta & tb) / len(ta | tb) #overlap divided by union

def parse_targets_to_list(x):
    if pd.isna(x):
        return []
    return [t.strip().upper() for t in re.split(r"[,;/|]+", str(x)) if t.strip()]

df["Target_A_list"] = df.get("Target_A", pd.Series("", index=df.index)).apply(parse_targets_to_list)
df["Target_B_list"] = df.get("Target_B", pd.Series("", index=df.index)).apply(parse_targets_to_list)

valid_genes = set(gene_df_all.columns)

def calc_target_expr(row_idx, target_list):

    matched = [g for g in target_list if g in valid_genes] #only keep targets we actually have expression data for

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

def get_morgan_fingerprint(smiles, radius=2, nBits=2048): #morgan fp captures circular neighbourhood around each atom
    if not smiles or pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)

fp_cache = {
    k: get_morgan_fingerprint(v) for k, v in master_smiles.items()
    } #precompute all fingerprints once to avoid recalculating per row

def calc_tanimoto(drug_a, drug_b):
    fp_a = fp_cache.get(normalize_key(drug_a))
     fp_b = fp_cache.get(normalize_key(drug_b))
    if fp_a is None or fp_b is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)

df["Tanimoto_Similarity"] = df.apply(lambda row: calc_tanimoto(row["Drug_A"], row["Drug_B"]), axis=1)

NON_GENE_COLS = [
    "Drug_A_enc", "Drug_B_enc", "Cell_Line_enc", "Cancer_Type_enc","Target_A_count", "Target_B_count", "Has_Target_A", "Has_Target_B",
    "same_target_flag", "total_targets", "target_diff", "target_jaccard", "A_target_expr_mean", "A_target_expr_max", "A_target_expr_sum","B_target_expr_mean", "B_target_expr_max", "B_target_expr_sum","Target_expr_mean_sum", "Target_expr_mean_diff"
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
) #xgboost params tuned to reduce overfitting on small dataset
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
        for conv, bn in zip(self.convs, self.bns): #apply conv then batchnorm then relu at each layer
            x = F.relu(bn(conv(x, ei)))
               x = F.dropout(x, p=self.dropout, training=self.training)
        return global_add_pool(x, b) #sum atom embeddings to get one vector per molecule

class DrugEncoder(nn.Module):
    def __init__(self, node_feat_dim, all_drugs, graph_cache, embed_dim=128):
        super().__init__()
        self.all_drugs = all_drugs
         self.embed_dim = embed_dim
         self.gcn_drugs = [d for d in all_drugs if d in graph_cache]
       
        self.fb_drugs = [d for d in all_drugs if d not in graph_cache] #drugs without smiles get a learned embedding instead
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
        return self.regressor(torch.cat([ea, eb], dim=1)).squeeze(-1) #concat both drug embeddings then predict

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
        return self.regressor(torch.cat([ea, eb, self.bio_enc(bio)], dim=1)).squeeze(-1) #fuse drug structure and gene expression


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
    
    X = np.concatenate([X_non_gene, gframe[sel].values], axis=1) #combine both feature sets
    sc = StandardScaler()

    return sc.fit_transform(X[tr]), sc.transform(X[te])

def run_sklearn_cv(data_df, y_arr, grp_arr, X_builder, model_fn, label):
    gkf = GroupKFold(n_splits=N_SPLITS)

    folds = []

    fold_test_data = []

    print(f"\n  {label}")
    for fi, (tr, te) in enumerate(gkf.split(data_df, y_arr, grp_arr), 1):
    assert not set(grp_arr[tr]) & set(grp_arr[te]) #make sure no drug pair appears in both train and test

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
        vi = perm[:int(0.15 * len(tr))] #use 15% of training data as validation
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
        sch = ReduceLROnPlateau(opt, "min", factor=0.5, patience=8) #halve lr if val loss plateaus for 8 epochs
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

            if vl < best_l: #save best weights based on validation loss
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
                nn.utils.clip_grad_norm_(model.parameters(), 1.0) #clip gradients to stabilise training
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

        model.load_state_dict(best_st) #restore best checkpoint before evaluating on test set
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

    print(f"CV Pearson : {res['Pearson_mean']:.4f} ± {res['Pearson_std']:.4f}")
    
    print(f"CV Spearman : {res['Spearman_mean']:.4f} ± {res['Spearman_std']:.4f}")
    
    print(f"CV C-Index : {res['CIndex_mean']:.4f} ± {res['CIndex_std']:.4f}")
    print(f"CV MAE : {res['MAE_mean']:.4f} ± {res['MAE_std']:.4f}")
    
    return res

def run_ensemble_cv(res_a, res_b, y_arr, weight_candidates=None, label="Ensemble"):
        if weight_candidates is None:
        weight_candidates = [(0.7,0.3),(0.6,0.4),(0.5,0.5),(0.4,0.6),(0.3,0.7)]

    best_mean = -999
    best_res = None

    print(f"\n  {label}")
    print(
        f" Model A: {res_a['Model']} (Pearson={res_a['Pearson_mean']:.4f})"
        )
    print(
        f" Model B: {res_b['Model']} (Pearson={res_b['Pearson_mean']:.4f})")

    for wa, wb in weight_candidates: #try different blending weights and keep the best
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

COLORS = ["#64748B","#3B82F6","#10B981","#F59E0B","#EF4444","#8B5CF6","#EC4899","#14B8A6"] #one color per model

def plot_bar(results, title, fname):
    labels = [r["Model"] for r in results]
    vals = [r["Pearson_mean"] for r in results]
    errs = [r["Pearson_std"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, vals, color=COLORS[:len(results)], edgecolor="white")
    ax.errorbar(labels, vals, yerr=errs, fmt="none", color="black", capsize=5) #error bars show std across folds
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
print("Genes : Top-300 by variance")

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
print("Models : XGBoost | XGBoost+Genes | GCN | GCN+Genes | Ensemble")

print("Genes : Top-300 by variance (per-fold, no leakage)")
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

print(f"\n Experiment 2 Results ({time.time()-t0:.0f}s) ──")
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
} #blue palette for consistent styling across all figures

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
