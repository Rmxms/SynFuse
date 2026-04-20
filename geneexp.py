import os
import pandas as pd
import numpy as np
from google.colab import drive

drive.mount('/content/drive')  #mount drive
PROJECT_PATH = "/content/drive/MyDrive/DrugGeneProject"  #project path
print("Project exists:", os.path.exists(PROJECT_PATH))  #check path
print("Files in project folder:")
print(os.listdir(PROJECT_PATH))  #list files
az_file = os.path.join(PROJECT_PATH, "Astrazeneca_Main.xlsx")  #az dataset
expr_file = os.path.join(PROJECT_PATH, "CCLE_expression.csv")  #expression
sample_file = os.path.join(PROJECT_PATH, "sample_info.csv")  #sample info

az = pd.read_excel(az_file)  #load az
expr = pd.read_csv(expr_file)  #load expr
sample = pd.read_csv(sample_file)  #load sample

print("AZ shape:", az.shape)
print("Expression shape:", expr.shape)
print("Sample shape:", sample.shape)
print("AZ columns:")
print(az.columns.tolist())
print("\nExpression columns:")
print(expr.columns.tolist()[:20])
print("\nSample columns:")
print(sample.columns.tolist())

display(az.head())  #preview az
display(expr.head())  #preview expr
display(sample.head())  #preview sample
#clean az columns
az_clean = az.rename(columns={
    "COMPOUND_A": "Drug_A",
    "COMPOUND_B": "Drug_B",
    "CELL_LINE": "Cell_Line",
    "SYNERGY_SCORE": "Synergy",
    "TARGET_A": "Target_A",
    "TARGET_B": "Target_B"
})[["Drug_A", "Drug_B", "Cell_Line", "Synergy", "Target_A", "Target_B", "CANCER_TYPE", "MUTATIONS"]]

print("AZ clean shape:", az_clean.shape)
display(az_clean.head())

import re

def clean_cell_line(x):  #normalize names as it caused many issues not to
    if pd.isna(x):
        return np.nan
    x = str(x).strip().lower()
    x = x.replace("-", "")
    x = x.replace("_", "")
    x = x.replace(" ", "")
    x = x.replace(":", "")
    x = x.replace(".", "")
    x = x.replace("/", "")
    return x

az_clean["Cell_Line_Clean"] = az_clean["Cell_Line"].apply(clean_cell_line)  #apply cleaning

sample_bridge = sample.copy()  #copy sample

#clean sample names
sample_bridge["cell_line_name_clean"] = sample_bridge["cell_line_name"].apply(clean_cell_line)
sample_bridge["stripped_cell_line_name_clean"] = sample_bridge["stripped_cell_line_name"].apply(clean_cell_line)
sample_bridge["CCLE_Name_clean"] = sample_bridge["CCLE_Name"].apply(lambda x: clean_cell_line(str(x).split("_")[0]) if pd.notna(x) else np.nan)

display(sample_bridge[[
    "DepMap_ID", "cell_line_name", "stripped_cell_line_name", "CCLE_Name", "cell_line_name_clean", "stripped_cell_line_name_clean", "CCLE_Name_clean"
]].head())

az_set = set(az_clean["Cell_Line_Clean"].dropna().unique())  #unique az names

#check overlap
for col in ["cell_line_name_clean", "stripped_cell_line_name_clean", "CCLE_Name_clean"]:
    sample_set = set(sample_bridge[col].dropna().unique())
    overlap = az_set & sample_set
    print(f"{col}: matched {len(overlap)} / {len(az_set)}")

mapping_parts = []  #store mappings

#build mapping table
for col in ["cell_line_name_clean", "stripped_cell_line_name_clean", "CCLE_Name_clean"]:
    temp = sample_bridge[[col, "DepMap_ID", "cell_line_name", "stripped_cell_line_name", "CCLE_Name"]].copy()
    temp = temp.rename(columns={col: "Cell_Line_Clean"})
    temp = temp.dropna(subset=["Cell_Line_Clean"])
    mapping_parts.append(temp)

mapping_df = pd.concat(mapping_parts, ignore_index=True)  #combine mappings

mapping_df = mapping_df.drop_duplicates(subset=["Cell_Line_Clean", "DepMap_ID"])  #remove duplicates

print("Mapping table shape:", mapping_df.shape)
display(mapping_df.head())

#merge az with mapping
az_mapped = az_clean.merge(
    mapping_df,
    on="Cell_Line_Clean",
    how="left"
)

print("AZ mapped shape:", az_mapped.shape)
print("Matched rows:", az_mapped["DepMap_ID"].notna().sum(), "out of", len(az_mapped))

display(az_mapped.head())

#check unmatched
unmatched = az_mapped[az_mapped["DepMap_ID"].isna()]["Cell_Line"].drop_duplicates().sort_values()
print("Unmatched cell lines:", len(unmatched))
print(unmatched.tolist()[:50])

#clean expression dataset
expr_clean = expr.rename(columns={"Unnamed: 0": "DepMap_ID"}).copy()

print("Expression clean shape:", expr_clean.shape)
display(expr_clean.iloc[:5, :8])

#final merge
merged_expr = az_mapped.merge(
    expr_clean, on="DepMap_ID", how="inner"
)

print("Merged expression shape:", merged_expr.shape)
display(merged_expr.iloc[:5, :12])
processed_path = os.path.join(PROJECT_PATH, "processed_data")
os.makedirs(processed_path, exist_ok=True)
merged_expr.to_csv(os.path.join(processed_path, "az_depmap_expression_merged.csv"), index=False)
print("Saved merged dataset.")
print("Matched rows:", az_mapped["DepMap_ID"].notna().sum(), "out of", len(az_mapped))
print("Merged shape:", merged_expr.shape)
print("Unmatched count:", az_mapped["DepMap_ID"].isna().sum())
print(unmatched.tolist()[:30])
all_drugs = pd.Series(
    pd.concat([merged_expr["Drug_A"], merged_expr["Drug_B"]]).dropna().unique(),
    name="Drug"
)

print("Number of unique drugs:", len(all_drugs))
display(all_drugs.head(20))
drug_smiles_df = pd.DataFrame({"Drug": all_drugs})
drug_smiles_df["SMILES"] = ""

display(drug_smiles_df.head())
print("Rows:", len(drug_smiles_df))

drug_smiles_path = os.path.join(processed_path, "drug_smiles_template.csv")
drug_smiles_df.to_csv(drug_smiles_path, index=False)
print("Saved:", drug_smiles_path)

#load smiles
drug_smiles_path = os.path.join(PROJECT_PATH, "drug_smiles.csv")
drug_smiles = pd.read_csv(drug_smiles_path)
print("Shape:", drug_smiles.shape)
print("Columns:", drug_smiles.columns.tolist())
display(drug_smiles.head())

#clean smiles
drug_smiles = drug_smiles[["Drug", "SMILES"]].copy()
drug_smiles["Drug"] = drug_smiles["Drug"].astype(str).str.strip()
drug_smiles["SMILES"] = drug_smiles["SMILES"].astype(str).str.strip()
print("drug_smiles shape:", drug_smiles.shape)
display(drug_smiles.head())

#compare coverage
all_drugs = set(pd.concat([merged_expr["Drug_A"], merged_expr["Drug_B"]]).dropna().astype(str).str.strip().unique())
smiles_drugs = set(drug_smiles["Drug"].dropna().astype(str).str.strip().unique())

print("Unique drugs in main dataset:", len(all_drugs))
print("Unique drugs in drug_smiles:", len(smiles_drugs))
print("Matched drugs:", len(all_drugs & smiles_drugs))
print("Missing drugs:", len(all_drugs - smiles_drugs))
print("Some missing drugs:", list(sorted(all_drugs - smiles_drugs))[:20])

#clean drug names
def clean_drug_name(x):
    if pd.isna(x):
        return x
    x = str(x).lower().strip()
    x = x.replace(" ", "")
    return x
merged_expr["Drug_A_clean"] = merged_expr["Drug_A"].apply(clean_drug_name)
merged_expr["Drug_B_clean"] = merged_expr["Drug_B"].apply(clean_drug_name)
drug_smiles["Drug_clean"] = drug_smiles["Drug"].apply(clean_drug_name)

#split names
def split_drug(x):
    if pd.isna(x):
        return x
    parts = str(x).split(",")
    return parts[-1].strip().lower().replace(" ", "")

merged_expr["Drug_A_clean"] = merged_expr["Drug_A"].apply(split_drug)
merged_expr["Drug_B_clean"] = merged_expr["Drug_B"].apply(split_drug)
drug_smiles["Drug_clean"] = drug_smiles["Drug"].apply(split_drug)

#final check
all_drugs = set(pd.concat([merged_expr["Drug_A_clean"], merged_expr["Drug_B_clean"]]).dropna().unique())
smiles_drugs = set(drug_smiles["Drug_clean"].dropna().unique())
print("Unique drugs:", len(all_drugs))
print("Matched:", len(all_drugs & smiles_drugs))
print("Missing:", len(all_drugs - smiles_drugs))
print("Some missing:", list(sorted(all_drugs - smiles_drugs))[:20])