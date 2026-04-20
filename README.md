# SynFuse – Drug Synergy Prediction

This repository contains the code used for our Topics in CS II project, where we worked on predicting drug synergy using a combination of biological data and machine learning models.

The project focuses on combining gene expression data with drug-related features to better understand how different drug pairs interact in cancer cell lines.

---

## Project Structure

There are two main files in this repository:

### 1. geneexp.py
This file handles the data preparation stage.

It includes:
- Loading the AstraZeneca drug combination dataset
- Loading CCLE gene expression data
- Matching cell lines across datasets using cleaned identifiers
- Merging everything into a single dataset
- Basic cleaning and formatting
- Exporting the final merged dataset for modeling

This step is important because the datasets do not align directly, so a lot of effort went into cleaning names and building a proper mapping.

---

### 2. Experiments.py
:contentReference[oaicite:0]{index=0}

This file contains the full modeling pipeline.

It includes:
- Data preprocessing and feature construction
- Gene expression feature selection (top-300 variance)
- Tabular models (Random Forest, MLP, XGBoost)
- Graph-based model (GCN for drug structures)
- Hybrid model (GCN + gene expression)
- Ensemble model combining predictions
- Evaluation using GroupKFold (to avoid data leakage)

It also handles:
- SMILES processing using RDKit
- Graph construction for drugs
- Fallback handling for missing structural data
- Training loops for both tabular and deep learning models

---

## Key Idea

Instead of relying on one type of data, this project combines:

- Gene expression (biological context)
- Drug structure (graph representation)
- Engineered features (targets, similarity, etc.)

The goal was to check whether combining these gives better predictions than using them separately.

---

## Evaluation Approach

We used a strict GroupKFold setup where entire drug pairs are held out.

This means:
- The model is always tested on unseen drug combinations
- Results reflect real generalization, not memorization

---

## Notes

- Some drugs do not have available SMILES data, so fallback embeddings are used
- Synergy values are noisy and have a wide range, which makes prediction harder
- Because of this, ranking metrics (Spearman, C-Index) are also used alongside Pearson

---

## Output

The main outputs include:
- Model performance metrics across folds
- Comparison between different model types
- Final ensemble results

---

## Running the Code

This project was developed in Google Colab with Google Drive integration.

To run:
1. Mount your Google Drive
2. Update file paths if needed
3. Run `geneexp.py` first to generate the dataset
4. Run `Experiments.py` for training and evaluation

---

## Authors

- Reena Sabouh  
- Ruqayyah Masad  

Supervised by Dr. Ayad Turky
