#Data set link - https://www.childhealthdata.org/dataset/download?rq=16465
# DGMF-Net for ADHD, ASD, and AuDHD Prediction

DSM-5 Guided Graph-Based Multi-View Feature Learning Framework for multiclass ADHD, ASD, and co-occurring ADHD+ASD prediction using the National Survey of Children's Health (NSCH) 2023-2024 dataset.

This repository contains an end-to-end research pipeline that combines clinical domain mapping, graph autoencoder embeddings, feature fusion, BorutaSHAP stability selection, Optuna NSGA-II hyperparameter optimization, CatBoost/XGBoost classifiers, and SHAP explainability.

## Project Status

The main workflow is implemented in `notebooks/DGMF_Net.ipynb` and supported by reusable modules under `src/`.

The repository intentionally does not track:

- Python virtual environments
- Raw and processed CSV datasets
- Generated model files
- Generated figures and output tables
- CatBoost training logs

These files are ignored because they are large, local, generated, or restricted by GitHub's 100 MB file limit.

## Research Objective

The goal is to predict four neurodevelopmental outcome classes:

| Label | Class |
| --- | --- |
| `0` | Healthy |
| `1` | ADHD |
| `2` | ASD |
| `3` | AuDHD |

The target labels are derived from NSCH diagnosis indicators:

- `ADHDind_2324`
- `AutismInd_2324`

The project is intended for research and educational use. It is not a clinical diagnostic tool.

## High-Level Pipeline

1. Load NSCH 2023-2024 data and variable labels.
2. Run data audit and exploratory analysis.
3. Clean NSCH special missing codes.
4. Create ADHD, ASD, AuDHD, and Healthy target classes.
5. Prepare data for TabDDPM synthetic data / imputation workflow.
6. Map selected NSCH variables to DSM-5 inspired domains.
7. Compute DSM-5 guided domain scores.
8. Build participant-level graph representations.
9. Train a graph autoencoder and extract graph embeddings.
10. Fuse DSM-5 features with graph embeddings.
11. Remove diagnosis leakage features.
12. Run BorutaSHAP bootstrap stability feature selection.
13. Optimize CatBoost and XGBoost with Optuna NSGA-II.
14. Select best Pareto-front trials.
15. Train final CatBoost and XGBoost models.
16. Evaluate final models with multiclass metrics.
17. Generate confusion matrices, ROC curves, SHAP plots, and output tables.
18. Save final models, label encoder, stable features, and hyperparameters.

## Repository Layout

```text
DGMF-Net-ADHD-ASD/
|-- data/
|   |-- raw/
|   |   `-- NSCH_2023_2024_Labels.xlsx
|   `-- processed/
|-- docs/
|   `-- CAHMI DRC 2023-2024 NSCH Codebook.pdf
|-- external/
|   `-- TabDDPM/
|-- notebooks/
|   `-- DGMF_Net.ipynb
|-- src/
|   |-- dsm5/
|   |-- feature_selection/
|   |-- fusion/
|   |-- graph/
|   |-- optimization/
|   |-- tabddpm/
|   |-- audit.py
|   |-- config.py
|   |-- data_loader.py
|   |-- eda.py
|   |-- preprocessing.py
|   `-- target.py
|-- .gitignore
|-- requirements.txt
`-- run_pipeline.py
```

Generated folders are created locally by the pipeline:

```text
models/
outputs/
outputs/figures/
outputs/tables/
outputs/logs/
outputs/reports/
figures/
```

## Data Requirements

The loader expects the following local files:

```text
data/raw/NSCH_2023_2024.csv
data/raw/NSCH_2023_2024_Labels.xlsx
```

Only the label workbook is tracked. The main CSV is ignored because it is larger than GitHub's file-size limit.

After target creation and preprocessing, generated CSV files are written under:

```text
data/processed/
```

Those processed CSV files are also ignored because they are generated and large.

## Environment Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the main dependencies. The current `requirements.txt` is empty, so install the libraries used by the source code and notebook:

```powershell
pip install numpy pandas matplotlib seaborn scikit-learn openpyxl
pip install catboost xgboost optuna shap joblib
pip install torch torch-geometric
```

Depending on your machine, `torch` and `torch-geometric` may require version-specific installation commands. If installation fails, use the official PyTorch and PyTorch Geometric installation selectors for your Python, CUDA, and OS configuration.

Initialize the TabDDPM submodule if needed:

```powershell
git submodule update --init --recursive
```

## Running the Main Notebook

Open:

```text
notebooks/DGMF_Net.ipynb
```

Run cells from top to bottom. The notebook is organized as a complete workflow, from data loading through final model export.

Important notebook sections include:

| Section | Purpose |
| --- | --- |
| 1-3 | Imports, setup, config |
| 4-7 | Data loading, audit, cleaning, target creation |
| 8 | TabDDPM preparation |
| 9-15 | DSM-5 mapping, scoring, graph construction, graph autoencoder |
| 16-19 | Feature fusion, leakage removal, stable feature selection |
| 25-31 | CatBoost/XGBoost Optuna optimization and Pareto selection |
| 33-35 | Final model training and evaluation |
| 37-38 | Visualizations and SHAP explainability |
| 39-41 | Model saving, result export, final verification |

## Core Modules

### Data Loading and Cleaning

- `src/data_loader.py`
  - Loads `NSCH_2023_2024.csv`
  - Loads `NSCH_2023_2024_Labels.xlsx`
  - Prints dataset summary information

- `src/preprocessing.py`
  - Cleans encoded missing values
  - Produces cleaned tabular data

- `src/audit.py`
  - Runs a dataset audit
  - Reports missing values, duplicate rows, and encoded missing patterns

- `src/eda.py`
  - Produces overview tables
  - Missing-value analysis
  - Numeric distribution plots
  - Numeric summary tables

### Target Construction

- `src/target.py`
  - Filters valid ADHD and Autism diagnosis codes
  - Creates four target classes:
    - Healthy
    - ADHD
    - ASD
    - AuDHD
  - Saves `data/processed/NSCH_Target_Dataset.csv`

### DSM-5 Guided Feature Mapping

- `src/dsm5/mapping.py`
  - Maps selected NSCH variables into DSM-5 inspired ADHD and ASD domains:
    - ADHD Inattention
    - ADHD Hyperactivity
    - ASD Social Communication
    - ASD Restricted/Repetitive Behavior

- `src/dsm5/scoring.py`
  - Converts special missing codes to `NaN`
  - Computes domain score averages
  - Median-imputes missing score values

### Graph Representation Learning

- `src/graph/constructor.py`
  - Builds graph edges from Pearson correlation between DSM-5 variables
  - Converts each participant into a PyTorch Geometric `Data` graph

- `src/graph/autoencoder.py`
  - Defines the graph convolutional encoder
  - Builds the graph autoencoder

- `src/graph/trainer.py`
  - Trains the graph autoencoder

- `src/graph/embeddings.py`
  - Extracts learned graph embeddings

### Feature Fusion

- `src/fusion/feature_fusion.py`
  - Combines DSM-5 guided features and graph embeddings into a fused feature table

### Feature Selection

- `src/feature_selection/dataset.py`
  - Removes diagnosis leakage variables such as direct ADHD/ASD indicator and treatment columns

- `src/feature_selection/shadow_features.py`
  - Creates shadow features for Boruta-style comparison

- `src/feature_selection/shap_importance.py`
  - Computes CatBoost SHAP feature importance

- `src/feature_selection/boruta_selector.py`
  - Selects original features based on shadow-feature thresholds

- `src/feature_selection/bootstrap_stability.py`
  - Runs bootstrap BorutaSHAP iterations
  - Computes stable features across bootstrap samples

- `src/feature_selection/stability.py`
  - Computes selection frequencies and stable feature lists

### Optimization and Evaluation

- `src/optimization/optimizer_catboost.py`
  - Runs Optuna NSGA-II optimization for CatBoost

- `src/optimization/optimizer_xgboost.py`
  - Runs Optuna NSGA-II optimization for XGBoost

- `src/optimization/objective_catboost.py`
  - Defines CatBoost multi-objective optimization objective

- `src/optimization/objective_xgboost.py`
  - Defines XGBoost multi-objective optimization objective

- `src/optimization/metrics.py`
  - Computes optimization metrics:
    - Accuracy
    - Macro recall
    - ROC-AUC
    - Matthews correlation coefficient

- `src/optimization/pareto_selector.py`
  - Selects the final trial from the Pareto front using weighted scoring

- `src/optimization/train_catboost.py`
  - Trains the final CatBoost model using selected hyperparameters

- `src/optimization/train_xgboost.py`
  - Trains the final XGBoost model using selected hyperparameters

- `src/optimization/evaluation.py`
  - Computes final metrics:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - ROC-AUC
    - PR-AUC
    - Balanced accuracy
    - MCC
  - Produces confusion matrix and classification report

- `src/optimization/visualization.py`
  - Saves confusion matrix plots
  - Saves multiclass ROC curves

## Output Artifacts

When the full notebook is executed, the project can generate:

```text
models/
|-- catboost_model.cbm
|-- xgboost_model.json
|-- label_encoder.pkl
|-- stable_features.pkl
`-- best_hyperparameters.pkl

outputs/tables/
|-- model_comparison.csv
|-- evaluation_metrics.csv
|-- catboost_classification_report.csv
|-- xgboost_classification_report.csv
|-- shap_feature_importance.csv
`-- final_summary.csv

outputs/figures/
|-- catboost_confusion_matrix.png
|-- xgboost_confusion_matrix.png
|-- catboost_roc_curves.png
|-- xgboost_roc_curves.png
|-- catboost_shap_summary.png
`-- catboost_shap_bar.png
```

These generated artifacts are ignored by Git. Re-run the notebook to regenerate them locally.

## Model Evaluation

The final evaluation compares CatBoost and XGBoost on the selected stable feature set.

The project reports:

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Macro ROC-AUC using one-vs-rest multiclass binarization
- Macro PR-AUC
- Balanced accuracy
- Matthews correlation coefficient
- Confusion matrices
- Multiclass ROC curves
- Classification reports

## Explainability

The notebook uses SHAP with the final CatBoost model:

- Computes SHAP values on a test subset
- Handles multiclass SHAP output shapes
- Produces summary and bar plots
- Exports mean absolute SHAP feature importance

This gives feature-level interpretability for the final fused representation.

## Leakage Control

The feature-selection pipeline removes direct diagnosis and treatment indicators before final model training. The leakage aliases include variables such as:

- `ADHDind_2324`
- `ADHDSevInd_2324`
- `ADHDMed_2324`
- `ADHDBehTreat_2324`
- `AutismInd_2324`
- `ASDSevInd_2324`
- `ASDMed_2324`
- `ASDBehTreat_2324`
- `ASDAge_2324`
- `ASDDrType_2324`
- `MEDB10ScrQ5_2324`

This is important because the target labels are derived from diagnosis indicators.

## Git and Artifact Policy

The repository tracks source code, notebook workflow, documentation, and small metadata files.

The repository does not track:

- `.venv/`
- `.venv_tabddpm/`
- `data/raw/*.csv`
- `data/processed/*.csv`
- `models/`
- `outputs/`
- `figures/`
- `notebooks/catboost_info/`

This keeps the repository small and GitHub-compatible.

If large datasets or trained models need to be shared, use one of:

- GitHub Releases
- Git LFS
- Zenodo
- OSF
- Google Drive
- Institutional storage

## Reproducibility Notes

To reproduce results on a new machine:

1. Clone the repository.
2. Initialize submodules.
3. Create a Python environment.
4. Install dependencies.
5. Place the NSCH CSV file in `data/raw/`.
6. Run `notebooks/DGMF_Net.ipynb` from top to bottom.
7. Confirm generated artifacts under `models/` and `outputs/`.

Example:

```powershell
git clone https://github.com/Deep-Ghosh6057/DGMF-NET-ADHD-ASD-AUDHD-PREDICTION.git
cd DGMF-NET-ADHD-ASD-AUDHD-PREDICTION
git submodule update --init --recursive

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install numpy pandas matplotlib seaborn scikit-learn openpyxl
pip install catboost xgboost optuna shap joblib
pip install torch torch-geometric
```

Then add:

```text
data/raw/NSCH_2023_2024.csv
```

and run the notebook.

## Known Practical Notes

- The NSCH CSV files are large and are not stored in Git.
- The notebook is the canonical workflow.
- `run_pipeline.py` is currently a placeholder and may be extended into a script entry point later.
- Graph construction and SHAP computation can be memory-intensive.
- Optuna optimization can be slow when increasing `n_trials`.
- CatBoost may create `catboost_info/` logs during training.

## Citation and Data Source

This project uses the National Survey of Children's Health 2023-2024 structure and codebook. Follow NSCH/CAHMI data-use and citation guidance when publishing results.


