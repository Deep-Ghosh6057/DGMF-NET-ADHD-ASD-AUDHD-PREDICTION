"""
===============================================================================
DGMF-Net: Data Loader Module
===============================================================================
Author : Deep Kumar Ghosh

Description:
    Loads the NSCH 2023-2024 dataset and variable labels.
===============================================================================
"""

from pathlib import Path
import pandas as pd


# =============================================================================
# Project Root
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"

DATA_FILE = DATA_DIR / "NSCH_2023_2024.csv"
LABEL_FILE = DATA_DIR / "NSCH_2023_2024_Labels.xlsx"


# =============================================================================
# Load Dataset
# =============================================================================

def load_nsch_data():
    """
    Load the NSCH dataset and variable labels.

    Returns
    -------
    df : pandas.DataFrame
        NSCH dataset

    labels : pandas.DataFrame
        Variable labels
    """

    print("=" * 80)
    print("Loading NSCH 2023-2024 Dataset...")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Check dataset exists
    # -------------------------------------------------------------------------
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"\nDataset not found:\n{DATA_FILE}")

    if not LABEL_FILE.exists():
        raise FileNotFoundError(f"\nVariable labels not found:\n{LABEL_FILE}")

    # -------------------------------------------------------------------------
    # Load files
    # -------------------------------------------------------------------------
    df = pd.read_csv(DATA_FILE)

    labels = pd.read_excel(LABEL_FILE)

    print("Dataset Loaded Successfully")
    print(f"Dataset Shape : {df.shape}")

    print("\nVariable Labels Loaded Successfully")
    print(f"Number of Variables : {labels.shape[0]}")

    return df, labels


# =============================================================================
# Dataset Summary
# =============================================================================

def dataset_summary(df):
    """
    Display basic information about the dataset.
    """

    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)

    print("\nDataset Shape")
    print(df.shape)

    print("\nDataset Information")
    print("-" * 80)
    df.info()

    print("\nMissing Values (Top 20)")
    print("-" * 80)
    print(df.isnull().sum().sort_values(ascending=False).head(20))

    print("\nFirst Five Rows")
    print("-" * 80)
    print(df.head())

    print("\nLast Five Rows")
    print("-" * 80)
    print(df.tail())

    print("\nColumn Names")
    print("-" * 80)
    print(df.columns.tolist())