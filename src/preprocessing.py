"""
===============================================================================
DGMF-Net : Data Cleaning Module
===============================================================================

"""

from pathlib import Path
import numpy as np
import pandas as pd


# =============================================================================
# Project Directories
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(exist_ok=True)


# =============================================================================
# Data Cleaning
# =============================================================================

def clean_data(df):
    """
    Perform basic data cleaning.
    """

    print("=" * 90)
    print("SECTION 5 : DATA CLEANING")
    print("=" * 90)

    # -------------------------------------------------------------------------
    # Create copy
    # -------------------------------------------------------------------------

    clean_df = df.copy()

    original_shape = clean_df.shape

    # -------------------------------------------------------------------------
    # Remove duplicate rows
    # -------------------------------------------------------------------------

    duplicate_rows = clean_df.duplicated().sum()

    clean_df = clean_df.drop_duplicates()

    print(f"\nDuplicate Rows Removed : {duplicate_rows}")

    # -------------------------------------------------------------------------
    # Replace encoded missing values
    # -------------------------------------------------------------------------

    print("\nReplacing Encoded Missing Values...")

    replacement_summary = {}

    missing_columns = {
    "HEIGHT": [9990, 9999],
    "WEIGHT": [9990, 9999]
}

    for column, values in missing_columns.items():

     if column in clean_df.columns:

        count = clean_df[column].isin(values).sum()

        replacement_summary[column] = int(count)

        clean_df[column] = clean_df[column].replace(values, np.nan)

    print("\nReplacement Summary")

    for col, cnt in replacement_summary.items():

        print(f"{col:<15} : {cnt}")

    # -------------------------------------------------------------------------
    # Final Shape
    # -------------------------------------------------------------------------

    print("\nOriginal Shape :", original_shape)

    print("Cleaned Shape  :", clean_df.shape)

    # -------------------------------------------------------------------------
    # Save dataset
    # -------------------------------------------------------------------------

    save_path = PROCESSED_DIR / "NSCH_2023_2024_Cleaned.csv"

    clean_df.to_csv(save_path, index=False)

    print("\nCleaned dataset saved successfully.")

    print(save_path)

    print("=" * 90)

    return clean_df


