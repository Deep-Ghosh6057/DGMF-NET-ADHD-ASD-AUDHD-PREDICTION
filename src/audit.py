"""
===============================================================================
DGMF-Net : Data Audit Module
===============================================================================
"""

import pandas as pd
import numpy as np


def run_data_audit(df):
    """
    Comprehensive Data Audit
    """

    print("=" * 90)
    print("SECTION 4 : DATA AUDIT")
    print("=" * 90)

    # ==========================================================
    # Dataset Shape
    # ==========================================================

    print("\nDataset Shape")
    print("-" * 90)
    print(df.shape)

    # ==========================================================
    # Duplicate Rows
    # ==========================================================

    duplicates = df.duplicated().sum()

    print("\nDuplicate Rows")
    print("-" * 90)
    print(duplicates)

    # ==========================================================
    # Data Types
    # ==========================================================

    print("\nData Types")
    print("-" * 90)
    print(df.dtypes.value_counts())

    # ==========================================================
    # Numeric / Categorical
    # ==========================================================

    numeric_cols = df.select_dtypes(include=np.number).columns

    categorical_cols = df.select_dtypes(exclude=np.number).columns

    print("\nNumeric Columns :", len(numeric_cols))
    print("Categorical Columns :", len(categorical_cols))

    # ==========================================================
    # Memory Usage
    # ==========================================================

    memory = df.memory_usage(deep=True).sum() / (1024**2)

    print("\nMemory Usage")
    print("-" * 90)
    print(f"{memory:.2f} MB")

    # ==========================================================
    # Constant Columns
    # ==========================================================

    constant_cols = [
        col for col in df.columns
        if df[col].nunique(dropna=False) == 1
    ]

    print("\nConstant Columns :", len(constant_cols))

    # ==========================================================
    # High Cardinality
    # ==========================================================

    high_cardinality = [
        col for col in df.columns
        if df[col].nunique() > 100
    ]

    print("High Cardinality Columns :", len(high_cardinality))

    # ==========================================================
    # Encoded Missing Values
    # ==========================================================

    print("\nEncoded Missing Value Report")
    print("-" * 90)

    encoded_missing = [
        99,
        999,
        9999,
        9990
    ]

    missing_report = {}

    for code in encoded_missing:

        count = (df == code).sum().sum()

        missing_report[code] = int(count)

        print(f"{code:<8} : {count}")

    # ==========================================================
    # Columns containing encoded missing values
    # ==========================================================

    print("\nColumns containing 9990")
    print("-" * 90)

    cols_9990 = [
        c for c in df.columns
        if (df[c] == 9990).any()
    ]

    print(cols_9990)

    print("\nColumns containing 9999")
    print("-" * 90)

    cols_9999 = [
        c for c in df.columns
        if (df[c] == 9999).any()
    ]

    print(cols_9999)

    print("\nColumns containing 999")
    print("-" * 90)

    cols_999 = [
        c for c in df.columns
        if (df[c] == 999).any()
    ]

    print(cols_999)

    print("\nColumns containing 99")
    print("-" * 90)

    cols_99 = [
        c for c in df.columns
        if (df[c] == 99).any()
    ]

    print(cols_99)

    # ==========================================================
    # Numeric Summary
    # ==========================================================

    print("\nNumeric Summary")
    print("-" * 90)

    print(df.describe().T.head(20))

    print("=" * 90)

    return {
        "duplicates": duplicates,
        "constant_columns": constant_cols,
        "high_cardinality": high_cardinality,
        "missing_report": missing_report,
        "columns_9990": cols_9990,
        "columns_9999": cols_9999,
        "columns_999": cols_999,
        "columns_99": cols_99,
    }


