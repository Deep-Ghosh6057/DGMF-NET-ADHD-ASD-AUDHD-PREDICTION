"""
===============================================================================
SECTION 8.1 : DATASET CONVERSION FOR TABDDPM
===============================================================================
"""

import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split

from src.config import PROJECT_ROOT


def prepare_tabddpm_dataset(
    df,
    target_col="TARGET",
    output_folder="NSCH_TabDDPM"
):
    """
    Convert NSCH dataframe into official TabDDPM dataset format.
    """

    print("=" * 90)
    print("SECTION 8.1 : TABDDPM DATASET PREPARATION")
    print("=" * 90)

    save_dir = (
        PROJECT_ROOT /
        "external" /
        "TabDDPM" /
        "data" /
        output_folder
    )

    save_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Remove helper column
    # ----------------------------------------------------------

    df = df.copy()

    if "TARGET_NAME" in df.columns:
        df = df.drop(columns=["TARGET_NAME"])

    # ----------------------------------------------------------
    # Separate X and y
    # ----------------------------------------------------------

    X = df.drop(columns=[target_col])

    y = df[target_col]

    # ----------------------------------------------------------
    # Train / Validation / Test Split
    # ----------------------------------------------------------

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    print(f"Training Samples   : {len(X_train):,}")
    print(f"Validation Samples : {len(X_val):,}")
    print(f"Testing Samples    : {len(X_test):,}")

    # ----------------------------------------------------------
    # Detect feature types
    # ----------------------------------------------------------

    numerical_cols = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_cols = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    print("\nNumerical Features :", len(numerical_cols))
    print("Categorical Features :", len(categorical_cols))

    # ----------------------------------------------------------
    # Save NumPy Arrays
    # ----------------------------------------------------------

    np.save(save_dir / "X_num_train.npy",
            X_train[numerical_cols].to_numpy())

    np.save(save_dir / "X_num_val.npy",
            X_val[numerical_cols].to_numpy())

    np.save(save_dir / "X_num_test.npy",
            X_test[numerical_cols].to_numpy())

    np.save(save_dir / "y_train.npy",
            y_train.to_numpy())

    np.save(save_dir / "y_val.npy",
            y_val.to_numpy())

    np.save(save_dir / "y_test.npy",
            y_test.to_numpy())

    if len(categorical_cols) > 0:

        np.save(save_dir / "X_cat_train.npy",
                X_train[categorical_cols].astype(str).to_numpy())

        np.save(save_dir / "X_cat_val.npy",
                X_val[categorical_cols].astype(str).to_numpy())

        np.save(save_dir / "X_cat_test.npy",
                X_test[categorical_cols].astype(str).to_numpy())

    # ----------------------------------------------------------
    # Generate info.json
    # ----------------------------------------------------------

    info = {

        "name": output_folder,

        "task_type": "multiclass",

        "n_classes": int(y.nunique()),

        "train_size": int(len(X_train)),

        "val_size": int(len(X_val)),

        "test_size": int(len(X_test)),

        "n_num_features": len(numerical_cols),

        "n_cat_features": len(categorical_cols),

        "target": target_col

    }

    with open(save_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    print("\nDataset saved to")

    print(save_dir)

    return save_dir

