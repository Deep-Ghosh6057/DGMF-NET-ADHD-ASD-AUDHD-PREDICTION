import pandas as pd


def select_features(importance_df):
    """
    Select original features whose SHAP importance is
    greater than the maximum shadow feature importance.
    """

    shadow_df = importance_df[
        importance_df["Feature"].str.startswith("Shadow_")
    ]

    threshold = shadow_df["Importance"].max()

    selected = importance_df[
        (~importance_df["Feature"].str.startswith("Shadow_")) &
        (importance_df["Importance"] > threshold)
    ]

    rejected = importance_df[
        (~importance_df["Feature"].str.startswith("Shadow_")) &
        (importance_df["Importance"] <= threshold)
    ]

    selected = selected.reset_index(drop=True)
    rejected = rejected.reset_index(drop=True)

    return selected, rejected, threshold