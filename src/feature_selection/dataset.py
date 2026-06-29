import pandas as pd


LEAKAGE_FEATURE_ALIASES = {
    "adhdind_2324",
    "adhdsevind_2324",
    "adhdmed_2324",
    "adhdbehtreat_2324",
    "autismind_2324",
    "asdsevind_2324",
    "asdmed_2324",
    "asdbehtreat_2324",
    "asdage_2324",
    "asddrtype_2324",
    "medb10scrq5_2324",
}


def remove_leakage_features(X):
    """Remove diagnosis leakage columns regardless of naming case."""

    existing = [
        column
        for column in X.columns
        if str(column).lower() in LEAKAGE_FEATURE_ALIASES
    ]

    return X.drop(columns=existing), existing


def prepare_feature_selection_data(fused_df, target_df):
    """
    Prepare X and y for feature selection.
    """

    X = fused_df.copy()
    X, existing = remove_leakage_features(X)

    print("Leakage columns removed:")
    print(existing)

    y = target_df["TARGET_NAME"].copy()

    return X, y

