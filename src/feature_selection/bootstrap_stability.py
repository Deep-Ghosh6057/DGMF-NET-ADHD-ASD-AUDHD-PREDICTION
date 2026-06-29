import numpy as np
import pandas as pd

from src.feature_selection.dataset import LEAKAGE_FEATURE_ALIASES


def _filter_leakage_features(features):
    return [
        feature
        for feature in features
        if str(feature).lower() not in LEAKAGE_FEATURE_ALIASES
    ]


def bootstrap_sample(
    X,
    y,
    random_state=None
):
    """
    Draw one bootstrap sample.
    """

    rng = np.random.default_rng(random_state)

    idx = rng.choice(
        len(X),
        size=len(X),
        replace=True
    )

    X_boot = X.iloc[idx].reset_index(drop=True)

    if isinstance(y, pd.Series):
        y_boot = y.iloc[idx].reset_index(drop=True)
    else:
        y_boot = pd.Series(y[idx])

    return X_boot, y_boot


from src.feature_selection.shadow_features import create_shadow_features
from src.feature_selection.shap_importance import compute_shap_importance
from src.feature_selection.boruta_selector import select_features


def run_one_iteration(
    X,
    y,
    seed
):
    """
    One Bootstrap BorutaSHAP iteration.
    """

    X_boot, y_boot = bootstrap_sample(
        X,
        y,
        random_state=seed
    )

    X_shadow = create_shadow_features(
        X_boot
    )

    _, importance_df = compute_shap_importance(
        X_shadow,
        y_boot
    )

    selected, _, _ = select_features(
        importance_df
    )

    return selected["Feature"].tolist()


def bootstrap_stability_selection(
    X,
    y,
    n_iterations=30,
    threshold=0.80,
    random_state=42
):
    """
    Perform Bootstrap BorutaSHAP Stability Selection.
    """

    selected_feature_lists = []

    for i in range(n_iterations):

        print(f"Bootstrap Iteration {i+1}/{n_iterations}")

        features = run_one_iteration(
            X,
            y,
            seed=random_state + i
        )

        selected_feature_lists.append(features)

    # Count selection frequency
    feature_counts = {}

    for feature_list in selected_feature_lists:

        for feature in feature_list:

            feature_counts[feature] = (
                feature_counts.get(feature, 0) + 1
            )

    stability_df = pd.DataFrame({

        "Feature": feature_counts.keys(),

        "Frequency": feature_counts.values()

    })

    stability_df["SelectionRate"] = (

        stability_df["Frequency"] /
        n_iterations

    )

    stability_df = stability_df.sort_values(

        "SelectionRate",

        ascending=False

    ).reset_index(drop=True)

    stable_features = stability_df[
        stability_df["SelectionRate"] >= threshold
    ]["Feature"].tolist()

    stable_features = _filter_leakage_features(stable_features)

    return stable_features, stability_df