import pandas as pd

from src.feature_selection.dataset import LEAKAGE_FEATURE_ALIASES


def _filter_leakage_features(features):
    return [
        feature
        for feature in features
        if str(feature).lower() not in LEAKAGE_FEATURE_ALIASES
    ]


def stability_selection(
    selected_feature_lists,
    threshold=0.80
):
    """
    Bootstrap Stability Selection

    Parameters
    ----------
    selected_feature_lists : list
        List of selected feature names from each bootstrap iteration.

    threshold : float
        Minimum selection frequency.

    Returns
    -------
    stable_features : list
    stability_df : DataFrame
    """

    feature_count = {}

    total_iterations = len(selected_feature_lists)

    # Count feature occurrences
    for feature_list in selected_feature_lists:

        for feature in feature_list:

            feature_count[feature] = (
                feature_count.get(feature, 0) + 1
            )

    stability_df = pd.DataFrame({

        "Feature": feature_count.keys(),

        "Frequency": feature_count.values()

    })

    stability_df["SelectionRate"] = (

        stability_df["Frequency"] /
        total_iterations

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