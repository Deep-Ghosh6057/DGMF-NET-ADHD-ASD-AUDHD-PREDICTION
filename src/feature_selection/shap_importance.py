import numpy as np
import pandas as pd
import shap

from catboost import CatBoostClassifier


def compute_shap_importance(
    X,
    y,
    random_state=42
):
    """
    Train a CatBoost model and compute mean absolute SHAP feature importance.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.

    y : array-like
        Encoded target labels.

    Returns
    -------
    model : CatBoostClassifier
        Trained CatBoost model.

    importance_df : pandas.DataFrame
        Feature importance ranked by mean absolute SHAP value.
    """

    model = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.05,
        loss_function="MultiClass",
        verbose=False,
        random_seed=random_state
    )

    model.fit(X, y)

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    # ---------------------------------------------------------
    # DEBUG INFORMATION
    # ---------------------------------------------------------
    print("=" * 60)
    print("SHAP DEBUG")

    print("Type :", type(shap_values))

    if isinstance(shap_values, list):
        print("Returned LIST")
        print("Length :", len(shap_values))
        print("First shape :", shap_values[0].shape)

    else:
        print("Returned ARRAY")
        print("Shape :", shap_values.shape)

    print("=" * 60)

    # ---------------------------------------------------------
    # Compute Feature Importance
    # ---------------------------------------------------------

    if isinstance(shap_values, list):

        # Older SHAP versions
        importance = np.mean(
            [
                np.abs(values).mean(axis=0)
                for values in shap_values
            ],
            axis=0
        )

    else:

        # SHAP >= 0.49
        if shap_values.ndim == 3:

            # (samples, features, classes)
            importance = np.abs(shap_values).mean(axis=(0, 2))

        else:

            # Binary classification
            importance = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    ).reset_index(drop=True)

    return model, importance_df