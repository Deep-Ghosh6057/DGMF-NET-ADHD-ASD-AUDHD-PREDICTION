import numpy as np
import pandas as pd


def create_shadow_features(X, random_state=42):
    """
    Create Boruta shadow features by randomly permuting
    every original feature.
    """

    rng = np.random.default_rng(random_state)

    shadow_features = {}

    for column in X.columns:
        shadow_features[f"Shadow_{column}"] = rng.permutation(
            X[column].values
        )

    shadow_df = pd.DataFrame(
        shadow_features,
        index=X.index
    )

    X_shadow = pd.concat(
        [X.reset_index(drop=True),
         shadow_df.reset_index(drop=True)],
        axis=1
    )

    return X_shadow