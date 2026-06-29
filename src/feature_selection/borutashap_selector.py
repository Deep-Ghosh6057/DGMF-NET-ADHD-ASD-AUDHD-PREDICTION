import pandas as pd
import numpy as np


def create_shadow_features(X, random_state=42):
    """
    Create shuffled shadow features.

    Parameters
    ----------
    X : pandas.DataFrame

    Returns
    -------
    X_shadow : pandas.DataFrame
    """

    rng = np.random.default_rng(random_state)

    shadow = X.copy()

    for column in shadow.columns:
        shadow[column] = rng.permutation(shadow[column].values)

    shadow.columns = [
        f"Shadow_{c}"
        for c in shadow.columns
    ]

    X_shadow = pd.concat(
        [X, shadow],
        axis=1
    )

    return X_shadow