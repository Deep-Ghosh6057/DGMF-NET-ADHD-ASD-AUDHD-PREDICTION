import numpy as np

from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold

from src.optimization.search_space import (
    xgboost_search_space
)

from src.optimization.metrics import (
    compute_metrics
)


def xgboost_objective(
    trial,
    X,
    y
):
    """
    Multi-objective function for XGBoost.
    """

    params = xgboost_search_space(
        trial
    )

    params.update({

        "objective": "multi:softprob",

        "num_class": len(np.unique(y)),

        "eval_metric": "mlogloss",

        "random_state": 42,

        "verbosity": 0,

        "tree_method": "hist"

    })

    skf = StratifiedKFold(

        n_splits=5,

        shuffle=True,

        random_state=42

    )

    accuracy_scores = []

    recall_scores = []

    roc_auc_scores = []

    mcc_scores = []

    for train_idx, valid_idx in skf.split(X, y):

        X_train_fold = X.iloc[train_idx]

        X_valid_fold = X.iloc[valid_idx]

        y_train_fold = y[train_idx]

        y_valid_fold = y[valid_idx]

        model = XGBClassifier(
            **params
        )

        model.fit(
            X_train_fold,
            y_train_fold
        )

        y_pred = model.predict(
            X_valid_fold
        )

        y_prob = model.predict_proba(
            X_valid_fold
        )

        metrics = compute_metrics(

            y_valid_fold,

            y_pred,

            y_prob

        )

        accuracy_scores.append(
            metrics["accuracy"]
        )

        recall_scores.append(
            metrics["recall"]
        )

        roc_auc_scores.append(
            metrics["roc_auc"]
        )

        mcc_scores.append(
            metrics["mcc"]
        )

    return (

        np.mean(accuracy_scores),

        np.mean(recall_scores),

        np.mean(roc_auc_scores),

        np.mean(mcc_scores)

    )