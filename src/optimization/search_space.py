"""
Hyperparameter search spaces for Optuna.
"""


def catboost_search_space(trial):

    params = {

        "depth": trial.suggest_int(
            "depth",
            4,
            10
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.30,
            log=True
        ),

        "iterations": trial.suggest_int(
            "iterations",
            200,
            1000
        ),

        "l2_leaf_reg": trial.suggest_float(
            "l2_leaf_reg",
            1.0,
            10.0
        )

    }

    return params


def xgboost_search_space(trial):

    params = {

        "n_estimators": trial.suggest_int(
            "n_estimators",
            200,
            1000
        ),

        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            10
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.30,
            log=True
        ),

        "subsample": trial.suggest_float(
            "subsample",
            0.6,
            1.0
        )

    }

    return params