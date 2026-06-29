from xgboost import XGBClassifier


def train_final_xgboost(
    X_train,
    y_train,
    params,
    random_state=42
):
    """
    Train the final XGBoost model using the
    best hyperparameters obtained from Optuna.
    """

    model = XGBClassifier(

        n_estimators=params["n_estimators"],

        max_depth=params["max_depth"],

        learning_rate=params["learning_rate"],

        subsample=params["subsample"],

        objective="multi:softprob",

        num_class=len(set(y_train)),

        eval_metric="mlogloss",

        random_state=random_state,

        verbosity=0,

        tree_method="hist"

    )

    model.fit(
        X_train,
        y_train
    )

    return model