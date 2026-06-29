from catboost import CatBoostClassifier


def train_final_catboost(
    X_train,
    y_train,
    params
):
    """
    Train the final CatBoost model using the
    best hyperparameters.
    """

    model = CatBoostClassifier(

        depth=params["depth"],

        learning_rate=params["learning_rate"],

        iterations=params["iterations"],

        l2_leaf_reg=params["l2_leaf_reg"],

        loss_function="MultiClass",

        verbose=False,

        random_seed=42

    )

    model.fit(

        X_train,

        y_train

    )

    return model