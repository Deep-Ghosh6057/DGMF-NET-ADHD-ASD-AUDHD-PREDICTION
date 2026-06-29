FEATURE_SELECTION_CONFIG = {

    # BorutaSHAP
    "importance_measure": "shap",
    "classification": True,

    # Bootstrap Stability
    "bootstrap_iterations": 30,
    "selection_threshold": 0.80,

    # Randomness
    "random_state": 42,

    # CatBoost
    "iterations": 200,
    "depth": 6,
    "learning_rate": 0.05,
}