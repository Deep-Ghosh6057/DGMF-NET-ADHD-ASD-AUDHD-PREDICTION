import optuna

from optuna.samplers import NSGAIISampler

from src.optimization.objective_xgboost import (
    xgboost_objective
)


def optimize_xgboost(
    X,
    y,
    n_trials=100,
    random_state=42
):

    sampler = NSGAIISampler(
        seed=random_state
    )

    study = optuna.create_study(

        directions=[
            "maximize",
            "maximize",
            "maximize",
            "maximize"
        ],

        sampler=sampler

    )

    study.optimize(

        lambda trial: xgboost_objective(
            trial,
            X,
            y
        ),

        n_trials=n_trials,

        show_progress_bar=True

    )

    return study