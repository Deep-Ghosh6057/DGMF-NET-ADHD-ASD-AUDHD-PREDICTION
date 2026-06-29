import optuna

from optuna.samplers import NSGAIISampler

from src.optimization.objective_catboost import (
    catboost_objective
)


def optimize_catboost(
    X,
    y,
    n_trials=100,
    random_state=42
):
    """
    Multi-objective optimization using Optuna + NSGA-II.
    """

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

        lambda trial: catboost_objective(
            trial,
            X,
            y
        ),

        n_trials=n_trials,

        show_progress_bar=True

    )

    return study