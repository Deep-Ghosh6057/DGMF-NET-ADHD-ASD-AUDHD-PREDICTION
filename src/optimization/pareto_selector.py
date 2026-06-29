import pandas as pd


def select_best_trial(study, weights=None):
    """
    Select the best trial from the Pareto front using
    a weighted score.
    """

    if weights is None:
        weights = [0.30, 0.20, 0.30, 0.20]

    rows = []

    for trial in study.best_trials:

        score = sum(
            w * v
            for w, v in zip(weights, trial.values)
        )

        rows.append({
            "Trial": trial.number,
            "Accuracy": trial.values[0],
            "Recall": trial.values[1],
            "ROC_AUC": trial.values[2],
            "MCC": trial.values[3],
            "Score": score,
            "Params": trial.params
        })

    pareto_df = pd.DataFrame(rows)

    pareto_df = pareto_df.sort_values(
        "Score",
        ascending=False
    ).reset_index(drop=True)

    best_trial = study.best_trials[0]

    best_score = -1

    for trial in study.best_trials:

        score = sum(
            w * v
            for w, v in zip(weights, trial.values)
        )

        if score > best_score:

            best_score = score

            best_trial = trial

    return best_trial, pareto_df

    