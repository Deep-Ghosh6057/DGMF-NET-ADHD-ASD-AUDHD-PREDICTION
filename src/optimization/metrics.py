import numpy as np

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    matthews_corrcoef,
    roc_auc_score
)

from sklearn.preprocessing import label_binarize


def compute_metrics(
    y_true,
    y_pred,
    y_prob
):
    """
    Compute multi-objective metrics.
    """

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro"
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred
    )

    classes = np.unique(y_true)

    y_true_bin = label_binarize(
        y_true,
        classes=classes
    )

    roc_auc = roc_auc_score(
        y_true_bin,
        y_prob,
        average="macro",
        multi_class="ovr"
    )

    return {

        "accuracy": accuracy,

        "recall": recall,

        "roc_auc": roc_auc,

        "mcc": mcc

    }