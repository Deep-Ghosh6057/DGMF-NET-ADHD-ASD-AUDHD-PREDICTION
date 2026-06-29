import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score
)

from sklearn.preprocessing import label_binarize


def evaluate_model(
    model,
    X_test, 
    y_test
):
    """
    Evaluate any trained classification model.
    Returns evaluation metrics, confusion matrix,
    and classification report.
    """

    # Predictions
    y_pred = model.predict(X_test)

    # Flatten predictions if needed
    if hasattr(y_pred, "flatten"):
        y_pred = y_pred.flatten()

    # Prediction probabilities
    y_prob = model.predict_proba(X_test)

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="macro"
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro"
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )

    balanced_acc = balanced_accuracy_score(
        y_test,
        y_pred
    )

    classes = sorted(pd.unique(y_test))

    y_test_bin = label_binarize(
        y_test,
        classes=classes
    )

    roc_auc = roc_auc_score(
        y_test_bin,
        y_prob,
        average="macro",
        multi_class="ovr"
    )

    pr_auc = average_precision_score(
        y_test_bin,
        y_prob,
        average="macro"
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    metrics = {

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1-Score": f1,

        "ROC-AUC": roc_auc,

        "PR-AUC": pr_auc,

        "Balanced Accuracy": balanced_acc,

        "MCC": mcc

    }

    return metrics, cm, report

