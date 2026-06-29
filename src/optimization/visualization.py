import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

from sklearn.preprocessing import label_binarize


# =============================================================================
# Project Root Directory
# =============================================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

FIGURE_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "figures"
)

os.makedirs(
    FIGURE_DIR,
    exist_ok=True
)


# =============================================================================
# Confusion Matrix
# =============================================================================

def plot_confusion_matrix(
    cm,
    class_names,
    title
):
    """
    Plot and save confusion matrix.
    """

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d"
    )

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Predicted Label",
        fontsize=12
    )

    ax.set_ylabel(
        "True Label",
        fontsize=12
    )

    ax.tick_params(
        axis="both",
        labelsize=11
    )

    plt.tight_layout()

    filename = (
        title.lower()
        .replace(" ", "_")
        + ".png"
    )

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            filename
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"Figure saved to: {os.path.join(FIGURE_DIR, filename)}"
    )


# =============================================================================
# ROC Curves
# =============================================================================

from sklearn.metrics import roc_curve, auc


def plot_multiclass_roc(
    model,
    X_test,
    y_test,
    class_names,
    title
):
    """
    Plot multiclass ROC curves in a single figure.
    """

    y_score = model.predict_proba(X_test)

    classes = np.unique(y_test)

    y_test_bin = label_binarize(
        y_test,
        classes=classes
    )

    plt.figure(figsize=(8, 7))

    for i in range(len(classes)):

        fpr, tpr, _ = roc_curve(
            y_test_bin[:, i],
            y_score[:, i]
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        colors = ["blue", "orange", "green", "red"]

        plt.plot(
            fpr,
            tpr,
            color=colors[i],
            linewidth=2,
            linestyle=["-","--","-.",":"][i],
            marker=["o","s","^","d"][i],
            markersize=3,
            markevery=30,
            label=f"{class_names[i]} (AUC={roc_auc:.4f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=2,
        label="Random Guess"
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel(
        "False Positive Rate",
        fontsize=12
    )

    plt.ylabel(
        "True Positive Rate",
        fontsize=12
    )

    plt.title(
        title,
        fontsize=16,
        fontweight="bold"
    )

    plt.legend(
        loc="lower right",
        fontsize=10
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()

    filename = (
        title.lower()
        .replace(" ", "_")
        + ".png"
    )

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            filename
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"Figure saved to: {os.path.join(FIGURE_DIR, filename)}"
    )

