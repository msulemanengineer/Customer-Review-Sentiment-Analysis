"""
Evaluation: how do we know the model is any good?

WHY NOT JUST ACCURACY?
Accuracy = (correct predictions) / (all predictions). It is easy to explain,
but it hides the *type* of mistake and it lies on imbalanced data.

Imagine 1,000 reviews where 950 are positive. A lazy model that always answers
"Positive" scores 95% accuracy while being completely useless -- it never
catches a single unhappy customer. That is why we also report precision,
recall and F1 per class, plus the confusion matrix which shows exactly which
mistakes were made.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Return all the headline numbers as a plain dict (easy to save as JSON).

    We report the "positive-class" metrics (how well we detect positive
    reviews) AND the macro average (the unweighted mean of both classes, so a
    small class counts as much as a large one).
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # precision = of the reviews we CALLED positive, how many really were?
        #             high precision = few false alarms
        "precision_positive": float(precision_score(y_true, y_pred, pos_label=1,
                                                    zero_division=0)),
        # recall = of the reviews that really WERE positive, how many did we find?
        #          high recall = few missed cases
        "recall_positive": float(recall_score(y_true, y_pred, pos_label=1,
                                              zero_division=0)),
        # F1 = harmonic mean of precision and recall; punishes a bad score in
        #      either one, so you cannot cheat by maximising just one of them.
        "f1_positive": float(f1_score(y_true, y_pred, pos_label=1,
                                      zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro",
                                                 zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro",
                                           zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
    }

    if y_proba is not None:
        # ROC-AUC judges the ranking quality across every possible threshold,
        # so it does not depend on the 0.5 cut-off we happen to use.
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics["confusion_matrix"] = {
        "true_negative": int(tn),   # negative review, correctly called negative
        "false_positive": int(fp),  # negative review, wrongly called positive
        "false_negative": int(fn),  # positive review, wrongly called negative
        "true_positive": int(tp),   # positive review, correctly called positive
    }
    return metrics


def print_report(name: str, y_true, y_pred) -> None:
    """Human-readable per-class report in the terminal."""
    print(f"\n--- {name} ---")
    print(classification_report(
        y_true, y_pred,
        labels=[0, 1],
        target_names=[config.LABEL_NAMES[0], config.LABEL_NAMES[1]],
        digits=4,
        zero_division=0,
    ))
    print("Confusion matrix (rows = actual, columns = predicted):")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"                 pred Neg   pred Pos")
    print(f"  actual Neg   {cm[0, 0]:>9} {cm[0, 1]:>10}")
    print(f"  actual Pos   {cm[1, 0]:>9} {cm[1, 1]:>10}")


def plot_confusion_matrix(y_true, y_pred, title: str, save_path=None,
                          close: bool = True):
    """Save the confusion matrix as a PNG so it can go in the README.

    We never call plt.show() here and by default we close the figure, so this
    is safe to call from a script with no screen attached. We also deliberately do
    NOT force the "Agg" backend at import time: matplotlib already falls back
    to Agg when there is no display, and forcing it would break inline plots
    for anyone importing this module inside a Jupyter notebook.

    close=True closes the figure when we are done, which is what a script
    wants. Pass close=False from a notebook, otherwise the figure is closed
    before Jupyter gets the chance to draw it.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    labels = [config.LABEL_NAMES[0], config.LABEL_NAMES[1]]

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    image = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1], labels=[f"Predicted\n{name}" for name in labels])
    ax.set_yticks([0, 1], labels=[f"Actual\n{name}" for name in labels])
    ax.set_title(title)

    # Write the count inside each cell, switching text colour for readability.
    threshold = cm.max() / 2
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{cm[row, col]}",
                    ha="center", va="center", fontsize=16,
                    color="white" if cm[row, col] > threshold else "black")

    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved confusion matrix -> {save_path}")
    if close:
        plt.close(fig)
    return cm


def save_metrics(payload: dict, path=None) -> None:
    """Write the measured numbers to reports/metrics.json."""
    path = path or config.METRICS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved metrics -> {path}")


def evaluate_saved_model() -> dict:
    """Re-score the saved model on the same held-out test set.

    This works because the split uses a fixed random_state, so rebuilding it
    gives the exact same 20% of rows every time.
    """
    from sklearn.model_selection import train_test_split

    from src import preprocessing
    from src.predict import load_artifacts

    model, vectorizer = load_artifacts()
    df = preprocessing.load_dataset()

    _, x_test, _, y_test = train_test_split(
        df["clean_review"],
        df[config.LABEL_COLUMN],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df[config.LABEL_COLUMN],
    )

    features = vectorizer.transform(x_test)
    y_pred = model.predict(features)
    y_proba = model.predict_proba(features)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)
    print_report("Saved model on held-out test set", y_test, y_pred)
    print(f"\nAccuracy: {metrics['accuracy']:.4f}  "
          f"F1 (macro): {metrics['f1_macro']:.4f}")
    return metrics


if __name__ == "__main__":
    evaluate_saved_model()
