"""Tests for the evaluation metrics.

We check the metrics against a tiny hand-worked example, so the numbers can be
verified with a pen and paper. This is the best way to be sure you actually
understand precision and recall.
"""

import json

import pytest

from src.evaluate import compute_metrics, plot_confusion_matrix, save_metrics

# A hand-built example, 10 predictions.
#            actual:  1  1  1  1  0  0  0  0  0  1
Y_TRUE = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]
Y_PRED = [1, 1, 1, 0, 0, 0, 0, 1, 1, 0]
#
# Worked out by hand:
#   true positive  (actual 1, predicted 1) = 3
#   false negative (actual 1, predicted 0) = 2
#   true negative  (actual 0, predicted 0) = 3
#   false positive (actual 0, predicted 1) = 2
#
#   accuracy  = (3 + 3) / 10            = 0.60
#   precision = TP / (TP + FP) = 3 / 5  = 0.60
#   recall    = TP / (TP + FN) = 3 / 5  = 0.60
#   F1        = 2 * (0.6 * 0.6) / 1.2   = 0.60


def test_confusion_matrix_counts():
    matrix = compute_metrics(Y_TRUE, Y_PRED)["confusion_matrix"]
    assert matrix["true_positive"] == 3
    assert matrix["false_negative"] == 2
    assert matrix["true_negative"] == 3
    assert matrix["false_positive"] == 2
    # Every prediction must land in exactly one cell.
    assert sum(matrix.values()) == len(Y_TRUE)


def test_accuracy_precision_recall_f1_match_hand_calculation():
    metrics = compute_metrics(Y_TRUE, Y_PRED)
    assert metrics["accuracy"] == pytest.approx(0.6)
    assert metrics["precision_positive"] == pytest.approx(0.6)
    assert metrics["recall_positive"] == pytest.approx(0.6)
    assert metrics["f1_positive"] == pytest.approx(0.6)


def test_perfect_predictions_score_one():
    metrics = compute_metrics(Y_TRUE, Y_TRUE)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert metrics["confusion_matrix"]["false_positive"] == 0
    assert metrics["confusion_matrix"]["false_negative"] == 0


def test_always_predicting_one_class_shows_why_accuracy_misleads():
    """The imbalanced-data trap, made concrete.

    9 of these 10 reviews are positive. A model that blindly answers
    "positive" every time gets 90% accuracy but has zero ability to spot an
    unhappy customer -- and recall for the negative class is 0. That gap is
    exactly why we report F1 and the confusion matrix too.
    """
    y_true = [1] * 9 + [0]
    y_pred = [1] * 10

    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == pytest.approx(0.9)      # looks great
    assert metrics["f1_macro"] < 0.5                      # actually poor
    assert metrics["confusion_matrix"]["true_negative"] == 0


def test_roc_auc_added_only_when_probabilities_given():
    assert "roc_auc" not in compute_metrics(Y_TRUE, Y_PRED)

    probabilities = [0.9, 0.8, 0.7, 0.4, 0.1, 0.2, 0.3, 0.6, 0.55, 0.45]
    with_proba = compute_metrics(Y_TRUE, Y_PRED, probabilities)
    assert 0.0 <= with_proba["roc_auc"] <= 1.0


def test_all_metrics_are_plain_json_types():
    """metrics.json must be writable -- no numpy types allowed."""
    metrics = compute_metrics(Y_TRUE, Y_PRED, [0.5] * 10)
    json.dumps(metrics)                       # raises TypeError if not clean


def test_save_metrics_writes_readable_json(tmp_path):
    path = tmp_path / "nested" / "metrics.json"
    save_metrics({"accuracy": 0.82}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"accuracy": 0.82}


def test_plot_confusion_matrix_writes_a_png(tmp_path):
    path = tmp_path / "cm.png"
    matrix = plot_confusion_matrix(Y_TRUE, Y_PRED, title="test", save_path=path)
    assert path.exists()
    assert path.stat().st_size > 0
    assert matrix.shape == (2, 2)
