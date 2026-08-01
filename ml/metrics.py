"""Classification metrics, computed in numpy so the training path stays lean.

WHAT
    Macro-F1, per-class precision/recall/F1, and a confusion matrix for the
    three-class down/flat/up problem.

WHY
    Macro-F1 is the headline metric for this project because accuracy is
    actively misleading on a deadbanded three-class target: a model that
    always predicts the majority class scores respectably on accuracy and
    zero on two of the three F1 components. Macro averaging weights each class
    equally regardless of how many samples it has, so a model cannot buy a
    good score by ignoring the rare-but-tradeable classes.

DESIGN DECISION — implemented here rather than imported from sklearn.
    sklearn is approved for the logistic-regression baseline only, and pulling
    it into the training loop would put it on the critical path of every stage
    that follows. These are twenty lines of numpy and they are worth owning:
    macro-F1's behaviour on an empty predicted class is a real decision (we
    define 0/0 as 0, the same convention sklearn uses with `zero_division=0`),
    and it is better to make that decision explicitly than to inherit it.
"""

from __future__ import annotations

import numpy as np

CLASS_NAMES = ("down", "flat", "up")


def confusion_matrix(true_labels: np.ndarray, predicted_labels: np.ndarray, class_count: int = 3) -> np.ndarray:
    """Rows are the truth, columns the prediction. `[i, j]` = truth i called j."""
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(matrix, (true_labels.astype(np.int64), predicted_labels.astype(np.int64)), 1)
    return matrix


def per_class_scores(true_labels: np.ndarray, predicted_labels: np.ndarray, class_count: int = 3) -> dict[str, np.ndarray]:
    """Precision, recall, F1 and support for each class.

    A class the model never predicts has undefined precision; a class absent
    from the truth has undefined recall. Both are reported as 0.0 rather than
    NaN, so that macro-F1 penalises a model for ignoring a class instead of
    quietly excluding it from the average.
    """
    matrix = confusion_matrix(true_labels, predicted_labels, class_count)
    true_positive = np.diag(matrix).astype(np.float64)
    predicted_total = matrix.sum(axis=0).astype(np.float64)
    actual_total = matrix.sum(axis=1).astype(np.float64)

    precision = np.divide(true_positive, predicted_total, out=np.zeros(class_count), where=predicted_total > 0)
    recall = np.divide(true_positive, actual_total, out=np.zeros(class_count), where=actual_total > 0)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros(class_count), where=denominator > 0)
    return {"precision": precision, "recall": recall, "f1": f1, "support": actual_total.astype(np.int64)}


def macro_f1(true_labels: np.ndarray, predicted_labels: np.ndarray, class_count: int = 3) -> float:
    """Unweighted mean of the per-class F1 scores. The project's headline metric."""
    return float(per_class_scores(true_labels, predicted_labels, class_count)["f1"].mean())


def accuracy(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    return float(np.mean(true_labels == predicted_labels))


def majority_class_baseline(train_labels: np.ndarray, test_labels: np.ndarray, class_count: int = 3) -> dict[str, float]:
    """Score of always predicting the most common training class.

    This is the number every result must beat before it is worth discussing.
    On a balanced three-class problem its macro-F1 is about 0.17 — low enough
    that beating it is easy, which is exactly why the queue-imbalance rule
    exists as a second, harder baseline.
    """
    majority = int(np.bincount(train_labels.astype(np.int64), minlength=class_count).argmax())
    predictions = np.full_like(test_labels, majority)
    return {
        "macro_f1": macro_f1(test_labels, predictions, class_count),
        "accuracy": accuracy(test_labels, predictions),
        "predicted_class": float(majority),
    }


def format_report(true_labels: np.ndarray, predicted_labels: np.ndarray, class_count: int = 3) -> str:
    """A small text table, for logs and notebooks."""
    scores = per_class_scores(true_labels, predicted_labels, class_count)
    lines = [f"{'class':>6} {'precision':>10} {'recall':>8} {'f1':>8} {'support':>9}"]
    for index in range(class_count):
        name = CLASS_NAMES[index] if index < len(CLASS_NAMES) else str(index)
        lines.append(
            f"{name:>6} {scores['precision'][index]:>10.3f} {scores['recall'][index]:>8.3f} "
            f"{scores['f1'][index]:>8.3f} {scores['support'][index]:>9,}"
        )
    lines.append(f"{'macro':>6} {'':>10} {'':>8} {scores['f1'].mean():>8.3f} {scores['support'].sum():>9,}")
    return "\n".join(lines)
