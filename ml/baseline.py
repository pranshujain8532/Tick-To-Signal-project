"""The baseline the deep model has to beat: logistic regression on one snapshot.

WHAT
    Fits multinomial logistic regression on the *final* feature row of each
    window — the current book state, 40 numbers, no history — and scores it
    with the same metrics, on the same splits, as the deep model.

WHY
    A deep model reported without a baseline is not a result. Almost all of
    the achievable score on this task comes from the instantaneous shape of
    the book, especially queue imbalance at the touch; a network that beats
    "always predict the majority class" has proven nothing, because that is a
    very low bar on a deadbanded three-class target. The interesting question
    is whether 100 timesteps and 320,000 parameters buy anything over 40
    numbers and 123.

    Framed as a challenge to the deep model, this baseline is doing its job
    whichever way it comes out: if the network wins, the margin is what the
    temporal structure is worth; if it does not, that is the finding, and it
    is a far more useful thing to know before Stages 6 and 7 spend weeks
    compressing and hand-rolling it.

DESIGN DECISION — sklearn, used here and nowhere else.
    CLAUDE.md limits dependencies and requires asking before adding one; this
    was raised and approved specifically for the baseline. It stays confined
    to this module: `ml/metrics.py` reimplements macro-F1 in numpy so the
    training path never imports sklearn, and nothing under `data_engine/`
    touches it.

DESIGN DECISION — the last snapshot only, not the flattened window.
    Rejected alternative: flatten all 100x40 = 4,000 inputs into the
    regression. That would be a 12,000-parameter model on ~20,000 training
    samples, which mostly measures how well ridge-penalised least squares can
    memorise overlapping windows — a strictly worse comparison, because the
    thing we want to isolate is *temporal structure*, and the honest way to
    isolate it is to deny the baseline any.

INFORMATION HORIZON
    The final row of a window is the book state at time t. Same horizon as the
    deep model, same labels, same embargoed splits — the comparison is only
    meaningful if nothing differs except the hypothesis class.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.metrics import accuracy, macro_f1, majority_class_baseline


@dataclass
class BaselineResult:
    """What the baseline scored, and what it was scored against."""

    macro_f1: float
    accuracy: float
    majority_macro_f1: float
    majority_accuracy: float
    predictions: np.ndarray
    coefficient_count: int


def fit_logistic_baseline(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    seed: int = 0,
    max_iterations: int = 2_000,
) -> BaselineResult:
    """Fit multinomial logistic regression and score it on the test split.

    `class_weight="balanced"` because the classes are only approximately
    balanced and the metric is macro-F1: without it the model quietly favours
    whichever class is most common in the training fold, which costs exactly
    the metric we report. The deep model gets the same treatment via a
    weighted loss, so neither is handed an advantage the other lacks.
    """
    from sklearn.linear_model import LogisticRegression  # imported here: see module docstring

    # `multi_class="multinomial"` is not passed: it was deprecated in sklearn
    # 1.5 and multinomial is now the only behaviour, so naming it only emits a
    # warning. Softmax over the three classes is what we want either way.
    model = LogisticRegression(
        max_iter=max_iterations,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(train_features, train_labels)
    predictions = model.predict(test_features)

    majority = majority_class_baseline(train_labels, test_labels)
    return BaselineResult(
        macro_f1=macro_f1(test_labels, predictions),
        accuracy=accuracy(test_labels, predictions),
        majority_macro_f1=majority["macro_f1"],
        majority_accuracy=majority["accuracy"],
        predictions=predictions,
        coefficient_count=int(model.coef_.size + model.intercept_.size),
    )


def queue_imbalance(features: np.ndarray, depth_levels: int = 10) -> np.ndarray:
    """The cheapest real microstructure signal, for a second, harder baseline.

    Sums the log-compressed sizes resting on each side of the book and returns
    `(bid - ask) / (bid + ask)`. Positive means more size resting on the bid,
    which is the classic weak predictor of an upward move.

    Assumes the project's feature layout — `[bid_price, bid_size, ask_price,
    ask_size]` per level — so sizes are columns 1 and 3 of each group of four.
    """
    bid_size = features[:, 1::4][:, :depth_levels].sum(axis=1)
    ask_size = features[:, 3::4][:, :depth_levels].sum(axis=1)
    total = bid_size + ask_size
    return np.divide(bid_size - ask_size, total, out=np.zeros_like(total), where=np.abs(total) > 1e-12)


def imbalance_rule_predictions(features: np.ndarray, dead_zone: float) -> np.ndarray:
    """Predict up/flat/down from queue imbalance alone, with a dead zone.

    A rule with no fitted parameters at all. It exists so that "the model beat
    logistic regression" cannot be the whole story: if a two-line rule gets
    most of the way there, the deep model's margin is what is actually being
    bought.
    """
    imbalance = queue_imbalance(features)
    predictions = np.full(len(features), 1, dtype=np.int64)  # flat
    predictions[imbalance > dead_zone] = 2  # up
    predictions[imbalance < -dead_zone] = 0  # down
    return predictions


def evaluate_imbalance_rule(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> dict[str, float]:
    """Pick the rule's dead zone on the TRAIN split, then score it on test.

    Choosing the threshold on the test split would be the same look-ahead sin
    the whole project is built to avoid, in miniature — a baseline tuned on
    the test set is not a baseline, it is a second model with an unfair
    advantage.
    """
    candidates = np.quantile(np.abs(queue_imbalance(train_features)), np.linspace(0.0, 0.9, 19))
    best_threshold = 0.0
    best_score = -1.0
    for threshold in candidates:
        score = macro_f1(train_labels, imbalance_rule_predictions(train_features, float(threshold)))
        if score > best_score:
            best_score, best_threshold = score, float(threshold)

    predictions = imbalance_rule_predictions(test_features, best_threshold)
    return {
        "macro_f1": macro_f1(test_labels, predictions),
        "accuracy": accuracy(test_labels, predictions),
        "dead_zone": best_threshold,
        "train_macro_f1": best_score,
    }
