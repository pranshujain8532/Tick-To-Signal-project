r"""Chronological walk-forward splits with an embargo gap.

WHAT
    Cuts a time-ordered sample index into train/test folds that move forward
    through time, with a gap between every train block and the test block that
    follows it. The gap is wide enough that no raw observation is read by both
    sides.

WHY
    Adjacent samples in this dataset are near-duplicates. Two windows one step
    apart share 99 of their 100 rows, and their labels are computed from
    overlapping forward means. A random split therefore places almost-identical
    samples on both sides of the train/test line, and the model scores well by
    recognising a neighbour rather than by predicting anything. That is the
    single most common way an ML-for-finance result turns out to be worthless,
    and it is invisible in the metrics — the numbers look excellent, which is
    exactly the problem.

THE TIMELINE

    Every sample sits at an index t, and touches raw rows on both sides of it:

        feature window                    label horizon
      |<---- W rows ---->|              |<---- k rows ---->|
      t-W+1            t                t+1              t+k
                         \_____________/
                          the sample at t

    So one sample at t reads rows [t - W + 1, t + k]. Two samples cannot share
    a single raw row unless their spans overlap, which gives the embargo
    directly: the last train sample's span must end before the first test
    sample's span begins.

        fold 0   [=========== train ===========][ embargo ][== test ==]
        fold 1   [================ train =================][ embargo ][== test ==]
        fold 2   [===================== train ====================][ embargo ][== test ==]
                 |                                                                       |
                 0                                                              n_samples

    The train block grows and the test block walks forward — an *anchored*
    walk-forward. The embargo is carved out of the end of the train block, not
    the start of the test block, so the test blocks tile the timeline exactly
    and every sample after the first fold is tested exactly once.

DESIGN DECISION — embargo covers the feature window as well as the label.
    The textbook purge (López de Prado) removes training samples whose *label*
    horizon reaches into the test period, which here means an embargo of k.
    We use `W + k` instead. Rejected alternative: the minimal k. An embargo of
    k stops a training label from being computed out of test-period data, but
    it still lets the last training sample's 100-row feature window overlap
    the first test sample's — so the two inputs are near-identical and the
    model can still score by recognition. Since W dominates k here (100 versus
    20), using the minimal embargo would leave most of the leak in place while
    looking rigorous. The cost is `W + k` samples discarded per fold, which on
    a tape of tens of thousands of rows is a rounding error.

DESIGN DECISION — anchored (expanding) train window, not a fixed-width rolling one.
    Rejected alternative: a fixed-size train block sliding forward, which
    keeps every fold's training set the same size and so makes fold scores
    directly comparable. We anchor instead because the honest question for
    this project is "how well does a model trained on everything up to now
    predict what happens next", which is the anchored setup. It also uses all
    the data, and data is the binding constraint at this stage. The
    consequence — later folds train on more data and so are not strictly
    comparable to earlier ones — is real, and is why fold scores get reported
    individually rather than only as a mean.

DESIGN DECISION — splits return index ranges, not arrays.
    Rejected alternative: hand back the sliced tensors. Returning ranges keeps
    this module free of any opinion about how the data is stored, makes the
    splits testable without building a dataset, and — the actual reason — lets
    `tests/test_labels.py` check the *index sets* for overlap directly, which
    is a far stronger assertion than checking that two arrays happen to
    differ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Split:
    """One fold. All bounds are inclusive sample indices.

    `train_end` is the last sample index the model may learn from; the
    `embargo` samples immediately after it belong to neither side.
    """

    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    embargo: int

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start + 1

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start + 1

    def train_indices(self) -> np.ndarray:
        return np.arange(self.train_start, self.train_end + 1)

    def test_indices(self) -> np.ndarray:
        return np.arange(self.test_start, self.test_end + 1)


def required_embargo(feature_window: int, label_horizon: int) -> int:
    """Smallest embargo for which no raw row is read by both train and test.

    A sample at index t reads raw rows `[t - feature_window + 1, t +
    label_horizon]`. For the last train sample and the first test sample to
    share nothing, the gap between their indices must exceed the length of
    that span minus one — which works out to `feature_window + label_horizon
    - 1` samples strictly between them.

    Callers should pass the *feature* window here, not the model's receptive
    field; if Stage 4 ever grows a longer context than the window it is fed,
    that longer number is the one that belongs in this calculation.
    """
    if feature_window < 1 or label_horizon < 0:
        raise ValueError(f"nonsensical horizon: feature_window={feature_window}, label_horizon={label_horizon}")
    return feature_window + label_horizon - 1


def walk_forward_splits(
    sample_count: int,
    fold_count: int,
    embargo: int,
    minimum_train_size: int = 1,
) -> list[Split]:
    """Build `fold_count` anchored walk-forward folds over `sample_count` samples.

    The timeline is cut into `fold_count + 1` contiguous blocks. Block 0 is
    train-only; block `i + 1` is the test set for fold `i`, and fold `i`
    trains on everything from index 0 up to `embargo` samples before its test
    block starts.

    Raises rather than silently returning fewer folds when the arithmetic does
    not fit — a split function that quietly hands back two folds when asked
    for five is how a five-fold result becomes a two-fold result in a report.
    """
    if fold_count < 1:
        raise ValueError(f"need at least one fold, got {fold_count}")
    if embargo < 0:
        raise ValueError(f"embargo must not be negative, got {embargo}")

    block_size = sample_count // (fold_count + 1)
    if block_size < 1:
        raise ValueError(f"{sample_count} samples cannot be cut into {fold_count + 1} non-empty blocks")

    splits = []
    for fold in range(fold_count):
        test_start = block_size * (fold + 1)
        # The final fold absorbs the remainder so no sample is silently dropped.
        test_end = sample_count - 1 if fold == fold_count - 1 else block_size * (fold + 2) - 1
        train_end = test_start - embargo - 1
        if train_end - 0 + 1 < minimum_train_size:
            raise ValueError(
                f"fold {fold}: an embargo of {embargo} leaves {max(train_end + 1, 0)} training "
                f"samples, below the required {minimum_train_size}. Use fewer folds, more data, "
                f"or a shorter feature window."
            )
        splits.append(
            Split(
                fold=fold,
                train_start=0,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                embargo=embargo,
            )
        )
    return splits


def rows_touched_by_samples(
    sample_indices: np.ndarray,
    feature_window: int,
    label_horizon: int,
) -> set[int]:
    """Every raw row index that the given samples read, features and labels alike.

    This is the function the leakage test is built on. Comparing the *rows*
    two sample sets touch is a much stronger check than comparing the sample
    indices themselves, because it catches the case where two samples are
    disjoint as indices but overlap as information — which is the entire
    failure mode this module exists to prevent.
    """
    touched: set[int] = set()
    for index in sample_indices:
        first = int(index) - feature_window + 1
        last = int(index) + label_horizon
        touched.update(range(first, last + 1))
    return touched


def describe(splits: list[Split], sample_count: int) -> str:
    """Render the folds as text, for the notebook and for logs."""
    lines = [f"{len(splits)} folds over {sample_count:,} samples, embargo {splits[0].embargo} samples"]
    for split in splits:
        lines.append(
            f"  fold {split.fold}: train [{split.train_start:>7,}, {split.train_end:>7,}] "
            f"({split.train_size:>7,})  embargo {split.embargo:>4}  "
            f"test [{split.test_start:>7,}, {split.test_end:>7,}] ({split.test_size:>7,})"
        )
    return "\n".join(lines)
