"""Forward-looking labels for supervised training.

WHAT
    The only module in this project permitted to read the future. Produces
    three-class direction targets (down / flat / up) from a smoothed mid-price
    move over a stated horizon, with a dead zone so that "flat" means "too
    small to be worth trading" rather than "exactly zero".

WHY
    Isolating every forward-looking line in one small file makes look-ahead
    auditable. If anything in `ml/features.py` ever needed to import from
    here, that would be immediately visible in a diff and immediately wrong.
    Everything else in the pipeline can then be reviewed under one rule: no
    future, ever.

DESIGN DECISION — smoothed means on both sides, not raw mid at t and t+k.
    This is the DeepLOB formulation. The label compares the average of the
    previous k mids against the average of the next k mids, rather than
    comparing two single observations. Rejected alternative: `sign(mid[t+k] -
    mid[t])`. A single mid is one draw from a very noisy process — at this
    sampling rate the mid moves by one tick constantly and most of that
    movement is quote flicker, not information. Comparing two point estimates
    labels the flicker; comparing two means averages it out and labels the
    drift. The cost is that the label is smoother than the thing you could
    actually trade, which is a real objection and is why Stage 5 marks
    positions against executable prices rather than against this.

DESIGN DECISION — a dead zone (alpha), not a two-class up/down split.
    Rejected alternative: binary direction, dropping the flat class. Binary
    labels force the model to call a direction on moves of one tick that are
    smaller than the spread — moves that cannot be captured after costs. The
    model then spends its capacity on the least profitable part of the
    distribution and reports impressive accuracy for it. The dead zone makes
    the model answer the question that pays: is the move big enough to matter?

DESIGN DECISION — label from mid, evaluate against executable prices.
    Mid is smooth and symmetric, which makes it a stable training target; it
    is also not a price anyone can trade at. So mid is used here and Stage 5's
    cost-aware evaluation marks against the touch. Rejected alternative:
    labelling directly off the executable side, which entangles the target
    with spread dynamics and makes the learning problem noisier than the
    thing we actually want to learn.

INFORMATION HORIZON — EXPLICIT, AND THE POINT OF THIS FILE

    The label at index t is a function of mids over

        [t - k + 1, t]      (the backward mean, m_minus)
        [t + 1, t + k]      (the forward mean, m_plus)

    so it reads the future by exactly k steps and no more. Consequences that
    every caller must respect:

      * The first k-1 and the last k rows of every file have no valid label
        and are DROPPED. They are never padded, never clipped to a shorter
        window, and never filled with `flat`. A padded label is a fabricated
        observation, and near a file boundary it is fabricated from the
        boundary itself.
      * Train and test blocks must be separated by an embargo of at least k
        rows so that no training label was computed from data lying inside
        the test block. `ml/splits.py` enforces a stronger bound that also
        covers the feature window.

    `tests/test_labels.py` asserts the horizon by mutation: changing the mid
    at t + k must be able to change the label at t, and changing the mid at
    t + k + 1 must never change it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Class encoding. Ordered so that the integer is the sign of the move, shifted
# by one — `label - 1` gives -1 / 0 / +1, which keeps any later sign-based
# arithmetic (PnL, information coefficient) free of a lookup table.
LABEL_DOWN = 0
LABEL_FLAT = 1
LABEL_UP = 2
LABEL_NAMES = ("down", "flat", "up")

# Chosen from the class-balance grid measured on our own captured BTCUSDT tape
# (14,109 snapshots, 483 s, ~29 snapshots/s). The grid is reproduced in
# notebooks/03_dataset_and_labels.ipynb, "Choosing k and alpha", and in
# docs/INTERVIEW_NOTES_stage3.md section 3. This pair gives
# down 0.345 / flat 0.334 / up 0.321 — an imbalance ratio of 1.08.
#
# WHY k IS THIS LARGE. The mid is unchanged between adjacent snapshots 99.1%
# of the time at this sampling density, so a short horizon has almost nothing
# to predict: at k=25 the smoothed return is *exactly zero* for 61% of rows
# and no alpha can rescue the balance (best imbalance 4.34). k=100 is the
# smallest horizon at which the three classes can be balanced at all. It is
# roughly 3.4 seconds of wall clock here, but it is defined in *event time* —
# 100 snapshots is 1,000 tape events — so it stretches and shrinks with market
# activity, which is the behaviour we want from a microstructure horizon.
#
# WHY THIS alpha. It is the value that splits the non-zero returns to put a
# third of the mass in the dead zone. In price terms it is a 0.585 USDT move
# at a 64,973 mid — about 117x the half-spread — so the up/down classes are
# moves comfortably larger than the cost of crossing, not spread noise.
#
# These are NOT the published FI-2010 values: those are calibrated to a
# different instrument at a different sampling rate, and reusing them here
# produced an almost entirely `flat` dataset.
#
# TODO(recalibrate): alpha is fitted to one session's volatility, so a calmer
# or wilder day shifts the balance. Re-derive on the full training corpus once
# Stage 4 has more than this pilot capture.
DEFAULT_SMOOTHING_K = 100
DEFAULT_ALPHA = 9e-6


@dataclass(frozen=True)
class LabelSet:
    """Labels plus the mask saying which of them are real.

    Kept together in one object because they are only ever correct together:
    handing back a bare label array invites a caller to index it without the
    mask, and the rows the mask excludes hold zeros that look exactly like a
    legitimate `down` label.
    """

    labels: np.ndarray  # [T] int8, meaningless where `valid` is False
    returns: np.ndarray  # [T] float64, the raw smoothed return before thresholding
    valid: np.ndarray  # [T] bool
    smoothing_k: int
    alpha: float

    def valid_labels(self) -> np.ndarray:
        return self.labels[self.valid]

    def class_counts(self) -> dict[str, int]:
        counted = self.valid_labels()
        return {name: int(np.count_nonzero(counted == index)) for index, name in enumerate(LABEL_NAMES)}

    def class_balance(self) -> dict[str, float]:
        total = int(np.count_nonzero(self.valid))
        if total == 0:
            return {name: 0.0 for name in LABEL_NAMES}
        return {name: count / total for name, count in self.class_counts().items()}


def smoothed_returns(mids: np.ndarray, smoothing_k: int) -> tuple[np.ndarray, np.ndarray]:
    """The DeepLOB smoothed return `l = (m_plus - m_minus) / m_minus`.

    Returns `(returns, valid)`. `m_minus` is the mean of mids `[t-k+1, t]` and
    `m_plus` the mean of mids `[t+1, t+k]`, so `returns[t]` reads the future
    by exactly k steps. Rows without a full window on both sides are invalid
    and hold zero — a value that must never be read, which is why the mask is
    returned alongside rather than being left for the caller to reconstruct.
    """
    if smoothing_k < 1:
        raise ValueError(f"smoothing_k must be at least 1, got {smoothing_k}")

    row_count = len(mids)
    returns = np.zeros(row_count, dtype=np.float64)
    valid = np.zeros(row_count, dtype=bool)
    if row_count < 2 * smoothing_k:
        return returns, valid

    prefix = np.concatenate([[0.0], np.cumsum(mids.astype(np.float64))])
    # window_mean[i] is the mean of mids[i : i + k].
    window_mean = (prefix[smoothing_k:] - prefix[:-smoothing_k]) / smoothing_k

    first = smoothing_k - 1
    last = row_count - smoothing_k - 1
    backward = window_mean[: last - first + 1]           # mean of [t-k+1, t]
    forward = window_mean[smoothing_k : smoothing_k + (last - first + 1)]  # mean of [t+1, t+k]

    returns[first : last + 1] = (forward - backward) / backward
    valid[first : last + 1] = True
    return returns, valid


def build_labels(
    mids: np.ndarray,
    smoothing_k: int = DEFAULT_SMOOTHING_K,
    alpha: float = DEFAULT_ALPHA,
) -> LabelSet:
    """Three-class direction labels from a mid series.

    INFORMATION HORIZON: `labels[t]` depends on `mids[t - k + 1 : t + k + 1]`.
    It reads the future by exactly `smoothing_k` steps. Rows within `k` of
    either end of the array are marked invalid and dropped by every consumer;
    they are never padded.
    """
    if alpha < 0:
        raise ValueError(f"alpha is a magnitude and must not be negative, got {alpha}")

    returns, valid = smoothed_returns(mids, smoothing_k)
    labels = np.full(len(mids), LABEL_FLAT, dtype=np.int8)
    labels[returns > alpha] = LABEL_UP
    labels[returns < -alpha] = LABEL_DOWN
    # Rows outside the valid span hold a zero return, which would otherwise
    # read as a confident `flat`; blank them so a mask-ignoring bug produces
    # obvious nonsense rather than a plausible majority class.
    labels[~valid] = LABEL_FLAT
    return LabelSet(labels=labels, returns=returns, valid=valid, smoothing_k=smoothing_k, alpha=alpha)


def class_balance_grid(
    mids: np.ndarray,
    smoothing_values: tuple[int, ...],
    alpha_values: tuple[float, ...],
) -> list[dict[str, float]]:
    """Class proportions for every (k, alpha) pair, for choosing the pair.

    Returned as a list of flat rows so the notebook can print it as a table
    without any plotting or dataframe dependency. Balance is not the only
    thing that matters — a large alpha gives balance by declaring everything
    flat — so each row also carries the count of usable labels and the
    imbalance ratio, which is what actually gets optimised.
    """
    rows = []
    for smoothing_k in smoothing_values:
        for alpha in alpha_values:
            label_set = build_labels(mids, smoothing_k, alpha)
            balance = label_set.class_balance()
            proportions = [balance[name] for name in LABEL_NAMES]
            rows.append(
                {
                    "k": smoothing_k,
                    "alpha": alpha,
                    "usable": int(np.count_nonzero(label_set.valid)),
                    "down": proportions[0],
                    "flat": proportions[1],
                    "up": proportions[2],
                    # 1.0 means perfectly balanced thirds; higher is worse.
                    "imbalance": max(proportions) / min(proportions) if min(proportions) > 0 else float("inf"),
                }
            )
    return rows


def label_horizon(smoothing_k: int = DEFAULT_SMOOTHING_K) -> int:
    """How many rows into the future a label reads. The embargo needs this."""
    return smoothing_k
