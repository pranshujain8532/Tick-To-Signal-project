"""Tests for `ml.labels` and `ml.splits` — including the mandatory leakage test.

Three groups:

  1. Label mechanics: does the smoothed return compute what the docstring
     says, and are boundary rows dropped rather than padded?
  2. The information horizon, asserted by *mutation*: a label must change when
     the mid at t+k changes and must never change when the mid at t+k+1 does.
     This is the only way to check a horizon claim that does not amount to
     re-reading the implementation.
  3. Leakage. Two forms, because either alone is weak:
       * A structural proof — the set of raw rows read by the training samples
         and the set read by the test samples must be disjoint, and must stop
         being disjoint if the embargo is one sample shorter. That second half
         is what makes it a proof of tightness rather than of sufficiency.
       * An empirical demonstration — a memorising 1-nearest-neighbour
         "model" scores far above chance under a random split of overlapping
         windows, and falls to chance under walk-forward with an embargo. This
         is the number that convinces a skeptic, because it shows the leak
         actually inflates a score.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.labels import (
    DEFAULT_ALPHA,
    DEFAULT_SMOOTHING_K,
    LABEL_DOWN,
    LABEL_FLAT,
    LABEL_NAMES,
    LABEL_UP,
    build_labels,
    class_balance_grid,
    label_horizon,
    smoothed_returns,
)
from ml.splits import (
    Split,
    required_embargo,
    rows_touched_by_samples,
    walk_forward_splits,
)


def random_walk_mids(count: int, seed: int = 42, start: float = 65_000.0) -> np.ndarray:
    """A mid series with the shape of a real one: small steps, no drift."""
    rng = np.random.default_rng(seed)
    return start + np.cumsum(rng.normal(0.0, 0.5, size=count))


# --------------------------------------------------------- label mechanics


def test_smoothed_return_matches_a_hand_computation():
    """k=2 on a short ramp, worked out by hand.

    mids      = [10, 12, 14, 16, 18, 20]
    at t=2:  m_minus = mean(mids[1:3]) = mean(12, 14) = 13
             m_plus  = mean(mids[3:5]) = mean(16, 18) = 17
             l       = (17 - 13) / 13
    """
    mids = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])

    returns, valid = smoothed_returns(mids, smoothing_k=2)

    assert valid[2]
    assert returns[2] == pytest.approx((17.0 - 13.0) / 13.0)


def test_a_rising_series_is_all_up_and_a_falling_one_all_down():
    rising = np.linspace(100.0, 200.0, 500)

    up_labels = build_labels(rising, smoothing_k=10, alpha=1e-6)
    down_labels = build_labels(rising[::-1].copy(), smoothing_k=10, alpha=1e-6)

    assert set(np.unique(up_labels.valid_labels()).tolist()) == {LABEL_UP}
    assert set(np.unique(down_labels.valid_labels()).tolist()) == {LABEL_DOWN}


def test_a_flat_series_is_entirely_flat():
    mids = np.full(300, 65_000.0)

    label_set = build_labels(mids, smoothing_k=10, alpha=1e-9)

    assert set(np.unique(label_set.valid_labels()).tolist()) == {LABEL_FLAT}


def test_a_wider_dead_zone_can_only_create_more_flat():
    mids = random_walk_mids(4_000)

    narrow = build_labels(mids, smoothing_k=20, alpha=1e-7).class_counts()["flat"]
    wide = build_labels(mids, smoothing_k=20, alpha=1e-4).class_counts()["flat"]

    assert wide > narrow


def test_class_balance_sums_to_one():
    balance = build_labels(random_walk_mids(3_000), 20, DEFAULT_ALPHA).class_balance()

    assert sum(balance.values()) == pytest.approx(1.0)
    assert set(balance) == set(LABEL_NAMES)


def test_negative_alpha_is_refused():
    with pytest.raises(ValueError, match="must not be negative"):
        build_labels(random_walk_mids(100), 10, alpha=-1e-6)


# ---------------------------------------------------- boundaries are dropped


def test_boundary_labels_are_dropped_never_padded():
    """The first k-1 and last k rows have no two-sided window and must not exist."""
    smoothing_k = 7
    mids = random_walk_mids(200)

    label_set = build_labels(mids, smoothing_k, DEFAULT_ALPHA)

    assert not label_set.valid[: smoothing_k - 1].any()
    assert not label_set.valid[-smoothing_k:].any()
    assert label_set.valid[smoothing_k - 1 : -smoothing_k].all()
    assert int(np.count_nonzero(label_set.valid)) == len(mids) - 2 * smoothing_k + 1


def test_a_series_shorter_than_the_window_yields_no_labels_at_all():
    label_set = build_labels(random_walk_mids(10), smoothing_k=20, alpha=DEFAULT_ALPHA)

    assert not label_set.valid.any()
    assert len(label_set.valid_labels()) == 0


def test_invalid_rows_are_not_silently_countable_as_flat():
    """A mask-ignoring caller must not read a plausible majority class."""
    label_set = build_labels(random_walk_mids(60), smoothing_k=10, alpha=DEFAULT_ALPHA)

    assert label_set.class_counts()["flat"] < int(np.count_nonzero(~label_set.valid)) + len(label_set.labels)
    assert sum(label_set.class_counts().values()) == int(np.count_nonzero(label_set.valid))


# ------------------------------------------------- the information horizon


def test_label_reads_exactly_k_steps_into_the_future():
    """Mutation test: t+k matters, t+k+1 never does.

    This is the assertion that makes the horizon claim in the docstring
    checkable. Re-reading the code proves nothing; changing one number in the
    future and watching which labels move proves it.
    """
    smoothing_k = 8
    mids = random_walk_mids(400, seed=3)
    target = 200

    baseline = build_labels(mids, smoothing_k, alpha=1e-7)

    # Push the forward mean the opposite way from whatever the baseline said,
    # so the label has to move. Bumping it the same way would leave an `up`
    # label `up` and prove nothing.
    direction = -1.0 if baseline.labels[target] == LABEL_UP else 1.0
    just_inside = mids.copy()
    just_inside[target + smoothing_k] += direction * 5_000.0
    inside = build_labels(just_inside, smoothing_k, alpha=1e-7)

    just_outside = mids.copy()
    just_outside[target + smoothing_k + 1] += direction * 5_000.0
    outside = build_labels(just_outside, smoothing_k, alpha=1e-7)

    assert inside.returns[target] != pytest.approx(baseline.returns[target]), "t+k is inside the horizon"
    assert inside.labels[target] != baseline.labels[target], "t+k is inside the horizon"
    assert outside.labels[target] == baseline.labels[target], "t+k+1 must be outside it"
    assert outside.returns[target] == pytest.approx(baseline.returns[target])


def test_label_reads_exactly_k_minus_one_steps_into_the_past():
    """The backward mean is [t-k+1, t]; t-k must not reach it."""
    smoothing_k = 8
    mids = random_walk_mids(400, seed=4)
    target = 200

    baseline = build_labels(mids, smoothing_k, alpha=1e-7)

    tampered = mids.copy()
    tampered[target - smoothing_k] += 5_000.0
    outside = build_labels(tampered, smoothing_k, alpha=1e-7)

    assert outside.returns[target] == pytest.approx(baseline.returns[target])


def test_label_horizon_reports_the_number_the_embargo_needs():
    assert label_horizon(DEFAULT_SMOOTHING_K) == DEFAULT_SMOOTHING_K


# ------------------------------------------------------------- split shapes


def test_required_embargo_covers_both_the_window_and_the_horizon():
    assert required_embargo(feature_window=100, label_horizon=20) == 119


def test_walk_forward_folds_move_forward_and_never_overlap():
    splits = walk_forward_splits(sample_count=10_000, fold_count=4, embargo=119)

    for split in splits:
        assert split.train_end < split.test_start
        assert split.train_end + split.embargo < split.test_start
    for earlier, later in zip(splits, splits[1:]):
        assert later.test_start > earlier.test_end
        assert later.train_end >= earlier.train_end, "anchored training set only grows"


def test_test_blocks_tile_the_timeline_without_gaps_or_repeats():
    splits = walk_forward_splits(sample_count=10_000, fold_count=4, embargo=119)

    tested = np.concatenate([split.test_indices() for split in splits])

    assert len(tested) == len(set(tested.tolist())), "no sample is tested twice"
    assert tested.max() == 9_999, "the final fold absorbs the remainder"


def test_impossible_geometry_raises_instead_of_returning_fewer_folds():
    """A split function that quietly returns 2 folds when asked for 5 is a trap."""
    with pytest.raises(ValueError, match="cannot be cut"):
        walk_forward_splits(sample_count=3, fold_count=5, embargo=0)

    with pytest.raises(ValueError, match="below the required"):
        walk_forward_splits(sample_count=1_000, fold_count=4, embargo=500, minimum_train_size=100)


def test_negative_embargo_is_refused():
    with pytest.raises(ValueError, match="must not be negative"):
        walk_forward_splits(sample_count=1_000, fold_count=2, embargo=-1)


# ------------------------------------------------- LEAKAGE: structural proof


def test_no_raw_row_is_read_by_both_train_and_test():
    """THE structural leakage test.

    A sample at index t reads raw rows [t - W + 1, t + k]. With the required
    embargo, the union of rows read by the training samples must not intersect
    the union read by the test samples — for every fold.
    """
    feature_window, horizon = 100, 20
    embargo = required_embargo(feature_window, horizon)
    splits = walk_forward_splits(sample_count=20_000, fold_count=4, embargo=embargo)

    for split in splits:
        train_rows = rows_touched_by_samples(split.train_indices(), feature_window, horizon)
        test_rows = rows_touched_by_samples(split.test_indices(), feature_window, horizon)
        assert train_rows.isdisjoint(test_rows), f"fold {split.fold} shares raw rows"


def test_one_sample_less_embargo_does_leak():
    """Proves the bound is TIGHT, not merely sufficient.

    A test that only shows "the embargo works" is satisfied by an absurdly
    large embargo. Showing that one sample less leaks proves the formula is
    the right one.
    """
    feature_window, horizon = 100, 20
    embargo = required_embargo(feature_window, horizon) - 1
    splits = walk_forward_splits(sample_count=20_000, fold_count=4, embargo=embargo)

    split = splits[0]
    train_rows = rows_touched_by_samples(split.train_indices(), feature_window, horizon)
    test_rows = rows_touched_by_samples(split.test_indices(), feature_window, horizon)

    assert not train_rows.isdisjoint(test_rows), "the embargo formula is one larger than it needs to be"


def test_a_zero_embargo_leaks_badly():
    feature_window, horizon = 100, 20
    splits = walk_forward_splits(sample_count=20_000, fold_count=4, embargo=0)

    split = splits[0]
    train_rows = rows_touched_by_samples(split.train_indices(), feature_window, horizon)
    test_rows = rows_touched_by_samples(split.test_indices(), feature_window, horizon)

    assert len(train_rows & test_rows) >= feature_window


# ---------------------------------------------- LEAKAGE: empirical proof


def _overlapping_windows(series: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows plus the index of each window's final row."""
    windows = np.lib.stride_tricks.sliding_window_view(series, window)
    end_indices = np.arange(window - 1, len(series))
    return windows, end_indices


def _nearest_neighbour_accuracy(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> float:
    """A deliberately memorising model: predict the nearest training sample's label.

    1-NN is chosen precisely because it cannot generalise. Any accuracy it
    achieves above chance is recognition, not prediction — which makes it the
    perfect instrument for detecting a leak.
    """
    predictions = np.empty(len(test_features), dtype=train_labels.dtype)
    for position, sample in enumerate(test_features):
        distances = np.sum((train_features - sample) ** 2, axis=1)
        predictions[position] = train_labels[int(np.argmin(distances))]
    return float(np.mean(predictions == test_labels))


def test_random_split_leaks_and_walk_forward_with_embargo_does_not():
    """THE empirical leakage demonstration.

    Overlapping windows over a random walk carry no real predictive signal, so
    an honest evaluation must score around chance. A random split scores far
    above it, because for every test window there is a train window one step
    away that is 95% the same numbers and carries almost the same label. The
    walk-forward split with an embargo removes those neighbours and the score
    collapses back to chance — which is the correct answer for a random walk.
    """
    window, smoothing_k = 100, 20
    mids = random_walk_mids(4_000, seed=17)

    label_set = build_labels(mids, smoothing_k, alpha=1e-7)
    windows, end_indices = _overlapping_windows(mids, window)
    usable = label_set.valid[end_indices]
    features = windows[usable]
    # Centre each window so 1-NN matches on shape, not on the walk's level.
    features = features - features.mean(axis=1, keepdims=True)
    labels = label_set.labels[end_indices][usable]

    rng = np.random.default_rng(5)
    shuffled = rng.permutation(len(features))
    cut = int(len(features) * 0.75)
    random_train, random_test = shuffled[:cut], shuffled[cut:][:400]
    random_accuracy = _nearest_neighbour_accuracy(
        features[random_train], labels[random_train], features[random_test], labels[random_test]
    )

    embargo = required_embargo(window, smoothing_k)
    split = walk_forward_splits(len(features), fold_count=3, embargo=embargo)[-1]
    honest_accuracy = _nearest_neighbour_accuracy(
        features[split.train_indices()],
        labels[split.train_indices()],
        features[split.test_indices()][:400],
        labels[split.test_indices()][:400],
    )

    # Thresholds are set from measured behaviour with margin, not guessed:
    # this configuration scores random ~0.96 against walk-forward ~0.61, with a
    # majority baseline of ~0.54.
    assert random_accuracy > 0.85, f"the leak should be large and obvious, got {random_accuracy:.3f}"
    assert random_accuracy > honest_accuracy + 0.25, (
        f"random split {random_accuracy:.3f} should badly beat walk-forward {honest_accuracy:.3f}"
    )
    assert honest_accuracy < 0.75, (
        f"walk-forward scored {honest_accuracy:.3f} on a RANDOM WALK, which is too high to be "
        "explained by the label's visible backward mean alone — suspect a leak in the split"
    )


def test_the_smoothed_label_is_partly_visible_inside_the_feature_window():
    """Why an honest split still scores above the majority baseline — with no leak.

    The label is `(m_plus - m_minus) / m_minus`, and `m_minus` — the mean of
    the previous k mids — lies *inside* the feature window. For a martingale,
    `E[m_plus | history] = mid[t]`, so the expected label is driven by
    `mid[t] - m_minus`, a quantity the model can see. Part of this label is
    therefore legitimately knowable with zero look-ahead, and the honest
    chance level sits above the majority rate.

    This is pinned down because it is misread in both directions: as proof of
    a leak when it is not, and as proof of predictive skill when it is only
    the label's own definition showing through. On a pure random walk — where
    there is no market signal whatsoever — this rule reaches ~0.74.
    """
    window, smoothing_k = 100, 20
    mids = random_walk_mids(4_000, seed=17)
    label_set = build_labels(mids, smoothing_k, alpha=1e-7)
    end_indices = np.arange(window - 1, len(mids))
    usable = label_set.valid[end_indices]
    labels = label_set.labels[end_indices][usable]
    returns = label_set.returns[end_indices][usable]

    windows = np.lib.stride_tricks.sliding_window_view(mids, window)[usable]
    deviation = windows[:, -1] - windows[:, -smoothing_k:].mean(axis=1)

    correlation = float(np.corrcoef(deviation, returns)[0, 1])
    predictions = np.where(deviation > 0, LABEL_UP, LABEL_DOWN)
    accuracy = float(np.mean(predictions == labels))
    majority = float(np.max(np.bincount(labels, minlength=3)) / len(labels))

    assert correlation > 0.4, (
        f"the window-visible part of the label should correlate strongly with it, got {correlation:.3f}"
    )
    assert accuracy > majority + 0.10, (
        f"a rule using only in-window data scores {accuracy:.3f} against a {majority:.3f} majority "
        "baseline, on data with no signal at all — this is the bar Stage 4 must actually beat"
    )


# ------------------------------------------------------- the balance grid


def test_class_balance_grid_covers_every_pair_and_reports_usable_counts():
    mids = random_walk_mids(3_000)

    rows = class_balance_grid(mids, smoothing_values=(10, 20), alpha_values=(1e-7, 1e-5))

    assert len(rows) == 4
    assert {(row["k"], row["alpha"]) for row in rows} == {(10, 1e-7), (10, 1e-5), (20, 1e-7), (20, 1e-5)}
    for row in rows:
        assert row["usable"] == len(mids) - 2 * row["k"] + 1
        assert row["down"] + row["flat"] + row["up"] == pytest.approx(1.0)
        assert row["imbalance"] >= 1.0


def test_the_chosen_defaults_are_actually_usable():
    """Guards against a future edit leaving the hardcoded pair nonsensical."""
    assert DEFAULT_SMOOTHING_K >= 1
    assert DEFAULT_ALPHA > 0

    label_set = build_labels(random_walk_mids(5_000), DEFAULT_SMOOTHING_K, DEFAULT_ALPHA)

    assert int(np.count_nonzero(label_set.valid)) > 0
    assert len(set(np.unique(label_set.valid_labels()).tolist())) == 3, "all three classes must appear"
