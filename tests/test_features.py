"""Tests for `ml.features`.

The important one is `test_normalisation_cannot_see_the_future`. Everything
else here checks that the tensor has the shape and content it claims; that one
checks the property the whole stage exists to guarantee, and it checks it by
mutation rather than by inspection — build the features, change the future,
rebuild, and demand the past came out bit-identical.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_engine.binfmt import snapshot_dtype
from ml.features import (
    DEFAULT_NORMALISATION_LOOKBACK,
    FEATURE_COUNT,
    FEATURES_PER_LEVEL,
    WINDOW_LENGTH,
    build_feature_matrix,
    build_input_tensor,
    build_windows,
    count_absent_levels,
    rolling_zscore,
    snapshot_mids_fixed,
)

PRICE_SCALE = 10 ** 8
QTY_SCALE = 10 ** 8
DEPTH = 10


def make_snapshots(count: int, seed: int = 7, depth: int = DEPTH) -> np.ndarray:
    """Synthetic snapshots with a wandering mid and plausible ladders."""
    rng = np.random.default_rng(seed)
    snapshots = np.zeros(count, dtype=snapshot_dtype(depth))
    mid_ticks = 6_400_000_000_000 + np.cumsum(rng.integers(-3, 4, size=count)) * 1_000_000

    level_offsets = (np.arange(depth) + 1) * 1_000_000
    for index in range(count):
        best_bid = mid_ticks[index] - 500_000
        best_ask = mid_ticks[index] + 500_000
        snapshots["bids"][index, :, 0] = best_bid - level_offsets + 1_000_000
        snapshots["asks"][index, :, 0] = best_ask + level_offsets - 1_000_000
        snapshots["bids"][index, :, 1] = rng.integers(1, 500, size=depth) * 1_000_000
        snapshots["asks"][index, :, 1] = rng.integers(1, 500, size=depth) * 1_000_000
    snapshots["local_ts_ns"] = np.arange(count, dtype=np.uint64) * 25_000_000
    return snapshots


# ----------------------------------------------------------- the mid itself


def test_mid_is_kept_as_twice_the_true_mid_to_stay_exact():
    """An odd-tick spread has a half-tick mid; rounding it would bias offsets."""
    snapshots = np.zeros(1, dtype=snapshot_dtype(DEPTH))
    snapshots["bids"][0, 0] = (100, 5)
    snapshots["asks"][0, 0] = (103, 5)

    doubled, valid = snapshot_mids_fixed(snapshots)

    assert doubled[0] == 203
    assert valid[0]


def test_one_sided_snapshot_is_marked_invalid():
    snapshots = np.zeros(2, dtype=snapshot_dtype(DEPTH))
    snapshots["bids"][0, 0] = (100, 5)   # no asks at all
    snapshots["bids"][1, 0] = (100, 5)
    snapshots["asks"][1, 0] = (101, 5)

    _doubled, valid = snapshot_mids_fixed(snapshots)

    assert valid.tolist() == [False, True]


# --------------------------------------------------------- feature contents


def test_feature_matrix_has_the_promised_shape_and_dtype():
    matrix, valid = build_feature_matrix(make_snapshots(50), PRICE_SCALE, QTY_SCALE)

    assert matrix.shape == (50, FEATURE_COUNT) == (50, 40)
    assert matrix.dtype == np.float32
    assert valid.all()


def test_layout_is_interleaved_per_level():
    """[bid_price, bid_qty, ask_price, ask_qty] per level, ten levels."""
    snapshots = np.zeros(1, dtype=snapshot_dtype(DEPTH))
    for level in range(DEPTH):
        snapshots["bids"][0, level] = (1_000_000_000 - level * 1_000_000, (level + 1) * QTY_SCALE)
        snapshots["asks"][0, level] = (1_000_000_002 + level * 1_000_000, (level + 11) * QTY_SCALE)

    matrix, _valid = build_feature_matrix(snapshots, PRICE_SCALE, QTY_SCALE)

    bid_offsets = matrix[0, 0::FEATURES_PER_LEVEL]
    bid_sizes = matrix[0, 1::FEATURES_PER_LEVEL]
    ask_offsets = matrix[0, 2::FEATURES_PER_LEVEL]
    ask_sizes = matrix[0, 3::FEATURES_PER_LEVEL]

    assert np.all(bid_offsets < 0), "bids sit below the mid"
    assert np.all(ask_offsets > 0), "asks sit above the mid"
    assert np.all(np.diff(bid_offsets) < 0), "bid offsets walk further below the mid"
    assert np.all(np.diff(ask_offsets) > 0), "ask offsets walk further above the mid"
    assert np.allclose(bid_sizes, np.log1p(np.arange(1, DEPTH + 1)), atol=1e-6)
    assert np.allclose(ask_sizes, np.log1p(np.arange(11, DEPTH + 11)), atol=1e-6)


def test_price_offsets_are_relative_so_a_level_shift_changes_nothing():
    """The whole point of mid-relative prices: shape survives a price move."""
    base = make_snapshots(20, seed=3)
    shifted = base.copy()
    shifted["bids"][:, :, 0] += 500 * PRICE_SCALE
    shifted["asks"][:, :, 0] += 500 * PRICE_SCALE

    base_matrix, _ = build_feature_matrix(base, PRICE_SCALE, QTY_SCALE)
    shifted_matrix, _ = build_feature_matrix(shifted, PRICE_SCALE, QTY_SCALE)

    assert np.allclose(base_matrix, shifted_matrix, atol=1e-5)


def test_absent_levels_become_zeros_and_are_counted():
    snapshots = make_snapshots(4, seed=11)
    snapshots["bids"][2, 7:, 0] = 0
    snapshots["bids"][2, 7:, 1] = 0

    matrix, _valid = build_feature_matrix(snapshots, PRICE_SCALE, QTY_SCALE)

    assert np.all(matrix[2, 7 * FEATURES_PER_LEVEL :: FEATURES_PER_LEVEL] == 0.0)
    assert count_absent_levels(snapshots) == 3


# ------------------------------------------------- the causality guarantee


def test_normalisation_cannot_see_the_future():
    """THE feature-side leakage test.

    Normalise a matrix, then rewrite every row after a cut point with garbage
    and normalise again. Every row at or before the cut must come out
    bit-identical. A dataset-wide scaler fails this immediately, which is the
    point — it is the check that would have caught the single most common
    look-ahead bug in ML-for-finance code.
    """
    lookback = 50
    matrix, _valid = build_feature_matrix(make_snapshots(400, seed=1), PRICE_SCALE, QTY_SCALE)
    cut = 250

    baseline, baseline_valid = rolling_zscore(matrix, lookback)

    tampered_input = matrix.copy()
    rng = np.random.default_rng(99)
    tampered_input[cut + 1 :] = rng.normal(500.0, 250.0, size=tampered_input[cut + 1 :].shape)
    tampered, tampered_valid = rolling_zscore(tampered_input, lookback)

    assert np.array_equal(baseline[: cut + 1], tampered[: cut + 1])
    assert np.array_equal(baseline_valid[: cut + 1], tampered_valid[: cut + 1])
    # And the tampering really did change something, or the test proves nothing.
    assert not np.array_equal(baseline[cut + 1 :], tampered[cut + 1 :])


def test_rolling_zscore_matches_a_naive_windowed_loop():
    """Prefix sums are fast; a loop is obviously correct. They must agree."""
    lookback = 30
    matrix, _valid = build_feature_matrix(make_snapshots(200, seed=5), PRICE_SCALE, QTY_SCALE)

    fast, valid = rolling_zscore(matrix, lookback)

    naive = np.zeros_like(fast)
    for row in range(lookback - 1, len(matrix)):
        window = matrix[row - lookback + 1 : row + 1].astype(np.float64)
        mean = window.mean(axis=0)
        deviation = window.std(axis=0)
        safe = np.where(deviation < 1e-12, 1.0, deviation)
        naive[row] = np.where(deviation < 1e-12, 0.0, (matrix[row] - mean) / safe)

    assert np.allclose(fast[valid], naive[valid], atol=1e-3, rtol=1e-3)


def test_warmup_rows_are_invalid_and_left_alone():
    """Back-filling short-window statistics would be a silent distribution shift."""
    lookback = 40
    matrix, _valid = build_feature_matrix(make_snapshots(100, seed=2), PRICE_SCALE, QTY_SCALE)

    normalised, valid = rolling_zscore(matrix, lookback)

    assert not valid[: lookback - 1].any()
    assert valid[lookback - 1 :].all()
    assert np.all(normalised[: lookback - 1] == 0.0)


def test_too_short_a_matrix_yields_nothing_valid():
    matrix, _valid = build_feature_matrix(make_snapshots(10, seed=4), PRICE_SCALE, QTY_SCALE)

    normalised, valid = rolling_zscore(matrix, lookback=50)

    assert not valid.any()
    assert np.all(normalised == 0.0)


def test_a_constant_feature_normalises_to_zero_not_nan():
    """Zero variance must not produce inf or NaN that poisons the tensor."""
    matrix = np.ones((80, FEATURE_COUNT), dtype=np.float32)

    normalised, valid = rolling_zscore(matrix, lookback=20)

    assert valid[19:].all()
    assert np.all(np.isfinite(normalised))
    assert np.all(normalised == 0.0)


def test_lookback_below_two_is_refused():
    with pytest.raises(ValueError, match="at least 2"):
        rolling_zscore(np.zeros((10, FEATURE_COUNT), dtype=np.float32), lookback=1)


# ---------------------------------------------------------- window assembly


def test_build_input_tensor_returns_one_window_of_the_right_shape():
    lookback = 60
    needed = WINDOW_LENGTH + lookback - 1
    snapshots = make_snapshots(needed, seed=6)

    tensor = build_input_tensor(snapshots, PRICE_SCALE, QTY_SCALE, lookback=lookback)

    assert tensor.shape == (WINDOW_LENGTH, FEATURE_COUNT)
    assert tensor.dtype == np.float32
    assert np.all(np.isfinite(tensor))


def test_build_input_tensor_refuses_insufficient_history():
    """Silently normalising against a shorter window would change the scale."""
    snapshots = make_snapshots(WINDOW_LENGTH, seed=8)

    with pytest.raises(ValueError, match="need .* snapshots"):
        build_input_tensor(snapshots, PRICE_SCALE, QTY_SCALE, lookback=DEFAULT_NORMALISATION_LOOKBACK)


def test_build_input_tensor_ends_at_the_last_snapshot():
    """The label attaches to the window's final row, so that row must be last."""
    lookback = 30
    snapshots = make_snapshots(WINDOW_LENGTH + lookback + 20, seed=9)

    full_matrix, _ = build_feature_matrix(snapshots, PRICE_SCALE, QTY_SCALE)
    full_normalised, _ = rolling_zscore(full_matrix, lookback)
    tensor = build_input_tensor(snapshots, PRICE_SCALE, QTY_SCALE, lookback=lookback)

    assert np.allclose(tensor[-1], full_normalised[-1], atol=1e-5)


def test_build_windows_reports_the_end_index_of_each_window():
    lookback = 20
    matrix, _ = build_feature_matrix(make_snapshots(150, seed=10), PRICE_SCALE, QTY_SCALE)
    normalised, valid = rolling_zscore(matrix, lookback)

    windows, end_indices = build_windows(normalised, valid, window_length=WINDOW_LENGTH)

    assert windows.shape[1:] == (WINDOW_LENGTH, FEATURE_COUNT)
    assert len(windows) == len(end_indices)
    # The first usable window ends once both the lookback and the window fit.
    assert int(end_indices[0]) == lookback - 1 + WINDOW_LENGTH - 1
    assert int(end_indices[-1]) == 149
    assert np.array_equal(windows[0][-1], normalised[end_indices[0]])


def test_build_windows_drops_windows_containing_warmup_rows():
    lookback = 40
    matrix, _ = build_feature_matrix(make_snapshots(200, seed=12), PRICE_SCALE, QTY_SCALE)
    normalised, valid = rolling_zscore(matrix, lookback)

    _windows, end_indices = build_windows(normalised, valid, window_length=WINDOW_LENGTH)

    assert int(end_indices.min()) >= lookback - 1 + WINDOW_LENGTH - 1


def test_build_windows_stays_a_view_when_validity_is_contiguous():
    """Overlapping windows must not be materialised: 2 MB of rows becomes 200 MB.

    Validity always forms a contiguous run in practice, because invalid rows
    come from the normalisation warmup at the front of the file. Selecting
    with a boolean mask would copy; slicing keeps the view.
    """
    lookback = 20
    matrix, _ = build_feature_matrix(make_snapshots(400, seed=13), PRICE_SCALE, QTY_SCALE)
    normalised, valid = rolling_zscore(matrix, lookback)

    windows, _end_indices = build_windows(normalised, valid, window_length=WINDOW_LENGTH)

    assert windows.base is not None, "expected a view onto the normalised matrix"
    assert windows.nbytes > normalised.nbytes, "the view describes far more bytes than it owns"


def test_build_windows_still_correct_when_validity_is_not_contiguous():
    """The fallback path: a hole in the middle forces a copy, and must be right."""
    lookback = 10
    matrix, _ = build_feature_matrix(make_snapshots(400, seed=14), PRICE_SCALE, QTY_SCALE)
    normalised, valid = rolling_zscore(matrix, lookback)
    valid[200:210] = False   # a synthetic hole no real tape would produce

    windows, end_indices = build_windows(normalised, valid, window_length=WINDOW_LENGTH)

    assert len(windows) == len(end_indices)
    for position, end in enumerate(end_indices):
        assert np.array_equal(windows[position][-1], normalised[end])
        assert valid[int(end) - WINDOW_LENGTH + 1 : int(end) + 1].all()


def test_build_windows_on_too_little_data_returns_empty_not_an_error():
    normalised = np.zeros((10, FEATURE_COUNT), dtype=np.float32)
    valid = np.ones(10, dtype=bool)

    windows, end_indices = build_windows(normalised, valid, window_length=WINDOW_LENGTH)

    assert windows.shape == (0, WINDOW_LENGTH, FEATURE_COUNT)
    assert len(end_indices) == 0
