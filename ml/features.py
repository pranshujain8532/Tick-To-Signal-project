"""Feature construction from order-book snapshots.

WHAT
    Turns a sequence of book snapshots (as stored by `data_engine.binfmt` and
    read back by `data_engine.replay`) into the float32 tensor the model
    consumes: `[T, 40]` per window — 10 levels a side, each contributing a
    mid-relative price and a log-compressed size — normalised by statistics
    computed only from the past.

WHY
    Raw price levels are non-stationary and instrument-specific. A model
    trained on them learns the price of BTC last Tuesday, and then fails the
    moment the level moves. Features exist to strip out that level dependence
    and leave the *shape* of the book, which is the part that generalises.

DESIGN DECISION — prices as offsets from the current mid, not raw prices.
    Rejected alternative: feed the raw level prices. Over a single day BTCUSDT
    moves thousands of dollars, so a raw price is a number that never repeats
    and never will again; the network would spend its capacity memorising a
    range instead of learning book shape. An offset from the current mid is
    directly comparable across time and across price regimes: "there is size
    resting two cents above the mid" means the same thing today and next
    month. The cost is that the absolute level is destroyed, which is
    deliberate — we are not trying to predict the price, only its direction.

DESIGN DECISION — `log1p` on quantities, not raw sizes.
    Resting size is heavy-tailed: most levels hold a fraction of a coin and a
    few hold hundreds. Raw sizes let a single whale order dominate the input
    scale and swamp every other feature. `log1p` compresses that tail while
    mapping zero to zero, so an empty level stays exactly zero rather than
    becoming a large negative number the way plain `log` would.

DESIGN DECISION — features interleaved per level, not grouped by field.
    The layout is `[bid_price_1, bid_qty_1, ask_price_1, ask_qty_1,
    bid_price_2, ...]`. Rejected alternative: all 10 bid prices, then all 10
    bid sizes, and so on. The model in Stage 4 is a convolutional front end
    whose first filters slide across this axis, so putting the four numbers
    that describe *the same level* next to each other means a small kernel
    can see a whole level at once. Grouping by field would force the first
    layer to learn a permutation before it could learn anything about
    microstructure.

DESIGN DECISION — causal rolling z-score, never a dataset-wide scaler.
    Rejected alternative: `sklearn.StandardScaler().fit(all_data)`, or any
    mean and standard deviation computed over the whole file. It is the
    default in most tutorials and it is a look-ahead bug wearing a helpful
    face: the scaler has seen the test set, so a test-set sample is normalised
    using statistics that partly describe itself. The reported accuracy is
    then real but unobtainable. Every statistic here is computed from a
    trailing window ending at the row it normalises. This costs accuracy on
    paper. The paper number was never real.

INFORMATION HORIZON
    Strictly backward-looking, and the bound is exact:

        the feature row at index t is a function of snapshots
        [t - normalisation_lookback + 1, t] and of nothing else.

    Rows with fewer than `normalisation_lookback` predecessors have no valid
    normalisation and are reported invalid rather than being back-filled with
    whatever statistics happen to be available. `tests/test_features.py`
    asserts the horizon directly by mutating the future and checking that
    every earlier row is bit-identical.
"""

from __future__ import annotations

import numpy as np

from data_engine.binfmt import DEFAULT_DEPTH_LEVELS

# 10 levels a side, each contributing (price offset, size) for bid and ask.
DEPTH_LEVELS = DEFAULT_DEPTH_LEVELS
FEATURES_PER_LEVEL = 4
FEATURE_COUNT = DEPTH_LEVELS * FEATURES_PER_LEVEL

# The model sees this many consecutive snapshots as one sample.
WINDOW_LENGTH = 100

# Trailing rows used to normalise one row. 500 snapshots is a few minutes of
# event time at the capture's snapshot cadence — long enough for the mean and
# variance to be stable, short enough to track a regime change rather than
# averaging over one.
DEFAULT_NORMALISATION_LOOKBACK = 500

# Guard for the z-score denominator. A feature that never moves inside the
# lookback (a level pinned at the same size all window) has zero variance;
# dividing by it would produce inf or NaN that then poisons every downstream
# tensor. Returning zero for such a feature is the honest answer: it carried
# no information over that window.
_MINIMUM_STANDARD_DEVIATION = 1e-12


def snapshot_mids_fixed(snapshots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mid price per snapshot in fixed-point units, plus a validity mask.

    Returns `(mid_times_two, valid)`. The mid is kept as *twice* the true mid
    so it stays an exact integer: the mid of an odd-tick spread is a half
    tick, and rounding it here would put a systematic bias into every price
    offset computed from it. Callers divide by two only when they leave
    integer land.
    """
    best_bid = snapshots["bids"][:, 0, 0].astype(np.int64)
    best_ask = snapshots["asks"][:, 0, 0].astype(np.int64)
    valid = (best_bid > 0) & (best_ask > 0)
    return best_bid + best_ask, valid


def build_feature_matrix(
    snapshots: np.ndarray,
    price_scale: int,
    qty_scale: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-snapshot raw features, before normalisation.

    Returns `(matrix, valid)` where `matrix` is `[T, 40]` float32 laid out as
    described in the module docstring, and `valid` is `[T]` bool, False for
    any snapshot with an empty side.

    Absent levels — the writer's `[0, 0]` zero-fill for a side shorter than
    `depth_levels` — become a zero offset and a zero size. That is a
    compromise: a zero offset reads as "a level sitting at the mid", which is
    not what an absent level means. It is tolerable only because it is
    vanishingly rare on a book we seed with 1000 levels a side, and
    `count_absent_levels` exists so the notebook can show that it is rare
    rather than asserting it.
    """
    mid_doubled, valid = snapshot_mids_fixed(snapshots)

    bid_prices = snapshots["bids"][:, :, 0].astype(np.float64)
    bid_sizes = snapshots["bids"][:, :, 1].astype(np.float64)
    ask_prices = snapshots["asks"][:, :, 0].astype(np.float64)
    ask_sizes = snapshots["asks"][:, :, 1].astype(np.float64)

    mid = (mid_doubled.astype(np.float64) / 2.0)[:, None]
    bid_offsets = np.where(bid_prices > 0, (bid_prices - mid) / price_scale, 0.0)
    ask_offsets = np.where(ask_prices > 0, (ask_prices - mid) / price_scale, 0.0)
    bid_volumes = np.log1p(bid_sizes / qty_scale)
    ask_volumes = np.log1p(ask_sizes / qty_scale)

    matrix = np.empty((len(snapshots), FEATURE_COUNT), dtype=np.float32)
    matrix[:, 0::FEATURES_PER_LEVEL] = bid_offsets
    matrix[:, 1::FEATURES_PER_LEVEL] = bid_volumes
    matrix[:, 2::FEATURES_PER_LEVEL] = ask_offsets
    matrix[:, 3::FEATURES_PER_LEVEL] = ask_volumes
    return matrix, valid


def count_absent_levels(snapshots: np.ndarray) -> int:
    """How many of the `T * 2 * depth` level slots were zero-filled padding."""
    absent_bids = int(np.count_nonzero(snapshots["bids"][:, :, 0] <= 0))
    absent_asks = int(np.count_nonzero(snapshots["asks"][:, :, 0] <= 0))
    return absent_bids + absent_asks


def rolling_zscore(
    matrix: np.ndarray,
    lookback: int = DEFAULT_NORMALISATION_LOOKBACK,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalise each row using only the `lookback` rows ending at that row.

    Returns `(normalised, valid)`. Row `t` is normalised by the mean and
    standard deviation of rows `[t - lookback + 1, t]` inclusive — so the
    statistics used on a row include that row and nothing after it. The first
    `lookback - 1` rows are marked invalid and left as zeros; back-filling
    them with short-window statistics would mean the first rows of every file
    are normalised differently from all the others, which is a silent
    distribution shift rather than a saving.

    Implemented with prefix sums, so the cost is O(T * F) regardless of
    lookback. Rejected alternative: `sliding_window_view`, which is clearer to
    read but materialises a `[T, lookback, F]` array — for a 40,000-row tape
    with a 500 lookback that is 3 GB, so it is not a real option.

    Accumulation is in float64 even though the output is float32: prefix sums
    of a long series lose precision in the low bits, and the variance is a
    difference of two large nearly-equal numbers, which is exactly where that
    loss shows up as a negative variance.
    """
    if lookback < 2:
        raise ValueError(f"lookback must be at least 2 to have a variance, got {lookback}")

    values = matrix.astype(np.float64)
    row_count = len(values)
    normalised = np.zeros_like(matrix, dtype=np.float32)
    valid = np.zeros(row_count, dtype=bool)
    if row_count < lookback:
        return normalised, valid

    padded_sum = _prefix_sums(values)
    padded_square_sum = _prefix_sums(values * values)

    window_sum = padded_sum[lookback:] - padded_sum[:-lookback]
    window_square_sum = padded_square_sum[lookback:] - padded_square_sum[:-lookback]
    mean = window_sum / lookback
    # A difference of two large near-equal sums can land just below zero from
    # rounding alone; clipping keeps the sqrt real without hiding a real bug,
    # because a genuinely negative variance would be large, not -1e-18.
    variance = np.maximum(window_square_sum / lookback - mean * mean, 0.0)
    deviation = np.sqrt(variance)

    usable = np.where(deviation < _MINIMUM_STANDARD_DEVIATION, 1.0, deviation)
    centred = (values[lookback - 1 :] - mean) / usable
    normalised[lookback - 1 :] = np.where(deviation < _MINIMUM_STANDARD_DEVIATION, 0.0, centred)
    valid[lookback - 1 :] = True
    return normalised, valid


def _prefix_sums(values: np.ndarray) -> np.ndarray:
    """Cumulative sums with a leading zero row, so windows are one subtraction."""
    return np.concatenate([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)], axis=0)


def build_input_tensor(
    snapshots: np.ndarray,
    price_scale: int,
    qty_scale: int,
    lookback: int = DEFAULT_NORMALISATION_LOOKBACK,
    window_length: int = WINDOW_LENGTH,
) -> np.ndarray:
    """One model input: `[window_length, 40]` float32 ending at the last snapshot.

    `snapshots` must supply enough history to normalise the whole window —
    `window_length + lookback - 1` rows — because the first row of the window
    needs its own trailing `lookback` rows behind it. Passing exactly
    `window_length` rows is refused rather than silently normalised against a
    shorter, differently-distributed window.

    INFORMATION HORIZON: the returned tensor is a function of
    `snapshots[-(window_length + lookback - 1):]` and nothing later. The last
    row corresponds to the last snapshot given, so a label attached to this
    tensor must look forward from that snapshot, never from inside it.
    """
    required = window_length + lookback - 1
    if len(snapshots) < required:
        raise ValueError(
            f"need {required} snapshots to build a {window_length}-row window with a "
            f"{lookback}-row normalisation lookback, got {len(snapshots)}"
        )
    matrix, _valid = build_feature_matrix(snapshots[-required:], price_scale, qty_scale)
    normalised, valid = rolling_zscore(matrix, lookback)
    window = normalised[-window_length:]
    if not valid[-window_length:].all():
        raise ValueError("window contains rows without a full normalisation lookback")
    return window


def build_windows(
    normalised: np.ndarray,
    row_valid: np.ndarray,
    window_length: int = WINDOW_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice a normalised matrix into every complete window, plus their end indices.

    Returns `(windows, end_indices)` where `windows` is `[N, window_length, 40]`
    and `end_indices[i]` is the row index of the *last* row of window `i` —
    the timestamp a label for that window must be attached to.

    The windows overlap by `window_length - 1` rows, which is the whole reason
    Stage 3 is paranoid about splitting: two adjacent windows are 99% the same
    numbers, so a random train/test split puts near-duplicates on both sides.
    See `ml/splits.py`.

    Built with `sliding_window_view`, so the result stays a *view* wherever
    possible: on a full tape the materialised form is
    `13,510 x 100 x 40 x 4 bytes` — over 200 MB for what is really 2 MB of
    underlying data, because overlapping windows repeat every row 100 times.
    Selecting valid windows with a boolean mask would force exactly that copy,
    so when the valid windows form a contiguous run — which they always do in
    practice, since invalidity comes from the normalisation warmup at the
    front — we slice instead, and the view survives. The general case falls
    back to fancy indexing and pays for the copy, because correctness first.
    """
    if len(normalised) < window_length:
        return (
            np.empty((0, window_length, normalised.shape[1]), dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )
    strided = np.lib.stride_tricks.sliding_window_view(normalised, window_length, axis=0)
    # sliding_window_view puts the window axis last; the model wants [N, T, F].
    windows = np.transpose(strided, (0, 2, 1))
    end_indices = np.arange(window_length - 1, len(normalised), dtype=np.int64)

    # A window is usable only if every row in it normalised cleanly.
    window_valid = np.lib.stride_tricks.sliding_window_view(row_valid, window_length).all(axis=1)
    selection = _contiguous_selection(window_valid)
    return windows[selection], end_indices[selection]


def _contiguous_selection(mask: np.ndarray) -> slice | np.ndarray:
    """A `slice` when the True values form one run, otherwise the index array.

    Slicing preserves a numpy view; boolean or fancy indexing always copies.
    For overlapping windows that difference is two orders of magnitude in
    memory, which is the difference between a notebook that runs on a laptop
    and one that does not.
    """
    positions = np.flatnonzero(mask)
    if len(positions) == 0:
        return positions
    if positions[-1] - positions[0] + 1 == len(positions):
        return slice(int(positions[0]), int(positions[-1]) + 1)
    return positions
