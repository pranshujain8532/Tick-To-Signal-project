"""Tests for `ml.eval`.

An evaluation module is the most dangerous kind of code in this project: a bug
here does not crash, it produces a plausible number that then goes in a README.
So the tests are built around cases where the right answer is known in advance
— a signal that is exactly the future, a fill that must be worse than the mid,
a fee that must exactly cancel a known edge.

The touch-price tests are the important ones. They pin down the single most
common way a student backtest lies to its author.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.eval import (
    block_shuffle,
    BASIS_POINTS,
    BINANCE_TAKER_FEE_BPS,
    breakeven_fee_bps,
    falsify,
    fee_sweep,
    fit_half_life,
    forward_return_by_rows,
    forward_return_by_time,
    ic_distribution,
    rows_for_horizon,
    signal_decay_curve,
    signal_from_probabilities,
    simulate_touch_price_trades,
    spearman_ic,
)


def make_book(rows: int, mid: float = 65_000.0, half_spread: float = 0.005):
    """A flat book: mids, bids and asks one tick apart."""
    mids = np.full(rows, mid)
    return mids, mids - half_spread, mids + half_spread


# --------------------------------------------------------------- the signal


def test_signal_is_up_minus_down_and_ignores_flat():
    probabilities = np.array([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1], [0.2, 0.6, 0.2]])

    signal = signal_from_probabilities(probabilities)

    assert signal == pytest.approx([0.6, -0.5, 0.0])


def test_signal_refuses_a_wrongly_shaped_input():
    with pytest.raises(ValueError, match=r"expected \[N, 3\]"):
        signal_from_probabilities(np.zeros((5, 2)))


# ------------------------------------------------------------------- the IC


def test_perfect_signal_scores_ic_of_one():
    returns = np.array([-3.0, -1.0, 0.5, 2.0, 7.0])

    assert spearman_ic(returns.copy(), returns) == pytest.approx(1.0)
    assert spearman_ic(-returns, returns) == pytest.approx(-1.0)


def test_ic_is_rank_based_so_a_monotone_transform_changes_nothing():
    """The reason Spearman and not Pearson: only the ordering should matter."""
    generator = np.random.default_rng(0)
    returns = generator.normal(size=400)
    signal = returns + 0.5 * generator.normal(size=400)

    linear_ic = spearman_ic(signal, returns)
    squashed_ic = spearman_ic(np.tanh(3 * signal), returns)

    assert linear_ic == pytest.approx(squashed_ic, abs=1e-12)


def test_ties_are_ranked_by_average_not_by_position():
    """Forward returns are often exactly zero; ties must not invent an order."""
    signal = np.array([1.0, 2.0, 3.0, 4.0])
    returns = np.array([0.0, 0.0, 0.0, 1.0])

    # With average ranks the three zeros are indistinguishable, so the only
    # information is that the largest signal had the largest return.
    assert spearman_ic(signal, returns) == pytest.approx(0.7745966, abs=1e-6)


def test_ic_of_a_constant_is_nan_not_zero():
    """Undefined is not the same as 'no signal', and must not be reported as it."""
    assert np.isnan(spearman_ic(np.ones(50), np.arange(50.0)))


def test_ic_distribution_reports_stability_not_just_a_mean():
    """Two signals with the same mean IC but different consistency must differ."""
    generator = np.random.default_rng(1)
    returns = generator.normal(size=2_000)

    steady = returns + generator.normal(size=2_000) * 3.0
    lumpy = generator.normal(size=2_000) * 3.0
    lumpy[:200] = returns[:200] * 10  # all the edge lives in one block

    steady_distribution = ic_distribution(steady, returns, block_size=200)
    lumpy_distribution = ic_distribution(lumpy, returns, block_size=200)

    assert steady_distribution.fraction_positive > lumpy_distribution.fraction_positive
    assert abs(steady_distribution.information_ratio) > abs(lumpy_distribution.information_ratio)


# ------------------------------------------------------------ forward returns


def test_forward_return_is_the_log_change_in_mid():
    mids = np.array([100.0, 101.0, 102.0, 103.0])

    returns = forward_return_by_rows(mids, np.array([0, 1]), horizon_rows=2)

    assert returns == pytest.approx([np.log(102 / 100), np.log(103 / 101)])


def test_returns_running_off_the_end_are_nan_not_clipped():
    """Clipping would silently shorten the horizon for the last samples."""
    mids = np.arange(100.0, 110.0)

    returns = forward_return_by_rows(mids, np.array([7, 8, 9]), horizon_rows=3)

    assert np.isnan(returns).all()


def test_horizon_in_milliseconds_uses_the_real_clock():
    """Event-time sampling means a fixed row count is not a fixed duration."""
    # 10 rows a second for the first half, then 100 a second.
    timestamps = np.concatenate(
        [np.arange(0, 10) * 100_000_000, 1_000_000_000 + np.arange(0, 100) * 10_000_000]
    ).astype(np.int64)

    slow = rows_for_horizon(timestamps, np.array([0]), horizon_ms=500)
    fast = rows_for_horizon(timestamps, np.array([50]), horizon_ms=500)

    assert int(slow[0]) - 0 == 5, "500ms is 5 rows in the slow stretch"
    assert int(fast[0]) - 50 == 50, "500ms is 50 rows in the fast stretch"


def test_a_backwards_clock_is_refused_rather_than_silently_wrong():
    """`searchsorted` cannot detect unsorted input; it just returns nonsense.

    This guard exists because the failure is invisible: an overflowed or
    NTP-stepped clock produces a decay curve that still plots, made entirely
    of noise. Caught originally by a test whose own timestamps overflowed
    int32 on Windows.
    """
    timestamps = np.array([0, 100, 200, 150, 300], dtype=np.int64)

    with pytest.raises(ValueError, match="non-decreasing"):
        rows_for_horizon(timestamps, np.array([0]), horizon_ms=0.1)


def test_a_flat_clock_is_allowed_because_snapshots_can_share_a_timestamp():
    """Equal timestamps are legitimate; only going backwards is not."""
    timestamps = np.array([0, 100, 100, 200], dtype=np.int64)

    targets = rows_for_horizon(timestamps, np.array([0]), horizon_ms=0.0001)

    assert int(targets[0]) == 1


def test_time_based_forward_return_matches_a_hand_computation():
    mids = np.array([100.0, 100.0, 100.0, 105.0, 105.0])
    timestamps = (np.arange(5) * 100_000_000).astype(np.int64)  # 100ms apart

    returns = forward_return_by_time(mids, timestamps, np.array([0]), horizon_ms=300)

    assert returns[0] == pytest.approx(np.log(105 / 100))


# --------------------------------------------------------------- decay fit


def test_half_life_recovers_a_known_exponential():
    horizons = np.array([100.0, 200.0, 400.0, 800.0, 1600.0])
    true_half_life = 500.0
    ics = 0.08 * np.exp(-np.log(2) * horizons / true_half_life)

    half_life, _rate, quality = fit_half_life(horizons, ics)

    assert half_life == pytest.approx(true_half_life, rel=1e-6)
    assert quality == pytest.approx(1.0, abs=1e-9)


def test_a_curve_still_rising_at_the_last_horizon_reports_no_half_life():
    """If the peak has not been reached, decay after it has not been measured.

    The fit starts at the peak, so a monotonically rising curve leaves nothing
    to fit. NaN is the honest answer — "measure longer horizons" — where a
    number would imply we had seen the signal turn over.
    """
    horizons = np.array([100.0, 200.0, 400.0])
    ics = np.array([0.02, 0.03, 0.05])

    half_life, _rate, _quality = fit_half_life(horizons, ics)

    assert np.isnan(half_life)


def test_a_signal_that_plateaus_after_its_peak_reports_infinite_half_life():
    """Better than a negative half-life, which would be nonsense on a chart."""
    horizons = np.array([100.0, 200.0, 400.0, 800.0])
    ics = np.array([0.02, 0.05, 0.05, 0.05])  # peaks at 200ms, then flat

    half_life, _rate, _quality = fit_half_life(horizons, ics)

    assert half_life == float("inf")


def test_the_fit_starts_at_the_peak_and_ignores_the_rising_part():
    """The rising limb must not drag the fitted decay rate toward zero.

    Our real curve climbs from 100 ms to a peak near 5 s before decaying, so a
    fit spanning the whole range would average a climb and a fall and describe
    neither.
    """
    horizons = np.array([100.0, 500.0, 1_000.0, 2_000.0, 4_000.0])
    # Rises to a peak at 1,000 ms, then halves every 1,000 ms.
    ics = np.array([0.10, 0.30, 0.40, 0.20, 0.05])

    half_life, _rate, quality = fit_half_life(horizons, ics)

    assert half_life == pytest.approx(1_000.0, rel=0.05)
    assert quality > 0.99


def test_half_life_is_nan_when_there_is_nothing_positive_to_fit():
    horizons = np.array([100.0, 200.0, 400.0])
    ics = np.array([-0.01, np.nan, -0.02])

    half_life, _rate, _quality = fit_half_life(horizons, ics)

    assert np.isnan(half_life)


def test_decay_curve_measures_a_planted_short_lived_signal():
    """A signal that predicts only the next few rows must decay on the curve."""
    generator = np.random.default_rng(3)
    rows_count = 6_000
    steps = generator.normal(0, 0.5, size=rows_count)
    mids = 65_000 + np.cumsum(steps)
    # dtype=int64 matters: np.arange defaults to int32 on Windows, and
    # 5999 * 34_000_000 overflows it silently, wrapping the clock backwards.
    timestamps = np.arange(rows_count, dtype=np.int64) * 34_000_000  # ~29/s

    # The signal knows the next 3 steps and nothing beyond them.
    rows = np.arange(100, rows_count - 400)
    lookahead = np.array([steps[row + 1 : row + 4].sum() for row in rows])
    signal = lookahead + generator.normal(0, 0.2, size=len(rows))

    curve = signal_decay_curve(signal, mids, timestamps, rows, (100.0, 250.0, 500.0, 1000.0, 2000.0))

    assert curve.ics[0] > curve.ics[-1], "IC must fall as the horizon lengthens"
    assert np.isfinite(curve.half_life_ms)


# ------------------------------------------------ touch prices: the key test


def test_buying_at_the_ask_and_selling_at_the_bid_costs_the_spread():
    """THE test. A flat market must lose exactly the spread, never break even.

    Prices do not move at all here, so a mid-marked backtest would report
    exactly zero PnL and conclude the rule is free. Executing at the touch
    shows the truth: every round trip pays the full spread, and that is before
    a single basis point of fees.
    """
    rows = 50
    mids, bids, asks = make_book(rows, mid=65_000.0, half_spread=0.005)
    entry_rows = np.arange(0, 10)
    exit_rows = entry_rows + 5
    signal = np.full(10, 0.9)  # confidently long every time

    result = simulate_touch_price_trades(signal, bids, asks, entry_rows, exit_rows, confidence=0.1, fee_bps=0.0)

    expected_loss_bps = -(asks[0] - bids[0]) / mids[0] * BASIS_POINTS
    assert result.trade_count == 10
    assert result.gross_bps_per_trade == pytest.approx(expected_loss_bps)
    assert result.gross_bps_per_trade < 0, "a flat market must not be free to trade"


def test_shorts_also_pay_the_spread():
    """Symmetry check: a short sells the bid and buys back the ask."""
    rows = 50
    mids, bids, asks = make_book(rows)
    entry_rows = np.arange(0, 10)
    signal = np.full(10, -0.9)

    result = simulate_touch_price_trades(signal, bids, asks, entry_rows, entry_rows + 5, 0.1, fee_bps=0.0)

    assert result.short_count == 10 and result.long_count == 0
    assert result.gross_bps_per_trade == pytest.approx(-(asks[0] - bids[0]) / mids[0] * BASIS_POINTS)


def test_a_move_larger_than_the_spread_is_profitable_before_fees():
    """The instrument must be able to detect a real edge, or it proves nothing."""
    rows = 50
    mids = np.concatenate([np.full(10, 65_000.0), np.full(rows - 10, 65_010.0)])
    bids, asks = mids - 0.005, mids + 0.005
    entry_rows = np.arange(0, 5)
    signal = np.full(5, 0.9)

    result = simulate_touch_price_trades(signal, bids, asks, entry_rows, entry_rows + 20, 0.1, fee_bps=0.0)

    assert result.gross_bps_per_trade > 0
    assert result.win_rate == 1.0


def test_fees_are_charged_on_both_legs():
    rows = 50
    mids, bids, asks = make_book(rows)
    entry_rows = np.arange(0, 8)
    signal = np.full(8, 0.9)

    free = simulate_touch_price_trades(signal, bids, asks, entry_rows, entry_rows + 5, 0.1, fee_bps=0.0)
    charged = simulate_touch_price_trades(signal, bids, asks, entry_rows, entry_rows + 5, 0.1, fee_bps=3.0)

    assert charged.net_bps_per_trade == pytest.approx(free.net_bps_per_trade - 6.0)


def test_the_confidence_threshold_actually_gates_trades():
    rows = 50
    _mids, bids, asks = make_book(rows)
    entry_rows = np.arange(0, 10)
    signal = np.linspace(-1.0, 1.0, 10)

    permissive = simulate_touch_price_trades(signal, bids, asks, entry_rows, entry_rows + 5, 0.05, 0.0)
    strict = simulate_touch_price_trades(signal, bids, asks, entry_rows, entry_rows + 5, 0.9, 0.0)

    assert permissive.trade_count > strict.trade_count


def test_trades_whose_exit_runs_off_the_tape_are_dropped():
    rows = 20
    _mids, bids, asks = make_book(rows)
    entry_rows = np.array([0, 5, 18])
    signal = np.full(3, 0.9)

    result = simulate_touch_price_trades(signal, bids, asks, entry_rows, entry_rows + 10, 0.1, 0.0)

    assert result.trade_count == 2, "the trade exiting past the end must not be counted"


def test_no_qualifying_trades_returns_nan_rather_than_zero_pnl():
    """Zero PnL from zero trades would read as 'breaks even', which is a lie."""
    rows = 20
    _mids, bids, asks = make_book(rows)

    result = simulate_touch_price_trades(np.zeros(5), bids, asks, np.arange(5), np.arange(5) + 3, 0.5, 0.0)

    assert result.trade_count == 0
    assert np.isnan(result.net_bps_per_trade)


# ------------------------------------------------------------ breakeven fee


def test_breakeven_fee_is_half_the_gross_edge():
    """A round trip pays the fee twice, so breakeven per side is half."""
    assert breakeven_fee_bps(4.0) == pytest.approx(2.0)


def test_a_negative_edge_gives_a_negative_breakeven_fee():
    """There is no fee low enough; reporting 0.0 would imply otherwise."""
    assert breakeven_fee_bps(-1.4) < 0


def test_breakeven_fee_actually_zeroes_the_simulated_pnl():
    """Closes the loop: the reported breakeven must reproduce zero net PnL."""
    rows = 200
    mids = np.concatenate([np.full(50, 65_000.0), np.full(rows - 50, 65_004.0)])
    bids, asks = mids - 0.005, mids + 0.005
    entry_rows = np.arange(0, 30)
    signal = np.full(30, 0.9)

    gross = simulate_touch_price_trades(signal, bids, asks, entry_rows, entry_rows + 60, 0.1, 0.0)
    breakeven = breakeven_fee_bps(gross.gross_bps_per_trade)
    at_breakeven = simulate_touch_price_trades(signal, bids, asks, entry_rows, entry_rows + 60, 0.1, breakeven)

    assert at_breakeven.net_bps_per_trade == pytest.approx(0.0, abs=1e-9)


def test_fee_sweep_is_monotonically_worse_with_higher_fees():
    rows = 200
    mids = np.concatenate([np.full(50, 65_000.0), np.full(rows - 50, 65_004.0)])
    bids, asks = mids - 0.005, mids + 0.005
    entry_rows = np.arange(0, 30)
    signal = np.full(30, 0.9)

    results = fee_sweep(signal, bids, asks, entry_rows, entry_rows + 60, 0.1, (0.0, 1.0, 2.0, 5.0, 7.5, 10.0))

    net = [r.net_bps_per_trade for r in results]
    assert net == sorted(net, reverse=True)
    assert results[0].trade_count == results[-1].trade_count, "only the fee changes"


def test_the_quoted_binance_tiers_are_the_ones_we_cite():
    """Guards against the documented fee table drifting from the code."""
    assert BINANCE_TAKER_FEE_BPS["VIP 0"] == 10.0
    assert BINANCE_TAKER_FEE_BPS["VIP 0 + BNB"] == 7.5
    assert BINANCE_TAKER_FEE_BPS["VIP 9"] == 4.0


# ------------------------------------------------------------ falsification


def test_falsification_collapses_a_genuine_signal():
    """The controls must all go to ~0 while the true IC stays high."""
    generator = np.random.default_rng(7)
    returns = generator.normal(size=4_000)
    signal = returns + generator.normal(size=4_000) * 0.8

    # block_size=100 gives 40 blocks to permute. With only a handful of blocks
    # the shuffled IC is noticeably noisy by luck alone, which is a property of
    # the control rather than of the signal.
    result = falsify(signal, returns, shift=100, seed=0, block_size=100)

    assert result.true_ic > 0.5
    assert abs(result.shifted_label_ic) < 0.05
    assert abs(result.block_shuffled_ic) < 0.05
    assert abs(result.shuffled_signal_ic) < 0.05
    assert result.reversed_signal_ic == pytest.approx(-result.true_ic, abs=1e-9)


def test_block_shuffle_leaves_no_block_in_its_original_position():
    """The derangement is what keeps the control unbiased.

    A plain permutation leaves on average one block in place, and that block
    keeps perfect signal-to-return alignment — leaking roughly `true_ic / N`
    into a control that is supposed to read zero. At 40 blocks that measured
    0.064 instead of ~0.00, which is large enough to be mistaken for a finding.
    """
    values = np.arange(1_000, dtype=np.float64)
    block_size = 50

    shuffled = block_shuffle(values, block_size=block_size, seed=3)

    original_blocks = values.reshape(-1, block_size)
    shuffled_blocks = shuffled.reshape(-1, block_size)
    in_place = sum(
        1 for index in range(len(original_blocks)) if np.array_equal(original_blocks[index], shuffled_blocks[index])
    )
    assert in_place == 0, "a block left in place keeps real alignment and biases the control"


def test_block_shuffle_preserves_local_structure_but_breaks_alignment():
    """The control that separates persistence from leakage.

    A plain element shuffle destroys the signal's autocorrelation as well as
    its alignment, so a near-zero IC from it proves only that the arithmetic
    works. Block shuffling keeps each block internally intact — so the shuffled
    series still *looks* like the signal — while moving it to the wrong time.
    """
    generator = np.random.default_rng(4)
    smooth = np.convolve(generator.normal(size=4_000), np.ones(50) / 50, mode="same")

    shuffled = block_shuffle(smooth, block_size=500, seed=1)

    assert len(shuffled) == len(smooth)
    assert sorted(shuffled.tolist()) == pytest.approx(sorted(smooth.tolist()))
    # Local smoothness survives: neighbouring values still resemble each other.
    original_roughness = float(np.mean(np.abs(np.diff(smooth))))
    shuffled_roughness = float(np.mean(np.abs(np.diff(shuffled))))
    assert shuffled_roughness < 3 * original_roughness
    fully_shuffled = np.random.default_rng(1).permutation(smooth)
    assert float(np.mean(np.abs(np.diff(fully_shuffled)))) > 5 * original_roughness


def test_a_persistent_signal_keeps_ic_at_a_small_shift_but_loses_it_at_a_large_one():
    """Why a non-zero shifted IC is not automatically a leak.

    A signal that genuinely predicts several steps ahead must still correlate
    with a slightly shifted return window. What distinguishes it from a leak is
    that the correlation *decays* as the shift grows. This is the exact
    behaviour observed on the real model, and it is asserted here so the
    interpretation in the notebook rests on a tested property.
    """
    generator = np.random.default_rng(21)
    steps = generator.normal(size=6_000)
    # Returns overlapping a 50-step forward window, so nearby windows share moves.
    returns = np.convolve(steps, np.ones(50) / 50, mode="same")
    signal = returns + generator.normal(size=6_000) * 0.3

    result = falsify(signal, returns, shift=25, seed=0, extra_shifts=(500, 2_000))

    assert result.shifted_ics[25] > 0.20, "a persistent signal survives a small shift"
    assert abs(result.shifted_ics[2_000]) < 0.1, "and must lose it at a large one"
    assert result.shifted_ics[25] > result.shifted_ics[500] > -0.2, "the decay must be monotone-ish"


def test_falsification_would_catch_a_leak():
    """If the 'signal' were the future itself, the shifted control stays high.

    This is the test that gives the falsification suite its teeth: it shows
    the shifted-label check can actually fail, rather than being a formality
    that passes on everything.
    """
    generator = np.random.default_rng(11)
    # A return series with strong autocorrelation, so a shifted pairing still
    # correlates — exactly the situation the control is designed to expose.
    noise = generator.normal(size=4_000)
    returns = np.convolve(noise, np.ones(400) / 400, mode="same")
    signal = returns.copy()

    result = falsify(signal, returns, shift=10, seed=0)

    assert abs(result.shifted_label_ic) > 0.5, "an autocorrelated series must trip the control"
