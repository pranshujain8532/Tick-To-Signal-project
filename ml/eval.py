"""Honest characterisation of the signal: IC, decay, and cost-aware PnL.

WHAT
    Turns model probabilities on held-out windows into the numbers a quant
    researcher would ask for: an information coefficient and its stability, a
    decay curve with a fitted half-life, a PnL simulation executed at touch
    prices net of taker fees, and the breakeven fee at which that PnL crosses
    zero. Plus two falsification tests that should return approximately
    nothing.

WHY
    Stage 4 produced a classification score. A classification score is not a
    characterisation: it says nothing about how long the edge persists,
    whether it is stable or comes from three lucky minutes, or whether it
    survives the cost of trading it. This module exists to answer the question
    "is this signal worth anything", and to be equally willing to answer no.

    **This module makes no strategy claims.** It measures a signal's
    properties. The simulated rule exists to convert an information
    coefficient into units of money so it can be compared against a fee, not
    because anyone should trade it.

DESIGN DECISION — fills at the touch, never at the mid.
    Every simulated trade buys at the *ask* and sells at the *bid*. Rejected
    alternative: mark both legs at the mid, which is what almost every student
    backtest does and is the single most common way a worthless signal is made
    to look profitable. The mid is the average of two prices, and it is not
    one you can trade at: to buy immediately you must cross to the ask, and to
    sell immediately you must hit the bid. Marking at mid silently hands the
    strategy half the spread on entry and half on exit — a full spread per
    round trip, free.

    On BTCUSDT that spread is one tick, which is only 0.0015 bps at a 65,000
    mid, so on *this* instrument the free half-spread is not what kills the
    result — the taker fee, roughly 6,500 times larger, is. The rule stands
    anyway: an execution model that is wrong in the strategy's favour is worth
    nothing even when the error is small, and on a wider-spread instrument the
    same code would be wrong by an amount that dominates everything.

DESIGN DECISION — the information coefficient, not accuracy.
    Rejected alternative: report classification accuracy and stop. Accuracy
    throws away two things that decide whether a signal is usable: the
    *magnitude* of the move it predicts, and the *confidence* of the
    prediction. A model that is 51% accurate on huge moves and wrong on small
    ones is valuable; one that is 60% accurate only on moves smaller than the
    spread is not, and accuracy cannot tell them apart. IC — the rank
    correlation between a continuous signal and the realised forward return —
    keeps both, and it is the number a quant desk actually quotes.

    Spearman rather than Pearson because forward returns at this timescale are
    heavy-tailed and mostly zero: a handful of large moves would dominate a
    Pearson correlation, so it would mostly measure whether the model called
    those few. Ranks make the statistic about ordering, which is what a signal
    is for.

DESIGN DECISION — IC computed per block, and the distribution reported.
    Rejected alternative: one IC over the whole test set. A single number
    hides whether the edge is steady or is one good afternoon. Splitting the
    test period into contiguous blocks and reporting the distribution shows
    stability, which for a tradeable signal matters more than the mean — a
    mean IC of 0.05 that is positive in 80% of blocks is a business; the same
    mean from two great blocks and eight flat ones is not.

INFORMATION HORIZON
    Everything here is computed on held-out walk-forward test windows only,
    using the embargo from `ml/splits.py`. Forward returns look ahead by
    construction — that is what they are for — but they are only ever compared
    against signals produced from data strictly before them, and the
    falsification tests exist to prove no leakage survives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Taker fees on Binance spot, checked 2026-07-28.
# https://www.binance.com/en/fee/schedule
#   VIP 0                     0.1000%  = 10.0 bps
#   VIP 0 paying fees in BNB  0.0750%  =  7.5 bps
#   VIP 9 (>$2bn 30d volume)  0.0400%  =  4.0 bps
# Quoted per side; a round trip pays twice.
BINANCE_TAKER_FEE_BPS = {
    "VIP 0": 10.0,
    "VIP 0 + BNB": 7.5,
    "VIP 4": 5.0,
    "VIP 9": 4.0,
}

BASIS_POINTS = 1e4


# ------------------------------------------------------------ scalar signal


def signal_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Collapse `[N, 3]` class probabilities into one scalar in [-1, 1].

    `s = P(up) - P(down)`, deliberately ignoring `P(flat)`. The flat class
    carries no directional information, so including it would only rescale the
    signal; leaving it out means `s` is directly interpretable as "how much
    more up than down", and its sign is the trade direction.
    """
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError(f"expected [N, 3] probabilities, got shape {probabilities.shape}")
    return probabilities[:, 2] - probabilities[:, 0]


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not bias the correlation.

    Ties matter here more than usual: forward returns over short horizons are
    frequently *exactly* zero because the mid did not move, so a naive ranking
    would impose an arbitrary order on a large block of identical values.
    """
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)

    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        if stop - start > 1:
            ranks[order[start:stop]] = ranks[order[start:stop]].mean()
        start = stop
    return ranks


def spearman_ic(signal: np.ndarray, forward_returns: np.ndarray) -> float:
    """Spearman rank correlation between signal and realised forward return.

    Returns NaN when either side is constant — a correlation with a constant
    is undefined, and returning 0.0 would quietly report "no signal" for what
    is really "no answer".
    """
    finite = np.isfinite(signal) & np.isfinite(forward_returns)
    if finite.sum() < 3:
        return float("nan")
    signal_ranks = _rank(signal[finite])
    return_ranks = _rank(forward_returns[finite])
    if signal_ranks.std() == 0 or return_ranks.std() == 0:
        return float("nan")
    return float(np.corrcoef(signal_ranks, return_ranks)[0, 1])


# ------------------------------------------------------- forward returns


def forward_return_by_rows(mids: np.ndarray, rows: np.ndarray, horizon_rows: int) -> np.ndarray:
    """Log return of the mid from each row to `horizon_rows` later.

    Log rather than simple returns so that horizons compose additively, which
    is what makes the decay curve's exponential fit meaningful. Rows whose
    horizon falls off the end of the session return NaN rather than being
    clipped to the last available row — clipping would silently shorten the
    horizon for exactly the samples at the end of the tape.
    """
    target = rows + horizon_rows
    valid = target < len(mids)
    returns = np.full(len(rows), np.nan)
    if not valid.any():
        return returns
    start_price = mids[rows[valid]]
    end_price = mids[target[valid]]
    positive = (start_price > 0) & (end_price > 0)
    computed = np.full(valid.sum(), np.nan)
    computed[positive] = np.log(end_price[positive] / start_price[positive])
    returns[valid] = computed
    return returns


def rows_for_horizon(timestamps_ns: np.ndarray, rows: np.ndarray, horizon_ms: float) -> np.ndarray:
    """For each row, the first row at least `horizon_ms` later in wall clock.

    The tape is sampled in *event* time — one snapshot per N book events — so a
    fixed number of rows is a different amount of wall clock depending on how
    busy the market was. Converting a millisecond horizon through the actual
    timestamps keeps the decay curve on a real clock, which is the only axis on
    which "half-life" means anything to a reader.

    Returns `len(timestamps)` for rows whose horizon runs past the end, so the
    caller's bounds check drops them.

    WHY THE MONOTONICITY CHECK. `searchsorted` on an unsorted array returns
    meaningless indices and raises nothing — it cannot tell that its input is
    not sorted, so a clock that went backwards produces a decay curve made of
    noise that still plots. This was found the hard way: a test built its
    timestamps with `np.arange(n) * 34_000_000`, which silently overflowed
    int32 on Windows and wrapped, and the only symptom was an IC that rose with
    horizon instead of falling. Two reasons this guard stays in production
    rather than only in the test: the tape's `local_ts_ns` is a *wall* clock,
    which NTP can step backwards, and any future resampling that concatenates
    sessions out of order would land here too.
    """
    timestamps_ns = np.asarray(timestamps_ns)
    if timestamps_ns.dtype != np.int64:
        timestamps_ns = timestamps_ns.astype(np.int64)
    if len(timestamps_ns) > 1 and np.any(np.diff(timestamps_ns) < 0):
        first_break = int(np.argmax(np.diff(timestamps_ns) < 0))
        raise ValueError(
            f"timestamps must be non-decreasing for a horizon lookup, but row {first_break + 1} "
            f"({timestamps_ns[first_break + 1]}) precedes row {first_break} ({timestamps_ns[first_break]}). "
            "A backwards clock step or an integer overflow would otherwise produce a plausible-looking "
            "decay curve made entirely of noise."
        )
    horizon_ns = np.int64(horizon_ms * 1e6)
    targets = np.searchsorted(timestamps_ns, timestamps_ns[rows] + horizon_ns, side="left")
    return targets.astype(np.int64)


def forward_return_by_time(
    mids: np.ndarray,
    timestamps_ns: np.ndarray,
    rows: np.ndarray,
    horizon_ms: float,
) -> np.ndarray:
    """Log mid return over a wall-clock horizon, NaN where it runs off the tape."""
    targets = rows_for_horizon(timestamps_ns, rows, horizon_ms)
    valid = targets < len(mids)
    returns = np.full(len(rows), np.nan)
    if not valid.any():
        return returns
    start_price = mids[rows[valid]]
    end_price = mids[targets[valid]]
    positive = (start_price > 0) & (end_price > 0)
    computed = np.full(valid.sum(), np.nan)
    computed[positive] = np.log(end_price[positive] / start_price[positive])
    returns[valid] = computed
    return returns


# ------------------------------------------------------------ IC stability


@dataclass
class ICDistribution:
    """Per-block ICs and the summary statistics that describe their stability."""

    block_ics: np.ndarray
    pooled_ic: float
    block_size: int

    @property
    def mean(self) -> float:
        return float(np.nanmean(self.block_ics)) if len(self.block_ics) else float("nan")

    @property
    def median(self) -> float:
        return float(np.nanmedian(self.block_ics)) if len(self.block_ics) else float("nan")

    @property
    def standard_deviation(self) -> float:
        return float(np.nanstd(self.block_ics)) if len(self.block_ics) else float("nan")

    @property
    def fraction_positive(self) -> float:
        finite = self.block_ics[np.isfinite(self.block_ics)]
        return float(np.mean(finite > 0)) if len(finite) else float("nan")

    @property
    def information_ratio(self) -> float:
        """Mean IC divided by its standard deviation across blocks.

        The stability number. A desk cares far more about this than about the
        mean: it is roughly how many blocks of edge you get per block of noise,
        and it is what decides whether a signal can be sized.
        """
        deviation = self.standard_deviation
        if not np.isfinite(deviation) or deviation == 0:
            return float("nan")
        return self.mean / deviation


def ic_distribution(signal: np.ndarray, forward_returns: np.ndarray, block_size: int = 500) -> ICDistribution:
    """IC computed within each contiguous block, plus the pooled IC.

    Blocks are contiguous in time rather than random, because the question is
    "was the edge there in every period" and a random partition would average
    that away.
    """
    block_count = max(1, len(signal) // block_size)
    block_ics = []
    for block in range(block_count):
        start = block * block_size
        stop = start + block_size if block < block_count - 1 else len(signal)
        block_ics.append(spearman_ic(signal[start:stop], forward_returns[start:stop]))
    return ICDistribution(
        block_ics=np.array(block_ics, dtype=np.float64),
        pooled_ic=spearman_ic(signal, forward_returns),
        block_size=block_size,
    )


# --------------------------------------------------------------- decay curve


@dataclass
class DecayCurve:
    """IC as a function of horizon, its peak, and the decay fitted after it."""

    horizons_ms: np.ndarray
    ics: np.ndarray
    sample_counts: np.ndarray
    half_life_ms: float
    decay_rate_per_ms: float
    fit_quality: float
    peak_horizon_ms: float = float("nan")
    peak_ic: float = float("nan")

    def as_rows(self) -> list[dict]:
        return [
            {"horizon_ms": float(h), "ic": float(i), "samples": int(n)}
            for h, i, n in zip(self.horizons_ms, self.ics, self.sample_counts)
        ]


def fit_half_life(horizons_ms: np.ndarray, ics: np.ndarray) -> tuple[float, float, float]:
    """Fit `IC(h) = IC0 * exp(-h / tau)` from the curve's PEAK onward.

    Returns `(half_life_ms, rate, R^2)`.

    WHY FROM THE PEAK AND NOT FROM ZERO. A textbook decay curve is monotone:
    the edge is largest at the shortest horizon and bleeds away. Ours is not,
    and pretending otherwise would produce a meaningless number. The model is
    trained on a *smoothed* label spanning ~100 snapshots, so it predicts an
    average over the next few seconds rather than the next instant — and the
    measured IC duly climbs from 100 ms, peaks near the label horizon, and only
    then decays. Fitting an exponential across the rising part would average a
    climb and a fall together and report a half-life that describes neither.

    So the peak is located first and the fit runs over horizons at or beyond
    it. The half-life then answers the question that actually matters for
    execution: once the signal is as good as it gets, how long do you have
    before half of it is gone?

    Only strictly positive ICs are used — the log of a negative IC is
    undefined, and a horizon where the signal has already decayed into noise
    says nothing about the rate.
    """
    finite = np.isfinite(ics)
    if finite.sum() < 2:
        return float("nan"), float("nan"), float("nan")

    peak_index = int(np.nanargmax(np.where(finite, ics, -np.inf)))
    usable = finite & (ics > 0)
    usable[:peak_index] = False
    if usable.sum() < 2:
        return float("nan"), float("nan"), float("nan")

    horizons = horizons_ms[usable].astype(np.float64)
    log_ics = np.log(ics[usable].astype(np.float64))
    slope, intercept = np.polyfit(horizons, log_ics, 1)
    # A perfectly flat curve fits a slope of ~-1e-19 rather than exactly 0, and
    # dividing by that yields a half-life of 1e18 ms, which prints as a number
    # and means "no decay". Anything within floating-point noise of zero is
    # reported as no decay, because that is what it is.
    if slope >= -np.finfo(np.float64).eps:
        return float("inf"), float(slope), _r_squared(horizons, log_ics, slope, intercept)

    half_life = float(np.log(2.0) / -slope)
    return half_life, float(slope), _r_squared(horizons, log_ics, slope, intercept)


def _build_decay_curve(horizons_ms, ics, counts) -> DecayCurve:
    """Assemble a `DecayCurve`, locating the peak the half-life is fitted from."""
    horizons_array = np.array(horizons_ms, dtype=np.float64)
    ics_array = np.array(ics, dtype=np.float64)
    half_life, rate, quality = fit_half_life(horizons_array, ics_array)

    finite = np.isfinite(ics_array)
    if finite.any():
        peak_index = int(np.nanargmax(np.where(finite, ics_array, -np.inf)))
        peak_horizon = float(horizons_array[peak_index])
        peak_ic = float(ics_array[peak_index])
    else:
        peak_horizon = float("nan")
        peak_ic = float("nan")

    return DecayCurve(
        horizons_ms=horizons_array,
        ics=ics_array,
        sample_counts=np.array(counts, dtype=np.int64),
        half_life_ms=half_life,
        decay_rate_per_ms=rate,
        fit_quality=quality,
        peak_horizon_ms=peak_horizon,
        peak_ic=peak_ic,
    )


def _r_squared(x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> float:
    predicted = slope * x + intercept
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - residual / total if total > 0 else float("nan")


def signal_decay_curve(
    signal: np.ndarray,
    mids: np.ndarray,
    timestamps_ns: np.ndarray,
    rows: np.ndarray,
    horizons_ms: tuple[float, ...],
) -> DecayCurve:
    """IC at each wall-clock horizon, with an exponential half-life fitted."""
    ics = []
    counts = []
    for horizon in horizons_ms:
        returns = forward_return_by_time(mids, timestamps_ns, rows, horizon)
        ics.append(spearman_ic(signal, returns))
        counts.append(int(np.isfinite(returns).sum()))

    return _build_decay_curve(horizons_ms, ics, counts)


# ------------------------------------------------------- cost-aware trading


@dataclass
class TradeSimulation:
    """The outcome of running the toy rule at one fee level."""

    fee_bps: float
    trade_count: int
    gross_bps_per_trade: float
    net_bps_per_trade: float
    total_net_bps: float
    win_rate: float
    cumulative_net_bps: np.ndarray = field(repr=False)
    long_count: int = 0
    short_count: int = 0


def simulate_trades_from_prices(
    signal: np.ndarray,
    entry_bid: np.ndarray,
    entry_ask: np.ndarray,
    exit_bid: np.ndarray,
    exit_ask: np.ndarray,
    tradeable: np.ndarray,
    confidence: float,
    fee_bps: float,
) -> TradeSimulation:
    """The execution core, taking per-sample prices rather than row indices.

    Exists because a test fold can span more than one capture session, and a
    row index only means something inside the session it came from. Passing
    prices that the caller has already resolved per session makes it
    impossible to index one tape's prices with another tape's rows — which is
    a mistake that would produce entirely fictional PnL without erroring.
    """
    usable = tradeable & (np.abs(signal) > confidence) & np.isfinite(entry_bid) & np.isfinite(exit_bid)
    if not usable.any():
        return TradeSimulation(fee_bps, 0, float("nan"), float("nan"), 0.0, float("nan"), np.zeros(0))

    direction = np.sign(signal[usable])
    is_long = direction > 0

    # Long: cross to the ask to get in, hit the bid to get out.
    # Short: hit the bid to get in, cross to the ask to get out.
    entry_price = np.where(is_long, entry_ask[usable], entry_bid[usable])
    exit_price = np.where(is_long, exit_bid[usable], exit_ask[usable])
    reference_mid = (entry_bid[usable] + entry_ask[usable]) / 2.0

    gross = np.where(is_long, exit_price - entry_price, entry_price - exit_price)
    gross_bps = gross / reference_mid * BASIS_POINTS
    net_bps = gross_bps - 2.0 * fee_bps  # taker fee on both legs

    return TradeSimulation(
        fee_bps=fee_bps,
        trade_count=int(usable.sum()),
        gross_bps_per_trade=float(np.mean(gross_bps)),
        net_bps_per_trade=float(np.mean(net_bps)),
        total_net_bps=float(np.sum(net_bps)),
        win_rate=float(np.mean(net_bps > 0)),
        cumulative_net_bps=np.cumsum(net_bps),
        long_count=int(np.sum(is_long)),
        short_count=int(np.sum(~is_long)),
    )


def resolve_prices_per_session(
    sessions: list,
    session_ids: np.ndarray,
    rows: np.ndarray,
    holding_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Entry and exit touch prices for samples spread across several sessions.

    Returns `(entry_bid, entry_ask, exit_bid, exit_ask, tradeable)`. A sample
    whose exit would fall past the end of *its own* session is marked
    untradeable rather than being allowed to borrow the next session's prices:
    the tapes are separated by resync gaps of seconds to minutes, so a position
    held across the join never existed.
    """
    count = len(rows)
    entry_bid = np.full(count, np.nan)
    entry_ask = np.full(count, np.nan)
    exit_bid = np.full(count, np.nan)
    exit_ask = np.full(count, np.nan)
    tradeable = np.zeros(count, dtype=bool)

    for session_id in np.unique(session_ids):
        selected = session_ids == session_id
        session = sessions[int(session_id)]
        entry_rows = rows[selected]
        exit_rows = entry_rows + holding_rows
        inside = exit_rows < len(session.best_bids)

        entry_bid[selected] = session.best_bids[entry_rows]
        entry_ask[selected] = session.best_asks[entry_rows]
        clipped = np.minimum(exit_rows, len(session.best_bids) - 1)
        exit_bid[selected] = np.where(inside, session.best_bids[clipped], np.nan)
        exit_ask[selected] = np.where(inside, session.best_asks[clipped], np.nan)
        tradeable[selected] = inside

    return entry_bid, entry_ask, exit_bid, exit_ask, tradeable


def pooled_forward_returns(
    sessions: list,
    session_ids: np.ndarray,
    rows: np.ndarray,
    horizon_ms: float,
) -> np.ndarray:
    """Forward return per sample, each computed inside its own session.

    Pooling across sessions is fine — a return is a per-sample number — but
    *computing* one across a session boundary is not, because the gap between
    tapes contains price moves nobody observed.
    """
    returns = np.full(len(rows), np.nan)
    for session_id in np.unique(session_ids):
        selected = session_ids == session_id
        session = sessions[int(session_id)]
        returns[selected] = forward_return_by_time(
            session.mids, session.timestamps_ns, rows[selected], horizon_ms
        )
    return returns


def pooled_decay_curve(
    signal: np.ndarray,
    sessions: list,
    session_ids: np.ndarray,
    rows: np.ndarray,
    horizons_ms: tuple[float, ...],
) -> DecayCurve:
    """Decay curve over samples that may come from several sessions."""
    ics = []
    counts = []
    for horizon in horizons_ms:
        returns = pooled_forward_returns(sessions, session_ids, rows, horizon)
        ics.append(spearman_ic(signal, returns))
        counts.append(int(np.isfinite(returns).sum()))

    return _build_decay_curve(horizons_ms, ics, counts)


def simulate_touch_price_trades(
    signal: np.ndarray,
    best_bids: np.ndarray,
    best_asks: np.ndarray,
    entry_rows: np.ndarray,
    exit_rows: np.ndarray,
    confidence: float,
    fee_bps: float,
) -> TradeSimulation:
    """Trade when the signal is confident; enter and exit at the touch.

    The rule, deliberately the simplest thing that can convert a signal into
    money: if `s > confidence` buy, if `s < -confidence` sell, hold for the
    label horizon, then close. No sizing, no stops, no netting, no queue
    position — this is a measuring instrument, not a strategy.

    Execution is the part that is not simplified:

        long   entry at the ASK, exit at the BID
        short  entry at the BID, exit at the ASK

    Both legs cross the spread, and both pay a taker fee. That is what
    immediate execution costs. Marking either leg at the mid would hand the
    rule half a spread it never earned.

    Returns PnL in basis points of the entry mid, so results are comparable
    across price levels and directly against a fee quoted in bps.
    """
    if len(signal) != len(entry_rows) or len(signal) != len(exit_rows):
        raise ValueError("signal, entry_rows and exit_rows must describe the same samples")

    inside = exit_rows < len(best_bids)
    clipped = np.minimum(exit_rows, len(best_bids) - 1)
    return simulate_trades_from_prices(
        signal,
        entry_bid=best_bids[entry_rows],
        entry_ask=best_asks[entry_rows],
        exit_bid=np.where(inside, best_bids[clipped], np.nan),
        exit_ask=np.where(inside, best_asks[clipped], np.nan),
        tradeable=inside,
        confidence=confidence,
        fee_bps=fee_bps,
    )


def breakeven_fee_bps(gross_bps_per_trade: float) -> float:
    """The per-side taker fee at which expected PnL per trade reaches zero.

    A round trip pays the fee twice, so the breakeven per-side fee is half the
    gross edge. Already-negative gross edge returns a negative number: there is
    no fee low enough, and reporting 0.0 would imply a free exchange might
    rescue it.
    """
    return gross_bps_per_trade / 2.0


def fee_sweep(
    signal: np.ndarray,
    best_bids: np.ndarray,
    best_asks: np.ndarray,
    entry_rows: np.ndarray,
    exit_rows: np.ndarray,
    confidence: float,
    fee_levels_bps: tuple[float, ...],
) -> list[TradeSimulation]:
    """Run the same trades at several fee levels. Only the fee changes."""
    return [
        simulate_touch_price_trades(
            signal, best_bids, best_asks, entry_rows, exit_rows, confidence, fee
        )
        for fee in fee_levels_bps
    ]


# ------------------------------------------------------------ falsification


@dataclass
class NullDistribution:
    """ICs produced by repeatedly destroying the signal-to-return alignment."""

    ics: np.ndarray
    true_ic: float

    @property
    def mean(self) -> float:
        return float(np.nanmean(self.ics))

    @property
    def standard_deviation(self) -> float:
        return float(np.nanstd(self.ics))

    @property
    def z_score(self) -> float:
        """How many null standard deviations the true IC sits above the null mean."""
        deviation = self.standard_deviation
        if not np.isfinite(deviation) or deviation == 0:
            return float("nan")
        return (self.true_ic - self.mean) / deviation

    @property
    def exceedance_rate(self) -> float:
        """Fraction of null draws at least as extreme as the true IC.

        The empirical p-value. Bounded below by `1 / trials`, so it should be
        read as "smaller than this" when it comes back at zero.
        """
        finite = self.ics[np.isfinite(self.ics)]
        if len(finite) == 0:
            return float("nan")
        return float(np.mean(np.abs(finite) >= abs(self.true_ic)))


def block_shuffle_null(
    signal: np.ndarray,
    forward_returns: np.ndarray,
    block_size: int = 500,
    trials: int = 200,
    seed: int = 0,
) -> NullDistribution:
    """Build the null distribution of the IC under destroyed alignment.

    WHY A DISTRIBUTION AND NOT ONE DRAW. A single block-shuffled IC is nearly
    uninformative here. The signal is autocorrelated over roughly the window
    length, so shuffling in blocks of 500 over a 9,300-sample test fold
    permutes only about 18 independent things — and the standard error of a
    correlation estimated from 18 effective observations is about 0.24. One
    draw landing at -0.14 therefore says nothing about whether the true IC is
    real; it is comfortably inside the noise. Repeating the shuffle turns that
    from a guess into a measurement, and the true IC can then be quoted in
    units of the null's own spread.
    """
    ics = np.array(
        [
            spearman_ic(block_shuffle(signal, block_size, seed + trial), forward_returns)
            for trial in range(trials)
        ]
    )
    return NullDistribution(ics=ics, true_ic=spearman_ic(signal, forward_returns))


@dataclass
class FalsificationResult:
    """What the sanity checks returned, and what each one is actually testing."""

    true_ic: float
    shifted_ics: dict[int, float]
    block_shuffled_ic: float
    shuffled_signal_ic: float
    reversed_signal_ic: float

    @property
    def shifted_label_ic(self) -> float:
        """IC at the smallest non-zero shift, for a quick summary line."""
        if not self.shifted_ics:
            return float("nan")
        return self.shifted_ics[min(self.shifted_ics)]


def block_shuffle(values: np.ndarray, block_size: int, seed: int = 0) -> np.ndarray:
    """Permute contiguous blocks, preserving local structure but destroying alignment.

    The control that a plain shuffle cannot provide. Our signal is strongly
    autocorrelated — adjacent windows share 99 of 100 rows — so shuffling
    element by element destroys that structure as well as the pairing, and the
    resulting near-zero IC only proves the arithmetic works. Shuffling whole
    blocks keeps the signal looking like itself while moving it to the wrong
    place in time, which isolates *alignment* as the thing being tested.

    Uses a *derangement* — a permutation with no fixed points — rather than a
    plain shuffle. With a plain permutation of N blocks, on average one block
    lands back where it started and keeps perfect signal-to-return alignment,
    which leaks a real correlation of roughly `true_ic / N` into a control that
    is supposed to read zero. Measured at 40 blocks that was enough to push the
    control to 0.06 when it should have been ~0.00. A control with a known bias
    is worse than no control.
    """
    generator = np.random.default_rng(seed)
    block_count = max(1, len(values) // block_size)
    blocks = [values[index * block_size : (index + 1) * block_size] for index in range(block_count)]
    blocks[-1] = values[(block_count - 1) * block_size :]
    order = _derangement(len(blocks), generator)
    return np.concatenate([blocks[index] for index in order])


def _derangement(count: int, generator: np.random.Generator) -> np.ndarray:
    """A permutation of `range(count)` with no element left in place.

    Rejection sampling would loop; instead any fixed point is swapped with its
    neighbour, which cannot create a new fixed point because the neighbour was
    not in place either (or becomes displaced by the swap).
    """
    if count < 2:
        return np.arange(count)
    order = generator.permutation(count)
    for position in range(count):
        if order[position] == position:
            partner = (position + 1) % count
            order[position], order[partner] = order[partner], order[position]
    return order


def falsify(
    signal: np.ndarray,
    forward_returns: np.ndarray,
    shift: int,
    seed: int = 0,
    extra_shifts: tuple[int, ...] = (),
    block_size: int = 500,
) -> FalsificationResult:
    """Controls that separate a real edge from a leak — and from mere persistence.

    * **Shifted returns**, at several shifts — pair each signal with a forward
      return from `shift` samples later, so the signal is asked about a window
      that starts after the one it was trained on.

      READ THIS ONE CAREFULLY. It does **not** go to zero for a persistent
      signal, and expecting it to is a mistake. Our signal is strongly
      autocorrelated, and it genuinely predicts returns several seconds out
      (the decay curve shows IC still positive at 20 s), so pairing it with a
      slightly later window *should* retain much of the correlation. What
      indicts a leak is not a non-zero value, it is a shifted IC that fails to
      **decay** as the shift grows — a leak is tied to a specific alignment and
      would collapse the moment that alignment breaks, whereas persistence
      fades smoothly. So the shift is swept, and the shape is what is read.

    * **Block shuffle** — move whole blocks of the signal to the wrong place in
      time while preserving their internal structure. This is the control that
      isolates alignment from persistence, and it *must* go to ~0.

    * **Element shuffle** — destroy everything. Anything but ~0 means the IC
      computation itself is broken.

    * **Reversed signal** — negate it; must mirror the true IC exactly, or the
      statistic is not symmetric.
    """
    true_ic = spearman_ic(signal, forward_returns)

    shifted_ics: dict[int, float] = {}
    for candidate in (shift,) + tuple(extra_shifts):
        if 0 < candidate < len(signal):
            shifted_ics[int(candidate)] = spearman_ic(signal[:-candidate], forward_returns[candidate:])

    generator = np.random.default_rng(seed)
    return FalsificationResult(
        true_ic=true_ic,
        shifted_ics=shifted_ics,
        block_shuffled_ic=spearman_ic(block_shuffle(signal, block_size, seed), forward_returns),
        shuffled_signal_ic=spearman_ic(generator.permutation(signal), forward_returns),
        reversed_signal_ic=spearman_ic(-signal, forward_returns),
    )
