# Interview notes — Stage 5: honest evaluation

What shipped: `ml/eval.py` (IC, stability, decay with a fitted half-life,
touch-price PnL, breakeven fee, a permutation-test falsification suite), 36 new
tests (184 total), `benchmarks/bench_evaluation.py`, three exported charts, and
[notebook 05](../notebooks/05_honest_evaluation.ipynb).

Files: [ml/eval.py](../ml/eval.py), [tests/test_eval.py](../tests/test_eval.py),
[benchmarks/bench_evaluation.py](../benchmarks/bench_evaluation.py).
Measured results: `benchmarks/evaluation_20260728T202537Z.json`.

**No strategy is claimed anywhere in this stage.** The simulated rule is a
measuring instrument that converts an IC into basis points so it can be
compared against an exchange fee.

---

## 1. The headline

| Question | Answer |
|---|---|
| Classifies better than chance? | Yes — macro-F1 **0.549** vs 0.371 majority accuracy |
| Ranks forward returns? | Yes — pooled IC **+0.421** |
| Is the edge stable? | **No** — per-block mean IC +0.073, std 0.351, IR **0.21** |
| How long does it last? | Peaks at 5 s; half-life **13.2 s** from the peak (R² 0.98) |
| Gross edge per trade, at the touch? | **+0.285 bps** |
| Breakeven taker fee? | **0.14 bps per side** |
| Survives retail fees? | **No** — 70× short of VIP 0, 28× short of VIP 9 |
| Statistically real? | Yes — **+4.3σ** vs a 200-draw permutation null, p < 0.005 |

**One sentence:** a real but small and unstable directional edge with a
multi-second half-life, roughly two orders of magnitude too small to pay
Binance's taker fee, therefore not tradeable as a taker by anyone at any
published tier.

---

## 2. Design decisions and the alternatives rejected

### 2.1 Fills at the touch, never at the mid — `eval.py:simulate_trades_from_prices`

Long buys the ask and sells the bid; short sells the bid and buys the ask.
Both legs cross, both pay taker fees.

Rejected: mark both legs at the mid. The mid is the average of two prices and
is not one anyone can trade at — to buy immediately you cross to the ask, to
sell you hit the bid. Mid-marking hands the strategy half the spread on entry
and half on exit, a full spread per round trip, free, on every trade. It never
announces itself; the backtest simply prints a better number.

On BTCUSDT the spread is only 0.0015 bps, so here the fee dominates the
half-spread by ~6,500×. The discipline is not instrument-specific: an execution
model wrong in the strategy's favour is worthless even when the error is small,
and on a wider-spread instrument it would dominate everything.

Pinned by `test_buying_at_the_ask_and_selling_at_the_bid_costs_the_spread`,
which trades a **flat** market: mid-marking would report exactly zero PnL and
call the rule free; touch pricing shows the loss.

### 2.2 Information coefficient over accuracy

Rejected: report accuracy. It discards the *magnitude* of the predicted move
and the *confidence* of the prediction, so it cannot distinguish a model that
is 51% right on large moves from one that is 60% right on moves smaller than
the spread — and only the first is worth anything.

Spearman rather than Pearson: forward returns here are heavy-tailed and
frequently exactly zero, so Pearson would mostly measure whether we called a
handful of large moves. Ranks make the statistic about ordering.

Tie handling is not incidental — the mid is unchanged 99% of the time, so a
naive ranking would impose arbitrary order on a large block of identical
zeros. `_rank` uses average ranks; `test_ties_are_ranked_by_average_not_by_position`.

### 2.3 IC computed per block, and the distribution reported

Rejected: one pooled IC. This turned out to be the single most consequential
decision in the stage — see §3.

### 2.4 The half-life is fitted from the PEAK, not from zero — `eval.py:fit_half_life`

Our decay curve is **not monotone**. It climbs from +0.25 at 100 ms to +0.44
at 5 s, then decays. The reason is the label: the model is trained on a
smoothed target spanning 2,736 ms, so it predicts an average over the next few
seconds rather than the next instant, and the IC peaks near the horizon it was
trained on.

Fitting an exponential across the whole range would average a climb and a fall
and report a rate describing neither. So the peak is located first and the fit
runs from there, answering the question that matters for execution: once the
signal is as good as it gets, how long before half of it is gone?

A curve still rising at the last horizon returns NaN, not a number —
"measure longer horizons" is the honest answer.

### 2.5 Horizons in wall-clock milliseconds, not row counts

The tape is sampled in *event* time, so a fixed number of rows is a different
duration depending on market activity. `rows_for_horizon` walks the real
timestamps. Half-life in rows would be uninterpretable to a reader.

### 2.6 A permutation null, not a single control draw — `eval.py:block_shuffle_null`

See §4. This replaced a single block-shuffle draw after that draw turned out
to be uninformative.

### 2.7 The block shuffle uses a derangement — `eval.py:_derangement`

A plain permutation of N blocks leaves on average one block in place, which
keeps perfect signal-to-return alignment and leaks roughly `IC/N` into a
control that is supposed to read zero. Measured at 40 blocks that bias was
**0.064** — large enough to be mistaken for a finding. A control with a known
bias is worse than no control.

### 2.8 Zero trades reports NaN, not zero PnL

`test_no_qualifying_trades_returns_nan_rather_than_zero_pnl`. Zero PnL from
zero trades reads as "breaks even", which is a different claim from "never
traded".

---

## 3. The finding that a single number would have hidden

**Pooled IC +0.421. Mean of per-block ICs +0.073.** Same data, factor of six.

Pooling ranks all 9,292 samples against each other at once, so slow common
drift shared across a whole block — signal generally high while returns are
generally positive for a few minutes — contributes enormously. Computing the IC
*within* each block removes that shared level and asks the harder question: on
any given stretch of a few minutes, does the signal rank the next few seconds
correctly?

Answer: weakly and inconsistently. Positive in **67% of 18 blocks**, standard
deviation **4.8× the mean**, information ratio **0.21**. A desk would call that
unsizeable — the edge is real but swamped by its own period-to-period variance.

The per-block number is the one that belongs in a report. The pooled IC is not
wrong; it answers a question nobody can trade, crediting the model for knowing
that one afternoon drifted up.

---

## 4. Two things I got wrong first, and what fixed them

### The shifted-label control does not go to zero, and I nearly called it a leak

The first run returned `shifted_label_ic = +0.297` against a true IC of
+0.421 — 70% of the signal surviving a shift that was supposed to destroy it.
That looks exactly like leakage surviving the Stage 3 embargo.

It is not, and the reasoning matters. Our signal is strongly autocorrelated
(adjacent windows share 99 of 100 rows) *and* genuinely predicts several
seconds out, as the decay curve independently shows. Pairing it with a slightly
later window **should** retain much of the correlation. What distinguishes
persistence from a leak is not whether the shifted IC is non-zero but whether
it **decays**:

| shift | IC |
|---|---|
| 100 rows (~3 s) | +0.297 |
| 300 rows | +0.032 |
| 900 rows | −0.057 |
| 2,000 rows | −0.045 |
| 4,000 rows | −0.016 |

A leak is tied to one alignment and would hold near +0.42 at every shift. This
collapses within 300 rows. The fix was to **sweep the shift** rather than
report one value, and `test_a_persistent_signal_keeps_ic_at_a_small_shift_but_loses_it_at_a_large_one`
pins the behaviour so the notebook's interpretation rests on a tested property.

### A single block-shuffled draw was uninformative

The same run reported a block-shuffled IC of **−0.144**, which looks like a
failed control. It is a −1σ draw: with blocks of 500 over 9,292 samples there
are only ~18 things to permute, so the null has standard deviation ≈ 0.11.

Replacing one draw with **200** turns a guess into a measurement: null mean
−0.035, std 0.107, and the true IC sits **+4.3σ above it, exceeded by 0.0% of
draws** (p < 0.005). That is the control that actually settles whether the
signal is real.

### Also caught here: a units error from Stage 1

The spread was recorded in Stage 1 as "~0.15 bps". It is **0.0015 bps** — one
basis point at a 64,680 mid is 6.47 USDT, and the spread is one cent. Wrong by
100×, corrected in [INTERVIEW_NOTES_stage1.md](INTERVIEW_NOTES_stage1.md) and
in notebooks 01 and 02. It surfaced because Stage 5's conclusion turns on the
ratio between spread and fee. The Stage 3 claim that the label's dead zone is
"117× the half-spread" was computed in price units and is unaffected.

---

## 5. The ten hardest questions

**Q1. Why touch prices rather than mid?**
The mid is not tradeable — you cross to the ask to buy and hit the bid to sell
— so mid-marking gifts the strategy a full spread per round trip. It is the
most common way a worthless signal is made to look profitable and it is
invisible in the output. Here the spread is tiny (0.0015 bps) so the fee
dominates, but an execution model that errs in the strategy's favour is
worthless regardless of magnitude.
→ `eval.py:simulate_trades_from_prices`, `test_buying_at_the_ask_and_selling_at_the_bid_costs_the_spread`.

**Q2. Why IC rather than accuracy?**
Accuracy discards move magnitude and prediction confidence, so it cannot
separate "51% right on big moves" from "60% right on moves smaller than the
spread". IC keeps both, and Spearman keeps it robust to the heavy tails and
the mass of exactly-zero returns at this timescale.
→ `eval.py:spearman_ic`.

**Q3. Pooled IC 0.42, per-block mean 0.07. Which do you report?**
The per-block one, with its distribution. Pooling lets slow common drift across
the fold inflate the correlation; per-block removes that and measures whether
the signal ranks correctly within a period, which is the only version anyone
can trade. Reporting the pooled figure alone would have overstated the edge
sixfold.

**Q4. Why does your decay curve rise before it falls?**
Because the label is smoothed over ~2.7 s, so the model predicts an average
over the next few seconds rather than the next tick, and the IC peaks near the
horizon it was trained on. That is a property of the Stage 3 label choice, not
a discovery about the market. It is also why the half-life is fitted from the
peak — fitting across the rise would describe neither limb.

**Q5. Is a 13-second half-life good or bad?**
Neither; it is a fact that determines what the signal is for. It means latency
is almost irrelevant to *this* signal — a few milliseconds costs a negligible
fraction of 13 seconds — so I will not claim Stage 7's microsecond work
rescues it. Latency matters for a maker deciding whether to pull a quote, and
for signals with microsecond half-lives. Stage 7 is the engineering
prerequisite for that class of strategy, not a fix for a 70× fee shortfall.

**Q6. Your win rate is 0.000 at a 1 bp fee. Is that a bug?**
No, and it is the most informative number in the sweep. The gross edge is
0.285 bps per trade against a 2 bps round-trip cost, so the fee exceeds the
edge on *every individual trade*, not merely on average. There is no threshold
or filter that recovers a profitable subset. That is the signature of being the
wrong order of magnitude rather than merely unlucky.

**Q7. Why is a negative after-cost result a strength?**
Because the alternative was never a profitable strategy, it was a *wrong* one.
Mid-price fills, a random split, a scaler fitted on all the data, or a
threshold tuned on the test set would each have produced a positive number that
would not survive a real exchange — and each is a mistake that gets caught in
an interview rather than in production. "0.14 bps against a 4 bps floor" is
actionable: stop optimising the model, change the venue economics or become a
maker. A fabricated Sharpe forecloses that decision.

**Q8. Your shifted control returns +0.30. How is that not leakage?**
Because it decays. A leak is tied to one alignment and would hold near the true
IC at every shift; this collapses to +0.03 by 300 rows and oscillates around
zero beyond. The signal is autocorrelated and genuinely predicts seconds ahead,
which is exactly what a persistent-but-honest signal looks like. The control
that settles it is the permutation test: +4.3σ above a 200-draw null.

**Q9. Who could monetise this?**
Not a taker, at any published tier — that is arithmetic, not skill. A **market
maker** could use it: the entire loss here is the cost of crossing, and a
passive quoter earns the spread instead of paying it, so a 0.285 bps
directional edge is a usable input to quote skew. Also participants with
exchange market-making fee exemptions, or the same signal on a wider-spread
instrument. All of those are different businesses, not this one with lower
costs.

**Q10. What is not measured, and what would you do next?**
**Capacity** is the big gap: every fill assumes the touch absorbs our size,
which is false at 2,788 trades over ~5 minutes, so all figures are an upper
bound. Next, in order: (a) capture far more data — 21 minutes across three
sessions is a pilot, and the per-block IC std of 0.35 is partly small-sample;
(b) rebuild the analysis for a **maker** fill model, since that is where the
economics could work; (c) add a queue-position model before believing any maker
number, because a passive fill you never got is a worse fiction than a
mid-price fill.

---

## 6. Method notes

- All figures on the **held-out test fold only** (samples 27,873–37,164 of the
  walk-forward split), with the Stage 3 embargo of 199 samples in force.
- The fold is read from the training run's own record rather than recomputed,
  so the block evaluated is byte-identical to the one the model never saw.
- The test fold spans two capture sessions. Forward returns and fills are
  resolved **per session**: no window, label, or held position crosses a resync
  gap, because the gap contains price moves nobody observed.
  → `eval.py:resolve_prices_per_session`, `pooled_forward_returns`.
- Confidence threshold for trading is the 70th percentile of |s| on the test
  fold. This is a *characterisation* choice, not a tuned parameter — the
  breakeven fee moves with it, and no threshold makes the signal survive 4 bps.
- Binance taker tiers checked 2026-07-28 against
  <https://www.binance.com/en/fee/schedule>: VIP 0 10 bps, VIP 0 + BNB 7.5 bps,
  VIP 9 4 bps. Pinned by `test_the_quoted_binance_tiers_are_the_ones_we_cite`.
