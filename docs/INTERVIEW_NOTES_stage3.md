# Interview notes — Stage 3: dataset, labels, and leakage prevention

What shipped: causal feature construction, DeepLOB-style smoothed labels with
a data-derived dead zone, anchored walk-forward splits with an embargo that
covers the feature window as well as the label horizon, and 47 new tests
(113 total) including a mandatory leakage test that quantifies the leak rather
than merely asserting its absence.

Files: [ml/features.py](../ml/features.py), [ml/labels.py](../ml/labels.py),
[ml/splits.py](../ml/splits.py), [tests/test_features.py](../tests/test_features.py),
[tests/test_labels.py](../tests/test_labels.py),
[notebooks/03_dataset_and_labels.ipynb](../notebooks/03_dataset_and_labels.ipynb).

Data: three BTCUSDT capture sessions, 2026-07-27, 39,259 snapshots at
`snapshot_interval=10` (~29 snapshots/s). Provenance in
[notebooks/sample_data/README.md](../notebooks/sample_data/README.md).

---

## 1. Design decisions and the alternatives rejected

### 1.1 Prices as offsets from the current mid — `features.py:build_feature_matrix`

Rejected: raw level prices. Over a day BTCUSDT moves thousands of dollars, so a
raw price is a number that never repeats; the network would spend capacity
memorising a range rather than learning book shape. Offsets are comparable
across time and price regimes.

Verified rather than asserted: shifting the whole book by $500 changes the
feature matrix by **exactly zero**, because offsets are integer differences
against an integer mid before anything becomes a float
(`test_price_offsets_are_relative_so_a_level_shift_changes_nothing`).

### 1.2 `log1p` on sizes — `features.py`

Resting size is heavy-tailed: most levels hold a fraction of a coin, a few hold
hundreds. Raw sizes let one whale order dominate the input scale. `log1p`
compresses the tail *and* maps zero to zero, so an absent level stays exactly
zero where plain `log` would produce a large negative number.

### 1.3 Features interleaved per level, not grouped by field

Layout is `[bid_price_1, bid_qty_1, ask_price_1, ask_qty_1, bid_price_2, ...]`.
Rejected: all ten bid prices, then all ten bid sizes. Stage 4's convolutional
front end slides filters along this axis, so the four numbers describing the
same level belong adjacent — a small kernel then sees a whole level at once.
Grouping by field would force the first layer to learn a permutation before it
could learn any microstructure.

### 1.4 Causal rolling z-score, never a dataset-wide scaler — `features.py:rolling_zscore`

Rejected: `StandardScaler().fit(all_data)`, the default in most tutorials. The
scaler then has seen the test set, so a test sample is normalised using
statistics that partly describe itself; the reported accuracy is real and
unobtainable. Row `t` is normalised using rows `[t - lookback + 1, t]` only.

Warmup rows are marked invalid, not back-filled: back-filling would mean the
start of every file is normalised differently from the rest, which is a silent
distribution shift rather than a saving.

Implemented with prefix sums (O(T·F) regardless of lookback) in float64.
Rejected `sliding_window_view` for the statistics: clearer to read, but it
materialises `[T, lookback, F]` — 3 GB for a 40,000-row tape.

### 1.5 The mid is kept as *twice* the true mid — `features.py:snapshot_mids_fixed`

The mid of an odd-tick spread is a half tick. Rounding it would put a
systematic bias into every price offset computed from it, so the integer
`bid + ask` is carried and halved only at the point where we leave integer land.

### 1.6 Smoothed labels, not next-tick direction — `labels.py`

The data decides this one. **99.4% of adjacent snapshots have an identical
mid**, and the moves that remain are one tick — 0.01 USDT — which is exactly
the spread. A next-tick label is therefore almost entirely degenerate, and the
part that is not is precisely the width of the cost of acting on it.

Comparing two k-observation means cancels quote flicker and leaves drift. The
cost, stated honestly: the label is smoother than anything tradeable, which is
why Stage 5 marks positions against executable prices rather than against this.

### 1.7 A dead zone, not a binary split

Rejected: two-class up/down. Binary labels force a call on moves smaller than
the spread — moves that cannot be captured after costs — so the model spends
capacity on the least profitable part of the distribution and reports
impressive accuracy for it.

### 1.8 Embargo covers the feature window, not just the label — `splits.py:required_embargo`

The textbook purge (López de Prado) removes training samples whose *label*
reaches into the test period: an embargo of `k`. We use `W + k - 1`.

Measured on our geometry, `W=100, k=100`: the textbook purge leaves **99 raw
rows shared** between train and test. It stops training labels being computed
from test data, but the last training window and the first test window remain
near-identical inputs, so a model can still score by recognition. Since `W` is
as large as `k` here, the minimal embargo removes roughly half the leak while
looking rigorous.

### 1.9 Anchored (expanding) train window, not fixed-width rolling

Rejected: fixed-size training blocks, which make fold scores directly
comparable. Anchored because the honest question is "how well does a model
trained on everything up to now predict what comes next", and because data is
the binding constraint. The consequence — later folds train on more data and so
are not strictly comparable — is real, and is why fold scores get reported
individually rather than only as a mean.

### 1.10 Splits return index ranges, not arrays

Keeps the module free of opinions about storage, and — the actual reason — lets
the leakage test check *index sets* for overlap directly, which is far stronger
than checking that two arrays happen to differ.

### 1.11 `build_windows` keeps a view when it can — `features.py:_contiguous_selection`

Overlapping windows repeat every row 100 times: 13,510 windows is 216 MB
materialised against 2 MB of underlying rows. Boolean masking always copies;
slicing preserves the view. Since invalidity always forms a contiguous prefix
(it comes from the normalisation warmup), the common path slices. The general
case falls back to fancy indexing and pays, because correctness first.

---

## 2. The ten hardest questions

**Q1. Where could look-ahead enter, and what stops it at each point?**
Three places. Normalisation — `rolling_zscore` uses only rows
`[t-lookback+1, t]`, verified by overwriting the future with noise and
demanding a bit-identical past. Labels — confined to `ml/labels.py`, horizon
`[t-k+1, t+k]` stated in the docstring and verified by mutation at `t+k` and
`t+k+1`. Splitting — embargo of `W+k-1`, with a companion test proving one
sample less does leak.
→ `features.py:rolling_zscore`, `labels.py:smoothed_returns`, `splits.py:required_embargo`.

**Q2. How much is the leak actually worth? Show me a number.**
+0.355 accuracy. A 1-nearest-neighbour model — chosen because it can only
memorise — scores **0.960 under a random split of overlapping windows on a pure
random walk**, data constructed so the future is independent of everything
observable. Walk-forward with the embargo scores 0.605. Every point of the
difference is the model finding a neighbouring window sharing 99 of its 100
rows and copying its label.
→ `tests/test_labels.py:test_random_split_leaks_and_walk_forward_with_embargo_does_not`, notebook 03 §8.

**Q3. Why is `W + k - 1` the right embargo, and how do you know it is not just a safe one?**
A sample at `t` reads raw rows `[t - W + 1, t + k]`, so two samples share
information exactly when those spans overlap. The test asserts zero shared rows
at the required embargo *and* that one sample less leaks — proving tightness,
not merely sufficiency. A test that only shows "the embargo works" is satisfied
by any absurdly large gap.
→ `test_one_sample_less_embargo_does_leak`.

**Q4. Your honest split beats the majority baseline on a random walk. Is that a leak?**
No, and this is the subtlest point in the stage. The label is
`(m_+ - m_-)/m_-` and `m_-` is the mean of the previous k mids, which lies
*inside* the feature window. For a martingale `E[m_+ | history] = mid_t`, so
`mid_t - m_-` predicts the label's sign and is fully observable with zero
look-ahead. On a random walk that visible component alone reaches ~0.74
accuracy and correlates 0.69 with the label. It is the label's own arithmetic
showing through, and it sets the floor Stage 4 must beat.
→ `test_the_smoothed_label_is_partly_visible_inside_the_feature_window`.

**Q5. How did you choose k and alpha, and how far do you trust them?**
From a class-balance grid over 4 k-values × 5 alphas on 14,109 captured
snapshots. `k=100, alpha=9e-6` minimises the imbalance ratio at 1.08
(0.345/0.334/0.321). `k` had to be that large because at `k=25` the smoothed
return is *exactly zero* for 61% of rows — no threshold can rebalance a mid
that has not moved. I trust the choice (it is also optimal on a second session)
but not the balance: it ranges 1.08–1.65 across three sessions captured within
an hour, so alpha is fitted to one period's volatility. Hence `TODO(recalibrate)`.
→ `labels.py` constants block, notebook 03 §6.

**Q6. Is the dead zone economically meaningful or just a statistical convenience?**
Both, and the second is a real caveat. It was chosen for class balance, but it
lands at a 0.585 USDT move — about 117× the half-spread — so up/down labels are
moves an order of magnitude larger than the cost of crossing, not spread noise.
That does not make the strategy profitable: Binance's retail taker fee dwarfs
both numbers, which is exactly what Stage 5's breakeven-fee analysis exists to
quantify.

**Q7. Why is `k` defined in snapshots rather than seconds?**
Because snapshots are sampled in event time — one anchor per 10 tape events —
so a horizon of 100 snapshots stretches and shrinks with market activity. That
is the behaviour we want from a microstructure horizon: 3.4 seconds in a busy
session and longer in a quiet one, always the same amount of *market*. The
consequence is that the same `k` is not the same wall-clock horizon across
sessions, which must be stated whenever the horizon is quoted in seconds.

**Q8. What sets the minimum size of your dataset?**
The embargo. With 802 labelled windows and a 199-sample embargo, fold 0 trains
on **one sample** — the embargo eats the entire first block. The same geometry
over one full session's 13,910 samples gives fold 0 3,278 training samples and
costs under 6% of the data. The embargo is therefore a floor on how much data
the project needs, not just a splitting detail.
→ notebook 03 §7, both split tables.

**Q9. You store zeros for absent book levels. Isn't that wrong?**
It is a compromise, and a zero offset does read as "a level sitting at the
mid", which is not what absent means. It is tolerable only because it is never
exercised: `count_absent_levels` reports **0** across the whole sample, since
the book is seeded with 1,000 levels a side and we only take the top 10. The
honest way to handle a defensive branch is to measure how often it fires rather
than assert it cannot — so the count is printed in the notebook.

**Q10. What would you change?**
Three things. Recalibrate alpha per-regime rather than globally, or replace the
fixed threshold with a volatility-scaled one so the balance is stable across
sessions. Capture far more data — three sessions totalling 21 minutes is a
pilot, and the embargo analysis shows why that matters. And add a
train/test class-balance drift check to the split output, since a balance shift
between folds is a distribution shift rather than a modelling failure, and the
two are easy to confuse when reading fold scores.

---

## 3. What Stage 3 measured

All from real captured data, reproduced in notebook 03.

### The structural fact that drives every label decision

| Quantity | Value |
|---|---|
| P(mid unchanged between adjacent snapshots) | **0.991 – 0.994** |
| P(mid moves up) / P(mid moves down) | 0.003 / 0.003 |
| Smallest non-zero mid move | 0.01 USDT — one tick, and equal to the spread |
| Snapshot rate | ~29 / second (`snapshot_interval=10`) |

### Class-balance grid (session 0, 14,109 snapshots)

| k | alpha | down | flat | up | imbalance |
|---|---|---|---|---|---|
| 25 | 1e-6 | 0.181 | 0.666 | 0.153 | 4.34 |
| 50 | 1e-6 | 0.274 | 0.482 | 0.243 | 1.98 |
| 100 | 5e-6 | 0.362 | 0.296 | 0.342 | 1.22 |
| **100** | **9e-6** | **0.345** | **0.334** | **0.321** | **1.08** ← chosen |
| 100 | 2e-5 | 0.292 | 0.452 | 0.256 | 1.77 |
| 200 | 9e-6 | 0.437 | 0.119 | 0.444 | 3.72 |

Full 20-row grid in the notebook. `alpha = 9e-6` is a 0.585 USDT move at a
64,973 mid — 117× the half-spread.

### Stability of the chosen pair across sessions

| Session | Snapshots | down | flat | up | imbalance |
|---|---|---|---|---|---|
| 0 | 14,109 | 0.345 | 0.334 | 0.321 | 1.08 |
| 1 | 19,614 | 0.242 | 0.358 | 0.400 | 1.65 |
| 2 | 5,536 | 0.412 | 0.273 | 0.316 | 1.51 |

The *choice* is stable — `(100, 9e-6)` is also optimal on session 1 — but the
achieved balance is not.

### The leakage measurement

1-nearest-neighbour on 3,881 overlapping windows over a pure random walk
(no signal by construction):

| Split | Accuracy |
|---|---|
| Random split | **0.960** |
| Walk-forward + 199-sample embargo | **0.605** |
| Majority baseline | 0.535 |
| In-window-only rule (no look-ahead) | ~0.74 |

**Leakage inflation: +0.355 accuracy on data containing no signal at all.**

### Raw rows shared between train and test, by embargo

| Embargo | Rationale | Shared rows |
|---|---|---|
| 199 | `W + k - 1` (ours) | **0** |
| 198 | one short | 1 |
| 100 | textbook purge, `k` only | 99 |
| 0 | none | 199 |
