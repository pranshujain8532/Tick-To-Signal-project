# Interview notes — Stage 4: the model, trained and honestly benchmarked

What shipped: a 319,715-parameter CNN + Inception + TCN written from scratch,
a plain PyTorch training loop with embargoed walk-forward splits and a
carved-off validation block, three baselines, an FI-2010 loader for a
literature-comparable number, and 35 new tests (148 total).

**Two headline numbers, and the pair is more informative than either alone.**

On our own captured data the deep model scores **macro-F1 0.549** against
**0.532** for multinomial logistic regression on a single snapshot — 320,000
parameters and a 100-step window bought **+1.7 points** over 123 parameters
and no history. That is a weak result and it is reported as one.

On FI-2010 the *identical* model scores **0.751**, which sits between the
published B(TABL) 0.692 and C(TABL) 0.776 on the same split. The architecture
is therefore not the bottleneck; the 23,324-sample training set is. Together
the two numbers say something more useful than either would separately: build
the capture pipeline first, then the model — which is the order this project
happens to have taken.

Files: [ml/model.py](../ml/model.py), [ml/train.py](../ml/train.py),
[ml/dataset.py](../ml/dataset.py), [ml/baseline.py](../ml/baseline.py),
[ml/fi2010.py](../ml/fi2010.py), [ml/metrics.py](../ml/metrics.py),
[tests/test_model.py](../tests/test_model.py),
[tests/test_dataset.py](../tests/test_dataset.py),
[notebooks/04_model_training.ipynb](../notebooks/04_model_training.ipynb).

Measured runs: `benchmarks/train_ours_btcusdt_20260728T102758Z.json`,
`benchmarks/baselines_20260728T102430Z.json`.

---

## 1. How each stage maps onto microstructure

The input is `[100, 40]`, laid out as `[bid_price, bid_size, ask_price,
ask_size]` per level for ten levels. The front end collapses the 40-wide axis
in three steps, and each one is a claim about the market, not just a shape.

| Stage | Kernel | Width | Fuses | The microstructure claim |
|---|---|---|---|---|
| `fuse_price_and_size` | `(1,2)` stride 2 | 40 → 20 | price with size | A level is a *pair*. "There is size at this price" is the atom; neither half means anything alone. After this, one number per side per level. |
| `fuse_sides` | `(1,2)` stride 2 | 20 → 10 | bid against ask | Per-level **imbalance** — how much rests on one side versus the other at the same depth. This is the strongest cheap predictor in microstructure, and the measured queue-imbalance baseline (0.395 macro-F1 from a single threshold) shows why. |
| `fuse_levels` | `(1,10)` | 10 → 1 | the ten levels | Whole-book **shape**: is depth stacked at the touch, or is the book thin in front and heavy behind? A thin touch in front of size is what precedes a move. |

Between each, two causal width-4 convolutions look along *time* at that width.
Separating "mix levels" from "mix time" is the DeepLOB idea and it is what
keeps each layer explainable — a single 3D kernel would do both at once and be
impossible to attribute.

The feature ordering is load-bearing: because the four numbers describing one
level are adjacent, a width-2 kernel sees exactly a (price, size) pair. Group
the features by field instead and the same kernel would fuse the price of
level 1 with the price of level 2, which means nothing.
→ `ml/model.py:LevelFusionBlock`, `ml/features.py` layout decision.

## 2. The receptive field, and why it is the number that justifies the TCN

**83 of the 100 timesteps**, and it is *measured*, not asserted:
differentiate the output with respect to the input and see which timesteps
have a non-zero gradient (`test_the_receptive_field_is_exactly_as_advertised`).

It decomposes as:

```
  front end   3 blocks x 2 convs x (4-1)          = 18
  inception   widest branch, kernel 5, (5-1)      =  4
  TCN         2 convs per block x (3-1) x d,
              d in (1, 2, 4, 8)  -> 2*2*(1+2+4+8) = 60
  current step                                    =  1
                                                   ---
                                                    83
```

Two consequences worth stating out loud. **Seventeen timesteps of every window
are dead weight** — the model cannot see them, so we feed 100 and use 83.
And the gradient is heavily concentrated on the most recent steps, which is
what a microstructure model *should* look like. If Stage 5's measured signal
half-life exceeds 83 events, this architecture needs a fifth dilation level,
and that decision gets made in writing rather than quietly.

## 3. Design decisions and the alternatives rejected

### 3.1 TCN, not LSTM — `model.py` module docstring

Rejected the canonical DeepLOB choice for three reasons, in the order they
matter here: (1) a recurrent step is sequentially dependent, so T timesteps
means T dependent matmuls, where a dilated stack is a fixed sequence of GEMMs
with a statically known access pattern — the thing that makes Stage 7's C++
path tractable; (2) causality everywhere allows **incremental inference** from
a ring buffer, advancing one tick instead of recomputing a 100×40 forward
pass; (3) reproducing convolution arithmetic between PyTorch and hand-written
C++ is doable, reproducing fused cuDNN recurrent kernels is not, and the
Stage 7 parity test is mandatory. Cost accepted: a bounded receptive field.

### 3.2 No transformer

At 100 timesteps and this parameter budget, attention is O(T²) where the
dilated stack is O(T), has no incremental form (every tick re-attends over the
whole window, destroying the ring-buffer trick), and adds nothing in a regime
where local structure dominates. We care about the p99, and attention's p99 is
worse for no accuracy we could point to.

### 3.3 Causal padding everywhere, with one honest caveat

Causality is **not** needed to prevent leakage — the window already ends at t
and the label looks forward from t, so `same` padding would leak nothing. It
is needed for incremental inference. The caveat: `BatchNorm` normalises over
batch *and time*, so during training the activation at step t is influenced by
the whole window. At inference it uses frozen running statistics, so the
deployed model — the one Stage 7 reimplements — is strictly causal.

### 3.4 Read out the last timestep, not a pooled average

Pooling would destroy the O(1) incremental update: a pooled value changes as
old timesteps expire, so it cannot be advanced in constant time. Reading the
final position also keeps the output a function of "state right now", which is
the semantically correct thing to predict from.

### 3.5 Early stopping on macro-F1, not on validation loss — vindicated by the data

Rejected the usual default. The measured curves show exactly why: training
loss collapses to 0.028 by epoch 8 while validation loss *rises* from 1.49 to
2.42, yet validation macro-F1 keeps climbing to its best value at epoch 8.
A run stopping on loss would have quit at epoch 1 with macro-F1 0.529 instead
of 0.657. The two metrics disagree because cross-entropy punishes confident
errors, and a model growing more confident about the majority class gets worse
loss while getting better at the minority classes — which is what macro-F1
rewards and what a trading signal needs.

### 3.6 A validation block carved out of train — a flaw I introduced and fixed

The first version early-stopped on the fold's test block and then reported
that same block. That is a quiet form of fitting to the test set: the stopping
epoch is selected by looking at the reported number, so the score is a maximum
over ~40 peeks rather than an unbiased estimate. Fixed by taking the last 15%
of the training block as validation with its own embargo, leaving the test
block untouched until the run ends.
→ `ml/train.py:carve_validation`.

### 3.7 Class-weighted loss, not resampling

Oversampling would duplicate temporally adjacent windows into the same epoch
and quietly reintroduce exactly the near-duplicate problem the Stage 3 embargo
exists to prevent. Weighting changes the gradient without changing the sample
set.

### 3.8 Determinism is off by default, and the run records that

Measured on this machine: `cudnn.deterministic=True` runs at **56 samples/s**
against **784 samples/s** with `cudnn.benchmark=True` — a **14× penalty** that
turns a 24-minute run into 5.5 hours. Seeds for Python, numpy and torch are
always set, so data order and initialisation are reproducible; only cuDNN's
kernel *selection* varies. Every saved run records the flag, because a number
is only as reproducible as the setting next to it says.
→ `ml/train.py:seed_everything`.

### 3.9 Batched gather instead of a per-sample Dataset

Per-sample `__getitem__` measured ~600 samples/s, which would have made
FI-2010 seven minutes an epoch. Because every window is a contiguous run of
rows, a whole batch is one fancy-index gather: **99,631 samples/s**, and the
data loading stopped being the bottleneck. `test_batched_gather_matches_the_per_sample_path_exactly`
pins the two paths together so the optimisation cannot silently change what
the model trains on.

---

## 4. The ten hardest questions

**Q1. Your model beats logistic regression by 1.7 points. Was it worth building?**
On our own data, on this evidence, barely — and saying so is the result. Most
of the achievable score lives in the instantaneous shape of the book, which
123 parameters already capture. But the FI-2010 run is the control experiment
that separates "the architecture is weak" from "the dataset is small": the
same model, loop and hyperparameters reach 0.751 there, ahead of published
B(TABL), on 216,255 training samples. So the honest reading is that the
architecture works and our capture is too short — which makes the next action
a longer capture, not a redesign. The system story (capture → format →
leakage-free labels → C++ inference) stands regardless; the *signal* story on
our own data is weak and is reported weak.

**Q2. Your validation score was 0.657 but test was 0.549. What is that gap?**
An 11-point generalisation gap across a 199-sample embargo, which says the
model is fitting session-specific structure rather than something durable.
Validation is the tail of the training block — temporally adjacent to training
data even after the embargo — while the test block sits further away and
includes a different capture session with a visibly different class balance
(session 1 is 0.239/0.359/0.402, session 2 is 0.394/0.267/0.339). Part of the
gap is genuine overfitting; part is distribution shift between sessions. Both
point at the same fix, which is more data.

**Q3. Training loss hits 0.028 by epoch 8. Isn't that a broken setup?**
It is severe overfitting, and it is expected given the sample geometry: 23,324
training windows that overlap by 99 of 100 rows are nothing like 23,324
independent samples — the effective sample size is closer to the number of
*non-overlapping* windows, around 233. A 320,000-parameter model memorises
that almost immediately. The mitigations in place are early stopping, weight
decay and dropout; the real fix is more data.

**Q4. How do you know the model can learn at all, independently of the data?**
The overfit-one-batch test: 32 samples with **random** labels, and the model
must reach 100% within 200 steps. Random labels are deliberate — there is no
pattern to find, so success proves only that gradients flow everywhere and the
optimiser can move the model where the loss demands. It separates "the
plumbing is broken" from "the problem is hard", which look identical on a
learning curve.
→ `tests/test_model.py:test_model_can_overfit_a_single_batch`.

**Q5. Why is macro-F1 the headline rather than accuracy?**
Because accuracy is actively misleading on a deadbanded three-class target: on
this test block, always predicting the majority class gives 33.1% accuracy but
macro-F1 of 0.166, because two of the three per-class F1 scores are zero.
Macro averaging weights each class equally, so a model cannot buy a good score
by ignoring the rare-but-tradeable classes. Our per-class F1 (down 0.601,
flat 0.407, up 0.640) also shows where the model is weakest — the flat class,
which is the one the dead zone defines.

**Q6. Your baselines — did you build them before or after the model?**
Before, and the ordering matters. A baseline computed after you know what you
need to beat is not a baseline. `benchmarks/bench_baselines.py` imports
`carve_validation` from `ml/train.py` rather than reimplementing the split,
because two copies of a splitting rule is exactly how a comparison quietly
stops being fair — both models fit on the same 23,324 samples and are scored
on the same untouched 9,292.

**Q7. Why does the queue-imbalance rule need a dead zone chosen on train?**
Because choosing it on test would be the same look-ahead sin the whole project
is built to avoid, in miniature — a baseline tuned on the test set is a second
model with an unfair advantage. The threshold (0.866) is picked by scanning
quantiles of |imbalance| on the training block only, and it transfers: train
macro-F1 0.365 against test 0.395.

**Q8. What does FI-2010 add that your own data does not?**
Comparability — and, as it turned out, a diagnosis. Every other number here is
measured on a tape only we have, so "macro-F1 0.549" is unfalsifiable by a
reader; FI-2010 is the standard public benchmark, so the same architecture on
it produces a number that can be placed against published work. It scored
**0.751**, between B(TABL) 0.692 and C(TABL) 0.776 on the same split. That
tells us the architecture is not what is limiting the BTCUSDT result: the same
model, same loop and same hyperparameters gained 20 points when given 216,255
training samples instead of 23,324. We use the 40 raw LOB features and the
dataset's own normalisation and division, because using the engineered
features or re-normalising would make the number better and meaningless.

**Q9. How is the training run reproducible, and how is it not?**
Python, numpy and torch seeds are fixed, so weight initialisation, dropout
masks and batch order are identical run to run. cuDNN kernel selection is not
pinned by default, because it costs 14× — so two runs agree on everything
except the last few decimal places of floating-point accumulation. The flag is
recorded in the saved JSON alongside device, GPU model, config, data
provenance and per-epoch history, so any number can be traced to the exact
conditions that produced it. `--deterministic` makes it bit-exact when that
matters, such as chasing a Stage 7 parity discrepancy.

**Q10. If this F1 were five points lower, what would you try, in order?**
1. **More data first, before touching the model.** 21 minutes of capture is a
   pilot. The effective sample size is ~233 independent windows; nothing about
   architecture is diagnosable at that scale, and the 11-point val/test gap is
   the signature of a data problem rather than a capacity problem.
2. **Shrink the model, not grow it.** With train loss at 0.028 the binding
   constraint is variance, not bias. Halve the channel widths (~80K
   parameters), raise dropout from 0.1 to 0.3, raise weight decay. If a
   smaller model scores the same, that is the answer and it also makes
   Stages 6 and 7 easier.
3. **Sample windows with a stride.** Training on every overlapping window
   feeds the same information a hundred times. A stride of 10 gives a tenth
   the samples with far more independent information per epoch.
4. **Check the label before the model.** Re-derive alpha per session rather
   than globally — Stage 3 measured the class balance moving from 1.08 to 1.65
   imbalance across sessions an hour apart, so part of what looks like model
   failure may be the target drifting.
5. **Only then architecture:** a fifth dilation level if the Stage 5 half-life
   says the 83-step field is too short, or dropping the inception block if
   ablation shows it earns nothing.

---

## 5. What Stage 4 measured

Machine: NVIDIA GeForce RTX 2050, Windows 11 (10.0.26200), Python 3.10.20,
torch 2.3.1+cu118. Non-deterministic cuDNN (`--deterministic` off), seed
20260728, batch 128, AdamW lr 1e-3, weight decay 1e-4, cosine schedule with 5%
warmup, patience 8.

Data: 3 BTCUSDT capture sessions, 37,165 labelled samples
(down 0.295 / flat 0.337 / up 0.369). Fold 2 of 3 walk-forward:
fit 23,324 → validation 4,151 → **test 9,292**, embargo 199 at every boundary.

### Results on the held-out test block

| Model | Input | Parameters | macro-F1 | accuracy |
|---|---|---|---|---|
| Majority class | — | 0 | 0.1657 | 0.3308 |
| Queue imbalance + dead zone | last snapshot | 1 | 0.3950 | 0.3974 |
| Logistic regression | last snapshot | 123 | 0.5317 | 0.5298 |
| **CNN + Inception + TCN** | **100 × 40 window** | **319,715** | **0.5493** | **0.5506** |

Per-class F1 for the deep model: down 0.601, flat 0.407, up 0.640. The flat
class is the weakest, which is the class the dead zone defines and the hardest
to call.

### Training dynamics

| | |
|---|---|
| Best validation macro-F1 | 0.6569 (epoch 8) |
| Epochs run | 17 of 40 (early stopped, patience 8) |
| Test macro-F1 at that checkpoint | 0.5493 |
| Validation → test gap | **−0.108** |
| Train loss at best epoch | 0.028 |
| Validation loss at best epoch | 1.779 (rising from 1.487 at epoch 0) |
| Epoch time | ~16.4 s |

### Throughput measurements taken along the way

| Path | Rate |
|---|---|
| Per-sample `Dataset.__getitem__` | ~600 samples/s |
| Batched fancy-index gather | 99,631 samples/s |
| Training, `cudnn.deterministic=True` | 56 samples/s |
| Training, `cudnn.benchmark=True` | 784 samples/s |

### FI-2010 — the literature-comparable number

Same architecture, same training loop, horizon k=10, `NoAuction_DecPre` with
the 40 raw LOB features. 216,255 fit / 38,197 validation / 139,488 test
samples; early stopped at epoch 14 of 15, best validation at epoch 5.

**Test macro-F1 0.7515** (accuracy 0.7531). Per-class F1: down 0.728,
flat 0.801, up 0.725.

Against published results on the same 7-training-day / final-3-day division
(DeepLOB, Zhang, Zohren & Roberts, [arXiv:1808.03668](https://arxiv.org/abs/1808.03668),
Table II, Setup 2), F1 %:

| Model | F1 % |
|---|---|
| CNN-I | 55.21 |
| LSTM | 66.33 |
| B(TABL) | 69.20 |
| **ours** | **75.15** |
| C(TABL) | 77.63 |
| DeepLOB | 83.40 |

**Three caveats, without which this comparison is dishonest.** The published
runs use the `Zscore` normalisation where we use `DecPre`; they train far
longer than 14 epochs on a laptop GPU; and the paper's F1 averaging convention
may not be our macro average. This is a floor on what the architecture can do
on this data, not a ranking claim.

**Why this row matters more than its position in that table.** The identical
model scores 0.751 on FI-2010 and 0.549 on our own capture. The architecture
is not the problem — data volume is. FI-2010 supplied 216,255 training samples
against our 23,324, and unlike ours those samples span five instruments and
ten days rather than 21 minutes of one. That is the strongest available
evidence for the "more data before more model" ordering in Q10, and it is why
the next action for Stage 4 is a longer capture rather than an architecture
change.
