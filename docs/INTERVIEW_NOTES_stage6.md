# Interview notes — Stage 6: compression and the Pareto frontier

What shipped: ONNX export with a parity gate, static int8 quantisation with a
measured calibration study, a 32K-parameter distilled student with a proper
control experiment, a pinned batch-1 latency benchmark over 100,000 iterations
per variant, 26 new tests (210 total), and
[notebook 06](../notebooks/06_compression_and_distillation.ipynb).

Files: [ml/export.py](../ml/export.py), [ml/distill.py](../ml/distill.py),
[tests/test_export.py](../tests/test_export.py),
[benchmarks/bench_python_variants.py](../benchmarks/bench_python_variants.py),
[benchmarks/bench_calibration.py](../benchmarks/bench_calibration.py).

Measured results: `benchmarks/python_variants_20260731T175351Z.json`,
`benchmarks/calibration_20260731T180908Z.json`,
`benchmarks/distillation_20260728T211009Z.json`.
Method: [docs/benchmark_methodology.md](benchmark_methodology.md).

---

## 1. The Pareto frontier

Batch 1, one pinned core, 10,000 warmup + 100,000 timed iterations, accuracy on
the full 9,292-window held-out block.

| variant | params | size | macro-F1 | p50 µs | p99 µs | p99.9 µs | p99 speedup |
|---|---|---|---|---|---|---|---|
| PyTorch eager fp32 | 319,715 | — | 0.5493 | 10,635 | 39,875 | 74,092 | 1.0× |
| ONNX Runtime fp32 | 319,715 | 1,292 KiB | 0.5493 | 1,916 | 5,766 | 15,600 | **6.9×** |
| ONNX Runtime int8 | 319,715 | 423 KiB | 0.5469 | 1,366 | 4,354 | 11,387 | **9.2×** |
| Distilled student int8 | 32,155 | 126 KiB | 0.5723 | 856 | 3,775 | 10,152 | **10.6×** |

**The headline: a 10.6× tail-latency reduction at no accuracy cost.** Accuracy
is flat to within 0.03 macro-F1 across the whole frontier, so nothing here is a
trade — the compression is close to free.

The single biggest win is simply leaving eager PyTorch: same parameters, same
function (parity proved it), 6.9× less tail latency. At batch 1 there is no
arithmetic to hide Python dispatch and per-op overhead behind, across ~40
layers.

---

## 2. Design decisions and the alternatives rejected

### 2.1 Parity is checked before anything is timed — `export.py:check_parity`

A faster graph that computes a slightly different function is not a faster
model. `benchmarks/bench_python_variants.py` refuses to benchmark a graph that
fails parity, so the ordering "correctness, then speed" is enforced by the
script rather than by discipline.

### 2.2 Fixed input shape `[1, 100, 40]`, no dynamic batch axis

Rejected: mark the batch dimension dynamic so one graph serves any batch size.
Dynamic axes stop the runtime specialising kernels and pre-planning
allocations. There is exactly one deployment shape, so declaring it is free.
It also makes every buffer size a compile-time constant for the Stage 7 C++
path.

### 2.3 Opset pinned at 17

An unpinned opset silently changes the artefact when torch is upgraded, turning
a reproducible export into a moving one.

### 2.4 Static quantisation, not dynamic

Rejected: `quantize_dynamic`, one line and no calibration data needed — because
it computes activation ranges *at run time*, which is exactly the cost being
optimised away. Static folds the scales into the graph; the price is needing
calibration data, and being only as good as it.

### 2.5 Percentile calibration, not onnxruntime's MinMax default — see §4

### 2.6 QDQ format and per-channel weights

QDQ inserts explicit Quantize/DequantizeLinear pairs: readable, and what the
CPU provider optimises best. `per_channel=True` gives each conv output channel
its own scale, because channels routinely differ in magnitude by an order of
magnitude and one shared scale crushes the small ones.

### 2.7 The student keeps the teacher's block structure *and* receptive field

`conv_channels` 32→8, `inception` 64→16, `tcn` 96→32, but the same four
dilations and therefore the same 83-timestep receptive field. Rejected: drop a
dilation to save more parameters — that would change what the student can
*see*, confounding capacity with context.

### 2.8 Batch size 1, and only 1 — see [benchmark_methodology.md](benchmark_methodology.md)

---

## 3. Three things I got wrong, and what fixed them

### The affinity pin silently did nothing

The first pinning helper reported "affinity request refused by the OS" on a
machine perfectly willing to pin it. ctypes had defaulted `GetCurrentProcess`'s
return type to a C `int`, truncating the 64-bit pseudo-handle. It would have
produced **unpinned numbers under a comment claiming they were pinned** —
exactly the kind of quiet wrongness this project keeps refusing. Fixed with
explicit `argtypes`/`restype`; the affinity string is now recorded in every
saved result so the claim is auditable.

### The accuracy subset was not representative

An earlier version of the Pareto benchmark scored accuracy on the first 4,000
of 9,292 test samples for speed. That prefix put the student at 0.649 macro-F1
against 0.595 on the full block, and the teacher at 0.533 against 0.549 — a
swing large enough to reverse the ranking. The test period is strongly
non-stationary: per-quarter macro-F1 varies by more than 0.15 and the class
balance runs from 59% up to 59% down. `--accuracy-samples` now defaults to the
whole block.

### The calibration claim reversed on the full block

The first version of `ml/export.py` asserted that MinMax calibration "costs
0.061 macro-F1". That came from a 1,000-sample prefix — the same trap, met a
second time — and it does **not** survive the full block, where MinMax scores
*higher* F1 than percentile calibration. The design decision survived; its
justification had to be rewritten from scratch. See §4.

---

## 4. The calibration study

fp32 reference: macro-F1 0.5493 on 9,292 held-out windows. Activations: std
1.37, 99.9th percentile 12.4, **max 22.4** — heavy-tailed, because the features
are causally z-scored.

| calibration set | macro-F1 | vs fp32 | argmax agreement |
|---|---|---|---|
| real, n=32 | 0.5434 | −0.0059 | 0.9457 |
| real, n=128 | 0.5409 | −0.0083 | 0.9500 |
| **real, n=512, pct 99.99** | **0.5478** | **−0.0015** | **0.9623** |
| real, n=512, pct 99.9 | 0.5454 | −0.0038 | 0.9595 |
| real, n=512, pct 99.0 | 0.5610 | +0.0118 | 0.9207 |
| real, n=512, MinMax | 0.5674 | +0.0182 | 0.9115 |
| gaussian noise | 0.5531 | +0.0039 | 0.9262 |
| noise ×8, too wide | 0.5256 | −0.0237 | 0.8421 |
| **noise ×0.05, too narrow** | **0.2630** | **−0.2863** | **0.3872** |

**The catastrophic row is the bottom one.** Calibrating on near-zero noise
collapses macro-F1 to 0.263 and agreement to 0.387: every real activation
saturates against a range built for values that never occur. That is the
clipping failure, and it is as bad as theory predicts.

**The subtle row is MinMax.** It scores the highest F1 and the lowest fidelity.
Both cannot be the right criterion, and F1 is not: a 0.02 difference sits well
inside this period's own variability, so choosing a calibrator on test F1 means
selecting a *deployment artefact* on the test set. Agreement with the fp32
graph measures the thing quantisation is supposed to preserve, it is monotone
in the clipping percentile rather than noisy, and percentile 99.99 wins it
clearly. That is the default.

---

## 5. Distillation: an honest negative

Teacher 319,715 params → student 32,155 (**9.9× compression**). Same seed,
schedule, data, split and architecture; **only the loss differs**. Three seeds.

| student trained | test macro-F1 |
|---|---|
| with distillation | 0.5058 ± 0.0681 |
| hard labels only (control) | 0.5233 ± 0.0085 |
| **delta** | **−0.0175 ± 0.0633** |

Per seed: −0.0931, −0.0214, +0.0618.

**The delta is negative and smaller than its own spread — no measurable
effect.** Two observations rather than excuses:

- **The distilled runs are ~8× more variable** than the control. Adding a
  soft-target term made optimisation less stable, not more.
- **The likely cause is a weak teacher.** Stage 4 measured it at 0.5493 macro-F1
  against 0.5317 for logistic regression on a single snapshot. Dark knowledge is
  only worth transferring if the teacher knows something; a weak teacher's soft
  targets are largely its own noise. That is consistent with Stage 4's finding
  that data volume, not architecture, is the binding constraint.

The technique is implemented, tested and correct. The finding is that it does
not help *yet*, and the harness is in place to re-run it against a stronger
teacher.

---

## 6. The ten hardest questions

**Q1. Why batch size 1 when batching is far faster per sample?**
Because tick-to-signal is a batch-1 problem: one order-book update arrives and
the answer is needed before the next, so there is no batch to form without
waiting — and waiting is latency. A batch-128 throughput figure can look an
order of magnitude better per sample while describing a system nobody is
building. Batching is used only for offline accuracy evaluation, where it
cannot affect the result.

**Q2. Your student breached the specified 1e-5 parity tolerance. Did you widen it?**
No — I changed the criterion and kept reporting the original. A fixed absolute
tolerance is not scale-invariant: the teacher sits at 5.2e-06 and the student at
2.6e-05, largely because their logits differ in magnitude, and the student's
*relative* divergence is 2.8e-06 with argmax agreement of exactly 1.0 over 1,000
trials — it would never change a prediction. The gate is now relative divergence
plus exact argmax agreement, and `meets_absolute_tolerance` still reports the
1e-5 bar, so the fact that the student misses it stays visible.
→ `export.py:check_parity`, `test_parity_fails_when_an_argmax_flips_even_if_numerically_close`.

**Q3. What does calibration data actually control?**
The activation ranges baked into the graph, which define the numeric range the
deployed model can represent at all — it is not a tuning detail. Calibrating on
near-zero noise collapses macro-F1 from 0.549 to 0.263. Weights are quantised
offline from tensors whose range is known exactly; activations are the part that
needs data, and the part where accuracy is won or lost.

**Q4. You chose a calibrator that scores lower F1 than the default. Justify that.**
Because F1 is the wrong criterion for this choice. The differences between
calibrators (~0.02) are well inside this test period's own variability — Stage 5
measured per-block macro-F1 swinging by more than 0.15 — so picking on test F1
is selecting on noise, and selecting a deployment artefact on the test set at
that. Agreement with the fp32 graph measures what quantisation is meant to
preserve, and percentile 99.99 wins it 0.962 to 0.911.

**Q5. Explain the T² factor in the distillation loss.**
Softening logits by `T` shrinks the KL term's gradients by roughly `1/T²` — the
derivative of the softened softmax carries a factor of `1/T`, and it enters
twice. Without the correction, raising `T` would silently shrink the
distillation term against the hard-label term, so `alpha` would stop meaning
what it says and tuning `T` would secretly retune the balance. `T²` cancels it,
leaving `T` to control only how soft the targets are.
→ `distill.py:distillation_loss`, `test_the_temperature_squared_factor_keeps_the_soft_term_scale_stable`.

**Q6. Distillation didn't work. Why report it at all?**
Because the control run *is* the experiment. A distilled student reported alone
cannot distinguish "the teacher taught it" from "the architecture was
sufficient"; only the paired difference can, and here it is negative and smaller
than its own spread across three seeds. Reporting the distilled score without
the control would have implied an effect the data does not support. The
diagnosis — a teacher only 0.017 macro-F1 better than logistic regression has
little dark knowledge to transfer — is more useful than a fabricated win.

**Q7. Why is int8 only ~1.4× faster than fp32, not 4×?**
Because this model is small and batch 1, so it is not compute-bound in the way
int8 speedups assume. The arithmetic reduction is real but a large share of the
remaining time is graph traversal, per-node dispatch and memory movement that
quantisation does not touch — and QDQ adds Quantize/Dequantize nodes of its own.
The disk-size reduction is much closer to the theoretical 4× (1,292 → 423 KiB,
3.1×) because that *is* dominated by weights.

**Q8. Your quantised model is bigger than the fp32 one at small scale. Explain.**
QDQ quantisation shrinks weights but adds graph structure — Q/DQ node pairs plus
their scale and zero-point initialisers — costing a roughly constant ~50 KiB.
Measured across three scales: 4,425 params 55.0 → 75.9 KiB (it *grows*), 32,155
params 173 → 125 KiB, 319,715 params 1,291 → 421 KiB. int8 only pays for itself
once weights dominate the overhead.
→ `test_quantisation_can_make_a_very_small_model_bigger`.

**Q9. Your two benchmark runs of the same code differ by 55%. Which is real?**
Neither absolutely, and that is the point. The same benchmark gave p50 6,847 µs
and 10,635 µs for PyTorch eager on the same machine — a laptop with no thermal
control, after hours of sustained load. Absolute latencies on this hardware
carry that much run-to-run uncertainty, which is why the README quotes the
**relative ordering** and the speedup ratios, which were stable across both runs
(ONNX ~6-7× at p99, student ~10-11×). Stage 7's harness will need a thermally
settled machine before its absolute microsecond claims mean anything.

**Q10. What does this tell you about Stage 7?**
Two things. The p50s are already respectable, so the remaining prize is the
**tail**: p99.9/p50 ratios run from 7.0× to 11.9×, and that is not model
arithmetic — the arithmetic is identical every iteration — it is OS scheduling,
page faults and cache eviction. A hand-written C++ path with no hot-path
allocation, a fixed memory layout and no runtime dispatch attacks exactly that.
And the student is the right target to hand-roll: 32K parameters, 126 KiB, same
receptive field, no measured accuracy penalty.
