# Benchmark methodology

How every number in the README was measured. The constitution requires this
document because an interviewer will probe each choice, and because a latency
figure without its method is not a measurement — it is an anecdote.

Read this before quoting any number from this project.

---

## 1. What is measured where

| Number | Produced by | Saved as |
|---|---|---|
| Capture throughput, resyncs, cross-checks | `data_engine.capture` heartbeat logs | `benchmarks/stage1_capture_*.md` |
| Storage size and replay throughput | `benchmarks/bench_binfmt.py` | `benchmarks/binfmt_*.json` |
| Baselines (majority, imbalance, logistic) | `benchmarks/bench_baselines.py` | `benchmarks/baselines_*.json` |
| Model macro-F1, training curves | `ml/train.py` | `benchmarks/train_*.json`, `*_history.csv`, `*_curves.png` |
| IC, decay, half-life, breakeven fee | `benchmarks/bench_evaluation.py` | `benchmarks/evaluation_*.json` + 3 PNGs |
| int8 calibration study | `benchmarks/bench_calibration.py` | `benchmarks/calibration_*.json` |
| Distillation delta | `python -m ml.distill --seeds 3` | `benchmarks/distillation_*.json` |
| **Inference latency (the Pareto frontier)** | `benchmarks/bench_python_variants.py` | `benchmarks/python_variants_*.json` |
| **C++ inference latency** | `inference_cpp/bench/bench.cpp` | `benchmarks/cpp_*.json` |
| Hero chart (accuracy vs p99) | `notebooks/07_latency_results_analysis.ipynb` | `benchmarks/hero_latency_vs_f1.png` |

No number is typed by hand into the README or a notebook. Every one is read
back out of a saved artefact, so a reader can check it without rerunning
anything.

---

## 2. Latency measurement — the part that gets probed

This governs `benchmarks/bench_python_variants.py` and, from Stage 7, the C++
harness. All of it is chosen to make the number *reproducible* rather than
flattering.

### Clock

`time.perf_counter_ns()` — the highest-resolution monotonic clock Python
exposes, and monotonic so it cannot step backwards under NTP. Resolution on
Windows is ~100 ns, which is three orders of magnitude below the quantities
being measured, so clock granularity is not a material error source here. It
will be in Stage 7, and that stage documents its own clock separately.

Each iteration is timed **individually** rather than timing a loop of N and
dividing. Dividing gives a mean and throws away the distribution, and the
distribution is the result.

### Batch size: 1, and only 1

This is the honest setting for tick-to-signal and the first thing an
interviewer should ask about. A trading system receives one order-book update
and must answer before the next arrives; there is no batch to form without
waiting, and waiting is latency. Batching amortises weight loading and fills
vector units, so a batch-128 figure can look an order of magnitude better *per
sample* while describing a system nobody is building.

Batching is used in exactly one place — offline accuracy evaluation, where it
cannot affect the result.

### Threading and core affinity

- `torch.set_num_threads(1)` and `torch.set_num_interop_threads(1)`
- onnxruntime `intra_op_num_threads = 1`, `inter_op_num_threads = 1`,
  `ExecutionMode.ORT_SEQUENTIAL`
- Process affinity pinned to one core: `SetProcessAffinityMask` on Windows,
  `os.sched_setaffinity` elsewhere.

Two reasons. At batch 1 the graph is far too small to fill several cores, so
threads add synchronisation cost for no parallel work. And without pinning the
OS migrates the process between cores; every migration costs a cold L1/L2 and
lands squarely in the tail, which is the part being measured.

> **A bug worth recording.** The first version of the pinning helper reported
> "affinity request refused by the OS" on a machine that was perfectly willing
> to pin it. The cause was ctypes defaulting `GetCurrentProcess`'s return type
> to a C `int`, truncating the 64-bit pseudo-handle. It would have produced
> unpinned numbers under a comment claiming they were pinned. `argtypes` and
> `restype` are now set explicitly, and the affinity string is recorded in
> every saved result so the claim is auditable.

### Warmup: 10,000 iterations, discarded

The first calls through any runtime are unrepresentative — lazy allocation,
one-time graph optimisation, kernel selection, cold instruction cache, and a
CPU still ramping its clock. Ten thousand is far past where those settle for a
model this size.

### Sample count: 100,000 timed iterations

Enough that the p99.9 is estimated from ~100 observations rather than one or
two. A p99.9 quoted from 1,000 iterations is a single sample and should not be
believed.

### Reported statistics: p50, p99, p99.9 — never a mean alone

A mean latency is a number no single inference ever experiences, and it hides
exactly the behaviour a deadline-sensitive system cares about. The mean is
recorded in the JSON for completeness and is not quoted in the README.

The full distribution is saved as a 200-bin histogram (clipped at the 99.99th
percentile so one outlier cannot flatten the plot), so the shape survives
without a 100,000-row file.

### Coordinated omission

This benchmark measures **service time**, not response time under load: it
issues the next inference as soon as the previous one returns, so it cannot
observe queueing delay that a real arrival process would produce. That is the
right choice for characterising the model itself, and it is a real limitation
of the number — a system fed by a live feed at 30+ messages/second could see
worse tails than these. Stated here rather than left for someone to find.

### What is NOT controlled

- A laptop (13th Gen Intel Core i5-13420H) running a desktop OS
- No thermal control; the CPU may downclock under sustained load
- No core isolation (`isolcpus`), no real-time scheduling priority
- Other user processes alive
- Turbo/boost states not pinned

Absolute numbers would improve on tuned hardware. **The relative ordering of
the variants is the durable result**, and that is how the README presents it.

---

## 3. The C++ latency harness (Stage 7b)

`inference_cpp/bench/bench.cpp`, with the statistics in `bench_report.hpp` and
the clock in `bench_clock.hpp`. Results land in `benchmarks/cpp_*.json`.

Everything in §2 still applies — batch 1, individual timing, no mean alone —
so this section covers only what is different or new.

### The clock, and a defect it caught

**Measure the clock's real resolution and print it before any number that
depends on it.** This is the first thing `bench` does, and on the first run it
paid for itself:

```
steady_clock: nominal period 1/1000000000 s, is_steady 1
              measured tick 997000 ns, timer pair overhead 0 ns
zero-duration samples: 19518        (of 20,000)
```

`std::chrono::steady_clock` on MinGW GCC 6.3 **advertises a one-nanosecond
period and resolves about one millisecond.** A `period` is a compile-time
ratio; nothing in the standard requires it to describe the hardware. Nineteen
thousand of twenty thousand forward passes had apparently taken zero time, and
the "p50 5,000 µs" the run reported was a 1 ms grid rather than a measurement.

The fix is `bench_clock.hpp`: `QueryPerformanceCounter` on Windows,
`clock_gettime(CLOCK_MONOTONIC)` on POSIX. Measured tick afterwards: **100 ns**,
timer-pair overhead below one tick. Both clocks are still measured on every run
and both go into the JSON (`clock_tick_ns`, `steady_clock_tick_ns`), so the
record carries the evidence rather than a comment asserting it.

The loop stores raw counter ticks and converts to nanoseconds after the run —
a divide between the two clock reads would be harness arithmetic charged to the
model.

`zero_samples` is in every record. If it is not 0, the clock was too coarse for
what was being timed and the run is void.

### What is inside the timed region — the tick-to-signal boundary

**Included:** the model forward pass, from a prepared `[40]` feature column (or
a `[100, 40]` window) to three logits. For the `full` variant that includes the
16 KB input `memcpy` that `StudentModel::forward` performs; the incremental
variant has no such copy, and that difference is real rather than an accounting
choice.

**Excluded, deliberately:**

- **Network round-trip to the exchange.** Four to five orders of magnitude
  larger than anything here, and not this code's responsibility.
- **Order-book maintenance** (`data_engine/book.py`'s `apply_diff`).
- **Feature construction.** This is the honest gap. `ml/features.py` builds the
  mid-relative prices, the `log1p` sizes and the causal rolling z-score, and
  that work is still Python-side; it is genuinely part of tick-to-signal and it
  is **not** in these numbers. Its cost in C++ is `TODO(measure)`. The
  arithmetic is O(40) per tick against the forward pass's tens of thousands of
  multiply-accumulates, so it is expected to be small — but "expected to be
  small" is not a measurement and is not reported as one.
- **Softmax.** The parity fixtures compare logits, so the timed path stops
  where the verified path stops.

### Real inputs, and two input modes

Zeros are not an acceptable benchmark input. They never produce denormals (and
this build does not enable flush-to-zero, since that would need `-ffast-math`,
refused project-wide), they take the same direction through every data-dependent
branch in `leaky_relu` and `maxpool`, and they make the tails meaningless. The
harness replays the same real held-out windows the parity test uses.

Two modes are measured because neither alone is honest:

| mode | what it does | why |
|---|---|---|
| `stream` | walks a rotating ~1 MB working set of real windows | input is cold-ish, as it would be if the box is shared |
| `hot` | replays one input forever | input stays in L1, as it would be right after the feature updater wrote it |

Production sits between the two. Measuring both bounds the effect instead of
assuming it away; the gap measured at baseline was about 10%.

### Pinning, priority, and a measured justification

Affinity is set programmatically — `SetThreadAffinityMask` on Windows,
`pthread_setaffinity_np` on Linux, where `taskset -c 3 ./bench …` is the
external equivalent. The mask actually applied is recorded, because Stage 6 had
a pinning call that silently succeeded and pinned nothing.

Priority is raised to `HIGH_PRIORITY_CLASS` (never `REALTIME` — that can starve
the input and storage stacks badly enough to need a hard reset). The
justification is measured, not assumed. Three runs of the identical binary:

| | p50 | p99 | run-to-run p99 spread |
|---|---|---|---|
| default priority | 31.8 / 31.9 / 31.9 µs | 124 / 187 / 143 µs | ±25% |
| high priority | 31.9 / 31.9 / 31.9 µs | 91.6 / 92.6 / 92.2 µs | ±0.5% |

The median never moved: that tail was other processes taking the core, not the
model. Without this, an optimisation worth 10% would have been invisible
underneath the noise.

### Settling, because boost clocks are worth 40%

`--settle 3` burns the CPU for three seconds before measuring anything. The
reason is a measurement, not superstition: the identical binary reports **p50
22.7 µs on an idle machine and p50 31.8 µs once it has been busy for a few
seconds** — a 1.40× spread caused entirely by boost clocks decaying. A "before"
taken in boost against an "after" taken in steady state would manufacture or
erase an improvement larger than most of Stage 7b's optimisations.

This makes runs comparable *to each other on one machine*. It does not make them
comparable to a different machine, and no amount of warmup would.

### Thermal state, without a temperature API

A fixed-work calibration kernel runs immediately before and immediately after
the timed loop, and both durations go into the JSON
(`calibration_before_s`, `calibration_after_s`). If the CPU down-clocked during
the run, the same arithmetic takes measurably longer the second time and the
ratio says by how much.

This is deliberately preferred to reading a thermal sensor. A temperature in
degrees does not tell you what happened to your throughput; a ratio of 1.39
does, and it needs no OS-specific API in a codebase whose selling point is
having no dependencies. **A pass whose two calibration numbers disagree by more
than a few percent is re-run rather than reported.**

### Warmup and sample counts

10,000 warmup iterations, discarded, as in §2.

| variant | timed iterations | why |
|---|---|---|
| `incremental` | 1,000,000 | ~32 µs each, so a full run is ~35 s. p99.9 rests on 1,000 observations. |
| `full` | 100,000 | ~5 ms each, so 1M would take 83 minutes and it is needed twice. p99.9 rests on 100 observations — thinner, and the count is in the JSON so nobody has to guess. |

The `full` variant is measured at the baseline and at the final optimised state
only. It exists to quantify one thing — the algorithmic win from not recomputing
a 100-row window — and re-measuring it after every pass would cost hours to
re-establish a comparison that does not change.

### Percentiles: nearest-rank, no interpolation

`percentile()` in `bench_report.hpp` returns an observed sample, never a blend
of two. Interpolating reports a duration the machine never produced, which is
the kind of invented number this project refuses everywhere else.

Saved per run: p50 / p90 / p99 / p99.9 / p99.99 / min / max / mean, a 206-point
quantile curve (0.5% steps through the body, finer through the tail), and a
256-bucket histogram spanning min…p99.9 with an explicit overflow count so a
single outlier cannot flatten the plot and nothing is silently discarded.

### Why p99.9 and not just p99

A tick-to-signal path runs on every book update. At the depth-stream rates
Stage 1 measured, p99 is a once-every-few-seconds event and p99.9 a
once-every-couple-of-minutes event; both happen constantly over a trading day.
p99.9 is also where the mechanisms a p50 optimisation cannot touch live — a page
fault, a scheduler preemption, an allocator slow path. Reporting only p99 lets a
change that trades a rare 10× spike for a small median win look like a win.

### Coordinated omission, and why a closed loop is right here

Coordinated omission is what happens when a load generator that should issue a
request every *T* stalls because the previous request was slow, and so never
records the queueing delay its own slowness caused. It is a property of
**open-loop** systems measured with a closed-loop harness.

This path has no queue. It is a synchronous function call made by the thread
that just received a book update; if it is slow, the next tick simply has not
arrived yet, so there is no request that "should" have been issued and was not.
What this harness measures is therefore **service time**, and it says so — it is
not a claim about end-to-end latency under a given tick arrival rate. If ticks
ever arrive faster than the model can serve them, the right instrument is an
open-loop harness with a fixed arrival schedule, and that is a different
program.

### Run labels

Each optimisation pass is a separate commit and a separate saved record:

| label | state of the code |
|---|---|
| `cpp_*_p0` | Stage 7a arithmetic, `-O3`, scalar. The baseline. |
| `cpp_*_p1` | weight pointers hoisted out of the hot path |
| `cpp_*_p2` | `-O3 -march=native` and `__restrict` |
| `cpp_*_p3` | loop order and layout |
| `cpp_*_p4` | AVX2 intrinsics on the widest inner loop |

Every record carries `build_name`, `build_flags`, `compiler`, `simd` and
`avx2_kernel_compiled_in`, so a JSON cannot be misattributed to the wrong build
six weeks later.

### What is still NOT controlled

Everything in §2's list — a laptop running a desktop OS, no core isolation, no
`isolcpus`, turbo states not pinned, other user processes alive. Pinning,
priority and settling remove a large part of the run-to-run variance; they do
not turn this into a tuned benchmarking host. **The relative ordering of the
variants is the durable result.**

---

## 4. Accuracy measurement

- Always on the **held-out test block** of the walk-forward split, with the
  Stage 3 embargo of 199 samples in force.
- The fold is read from the training run's own saved record rather than
  recomputed, so the block scored is byte-identical to the one the model never
  saw.
- Scored on the **whole block**, never a prefix.

> **A trap worth recording.** An earlier version of the Pareto benchmark scored
> accuracy on the first 4,000 of 9,292 test samples for speed. That subset is
> not representative: it put the student at 0.649 macro-F1 against 0.595 on the
> full block, and the teacher at 0.533 against 0.549. The test period is
> strongly non-stationary — per-quarter macro-F1 swings by more than 0.15 and
> the class balance ranges from 59% up to 59% down — so any prefix is its own
> little regime. `--accuracy-samples` now defaults to the whole block.

---

## 5. Reproducibility

- `seed_everything` fixes Python, numpy and torch RNGs.
- `torch.backends.cudnn.deterministic` is **off by default** and its state is
  recorded in every saved run. Enabling it was measured at a **14x** slowdown
  (56 vs 784 samples/s), turning a 24-minute run into 5.5 hours. Seeds still
  pin data order and initialisation, so runs are reproducible in distribution;
  bit-exact GPU reproducibility is available with the flag and is not the
  default. Every result JSON records which was used, so a number always
  carries its own reproducibility guarantee.
- Every saved artefact records platform, processor, Python, torch and numpy
  versions, plus the device.
- Cross-device reproducibility is explicitly **not** claimed: cuDNN selects
  different kernels on different hardware.

---

## 6. Statistical care

Three places where a single number would have been misleading, and what was
done instead:

**Distillation delta** (Stage 6) — one paired run cannot separate a small
effect from run-to-run noise, so the comparison runs across 3 seeds and reports
mean ± spread. The measured delta is smaller than its own spread, which is why
it is reported as "no measurable effect" rather than as a small negative.

**Information coefficient** (Stage 5) — a single block-shuffled control draw
landed at −0.14 and meant nothing, because with ~18 permutable blocks the null
has σ ≈ 0.11. Replaced with a 200-draw permutation null: mean −0.035, σ 0.107,
true IC +4.3σ above it.

**IC stability** (Stage 5) — the pooled IC (+0.42) and the mean per-block IC
(+0.07) differ by a factor of six, because pooling lets slow common drift
inflate the correlation. Both are reported; the per-block figure is the one
that matters.

---

## 7. How to reproduce everything

```bash
# Capture (needs network; ~20 min for the Stage 3 dataset)
python -m data_engine.capture --symbol btcusdt --out data/stage3 \
    --snapshot-interval 10 --max-messages 45000

# Storage, baselines, training
python benchmarks/bench_binfmt.py --data-dir data/ --repeats 7
python benchmarks/bench_baselines.py --tapes "data/stage3/*.tape"
python -m ml.train --tapes "data/stage3/*.tape" --epochs 40 --device cuda

# Signal characterisation
python benchmarks/bench_evaluation.py

# Compression (the Pareto frontier). ~20 minutes; do not run anything else
# on the machine while this is timing.
python -m ml.distill --epochs 30 --seeds 3
python benchmarks/bench_calibration.py
python benchmarks/bench_python_variants.py --iterations 100000 --warmup 10000

# The C++ path. Export first, then TEST before timing anything.
python -m ml.export_weights --real-data
cd inference_cpp
cmake --preset release-avx2 && cmake --build --preset release-avx2
ctest --preset release-avx2          # parity + equivalence, or the numbers mean nothing
./build/release-avx2/bench.exe --variant incremental --iterations 1000000 \
    --weights artifacts/student_weights.ttsw --fixtures artifacts/student_fixtures.ttsf
./build/release-avx2/bench.exe --variant full --iterations 100000 \
    --weights artifacts/student_weights.ttsw --fixtures artifacts/student_fixtures.ttsf
```

The last two are the reason this document exists. Running them while the machine
is busy will produce worse tails, and the result will not say so — which is
exactly why the method, not just the number, has to be written down.

**Comparing two C++ builds is the part that goes wrong.** Build both into
separate directories *first*, let the machine cool, then run them *alternately*,
and check that `calibration_before_s` agrees between the two records before
believing any difference. Doing it any other way once turned a 34% improvement
into an apparent 52% regression; `benchmarks/cpp_incremental_p3_throttled.json`
and `benchmarks/cpp_incremental_p3_stream.json` are the same binary in the two
machine states, 3.24× apart.


---

## 8. The dashboard render measurement (Stage 8b)

The dashboard claims it renders at the display's refresh rate inside a 4 ms
per-frame budget — measured at **144.7 fps with a p50 of 1.70 ms** on a 144 Hz
GPU-composited browser. That is a
latency claim on a project whose whole thesis is that latency claims must be
measured, so it is measured the same way as everything else here: stated
instrument, stated conditions, stated exclusions.

### What is timed

`performance.now()` around the body of one `requestAnimationFrame` callback in
`serving/dashboard/js/render.js:50` — committing one tape column and calling
`draw` on every panel. It is **scripting time only**.

**Excluded, and not small:** style, layout, paint and compositing, all of which
the browser does after the callback returns; websocket parsing, which happens in
`onmessage` outside the loop; and the server's own work. This is service time for
the render, in the same sense §3 uses for the C++ forward pass, and for the same
reason: it is what this code controls.

### The instrument

`performance.now()` is specified to be monotonic and is clamped by browsers to
100 µs resolution or coarser for cross-origin isolation reasons. Against a p50 of
~1 ms that is a 10% quantisation floor on a single sample, which is why the
reported figures are **percentiles over a 128-frame ring** rather than any
individual frame: the p50 of 128 samples is not meaningfully affected by 100 µs
of quantisation, and the meter refreshes at 4 Hz rather than per frame.

Frame rate is a smoothed rAF interval (exponential mean, α = 0.1), not `1/Δt` of
the last frame — an instantaneous figure flickering between 58 and 62 reads as
instability rather than as measurement noise.

### Conditions

| | |
|---|---|
| browser | headless Chrome 150.0.7871.188, `--headless=new`, **`--enable-gpu`** |
| driver | Chrome DevTools Protocol over a real wall clock |
| viewport | 1440x900, devicePixelRatio 1 (and a separate run at 2) |
| server | the `docker compose up` container on `localhost:8000`, demo mode, 10x |
| duration | 45–60 s of continuous rendering after the boot sequence |
| machine | the same laptop as every other measurement in this document |

**GPU rasterisation must be enabled, and the reason is a factor of three.**
Headless Chrome defaults to software rasterisation, and this page composites two
full-width canvas layers per frame. The same build measures **~40 fps under
software raster and 144.7 fps with the GPU**, with the render loop itself
unchanged at ~1.7 ms p50 in both — so the software number measures the
compositor, not the code, and reporting it would understate the product on any
real machine. Both numbers are stated wherever the frame rate is quoted.

**`--virtual-time-budget` must not be used.** It was, in the first attempt, and
it fast-forwards timers while starving `requestAnimationFrame`: the page produced
a single column in nine seconds of virtual time and the resulting screenshot
showed an unpopulated dashboard. Any headless measurement of a rAF loop has to
run on a real clock.

### What is reported, and what a single number would hide

Both **p50 and max** over the trailing 128 frames, permanently, in the header —
never a mean. A p50 of 1.00 ms with a max of 5.60 ms is a real dropped frame
somebody can see, and averaging it away would be the same error the latency panel
refuses to make about p99 versus mean.

The slow-frame hunt that produced the current numbers used a temporary
`console.warn` on any frame over 12 ms, printing per-panel timings. That
instrumentation was removed once the cause was found and fixed; it is described
in `docs/INTERVIEW_NOTES_stage8b.md` §1.2 rather than left in the render loop.

### Not controlled

A desktop OS with a browser on it. No core isolation, no thermal control, and
the compositor is sharing the machine with the server being measured — the
serving p50 shown in these runs is 465 µs against the container and 939–2,043 µs
against a local process competing with Chrome for the same cores, which is a
good illustration of why §2's "what is NOT controlled" applies here too.
