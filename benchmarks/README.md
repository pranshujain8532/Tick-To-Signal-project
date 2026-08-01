# benchmarks/ — every measurement this project publishes

Each file here was written by a script, carries a UTC timestamp or an
optimisation-pass label in its name, and records the machine it ran on. Nothing
in the README, the notebooks or the dashboard states a number that does not come
from one of these files — `serving/records.py` reads them at request time so the
running dashboard cannot drift from them either.

**How to read a latency record.** Every one carries the clock's measured
resolution, the warmup and iteration counts, whether the process was pinned, and
a fixed-workload calibration taken before and after the run. If
`calibration_before_s` and `calibration_after_s` disagree by more than a few
percent, the machine changed speed during the run and the numbers are void by
inspection. Method: [../docs/benchmark_methodology.md](../docs/benchmark_methodology.md).

---

## Capture and storage

| File | What it records |
|---|---|
| [stage1_capture_20260727.md](stage1_capture_20260727.md) | The live capture run: 70.7 msgs/s over 283 s, 0 drops, 0 resyncs, and the 60-second REST cross-checks against the exchange (2 PASS / 2 SKEW / 0 FAIL) |
| [binfmt_20260727T162516Z.json](binfmt_20260727T162516Z.json) | The binary tape against JSON: 1.57× smaller than raw, **6.56× larger than gzipped**, 4.86× faster sequential, 261× faster vectorised |
| [bench_binfmt.py](bench_binfmt.py) | The script that produced it |

## Training and baselines

| File | What it records |
|---|---|
| [train_ours_btcusdt_20260728T102758Z.json](train_ours_btcusdt_20260728T102758Z.json) | The teacher: CNN+TCN, 319,715 params, test macro-F1 **0.5493** |
| [train_ours_btcusdt_20260728T102758Z_curves.png](train_ours_btcusdt_20260728T102758Z_curves.png) · [_history.csv](train_ours_btcusdt_20260728T102758Z_history.csv) | Its loss and validation curves per epoch, and the same data as a table |
| [baselines_20260728T102430Z.json](baselines_20260728T102430Z.json) | The three baselines the teacher is judged against: majority class 0.1657, queue-imbalance rule 0.3950, **logistic regression 0.5317** |
| [bench_baselines.py](bench_baselines.py) | The script that produced them |
| [train_fi2010_20260728T111135Z.json](train_fi2010_20260728T111135Z.json) | The same architecture on the public FI-2010 benchmark: **0.751** at h=10 |
| [train_fi2010_20260728T111135Z_curves.png](train_fi2010_20260728T111135Z_curves.png) · [_history.csv](train_fi2010_20260728T111135Z_history.csv) | FI-2010 curves and per-epoch history |

## Evaluation — the numbers that decide the project

| File | What it records |
|---|---|
| [evaluation_20260728T202537Z.json](evaluation_20260728T202537Z.json) | Per-block IC (+0.073, IR 0.21, 12 of 18 positive), the pooled contrast (+0.421), the 200-draw permutation null (+4.25 σ), the decay fit (half-life 13.2 s, R² 0.98), and the trade simulation (+0.285 bps gross, 0.142 bps/side breakeven) |
| [evaluation_20260728T202537Z_ic_blocks.png](evaluation_20260728T202537Z_ic_blocks.png) | The 18 block ICs around zero — the picture the dashboard's stability panel draws |
| [evaluation_20260728T202537Z_decay.png](evaluation_20260728T202537Z_decay.png) | IC against horizon: the rise to 5 s and the fit from the peak onward |
| [evaluation_20260728T202537Z_pnl.png](evaluation_20260728T202537Z_pnl.png) | Net edge against fee, crossing zero at 0.142 bps/side and every published tier far to the right of it |
| [bench_evaluation.py](bench_evaluation.py) | The script that produced them |

## Compression

| File | What it records |
|---|---|
| [distillation_20260728T211009Z.json](distillation_20260728T211009Z.json) | **The three-seed study.** Distilled 0.5058 ± 0.0681 against a scratch-trained control 0.5233 ± 0.0085 — distillation shows no measurable benefit, and the distilled variance is 8× the control's |
| [distillation_20260728T210128Z.json](distillation_20260728T210128Z.json) | The earlier **single-seed** run that the three-seed study exists to correct. Kept deliberately: it is the record of a result that looked fine until it was repeated |
| [calibration_20260731T180908Z.json](calibration_20260731T180908Z.json) | Post-quantisation calibration, and the run where scoring on a 4,000-sample prefix reversed the conclusion |
| [python_variants_20260801T082226Z.json](python_variants_20260801T082226Z.json) | The Python side of the latency frontier: eager fp32, ONNX fp32, ONNX int8, student fp32, student int8 |
| [python_variants_20260731T175351Z.json](python_variants_20260731T175351Z.json) | The previous frontier run, superseded by the one above. `serving/records.py` deliberately serves the newest matching record, and keeping the older one is what makes "newest wins" auditable |
| [bench_calibration.py](bench_calibration.py) · [bench_python_variants.py](bench_python_variants.py) | The scripts |
| [hero_latency_vs_f1.png](hero_latency_vs_f1.png) | Accuracy against p99 latency across variants |

## The C++ optimisation passes

Eleven records, and the sequence is the argument. `p0` is the first *correct*
implementation; each later pass is one change, measured. All are on the same
1,000 held-out windows and every build passes the parity and equivalence tests —
a pass that changed the answer would have failed before it was timed.

| Record | Pass | p50 | What changed |
|---|---|---|---|
| [cpp_full_p0_stream.json](cpp_full_p0_stream.json) | p0 | 5,596.6 µs | The first correct hand-written full recompute. **6.5× slower than ONNX Runtime** on the same model |
| [cpp_full_p1_stream.json](cpp_full_p1_stream.json) | p1 | 5,811.0 µs | Full recompute after the loop-order work — no better, which is why the algorithm had to change |
| [cpp_full_p4_stream.json](cpp_full_p4_stream.json) | p4 | 2,118.5 µs | Full recompute with every micro-optimisation and AVX2. Still slower than ONNX int8 |
| [cpp_incremental_p0_stream.json](cpp_incremental_p0_stream.json) | p0 | 31.9 µs | The ring buffer: advance one tick instead of recomputing a window. **This one change is the 193×** |
| [cpp_incremental_p0_hot.json](cpp_incremental_p0_hot.json) | p0 | 28.7 µs | The same pass measured with a cached input — the best case, kept separate from the streaming case on purpose |
| [cpp_incremental_p1_stream.json](cpp_incremental_p1_stream.json) | p1 | 22.5 µs | Loop order changed so the inner loop runs over output channels |
| [cpp_incremental_p2_stream.json](cpp_incremental_p2_stream.json) | p2 | 21.2 µs | `-march=native`. Almost nothing, which is the point |
| [cpp_incremental_p3_stream.json](cpp_incremental_p3_stream.json) | p3 | 14.8 µs | Weight layout: transposed so the hot loop reads contiguously |
| [cpp_incremental_p3_throttled.json](cpp_incremental_p3_throttled.json) | p3 | 44.9 µs | **The same binary as the row above**, run on a thermally loaded machine. 3.24× apart. This pair is why every record now carries a before/after calibration |
| [cpp_incremental_p4_stream.json](cpp_incremental_p4_stream.json) | p4 | **11.0 µs** | Hand-written AVX2 intrinsics for the 1-D kernel. **The shipped path** |
| [cpp_incremental_p4_hot.json](cpp_incremental_p4_hot.json) | p4 | 9.1 µs | The shipped path with a cached input. The 11.0 µs streaming figure is the one quoted everywhere, because a live book does not hand you the same column twice |

**What all eleven exclude:** feature construction. The timed region is the model
forward pass from a prepared `[40]` column to three logits. Its C++ cost is the
one `TODO(measure)` in this repository — see the Known gaps section of the
[README](../README.md).

## Serving and the dashboard

No file here. The serving latency percentiles are *rolling* and measured in the
live process, so they are served by `/latency` rather than written to disk — a
committed record would be a snapshot of one machine's load at one moment
pretending to be a property of the system. The dashboard's own render budget is
measured by the on-screen meter and the method is
[benchmark_methodology.md §8](../docs/benchmark_methodology.md).
