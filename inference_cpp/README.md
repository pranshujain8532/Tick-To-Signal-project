# inference_cpp — the hand-written forward pass

A dependency-free C++ implementation of the distilled student model, verified
against PyTorch on 1,000 real held-out windows.

**No libtorch, no BLAS, no onnxruntime, no Eigen.** Only the C++ standard
library. That constraint is the point: this is the artefact that shows the
model is understood well enough to rebuild from scratch, and it is the base the
Stage 7b latency work optimises.

**Stage 7a is correctness only.** There is deliberately no performance work in
this code — no blocking, no unrolling, no intrinsics, no `-ffast-math`. Every
loop is the most obvious one that produces the right numbers. Optimising before
a verified reference exists means debugging arithmetic and performance at the
same time, and chasing a numerical disagreement that turns out to be the
compiler reassociating a sum is a miserable way to spend a day.

---

## Results

| build | ops test | parity test | max \|diff\| | argmax agreement |
|---|---|---|---|---|
| `debug-O0` | pass | pass | 2.38e-05 | 1000 / 1000 |
| `release-O3` | pass | pass | 2.31e-05 | 1000 / 1000 |

Tolerance is 1e-4. The observed difference is ~2.3e-05 absolute, 2.2e-06
relative to the logit scale — and most of that is inherited from folding
BatchNorm in float32, not from the C++ arithmetic itself (the fold alone
accounts for 1.98e-05).

Both optimisation levels are tested because the -O3 result is what proves the
optimiser did not change the answer.

---

## Building

Requires CMake ≥ 3.15 and a C++14 compiler. Developed against GCC 6.3 (MinGW)
and CMake 4.4.

**First, generate the artefacts** (from the repository root, in the Python
environment — see the main README):

```bash
python -m ml.export_weights --real-data
```

That writes into `inference_cpp/artifacts/`:

| file | what it is |
|---|---|
| `student_weights.ttsw` | 48 tensors, BatchNorm already folded in |
| `student_fixtures.ttsf` | 1,000 (input, output) pairs from PyTorch |
| `op_fixtures.ttsw` | per-operator inputs and expected outputs |

**Then build and test.** With presets:

```bash
cmake --preset debug-O0     && cmake --build --preset debug-O0     && ctest --preset debug-O0
cmake --preset release-O3   && cmake --build --preset release-O3   && ctest --preset release-O3
```

Or without, naming the generator explicitly:

```bash
cmake -S . -B build/debug-O0 -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug-O0
ctest --test-dir build/debug-O0 --output-on-failure
```

On Windows you may need to point CMake at the toolchain if it is not on `PATH`:

```bash
-DCMAKE_MAKE_PROGRAM="D:/bin/mingw32-make.exe" -DCMAKE_CXX_COMPILER="D:/bin/g++.exe"
```

Run the tests directly if you prefer their full output:

```bash
./build/debug-O0/ops_test.exe    artifacts/op_fixtures.ttsw
./build/debug-O0/parity_test.exe artifacts/student_weights.ttsw artifacts/student_fixtures.ttsf
```

> **Note on this machine.** A Windows Application Control policy intermittently
> blocks freshly linked executables (`An Application Control policy has blocked
> this file`). Re-linking the target clears it. It is an environment quirk, not
> a build problem — the same policy is why the project's Python runs under the
> `py310` conda environment.

---

## Layout

```
src/
  tensor.hpp    owning float buffer, 64-byte aligned. No templates, no broadcasting.
  weights.hpp   TTSW reader interface
  weights.cpp   ...and its implementation, validating every field it reads
  ops.hpp       the seven operators this model needs, and the layout convention
  ops.cpp       plain nested loops, correctness-obvious
  model.hpp     architecture constants and the buffer inventory
  model.cpp     the forward pass; every buffer allocated in the constructor
tests/
  ops_test.cpp     each operator against its own PyTorch fixture
  parity_test.cpp  the whole model against 1,000 PyTorch fixtures
```

### Conventions worth knowing before reading the code

**Layout.** Activations are channel-major and contiguous, with the batch
dimension gone: `[channels][height][width]` indexed
`c * height * width + h * width + w`, and `[channels][length]` for 1-D.
Convolution weights keep PyTorch's own `[out][in][kh][kw]` order so the exporter
never transposes — one fewer transformation to get wrong.

**Causal padding is done by skipping, not by padding buffers.** Every
time-axis convolution is left-padded with zeros; the loops compute the source
index and skip anything negative, which is arithmetically identical and avoids
an allocation and a copy per layer.

**Max-pool is the exception.** PyTorch zero-pads *before* pooling, so a window
overhanging the start takes the max of the real values **and zero** — not the
max of the real values alone. Skipping padded taps would return the largest
negative value instead, and would disagree only near `t = 0` and only on
negative activations. `ops_test` has a case with deliberately negative inputs
for exactly this reason.

**Zero heap allocation after construction.** `StudentModel::forward` allocates
nothing; all ~275 KB of activation buffers are sized in the constructor. The
consequence is that a `StudentModel` is not thread-safe — construct one per
thread.

---

## Stage 7b — the streaming path and the harness

```
src/
  incremental.hpp   which layers can cache and why; the ring-buffer design
  incremental.cpp   ...and the column operators, one output timestep each
  fixture_io.hpp    the TTSF reader, shared by both tests and the benchmark
tests/
  incremental_test.cpp  streaming output == full recompute, bit for bit
bench/
  bench_clock.hpp   a monotonic clock that actually resolves microseconds
  bench_report.hpp  percentiles, histogram, JSON
  bench.cpp         the harness: pinning, warmup, settle, what is timed
```

`IncrementalModel::push_tick` takes one `[40]` feature column and produces the
logits for that tick, keeping each layer's recent activations in a ring sized to
that layer's reach:

| | p50 | p99 | p99.9 | state |
|---|---|---|---|---|
| `full` — recompute `[100, 40]` | 2,118.5 µs | 2,766.6 µs | 3,396.2 µs | 281,600 B |
| `incremental` — one tick | **11.0 µs** | **16.6 µs** | **86.9 µs** | **18,560 B** |

The two agree **bit for bit**, on 333 comparisons, on all four builds. That
depends on the receptive field (83) being shorter than the window (100) — see
the note at the top of `incremental.hpp`.

### Builds

Four presets now. Parity and equivalence must pass on all of them; that is what
makes the optimisation work checkable rather than hopeful.

| preset | flags | role |
|---|---|---|
| `debug-O0` | `-O0 -g` | reference. The compiler has barely touched the arithmetic. |
| `release-O3` | `-O3` | portable optimised. Runs on any x86 part. |
| `release-native` | `+ -march=native` | AVX2/FMA for this CPU. |
| `release-avx2` | `+ TTS_AVX2=ON` | the hand-written kernel. What the benchmarks use. |

All four carry `-msse2 -mfpmath=sse -ffp-contract=fast`, and none carries
`-ffast-math`. The `-mfpmath=sse` is not cosmetic: 32-bit MinGW defaults to x87,
which computes in 80-bit registers and rounds to 32 bits only on store, so two
loops doing the same additions in the same order can still disagree. That cost
two debugging sessions and is written up in `CMakeLists.txt`.

```bash
cmake --preset release-avx2 && cmake --build --preset release-avx2 && ctest --preset release-avx2
```

### Benchmarking

```bash
./build/release-avx2/bench.exe --variant incremental --iterations 1000000 \
    --weights artifacts/student_weights.ttsw --fixtures artifacts/student_fixtures.ttsf
```

It pins a core, raises priority, burns the CPU for `--settle` seconds so boost
clocks are not part of the result, measures the clock's *real* resolution before
trusting it, and writes the full distribution to `benchmarks/`. Read
`docs/benchmark_methodology.md` §3 before quoting any number it prints —
particularly the part about what the timed region excludes.

## What is not here yet

A C++ feature updater. The timed region starts from a prepared `[40]` column;
building that column from a book is still Python-side, and its cost is
`TODO(measure)`. Stage 8 is serving.
