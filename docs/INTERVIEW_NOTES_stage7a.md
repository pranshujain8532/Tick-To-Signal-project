# Interview notes — Stage 7a: C++ correctness

What shipped: a dependency-free C++ forward pass for the 32,155-parameter
distilled student, a custom binary weight format with BatchNorm folded in at
export, per-operator fixtures, and a 1,000-fixture parity test that passes at
both `-O0` and `-O3`. Plus 13 Python tests (221 total).

Files: [ml/export_weights.py](../ml/export_weights.py),
[inference_cpp/src/](../inference_cpp/src/),
[inference_cpp/tests/](../inference_cpp/tests/),
[tests/test_export_weights.py](../tests/test_export_weights.py).
Build instructions: [inference_cpp/README.md](../inference_cpp/README.md).

**No performance work in this stage, by design.** See §5.

---

## 1. The result

| build | ops test | parity test | max \|diff\| | relative | argmax agreement |
|---|---|---|---|---|---|
| `debug-O0` | pass | pass | 2.384e-05 | 2.21e-06 | 1000 / 1000 |
| `release-O3` | pass | pass | 2.313e-05 | 2.14e-06 | 1000 / 1000 |

Tolerance 1e-4, on 1,000 **real held-out windows** (not random noise). Both
optimisation levels pass, which is what proves the optimiser did not change the
answer.

Where the difference comes from, in order of size:

| source | contribution |
|---|---|
| BatchNorm folded in float32 | 1.98e-05 |
| everything else (C++ summation order vs PyTorch's vectorised kernels) | ~4e-06 |

That decomposition matters: **most of the divergence was already present in
Python before any C++ ran.** The hand-written arithmetic contributes about
4e-06 on a logit scale of 10.8.

---

## 2. The BatchNorm folding math

At inference, BatchNorm applies per output channel `c`:

```
BN(y_c) = gamma_c * (y_c - mean_c) / sqrt(var_c + eps) + beta_c
```

and the convolution feeding it computes `y_c = sum_i W_ci * x_i + b_c`.
Substituting and collecting terms in `x`:

```
s_c   = gamma_c / sqrt(var_c + eps)      one scalar per output channel
W'_ci = W_ci * s_c
b'_c  = (b_c - mean_c) * s_c + beta_c
```

so `BN(conv(x)) == conv'(x)` **exactly**, for every input. The transformation is
a per-output-channel rescale of the weights plus an affine correction to the
bias. Where PyTorch omits the convolution bias (which it does whenever a
BatchNorm follows), `b_c` is zero and the formula still holds.

**The Inception block is the interesting case.** Its norm sits after a
*concatenation* of three branches, so it looks unfoldable. But concatenation
only stacks channels and BatchNorm is per-channel, so the norm's parameters
split cleanly by branch: channels `[0, 16)` belong to `conv_three`, `[16, 32)`
to `conv_five`, `[32, 48)` to `pool_project`. Folding each slice into the
convolution that produced it is exact, not an approximation.
→ `ml/export_weights.py:fold_inception`.

`reduce_three` and `reduce_five` are **not** folded: they are followed by an
activation, not a norm, so they keep their own bias.

**Verified, not assumed.** `verify_folding` runs the folded model against the
original before anything is written to disk. Measured:

| precision | max \|diff\| |
|---|---|
| fold and evaluate in float64 | **4.09e-14** |
| fold and evaluate in float32 | 1.98e-05 (2.31e-06 relative) |

The float64 figure is machine epsilon for a 40-layer network — that is the
proof the algebra and the channel slicing are right. The float32 figure is nine
orders larger *because float32 has nine fewer digits*, and it accumulates
smoothly with depth (per-stage relative error stays flat at 1.5–2.3e-06 from
the first block to the logits). An algebra error would jump at one stage and
would not shrink in float64.

---

## 3. Design decisions and the alternatives rejected

### 3.1 Fold BatchNorm in Python, not in C++
Rejected: implement a BatchNorm operator in C++. Folding is exact, free at
inference, and removes an entire operator plus four parameter tensors per layer
from the code that has to be hand-written and hand-verified. It also removes
the epsilon and running-statistics bookkeeping from the C++ side — two more
things that could silently differ from PyTorch.

### 3.2 A bespoke binary format, not pickle / npz / ONNX
Rejected: `torch.save` (needs libtorch), `.npz` (needs a zip reader and numpy's
format), ONNX (needs protobuf). The C++ side is meant to have *no* dependencies,
so the file must be readable with `fread` and length checks. TTSW is a magic, a
version, then per tensor: name length, name, dtype tag, rank, dims, float32
data. Same "explain every byte" discipline as the Stage 2 tape format.

### 3.3 No templates, no broadcasting, no general shapes
Rejected: a templated `Tensor<T, Rank>` with strides and numpy-style
broadcasting. That is what a *library* needs. This runs exactly one model whose
every tensor is float32 and whose every shape is known at compile time. A
broadcasting engine would be hundreds of lines the parity test cannot exercise
— hundreds of lines that could be wrong without anyone noticing.

### 3.4 Causal padding by skipping, not by padded buffers
Every time-axis convolution is left-padded with zeros. Rather than materialise
a padded copy, the loops compute the source index and skip negatives. Identical
arithmetic (padded values are zero and contribute nothing), minus an allocation
and a copy per layer.

### 3.5 64-byte alignment now, SIMD later
64 bytes is the x86-64 cache line and the alignment AVX-512 loads require.
Doing it at every allocation costs nothing today and means Stage 7b's
vectorisation is a change to the inner loop rather than to every buffer.

### 3.6 Zero heap allocation after construction — see §4

---

## 4. Why the zero-allocation design

`StudentModel::forward` allocates nothing. All ~275 KB of activation buffers are
sized in the constructor. Three reasons, in the order they matter:

1. **Determinism.** An allocator can take a lock, hit a slow path, or ask the OS
   for pages. Those events are rare and large — which puts them precisely in the
   p99.9 that Stage 6 measured (p99.9/p50 ratios of 7–12×) and that Stage 7b
   exists to improve. A forward pass that *cannot* allocate cannot be surprised
   by the allocator.
2. **Cache behaviour.** Reusing the same buffers keeps a 275 KB working set
   resident and warm instead of touching fresh cold pages each call.
3. **It is a real constraint that shapes the design.** Permanent buffers mean
   shapes must be known at construction — which is only true because the input
   shape is fixed at `[1, 100, 40]`. That is the payoff for the Stage 6 decision
   not to export a dynamic batch axis.

**The cost, stated rather than discovered:** a `StudentModel` is not
thread-safe. Two threads calling `forward` on one instance would overwrite each
other's activations. For a batch-1 tick-to-signal path the answer is one
instance per thread, and the header says so.

---

## 5. Why no performance work in this stage

Optimising before a verified reference exists means debugging arithmetic and
performance simultaneously. The specific trap is `-ffast-math`: it lets the
compiler reassociate floating-point sums, and reassociation alone can move a
reduction's result by more than this project's parity tolerance. The build would
simply start failing with no code change to blame.

So: `-ffast-math` is never enabled, in either preset. The two presets differ in
exactly one thing, the `-O` level, and both must pass parity. Stage 7b will
write vectorisation explicitly rather than begging the compiler for it — and
this parity test is what will keep that honest.

---

## 6. The ten hardest questions

**Q1. Derive the BatchNorm folding, and say when it fails.**
`BN(y) = gamma*(y-mean)/sqrt(var+eps) + beta` with `y = Wx + b` gives
`W' = W * s`, `b' = (b-mean)*s + beta` where `s = gamma/sqrt(var+eps)` — a
per-output-channel rescale plus an affine bias fix, exact for every input. It
fails if anything non-linear sits between the convolution and the norm, or if
the norm is in *training* mode where the statistics depend on the batch. Neither
applies here: every norm follows its conv directly, and inference uses frozen
running statistics.

**Q2. The Inception norm follows a concatenation. How is that foldable?**
Concatenation only stacks channels and BatchNorm is per-channel, so the norm's
parameters partition by branch — `[0,16)` to `conv_three`, `[16,32)` to
`conv_five`, `[32,48)` to `pool_project`. Each slice folds into the convolution
that produced those channels. Getting the slice *order* wrong would mix one
branch's statistics into another and still produce plausible logits, which is
why there is a test asserting the fold produces exactly those three branch
tensors and no surviving norm.
→ `test_inception_norm_splits_across_its_three_branches`.

**Q3. Why fixture-driven testing rather than hand-written expectations?**
There is no useful way to write down by hand what a 32,155-parameter network
should output. Any expectation would have been produced by running something,
so the honest move is to name that something and pin it: the fixtures come from
the same PyTorch model object that exported the weights, in the same process,
so weights and expectations cannot drift. It also makes the test an *oracle*
rather than a change detector — it fails against the definition of correct, not
against a previous run of possibly-wrong code.

**Q4. Why 1e-4 and not bit-exact?**
Bit-exactness with PyTorch is not achievable and not worth chasing: its
convolutions dispatch to vectorised kernels that accumulate in a different
order, and float addition is not associative. The question is whether a
difference is rounding or a bug, and rounding here sits at 2.3e-05 — three
orders below the threshold — so 1e-4 discriminates cleanly while tolerating a
legitimately different summation order. Argmax agreement is asserted separately
because that is what actually changes a prediction.

**Q5. What is the single most likely bug in a hand-written conv, and how did you catch it?**
Padding semantics. The specific one here is **max-pool**: PyTorch zero-pads
*before* pooling, so a window overhanging the start competes against zeros. An
implementation that "skips" padded taps — which is correct for convolution,
where a zero contributes nothing to a sum — returns the largest *negative* value
instead of 0.0. It disagrees only near `t=0` and only on negative activations,
so it survives casual testing. The op fixture for max-pool uses deliberately
negative inputs, and there is a Python test asserting the fixture really is
negative, so the case cannot quietly stop testing anything.
→ `ops.cpp:maxpool2d_causal_time`, `test_the_maxpool_fixture_is_actually_negative`.

**Q6. Why per-operator tests as well as full-model parity?**
Full-model parity tells you the network disagrees; it does not tell you which of
seven operators is responsible, and bisecting a 40-layer forward pass by hand is
exactly the tedium worth a file to avoid. The op fixtures are shaped to hit the
same code paths the model uses — the strides, dilations and kernel heights are
the ones that actually occur.

**Q7. Justify zero heap allocation on the hot path.**
Determinism first: an allocator can lock, hit a slow path, or fault in pages,
and those events are rare and large — exactly the p99.9 behaviour Stage 6
measured and Stage 7b targets. Then cache residency: the same 275 KB working set
stays warm. And it forces the design to admit that the input shape is fixed,
which is what makes buffer sizes compile-time constants. The cost is
thread-safety, and the answer is one model instance per thread.

**Q8. Why test at -O3 as well as -O0?**
Because they answer different questions. `-O0` establishes that the *code* is
right, with the compiler having barely touched the arithmetic. `-O3` establishes
that the *optimiser* did not change the answer — vectorisation and
reassociation opportunities the compiler takes on its own are still capable of
moving a float result. The two presets differ in exactly one flag so a failure
at `-O3` can only be the optimiser.

**Q9. You claim no dependencies. What exactly does the C++ include?**
`<cstdio>`, `<cstdlib>`, `<cstring>`, `<cmath>`, `<limits>`, `<string>`,
`<vector>`, `<map>`, `<stdexcept>` — the C++ standard library and nothing else.
No libtorch, no BLAS, no Eigen, no onnxruntime. The weight file is read with
`fread`, alignment is hand-rolled over-allocation rather than `std::aligned_alloc`
(C++17, and absent from this MinGW libc), and the whole thing builds with a
C++14 compiler.

**Q10. What would you do differently, and what is next?**
Two things I would change: the weight file has no checksum, so a corrupted
tensor would be detected only as a parity failure rather than at load; and the
fixtures are 16 MB of float32, which is fine locally but would want compression
or a smaller sample if this were ever committed. Next is Stage 7b — SIMD, loop
ordering, and an incremental ring-buffer path that advances one tick instead of
recomputing the whole 100×40 window, measured against the Stage 6 frontier
(ONNX int8 at p99 3,775 µs). The parity test is the safety net for all of it.

---

## 7. Method notes

- Fixtures are 1,000 **real held-out windows** from the Stage 3 capture, not
  random noise. Stage 6 measured the feature distribution as heavy-tailed (bulk
  near ±1.4, extremes near ±22), and large magnitudes are where two
  implementations first disagree numerically.
- Toolchain: GCC 6.3 (MinGW, 32-bit), CMake 4.4, C++14. The old compiler is why
  C++14 rather than 17; nothing here needs 17.
- The 32-bit toolchain is worth flagging for Stage 7b: a 64-bit compiler would
  give twice the SIMD registers and a saner ABI for the performance work.
- A Windows Application Control policy intermittently blocks freshly linked
  executables on this machine. Re-linking clears it. It is the same policy that
  forces the project's Python onto the `py310` conda environment.
