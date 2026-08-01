# Stage 7b — speed, the streaming path, and the harness

Stage 7a produced a C++ forward pass that computes the right answer. This stage
made it fast and, more importantly, built the instrument that decides whether
"fast" is true.

The headline is uncomfortable and is the most useful thing here:

> **Four passes of micro-optimisation are worth 2.05×. Not recomputing the
> window is worth 193×. And the first working hand-written C++ was 6.5× *slower*
> than ONNX Runtime.**

---

## 1. The decisions, and what was rejected

### Streaming state: one ring per layer input, sized to that layer's reach

`incremental.hpp:70`. Each ring holds exactly the columns its consumer can still
reach — four for a kernel-4 convolution, `2 × dilation + 1` for a dilated
kernel-3 one. Total state is **18,560 bytes** against the full path's **281,600**.

**Rejected:** keep the full `[C, 100, W]` activation buffers and shift them left
one column per tick. Simpler to read, and it copies ~275 KB per tick — more
memory traffic than the arithmetic it saves.

**Rejected:** a single flat arena with hand-computed offsets. Faster to write,
impossible to review. `ColumnRing::peek(steps_back)` (`incremental.cpp:207`) is
four lines and its off-by-one is checkable by eye, which matters because a ring
indexed one slot late produces *plausible* logits.

### Weight pointers resolved in the constructor

`incremental.hpp:83`. `StudentModel::forward` had claimed "zero heap allocation"
while building a `std::string` per layer to look weights up by name — about
fifty allocations per call, under a docstring saying there were none. Fixed in
both classes.

Both classes resolve the same names **independently** rather than sharing a
table. The shape argument to each `resolve()` call *is* the assertion, so two
classes asserting the same shapes is two chances to catch a divergence from
`ml/model.py`; a shared table would be a third place for the architecture to be
described.

### Loop order: the output axis innermost

`ops.cpp:8`, `incremental.cpp:142`. Stage 7a summed over `(in_channel, kernel)`
for one output element at a time, which left a single float accumulator absorbing
32–96 **dependent** additions — a ~4-cycle latency chain with nothing to overlap.
Inverting the nest makes the inner loop `out[i] += in[i] * coefficient`:
contiguous, and element-wise rather than a reduction.

**Rejected:** multiple accumulators to break the chain. That reassociates the
sum and changes the result, which would have cost bit-identity with PyTorch's
ordering for a smaller win than the reorder gave.

### Transposed 1-D weights

`incremental.cpp:245`. `conv1d_column` must run its inner loop over output
channels — the feature axis is width 1 by then, so there is no other axis. In
PyTorch's `[out][in][k]` layout consecutive output channels are 384 bytes apart,
and **a strided read cannot be vectorised at all**. A `[in][k][out]` copy is
built once at construction.

**Rejected:** changing the file format. `ml/export_weights.py` still writes
PyTorch's order, so there is one layout on disk and exactly one place that
derives another. `conv2d_column` is deliberately *not* transposed — it already
runs the output width innermost, where the file's layout is already contiguous.

### AVX2 behind a compile-time flag, scalar path retained

`incremental.cpp:114`, `CMakeLists.txt:122`. `-DTTS_AVX2=ON` selects intrinsics;
anything else — including any non-x86 target — gets the plain loop. Both are
compiled and both run the full test suite, via the `release-avx2` preset.

**Rejected:** runtime dispatch on CPUID. That is a second code path, a function
pointer on the hot loop, and a combinatorial test matrix, for a binary that is
built on the machine it runs on.

### `-ffast-math` still refused; FMA contraction explicitly allowed

`CMakeLists.txt:104`. The distinction is the whole floating-point position of
this project: **contraction removes a rounding step and never changes which
numbers are added together; reassociation changes the arithmetic.** The first is
fine and is now pinned explicitly with `-ffp-contract=fast` rather than
inherited from GCC's default, because the hand-written kernel's use of
`_mm256_fmadd_ps` depends on it.

---

## 2. Two findings about the instrument, not the code

**The clock lied by six orders of magnitude.** `std::chrono::steady_clock` on
MinGW GCC 6.3 advertises a 1 ns period and resolves ~1 ms. The first run of the
harness reported 19,518 of 20,000 forward passes as taking *zero nanoseconds*,
and its `p50 5,000 µs` was a millisecond grid rather than a measurement. Caught
because the harness measures and prints the clock's real resolution before
anything that depends on it (`bench_clock.hpp:8`).

**The CPU's own clock varied 3.24×.** A fixed-work calibration kernel
(`bench.cpp:242`) runs before and after every timed loop. A cooled machine ran it
in 0.019 s; after an hour of benchmarking, 0.061 s. This once made a 34%
improvement look like a **52% regression** — the loop-reordering pass was one
decision away from being reverted. `benchmarks/cpp_incremental_p3_throttled.json`
and `cpp_incremental_p3_stream.json` are the same binary in the two states.

---

## 3. The ten hardest questions

### Q1. You say the streaming path is *bit-identical* to the full recompute. Prove that is not luck.

Three separate reasons, and the test checks the conjunction of all of them
(`incremental_test.cpp:26`).

1. **Causality.** Every layer's output at *t* depends only on inputs at *t' ≤ t*
   (`incremental.hpp:18`), so nothing computed for *t* can be invalidated by a
   tick at *t+1*. That is what makes caching sound at all.
2. **Order.** The column operators accumulate in the same `(in_channel, kernel)`
   order as `ops.cpp`. Float addition is not associative, so matching the order
   is what turns "close" into "equal".
3. **Padding.** Where the full version *skips* a zero-padded tap, the column
   version multiplies the ring's zero and adds it. `sum + 0.0f * w == sum` is
   exact in IEEE-754.

333 comparisons across four build configurations, `max |diff| 0.000e+00`.

### Q2. The two paths see different inputs — one sees 100 rows, one has seen the whole stream. Why do they agree?

Because the **receptive field is 83 and the window is 100**
(`incremental_test.cpp:38`). Output at *t* cannot reach before *t−82*, so the
streaming model's longer memory is unreachable and the full path's extra 17
timesteps cannot affect its own answer.

This is a real dependency, not a coincidence. If `ml/model.py` ever grows the
receptive field past 100, this test *should* fail, and the equivalence would have
to be restated as a tolerance. `TickToSignalNet.receptive_field()` is what pins
the number.

### Q3. `-march=native` broke your equivalence test. Walk me through the diagnosis.

`CMakeLists.txt:40` records it. The obvious suspect was FMA contraction, since
`-march=native` enables FMA and FMA rounds once where multiply-then-add rounds
twice. **`-ffp-contract=off` changed nothing**, which ruled it out — and that
negative result is the useful part, because it forced a real diagnosis instead of
a plausible one.

Three isolating builds found the cause:

| flags | bit-identical |
|---|---|
| `-msse2 -mfpmath=sse` | 100% |
| `-msse2 -mfpmath=387` | 0% (5.2e-06) |
| `-mavx2 -mfma -mfpmath=387` | 0% (4.5e-06) |

32-bit MinGW defaults to x87, which computes in 80-bit registers and rounds to 32
bits only on store. Two loops doing the *same* additions in the *same* order can
therefore land on different `float32` values depending on whether an intermediate
stayed in a register — and on whether one of them got vectorised with SSE, which
does round at 32 bits.

`-mfpmath=sse` is now global. I had first applied it only to the native preset,
reasoning that a reference build should keep the toolchain's defaults; the loop
reorder then broke `release-O3` the same way. **A reference build whose float32
results depend on register allocation is not a reference.** `-O0` and `-O3` now
agree to the digit on parity (2.527237e-05 both), which they did not before.

### Q4. You reordered floating-point loops and claim the result is unchanged. That is normally false.

It is false when you reorder the *additions*. This reorder does not
(`ops.cpp:30`). For each individual output element the sequence is still bias,
then in_channel, then kernel tap — that nest is untouched. What moved is *which
output element* is being worked on, and choosing an output element is not a
floating-point operation.

This is also exactly why the compiler may then vectorise it without
`-ffast-math`: eight independent outputs accumulating in parallel requires no
reassociation, whereas vectorising a reduction does. Getting a vectorisable loop
*and* bit-identity out of the same change is the point of doing it this way
rather than reaching for the flag.

### Q5. Your hand-written AVX2 kernel beat GCC by 8%. Is that worth the maintenance?

On its own, marginally — and the honest comparison is that the **layout change
beside it was worth 19%**. GCC auto-vectorises the transposed loop perfectly
well. What it could not do was transpose the weights, because that is a
data-structure decision and the compiler is not allowed to make it.

The intrinsics stay only because all three of these hold: the scalar path is
retained and is the default (`incremental.cpp:131`), a CI preset compiles and
tests both, and `incremental_test` proves the vector kernel is bit-identical.
Without those, a hand-written kernel is a liability.

### Q6. Why `_mm256_fmadd_ps` rather than a separate multiply and add?

Because the surrounding code already contracts. Disassembling the scalar loop
under `-march=native` shows **16 `vfmadd` and zero `vmulps`/`vaddps`** — GCC's
C++ default is `-ffp-contract=fast`. Writing `mul` + `add` here would have made
this one kernel the odd one out and broken bit-identity with everything around
it.

That default is now pinned explicitly (`CMakeLists.txt:116`) because it became
load-bearing. A toolchain defaulting to `off` would otherwise break the
equivalence test with no source change to blame.

### Q7. Why p99.9 and not p99, and why never a mean?

A tick-to-signal path runs on every book update, so p99 is a
once-every-few-seconds event and p99.9 a once-every-couple-of-minutes event; both
happen constantly over a trading day. p99.9 is also where the mechanisms a
median optimisation cannot touch live — page faults, preemption, allocator slow
paths. Reporting only p99 lets a change that trades a rare 10× spike for a small
median win look like a win.

The clearest example is in this stage. Hoisting the weight lookups moved the p50
by **0%** after normalising for machine speed, and moved p99.99 by **2.4×** and
the max by **2.5×** — precisely because an allocator is fast almost always and
occasionally takes a lock. A mean would have reported nothing.

p99.9 also needs samples to exist: at 1M iterations it rests on 1,000
observations, at 100k on 100. Every record carries its iteration count.

### Q8. Your harness is closed-loop. Isn't that coordinated omission?

No, and the distinction is worth being precise about (`bench.cpp:61`).
Coordinated omission is what happens when a load generator that *should* issue a
request every *T* stalls because the previous one was slow, and so never records
the queueing delay its own slowness caused. It is a property of **open-loop**
systems measured with a closed-loop harness.

This path has no queue. It is a synchronous call made by the thread that just
received a book update; if it is slow, the next tick has not arrived yet. There
is no request that should have been issued and was not.

What the harness measures is therefore **service time**, and it says so. It is
not a claim about end-to-end latency under a given arrival rate. If ticks ever
arrived faster than the model could serve them, the right instrument would be an
open-loop harness with a fixed arrival schedule — a different program.

### Q9. What exactly is inside your 11 µs, and what is not?

Stated at `bench.cpp:9` rather than left to be inferred.

**In:** the forward pass, from a prepared `[40]` feature column to three logits.
For the `full` variant it also includes the 16 KB input `memcpy`; the incremental
variant has no such copy, and that difference is real rather than an accounting
choice.

**Out, and the honest gap:** **feature construction is not in the number.**
`ml/features.py` builds the mid-relative prices, `log1p` sizes and causal rolling
z-score, and that is still Python-side. It is genuinely part of tick-to-signal.
Its C++ cost is `TODO(measure)`. The arithmetic is O(40) per tick against ~46,000
multiply-accumulates in the forward pass, so it should be small — but "should be
small" is not a measurement and is not reported as one.

Also out: exchange round-trip (4–5 orders of magnitude larger, and not this
code's responsibility), book maintenance, and softmax — the timed path stops
where the parity-verified path stops.

### Q10. Given all this, what is the actual takeaway about hand-written C++?

That it is not, by itself, a performance technique. The first correct
hand-written full-window pass ran at **p50 5,596.6 µs against ONNX Runtime's 856
µs on the same model** — 6.5× slower, because ONNX Runtime ships vectorised
kernels that a naive nested loop does not match.

Everything this path won, it won from two things: an **algorithm** the runtime
could not express (advance one tick instead of recomputing a window, ~193×) and
**layout and loop shape** the compiler was not permitted to change (~2×). The
language was the vehicle, not the reason.

The corollary is the reason Stage 7a came first: none of these changes would have
been safe to make without a parity test and an equivalence test that fail on a
changed answer rather than on a changed number.

---

## 4. Things that would be done differently

* **Measure the clock first, always.** It is now the first thing `bench` prints,
  and `zero_samples` is in every record so a coarse-clock run is void by
  inspection rather than by luck.
* **The `full` variant should have been re-measured after every pass.** It was
  measured at p0, p1 and p4 only, on the grounds that a 100k-iteration run costs
  10 minutes. That is a defensible budget decision and it does leave a gap in the
  record; the incremental path — the one that ships — was measured at every pass.
* **`bench.cpp` is 340 lines and does three jobs** (harness, statistics, clock),
  split across three files. A fourth split, separating option parsing, would have
  been better than the current `parse()` being the least readable function in the
  stage.
* **The AVX2 kernel covers one loop.** `conv2d_column`'s width-20 inner loop is
  the next-largest target at ~22% of the multiply-accumulates, and 20 is not a
  multiple of 8 so it needs remainder handling. It was left alone because the
  measured gain from intrinsics elsewhere was 8%, and 8% of 22% is not worth a
  masked tail.
