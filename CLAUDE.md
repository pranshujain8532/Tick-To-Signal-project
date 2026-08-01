# CLAUDE.md — Tick-to-Signal Project Constitution

## What this project is
An end-to-end low-latency ML microstructure system: live order-book capture →
custom binary storage → from-scratch deep LOB model → honest cost-aware
evaluation → compression (ONNX/int8/distillation) → hand-rolled C++ inference
path with microsecond-level measured latency → FastAPI serving + dashboard.

The author is a student preparing to defend every line in HFT/FAANG
interviews. Optimize for UNDERSTANDABILITY and CORRECTNESS first, speed second,
cleverness never.

## Code style — non-negotiable
1. Write code like a thoughtful human, not a code generator:
   - Every module starts with a docstring explaining WHAT it does, WHY it
     exists, and the key DESIGN DECISION with the alternative that was
     rejected (e.g. "TCN instead of LSTM because...").
   - Comments explain WHY, never narrate WHAT ("# skip stale diffs that were
     already reflected in the snapshot" — good; "# loop over list" — bad).
   - Small functions (<40 lines), descriptive names, full type hints.
   - No clever one-liners, no walrus-operator tricks, no nested
     comprehensions beyond one level. If a line needs 10 seconds to parse,
     rewrite it.
2. NO over-engineering: no abstract base classes until a second
   implementation exists, no dependency-injection frameworks, no config
   systems beyond a simple dataclass, no dead code, no premature interfaces.
3. Dependencies: stdlib + numpy + pytorch + onnxruntime + fastapi + websockets
   + pytest. Ask before adding anything else.

## Presentation-layer exception (Stage 8 only)
The "simplicity wins" rule targets BUILD SYSTEMS and FRAMEWORKS, not visual
quality. For serving/dashboard/ specifically:
- Still forbidden: npm, bundlers, React/Vue/Svelte, CSS frameworks,
  component libraries, any build step. The dashboard must run by opening a
  file or hitting a static route.
- Now required: production-grade craft. Canvas rendering for high-frequency
  layers, a render loop decoupled from the data loop, preallocated
  typed-array ring buffers, devicePixelRatio handling, a stated performance
  budget enforced by an on-screen meter.
- Rationale to carry into comments: this dashboard renders a live order book
  and a latency histogram for a project whose entire thesis is tail latency.
  A DOM-thrashing dashboard would become the slowest component in the
  system it measures.

## Honesty rules specific to the dashboard
The dashboard is the only artefact most people will ever see. It is
therefore the easiest place to accidentally claim something the measurements
do not support. Three claims are FORBIDDEN, by construction, not by care:
1. That low latency is what makes this signal tradeable. It is not — the
   measured half-life is 13.2 s and the fee shortfall is 70x.
2. That the microsecond figure is what the dashboard is running. It is not —
   serving is ONNX int8 at ~856 us p50; the C++ path is ~11 us measured by a
   separate harness.
3. That the pooled IC (+0.421) is the edge. It is not — the per-block mean
   is +0.073 with IR 0.21, and Stage 5 established the pooled figure answers
   a question nobody can trade.
Every panel that could imply one of these must carry its correction as
permanent visible text, never a tooltip, never collapsible.

Everywhere else in the repo, the original constitution is unchanged.

## Notebook rules
- All research/ML/analysis code lives in `notebooks/` as .ipynb.
- Cell pattern, strictly: (markdown: concept + why) → (code: one focused
  step) → (markdown: interpret the output we just saw). Never more than
  ~25 lines of code per cell.
- Each notebook ends with a "## Interview checkpoint" markdown cell: 5
  questions an interviewer would ask about this notebook, each with a
  2-3 sentence answer.
- Notebook-first, module-second: production logic is developed in the
  walkthrough notebook, then mirrored into the .py module. If you change a
  module, update its notebook in the SAME session. Only `data_engine/capture.py`
  and `serving/api.py` contain logic that cannot literally run in a notebook
  (long-running loops); everything else must be importable and demonstrated
  in its notebook.

## Correctness rules
- Every module gets pytest tests in `tests/`. Book invariants, format
  round-trips, label-leakage checks, and C++/PyTorch parity are MANDATORY
  tests, not optional.
- Never fabricate results. Benchmarks are produced by scripts, saved under
  `benchmarks/` with timestamps and machine info, and referenced from the
  README. If a number is a placeholder, mark it `TODO(measure)`.
- No look-ahead anywhere in the ML pipeline. Any function touching labels
  must state its information horizon in the docstring.

## Documentation rules
- After each stage, write `docs/INTERVIEW_NOTES_stageN.md`: the design
  decisions made, alternatives rejected, and the 10 hardest questions an
  interviewer could ask about this stage — with answers and file:line
  pointers into the code.
- `docs/benchmark_methodology.md` records exactly how every latency number
  was measured (clock, pinning, warmup, iteration count, thermal state).

## Definition of Done for any stage
tests green + notebook runs top-to-bottom + interview notes written +
README results table updated. All four, or the stage is not done.
