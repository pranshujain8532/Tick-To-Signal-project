# Tick-to-Signal — Implementation Plan

**One-line pitch:** *"I capture live order-book data myself, train my own deep LOB model from scratch, then compress and compile it into a C++ inference path with a measured p99 tick-to-signal latency in microseconds — and I evaluate the signal honestly, after fees."*

This document is the master plan. The companion documents are:
- `02_CLAUDE_CODE_PROMPTS.md` — the exact prompts you paste into Claude Code, stage by stage.
- `03_PROJECT_EXPLAINED.md` — every concept in simple language + interview Q&A.

---

## 0. Ground rules (read first)

**Hardware assumed:** one laptop (8+ GB RAM), free GPU (Kaggle / Colab) for training only. Everything else runs on the laptop. The model is deliberately tiny (~300K params) — *small-and-fast is the thesis of the project*, not a compromise.

**The notebook rule (important):** You asked for all Python in `.ipynb`. That works for research code, but two components **must** be plain `.py` to be credible to interviewers: the 24/7 capture daemon and the FastAPI service (nobody runs a websocket daemon inside a notebook, and an HFT interviewer will notice). The solution used throughout this plan:

> **Notebook-first, module-second.** Every production `.py` module has a paired *walkthrough notebook* that builds the same logic cell-by-cell with full explanations. You study the notebook; the daemon runs the module. The notebook is your interview prep; the module is your production credibility. Both are kept in sync.

**The honesty rule:** This project never claims a profitable trading strategy. It claims a *characterized signal* and a *measured system*. "My signal is real but dies after retail fees — here's the breakeven fee analysis" is worth more in an HFT interview than any fake Sharpe ratio.

**The interview rule:** you must be able to defend every line. The Claude Code prompts enforce heavy why-comments, design-rationale docstrings, and an `INTERVIEW_NOTES.md` per stage listing the questions an interviewer would ask about that stage and where the answer lives in the code.

---

## 1. Final repository structure

```
tick-to-signal/
├── CLAUDE.md                      # Claude Code's constitution (Stage 0)
├── README.md                      # Results-first readme: numbers, charts, methodology
├── requirements.txt
├── docker-compose.yml
│
├── data_engine/                   # Stage 1–2: production modules (.py)
│   ├── capture.py                 # websocket capture daemon (runs 24/7)
│   ├── book.py                    # order-book reconstruction from diffs
│   ├── binfmt.py                  # custom fixed-width binary storage format
│   └── replay.py                  # mmap reader + historical replayer
│
├── ml/                            # Stage 3–6: modules mirrored from notebooks
│   ├── features.py                # snapshot tensors, normalization
│   ├── labels.py                  # smoothed-mid labels with dead zone
│   ├── model.py                   # CNN + TCN model (from scratch)
│   ├── train.py                   # training loop (also runnable on Kaggle)
│   └── distill.py                 # teacher→student distillation
│
├── notebooks/                     # THE interview-prep artifacts
│   ├── 01_orderbook_capture_walkthrough.ipynb
│   ├── 02_binary_format_design.ipynb
│   ├── 03_dataset_and_labels.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_honest_evaluation.ipynb
│   ├── 06_compression_and_distillation.ipynb
│   └── 07_latency_results_analysis.ipynb
│
├── inference_cpp/                 # Stage 7: the differentiator
│   ├── src/
│   │   ├── weights.hpp/.cpp       # loads custom weight file
│   │   ├── tensor.hpp             # minimal aligned-buffer tensor
│   │   ├── ops.hpp/.cpp           # conv1d, relu, softmax — hand-rolled
│   │   ├── model.hpp/.cpp         # forward pass, zero alloc on hot path
│   │   ├── incremental.hpp/.cpp   # ring-buffer feature updates per tick
│   │   └── bench.cpp              # latency harness (percentiles, pinning)
│   ├── tests/parity_test.cpp      # bit-level parity vs PyTorch outputs
│   └── CMakeLists.txt
│
├── serving/                       # Stage 8
│   ├── api.py                     # FastAPI: /signal, /book, /latency-stats
│   └── dashboard/                 # single-page live dashboard
│
├── tests/                         # pytest for every Python module
├── benchmarks/                    # saved latency histograms, Pareto data (CSV/JSON)
└── docs/
    ├── INTERVIEW_NOTES_stage1.md ... stage8.md
    └── benchmark_methodology.md   # how numbers were measured (interviewers WILL probe)
```

---

## 2. Stage-by-stage plan

Effort estimates assume part-time work with Claude Code doing the typing and you doing the reviewing/understanding. **Do not start a stage before the previous stage's Definition of Done is fully green.**

### Stage 0 — Scaffold (1–2 days)
Repo skeleton, `CLAUDE.md` constitution, environment, CI-style test runner script.
**Definition of Done:** repo builds, `pytest` runs (0 tests is fine), README skeleton with placeholder results tables.

### Stage 1 — Capture engine + order-book reconstruction (1 week)
Connect to Binance combined websocket streams (`btcusdt@depth@100ms` + `btcusdt@trade`), implement the documented snapshot-sync algorithm (buffer diffs → REST snapshot → discard stale diffs → apply), maintain a full local book, detect sequence gaps, auto-resync, archive raw messages.
**Key numbers produced:** sustained msgs/sec, uptime, resync count.
**Definition of Done:**
- Book invariants tested: bids strictly descending, asks strictly ascending, best_bid < best_ask always.
- Periodic cross-check of local book vs fresh REST snapshot (top-10 levels match).
- Runs 24h unattended on the laptop without gap-related corruption.
- Walkthrough notebook 01 rebuilds a book from an archived message sample, cell by cell.

### Stage 2 — Custom binary storage format (3–5 days)
Design a fixed-width binary record format (int64 fixed-point prices/quantities, ns timestamps), a small file header (magic, version, symbol, tick size), periodic full-snapshot records for fast seeking, and an mmap-based zero-copy reader exposed as a numpy structured array.
**Key numbers produced:** bytes/event vs raw JSON (expect ~10–20× smaller), read throughput (M events/sec via mmap).
**Definition of Done:** round-trip test (write → read → identical), seek test, throughput benchmark saved to `benchmarks/`, notebook 02 explains every byte of the layout and *why* (this is a favorite systems-interview topic).

### Stage 3 — Dataset + labels (4–6 days)
Convert replayed books into model inputs: top-10 levels × (price, qty) × 2 sides = 40 features × 100 timesteps. Labels: smoothed mid-price direction (mean of next *k* mids vs mean of previous *k* mids, threshold α → down/flat/up). Chronological walk-forward splits with an **embargo gap ≥ label horizon** between train and test (leakage prevention — interviewers probe this hard).
**Definition of Done:** a leakage unit test (no test-set label can be computed from any train-window data), class-balance report, notebook 03 with label-choice rationale.

### Stage 4 — Model training from scratch (1–1.5 weeks)
DeepLOB-family CNN feature extractor + **TCN (dilated causal convolutions) head instead of an LSTM** — deliberately, because a TCN is pure convolutions, which (a) you can hand-roll in C++ later and (b) supports incremental per-tick updates via a ring buffer. ~300K params, trains in hours on a free Kaggle GPU. Also train/evaluate on the public **FI-2010** benchmark so you have one number comparable to published papers.
**Key numbers produced:** macro-F1 vs logistic-regression baseline and vs published FI-2010 baselines.
**Definition of Done:** training reproducible from a seed, loss curves saved, baseline comparison table in README, notebook 04 explains every architectural choice with a "why not X" note (why not LSTM, why not transformer).

### Stage 5 — Honest evaluation (1 week — do not rush this)
The stage that separates you from every other student project. Compute: hit rate by horizon, **information coefficient**, **signal decay curve** (edge vs milliseconds elapsed — expect it to die fast; that's the point), PnL simulation **after taker fees + half-spread crossing**, and the **breakeven fee level**.
**Definition of Done:** notebook 05 with the decay chart and the cost-adjusted PnL curve, and a written conclusion that says out loud whether the signal survives retail fees (it likely won't — say so, and quantify the fee tier at which it would).

### Stage 6 — Compression: ONNX → int8 → distillation (1 week)
Export to ONNX, benchmark ONNX Runtime, apply static int8 quantization (with a calibration set), then distill into a ~30–50K-param student (KL + CE loss, temperature). Produce the **accuracy-vs-latency Pareto table** with 4 rows: PyTorch eager / ONNX fp32 / ONNX int8 / distilled student.
**Definition of Done:** Pareto data saved in `benchmarks/`, accuracy degradation quantified at each step, notebook 06 explains quantization and distillation mechanics.

### Stage 7 — C++ hot path (2–3 weeks — the centerpiece, budget accordingly)
Export weights to a custom binary format. Hand-roll the forward pass in C++: aligned buffers, zero heap allocation on the hot path, incremental feature updates (ring buffer — a new tick updates state instead of recomputing 100×40 inputs), `-O3 -march=native`, optional explicit SIMD intrinsics for the inner conv loops.
**Non-negotiable:** a parity test asserting C++ outputs match PyTorch within 1e-4 on 1,000 random inputs — correctness before speed, always.
Then the latency harness: pinned core, 10k-iteration warmup, monotonic clock, full histograms, **p50/p99/p99.9 (never averages)**, methodology documented in `docs/benchmark_methodology.md` (clock resolution, thermal throttling, coordinated omission — an HFT interviewer will probe each one; being probe-able is the goal).
**Definition of Done:** parity test green, latency histograms saved, Pareto chart now has 5 rows, methodology doc written.

### Stage 8 — Serving + dashboard + final README (4–6 days)
FastAPI service (`/signal`, `/book`, `/latency-stats`), Dockerized, single-page live dashboard (book ladder + signal + rolling latency percentiles). Final README rewritten **results-first**: the Pareto chart at the top, then the decay curve, then methodology links.
**Definition of Done:** `docker compose up` gives a working live dashboard; README contains every headline number with a link to the notebook/benchmark that produced it.

---

## 3. Milestone checkpoints (your "am I on track" tests)

| Checkpoint | You should be able to say... |
|---|---|
| After Stage 1 | "My capture daemon ran 24h, processed N million messages, resynced K times, and my book matches the exchange snapshot." |
| After Stage 2 | "My format stores an event in X bytes vs Y bytes of JSON, and I can replay Z million events/sec via mmap." |
| After Stage 4 | "My 300K-param model gets F1 = A on my data and B on FI-2010, vs C for logistic regression." |
| After Stage 5 | "My signal's edge has a half-life of ~X ms and breaks even at a fee of Y bps — below retail tiers, and here's why that's still a real result." |
| After Stage 7 | "Tick-to-signal p99: PyTorch ~N ms → ONNX int8 ~N µs → my C++ path ~N µs, at <1% F1 loss." |

---

## 4. Risk table

| Risk | Likelihood | Mitigation / fallback |
|---|---|---|
| C++ forward pass hard to get bit-correct | Medium | Parity test from day one of Stage 7; implement ops one at a time, testing each against PyTorch. **Fallback:** stop at ONNX Runtime int8 with the same rigorous harness — still a sub-ms story with a Pareto chart. |
| Model shows no predictive power | Low–Medium | FI-2010 first (published baselines prove the task is learnable); if live-data F1 is weak, that's *reportable* — the system story stands regardless. |
| Laptop can't run 24/7 capture | Medium | Capture in sessions (e.g., 8h/day for 2 weeks); the format supports multi-file datasets. Or a free-tier cloud VM for capture only. |
| Binance API changes / geo-restrictions | Low | Capture layer is exchange-agnostic behind an interface; Bybit/OKX adapters are ~50 lines each. |
| Notebook/module drift | Medium | Rule in CLAUDE.md: any change to a `.py` module requires updating its walkthrough notebook in the same session. |

---

## 5. Data budget (so nothing surprises you)

- `btcusdt@depth@100ms` + trades ≈ roughly 1–3 GB/day as raw JSON; your binary format should cut that ~10–20×.
- **2–3 weeks of capture is enough** for training + honest walk-forward evaluation. Start capture during Stage 2 so data accumulates while you build Stages 3–4.
- Keep raw JSON for the first 2 days only (for format-validation tests); after that, archive binary only.

## 6. Resume bullets this plan produces (templates — fill with YOUR numbers)

- Built a live market-data capture engine processing N msgs/sec from exchange websocket feeds with gap detection, auto-resync, and a custom binary format achieving X× compression and Y M-events/sec mmap replay.
- Trained a 300K-parameter CNN+TCN limit-order-book model from scratch (macro-F1 A vs B baseline on FI-2010), with leakage-free walk-forward evaluation, signal-decay analysis, and cost-adjusted breakeven-fee characterization.
- Compressed and re-implemented the model as a dependency-free C++ inference path (quantization + distillation + SIMD), reducing p99 tick-to-signal latency from N ms to N µs at <1% accuracy loss, with a documented benchmark methodology.
