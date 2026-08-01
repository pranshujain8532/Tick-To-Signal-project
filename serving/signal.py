"""Turn a stream of book anchors into a signal, and time the part that is ours.

WHAT
    `SignalEngine` keeps the rolling history one model input needs, builds the
    `[100, 40]` window with `ml/features.py`, runs it through the quantised ONNX
    student, and reports how long the forward pass took. `MeasuredClock`
    establishes, at startup, whether the clock can resolve that duration at all.

WHY THIS IS NOT INSIDE api.py
    The constitution exempts `serving/api.py` from notebook coverage because a
    server loop cannot run in a cell. The prediction path is not a server loop —
    it is a pure function of the last 599 anchors — so it lives here, where it is
    importable, testable and demonstrable.

WHAT IS TIMED, AND WHAT IS NOT — read this before quoting `serving_infer_us`.
    `serving_infer_us` covers `session.run(...)` and nothing else: the ONNX int8
    forward pass for one `[1, 100, 40]` input. It EXCLUDES feature construction,
    which is the rolling z-score and the mid-relative price arithmetic in
    `ml/features.py`, and which is real work this process does on every anchor.
    Feature time is measured separately and reported by `/latency` under its own
    key, because folding it into one number would make the serving figure
    incomparable with the Stage 6 frontier that was measured the same way.

    THE C++ FIGURE IS A DIFFERENT MEASUREMENT OF A DIFFERENT THING. Stage 7b's
    ~11 us covers a hand-written forward pass from an already-prepared [40]
    feature column to three logits, measured by `inference_cpp/bench/bench.cpp`
    on a pinned core at high priority over a million iterations. It is not what
    this process runs, it is not measured by this process, and no endpoint here
    reports it as if it were. CLAUDE.md forbids that confusion by name.

DESIGN DECISION — a ring of raw snapshots, not a ring of finished features.
    `ml/features.py` normalises with a causal rolling z-score over 500 rows, so
    row t's features depend on rows t-499..t. Caching finished feature rows would
    therefore be wrong: the normalisation of an old row does not change, but the
    module recomputes the window from raw snapshots as one array and that is the
    path Stage 3 tested. Rejected alternative: an incremental z-score updated per
    tick, which is what `inference_cpp` does for the model and would be the right
    move if this path were hot. It is not hot — it is ~800 us of ONNX against
    ~40 us of numpy — so the version that provably matches training wins.

DESIGN DECISION — history is destroyed at a session boundary, never carried.
    `reset()` is called on every `SessionBoundary`, so the 599-anchor warmup is
    paid again after every resync, loop wrap and seek. During those 599 anchors
    there is NO signal and the engine says so rather than repeating its last
    answer. Carrying the ring across a boundary would normalise one session's
    book against another session's statistics, across a gap containing price
    moves nobody observed — the same error `ml/eval.py:resolve_prices_per_session`
    exists to prevent.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ml.features import (
    DEFAULT_NORMALISATION_LOOKBACK,
    DEPTH_LEVELS,
    FEATURE_COUNT,
    WINDOW_LENGTH,
    build_input_tensor,
)

LOGGER = logging.getLogger("serving.signal")

# One input needs `window_length + lookback - 1` snapshots: the first row of the
# window needs its own full lookback behind it. 100 + 500 - 1 = 599.
HISTORY_ROWS = WINDOW_LENGTH + DEFAULT_NORMALISATION_LOOKBACK - 1


def snapshot_dtype(depth: int = DEPTH_LEVELS) -> np.dtype:
    """The minimal structured dtype `ml/features.py` reads.

    Only `bids` and `asks` — the feature builder never touches the tape's other
    snapshot fields, and declaring fields this module does not fill would invite
    someone to read a zero as data.
    """
    return np.dtype([("bids", np.int64, (depth, 2)), ("asks", np.int64, (depth, 2))])


class MeasuredClock:
    """A monotonic clock that reports what it can actually resolve.

    WHY THIS EXISTS, with the number that forced it: Stage 7b's C++ harness found
    `std::chrono::steady_clock` advertising a one-nanosecond period and resolving
    about one millisecond on that toolchain, which recorded 19,518 of 20,000
    forward passes as taking exactly zero time. The published "p50 5,000 us" was
    a 1 ms grid, not a measurement.

    Python's `perf_counter_ns` is a different clock on a different stack and is
    expected to be fine here — but "expected to be fine" is what that C++ run
    also assumed. So this measures, logs, and exposes the result, and
    `/latency` refuses to publish percentiles it cannot resolve.
    """

    def __init__(self, probes: int = 200_000) -> None:
        self.resolution_ns = self._measure(probes)
        self.overhead_ns = self._overhead(probes // 2)

    @staticmethod
    def _measure(probes: int) -> int:
        """Smallest non-zero gap between two back-to-back reads."""
        smallest = 0
        for _ in range(probes):
            first = time.perf_counter_ns()
            second = time.perf_counter_ns()
            gap = second - first
            if gap > 0 and (smallest == 0 or gap < smallest):
                smallest = gap
        return smallest

    @staticmethod
    def _overhead(probes: int) -> int:
        """Median cost of a read pair: the noise floor under every sample."""
        samples = []
        for _ in range(probes):
            first = time.perf_counter_ns()
            samples.append(time.perf_counter_ns() - first)
        samples.sort()
        return samples[len(samples) // 2]

    def can_resolve(self, duration_ns: float, ratio: int = 100) -> bool:
        """Is `duration_ns` at least `ratio` times the clock's resolution?

        A hundred ticks per measurement means quantisation contributes about 1%,
        which is far below this machine's run-to-run variance. Below that the
        percentiles describe the clock rather than the code.

        Returns a plain `bool`, not whatever numpy hands back from comparing a
        float64 — a `numpy.bool_` reaches the JSON encoder as an unserialisable
        object and takes the whole endpoint down with it.
        """
        return bool(self.resolution_ns > 0 and duration_ns >= ratio * self.resolution_ns)

    def describe(self) -> dict[str, Any]:
        return {
            "clock": "time.perf_counter_ns",
            "resolution_ns": self.resolution_ns,
            "read_pair_overhead_ns": self.overhead_ns,
        }


@dataclass(frozen=True)
class SignalResult:
    """One inference, or an honest statement that there was not one yet."""

    ready: bool
    probabilities: tuple[float, float, float] | None
    score: float | None
    infer_ns: int
    feature_ns: int
    anchors_until_ready: int


class SignalEngine:
    """Rolling history plus the quantised student, with the boundary enforced."""

    def __init__(self, model_path: Path | str, price_scale: int, qty_scale: int,
                 depth: int = DEPTH_LEVELS) -> None:
        self.model_path = Path(model_path)
        self.price_scale = price_scale
        self.qty_scale = qty_scale
        self.session = _open_session(self.model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.clock = MeasuredClock()

        # Preallocated ring of raw snapshots. Sized once; `push` never allocates
        # it again, so the steady-state cost is the ordered copy in `_ordered`.
        self._ring = np.zeros(HISTORY_ROWS, dtype=snapshot_dtype(depth))
        self._write = 0
        self._filled = 0
        self._warm_up_forward_pass()

    def _warm_up_forward_pass(self) -> None:
        """Run one throwaway inference so the first real one is not an outlier.

        The first call through any runtime allocates arenas, selects kernels and
        touches cold instruction cache. Stage 6 discards 10,000 iterations for
        this reason; one is enough here because the serving percentiles are
        rolling and a single outlier ages out, but zero would put a 10x sample in
        the histogram at second one of every demo.
        """
        probe = np.zeros((1, WINDOW_LENGTH, FEATURE_COUNT), dtype=np.float32)
        self.session.run(None, {self.input_name: probe})

    def reset(self) -> None:
        """Forget everything. Called on every session boundary."""
        self._write = 0
        self._filled = 0
        self._ring[:] = 0

    @property
    def ready(self) -> bool:
        return self._filled >= HISTORY_ROWS

    @property
    def anchors_until_ready(self) -> int:
        """How many more anchors before a window can be built. 0 once warm."""
        return max(0, HISTORY_ROWS - self._filled)

    def push(self, bids: np.ndarray, asks: np.ndarray) -> SignalResult:
        """Add one anchor and, once warm, produce a signal for it."""
        self._ring["bids"][self._write] = bids
        self._ring["asks"][self._write] = asks
        self._write = (self._write + 1) % HISTORY_ROWS
        self._filled = min(self._filled + 1, HISTORY_ROWS)

        if not self.ready:
            return SignalResult(False, None, None, 0, 0, HISTORY_ROWS - self._filled)

        feature_start = time.perf_counter_ns()
        window = build_input_tensor(self._ordered(), self.price_scale, self.qty_scale)
        feature_ns = time.perf_counter_ns() - feature_start

        model_input = window[None, :, :].astype(np.float32)
        infer_start = time.perf_counter_ns()
        logits = self.session.run(None, {self.input_name: model_input})[0][0]
        infer_ns = time.perf_counter_ns() - infer_start

        probabilities = _softmax(logits)
        score = float(probabilities[2] - probabilities[0])
        return SignalResult(
            ready=True,
            probabilities=(float(probabilities[0]), float(probabilities[1]), float(probabilities[2])),
            score=score,
            infer_ns=infer_ns,
            feature_ns=feature_ns,
            anchors_until_ready=0,
        )

    def _ordered(self) -> np.ndarray:
        """The ring as one oldest-first array, which is what the features want.

        Two slices and one concatenate: 599 records, about 96 KB. Cheap enough
        that avoiding it would be optimising the wrong thing — the ONNX pass next
        door costs twenty times more.
        """
        if self._write == 0:
            return self._ring
        return np.concatenate((self._ring[self._write :], self._ring[: self._write]))

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.model_path.name,
            "variant": "ONNX int8 distilled student",
            "history_rows_required": HISTORY_ROWS,
            "window_length": WINDOW_LENGTH,
            "normalisation_lookback": DEFAULT_NORMALISATION_LOOKBACK,
            **self.clock.describe(),
        }


def _open_session(model_path: Path):
    """Open the ONNX session, single-threaded.

    THE SETTINGS ARE ml/export.py:make_session's, and the duplication is
    deliberate. Importing that function would import `ml.export`, which imports
    torch at module scope — dragging a ~2 GB dependency into a container whose
    entire job is to run a 126 KB quantised graph. The serving image ships
    numpy, onnxruntime, fastapi and uvicorn and nothing else, and that is the
    whole point of having exported to ONNX in Stage 6: the deployment artefact
    does not need the framework that trained it.

    One thread, not several, and that is a measurement decision rather than a
    performance one. At batch 1 this graph cannot fill several cores, so extra
    threads add synchronisation cost and make the latency depend on what else
    the machine is doing. It is also what the Stage 6 frontier was measured
    with, so `/latency` and `/pareto` describe the same configuration.

    If `make_session` ever changes, this must change with it. There is no way to
    assert that across the torch boundary without importing torch, so it is
    stated here instead.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Stable softmax. The model emits logits; the dashboard wants probabilities.

    Done here rather than in the graph because the Stage 6 parity fixtures
    compare logits, and a softmax in the exported graph would have hidden a
    constant offset between PyTorch and ONNX from that check.
    """
    shifted = logits - np.max(logits)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()
