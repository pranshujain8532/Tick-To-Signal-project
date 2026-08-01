"""Latency and accuracy of every Python-land inference variant, on one core.

    python benchmarks/bench_python_variants.py --iterations 100000

Produces the accuracy-vs-latency Pareto data for four variants — PyTorch eager,
ONNX Runtime fp32, ONNX Runtime int8, and the distilled student in int8 — and
saves percentiles, a histogram and macro-F1 for each to `benchmarks/`.

This is the last stop before Stage 7 hand-writes the forward pass in C++, so
the numbers here are the bar that work has to clear.

=============================== METHODOLOGY ===============================

Every choice below exists to make the number reproducible rather than
flattering. The same recipe is recorded in `docs/benchmark_methodology.md`.

**Batch size 1, and only 1.** This is the honest setting for tick-to-signal
and the one an interviewer will probe. A trading system does not receive a
batch of 128 order-book states and predict them together; it receives one tick
and must answer before the next one. Batching amortises weight loading and
fills vector units, so a batch-128 throughput number can be an order of
magnitude better per sample — and it describes a system nobody is building.
Quoting it would be measuring the wrong thing well.

**One thread, pinned to one core.** `torch.set_num_threads(1)` and
onnxruntime's `intra_op_num_threads = 1` stop the runtimes from spawning
worker pools; process affinity then pins execution to a single core via
`SetProcessAffinityMask` on Windows or `os.sched_setaffinity` elsewhere.
Without pinning, the OS migrates the process between cores, and every
migration costs a cold L1/L2 and shows up in the tail — which is exactly the
part of the distribution we care about. At batch 1 this graph is far too small
to fill several cores anyway, so threads would add synchronisation cost for no
parallel work.

**10,000 warmup iterations, discarded.** The first calls through any runtime
are unrepresentative: lazy allocation, one-time graph optimisation, JIT
kernel selection, cold instruction cache, and a CPU still ramping its clock.
Ten thousand is far past where those settle for a model this size.

**100,000 timed iterations, `time.perf_counter_ns`.** The highest-resolution
monotonic clock Python exposes. Each iteration is timed individually rather
than timing a loop and dividing, because the whole point is the *shape* of the
distribution — a mean hides the tail, and the tail is what a trading system
lives with.

**p50, p99, p99.9 — never a mean.** A mean latency is a number no request ever
experiences. The p99.9 is what a system sees several times a second at market
data rates.

**What is NOT controlled:** this is a laptop running a desktop OS, with no
thermal management, no isolated cores, no real-time scheduling priority, and
other processes alive. Absolute numbers would be better on tuned hardware; the
*relative* ordering of the four variants is the durable result. Stated here
rather than discovered later.
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.dataset import build_sample_index, gather_windows, load_sessions  # noqa: E402
from ml.distill import STUDENT_CONFIG, build_student  # noqa: E402
from ml.export import (  # noqa: E402
    DEPLOY_FEATURES,
    DEPLOY_WINDOW,
    check_parity,
    export_to_onnx,
    file_size_kib,
    make_session,
    quantize_to_int8,
)
from ml.metrics import macro_f1  # noqa: E402
from ml.model import ModelConfig, TickToSignalNet  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = REPO_ROOT / "benchmarks"
ARTIFACT_DIR = REPO_ROOT / "artifacts"

CALIBRATION_WINDOWS = 512
WARMUP_ITERATIONS = 10_000


def pin_to_single_core(core: int = 0) -> str:
    """Restrict this process to one core. Returns a description for the record.

    Core migration is the single largest source of tail noise in a
    single-threaded microbenchmark: each move costs a cold L1 and L2, and shows
    up precisely in the p99.9 this benchmark exists to measure.

    The explicit `argtypes`/`restype` are not decoration. Without them ctypes
    assumes a C `int` return, so `GetCurrentProcess`'s 64-bit pseudo-handle is
    truncated and `SetProcessAffinityMask` silently fails — the first version
    of this function reported "affinity request refused by the OS" on a machine
    that was perfectly willing to pin it, which would have quietly produced
    unpinned numbers under a comment claiming they were pinned.
    """
    try:
        if os.name == "nt":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            kernel32.SetProcessAffinityMask.restype = ctypes.c_int
            if kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), ctypes.c_size_t(1 << core)):
                return f"SetProcessAffinityMask(core {core})"
            return f"affinity refused by the OS (error {ctypes.get_last_error()})"
        os.sched_setaffinity(0, {core})
        return f"sched_setaffinity(core {core})"
    except Exception as error:  # noqa: BLE001 - a failed pin must not abort the run
        return f"unpinned ({type(error).__name__}: {error})"


def percentiles(samples_ns: np.ndarray) -> dict[str, float]:
    """The percentiles a latency-sensitive system actually cares about."""
    levels = {"p50": 50, "p90": 90, "p99": 99, "p99_9": 99.9, "p99_99": 99.99}
    result = {name: float(np.percentile(samples_ns, level)) / 1000.0 for name, level in levels.items()}
    result["min_us"] = float(samples_ns.min()) / 1000.0
    result["max_us"] = float(samples_ns.max()) / 1000.0
    # Reported for completeness and immediately deprecated in the text: a mean
    # latency is a number no single inference ever experiences.
    result["mean_us"] = float(samples_ns.mean()) / 1000.0
    return {key.replace("p", "p") if key.endswith("_us") else f"{key}_us": value for key, value in result.items()}


def histogram(samples_ns: np.ndarray, bins: int = 200) -> dict:
    """A binned histogram, so the full shape survives without a 100k-row file."""
    clipped = np.clip(samples_ns, None, np.percentile(samples_ns, 99.99))
    counts, edges = np.histogram(clipped, bins=bins)
    return {
        "counts": counts.tolist(),
        "edges_us": (edges / 1000.0).tolist(),
        "clipped_at_us": float(np.percentile(samples_ns, 99.99)) / 1000.0,
    }


def time_callable(run_once, iterations: int, warmup: int) -> np.ndarray:
    """Warm up, then time each call individually with the monotonic clock."""
    for _ in range(warmup):
        run_once()
    samples = np.empty(iterations, dtype=np.int64)
    for index in range(iterations):
        started = time.perf_counter_ns()
        run_once()
        samples[index] = time.perf_counter_ns() - started
    return samples


def torch_runner(model: TickToSignalNet, window: np.ndarray):
    """A closure that performs exactly one batch-1 forward pass."""
    tensor = torch.from_numpy(window[None, :, :].copy())
    model = model.eval()

    def run_once():
        with torch.no_grad():
            model(tensor)

    return run_once


def onnx_runner(session, window: np.ndarray):
    name = session.get_inputs()[0].name
    single = window[None, :, :].astype(np.float32)

    def run_once():
        session.run(None, {name: single})

    return run_once


def torch_accuracy(model: TickToSignalNet, windows: np.ndarray, truth: np.ndarray) -> float:
    model = model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(windows)).numpy().argmax(axis=1)
    return macro_f1(truth, predictions)


def onnx_accuracy(session, windows: np.ndarray, truth: np.ndarray) -> float:
    """Accuracy through the deployed batch-1 graph, one window at a time."""
    name = session.get_inputs()[0].name
    predictions = np.empty(len(windows), dtype=np.int64)
    for index, window in enumerate(windows):
        output = session.run(None, {name: window[None, :, :].astype(np.float32)})[0]
        predictions[index] = int(output[0].argmax())
    return macro_f1(truth, predictions)


def build_artifacts(teacher: TickToSignalNet, student: TickToSignalNet, calibration: np.ndarray) -> dict[str, Path]:
    """Export and quantise everything the benchmark needs, checking parity first."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "teacher_fp32": export_to_onnx(teacher, ARTIFACT_DIR / "teacher_fp32.onnx"),
        "student_fp32": export_to_onnx(student, ARTIFACT_DIR / "student_fp32.onnx"),
    }
    for name, model in (("teacher", teacher), ("student", student)):
        report = check_parity(model, paths[f"{name}_fp32"], trials=1_000, tolerance=1e-5)
        print(f"  {name}: {report.summary()}")
        if not report.passed:
            raise SystemExit(f"{name} failed ONNX parity; refusing to benchmark a graph that is not the model")

    paths["teacher_int8"] = quantize_to_int8(
        paths["teacher_fp32"], ARTIFACT_DIR / "teacher_int8.onnx", calibration
    )
    paths["student_int8"] = quantize_to_int8(
        paths["student_fp32"], ARTIFACT_DIR / "student_int8.onnx", calibration
    )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERATIONS)
    parser.add_argument("--core", type=int, default=0)
    parser.add_argument(
        "--accuracy-samples",
        type=int,
        default=0,
        help=(
            "0 (the default) scores the whole held-out block. A subset is a trap here: "
            "the test period is strongly non-stationary, and the first 4,000 of 9,292 "
            "samples put the student at 0.649 macro-F1 against 0.595 on the full block"
        ),
    )
    parser.add_argument("--teacher-run", default=None)
    parser.add_argument("--student-checkpoint", default="checkpoints/student_distilled.pt")
    args = parser.parse_args()

    affinity = pin_to_single_core(args.core)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print(f"affinity: {affinity}; torch threads: {torch.get_num_threads()}")

    run_path = Path(args.teacher_run or sorted(glob.glob(str(BENCHMARK_DIR / "train_ours_*.json")))[-1])
    teacher_run = json.loads(run_path.read_text(encoding="utf-8"))
    teacher = TickToSignalNet(ModelConfig())
    teacher.load_state_dict(torch.load(REPO_ROOT / teacher_run["artefacts"]["checkpoint"], map_location="cpu"))
    student = build_student()
    student.load_state_dict(torch.load(REPO_ROOT / args.student_checkpoint, map_location="cpu"))

    sessions = load_sessions(sorted(glob.glob(teacher_run["data"]["tapes"])))
    sample_index = build_sample_index(sessions)
    fold = teacher_run["data"]["fold"]
    train_positions = np.arange(fold["train_start"], fold["train_end"] + 1)
    test_positions = np.arange(fold["test_start"], fold["test_end"] + 1)
    if args.accuracy_samples:
        test_positions = test_positions[: args.accuracy_samples]

    generator = np.random.default_rng(0)
    calibration = gather_windows(
        sessions, sample_index, generator.choice(train_positions, size=CALIBRATION_WINDOWS, replace=False)
    )
    windows = gather_windows(sessions, sample_index, test_positions)
    truth = sample_index.labels[test_positions]

    print("\nexporting and checking parity before any timing:")
    paths = build_artifacts(teacher, student, calibration)

    sample_window = windows[0]
    variants = [
        ("pytorch_eager_fp32", teacher.parameter_count(), None,
         torch_runner(teacher, sample_window), lambda: torch_accuracy(teacher, windows, truth)),
    ]
    for label, key, parameters in (
        ("onnx_fp32", "teacher_fp32", teacher.parameter_count()),
        ("onnx_int8", "teacher_int8", teacher.parameter_count()),
        # student_fp32 is the rung this frontier was missing, and Stage 7b is why
        # it matters: the hand-written C++ path implements exactly this model
        # (fp32, BatchNorm folded), so without it the C++ row on the Pareto chart
        # would have to borrow student_int8's accuracy, which is a different
        # numerical configuration. Measured here rather than assumed equal.
        ("student_fp32", "student_fp32", student.parameter_count()),
        ("student_int8", "student_int8", student.parameter_count()),
    ):
        session = make_session(paths[key])
        variants.append(
            (label, parameters, paths[key], onnx_runner(session, sample_window),
             lambda s=session: onnx_accuracy(s, windows, truth))
        )

    results = []
    for label, parameters, path, run_once, accuracy_fn in variants:
        print(f"\n{label}: measuring accuracy on {len(windows):,} held-out windows...")
        score = accuracy_fn()
        print(f"  macro-F1 {score:.4f}; timing {args.iterations:,} iterations after {args.warmup:,} warmup...")
        started = time.perf_counter()
        samples = time_callable(run_once, args.iterations, args.warmup)
        stats = percentiles(samples)
        print(
            f"  p50 {stats['p50_us']:.1f} us | p99 {stats['p99_us']:.1f} us | "
            f"p99.9 {stats['p99_9_us']:.1f} us  ({time.perf_counter() - started:.0f}s)"
        )
        results.append(
            {
                "variant": label,
                "parameters": int(parameters),
                "artifact": path.name if path else "in-memory (torch)",
                "size_kib": file_size_kib(path) if path else None,
                "macro_f1": score,
                "latency_us": stats,
                "histogram": histogram(samples),
            }
        )

    payload = {
        "benchmark": "python_variants",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "method": {
            "batch_size": 1,
            "affinity": affinity,
            "torch_threads": torch.get_num_threads(),
            "onnxruntime_intra_op_threads": 1,
            "warmup_iterations": args.warmup,
            "timed_iterations": args.iterations,
            "clock": "time.perf_counter_ns",
            "accuracy_samples": int(len(windows)),
            "not_controlled": "laptop, desktop OS, no thermal control, no core isolation",
        },
        "teacher_run": run_path.name,
        "variants": results,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = BENCHMARK_DIR / f"python_variants_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n{'variant':<22} {'params':>9} {'macro-F1':>9} {'p50 us':>9} {'p99 us':>9} {'p99.9 us':>10}")
    for entry in results:
        latency = entry["latency_us"]
        print(
            f"{entry['variant']:<22} {entry['parameters']:>9,} {entry['macro_f1']:>9.4f} "
            f"{latency['p50_us']:>9.1f} {latency['p99_us']:>9.1f} {latency['p99_9_us']:>10.1f}"
        )
    print(f"\nsaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
