"""Does the calibration set actually decide int8 accuracy? Measure it.

    python benchmarks/bench_calibration.py

Static int8 quantisation picks its activation scales by watching the model run
over calibration inputs, so those inputs define the numeric range the deployed
graph can represent at all. That is easy to assert and easy to get wrong, so
this script measures it: the same fp32 graph is quantised several times, each
with a different calibration set or calibrator, and each result is scored on
the held-out block.

It exists because the first version of `ml/export.py` asserted "use real data"
and stopped there — which turned out to be incomplete in an instructive way.
The saved JSON is what notebook 06 and the interview notes quote.
"""

from __future__ import annotations

import argparse
import glob
import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.dataset import build_sample_index, gather_windows, load_sessions  # noqa: E402
from ml.export import (  # noqa: E402
    DEPLOY_FEATURES,
    DEPLOY_WINDOW,
    export_to_onnx,
    make_session,
    quantize_to_int8,
    run_onnx,
)
from ml.metrics import macro_f1  # noqa: E402
from ml.model import ModelConfig, TickToSignalNet  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = REPO_ROOT / "benchmarks"
ARTIFACT_DIR = REPO_ROOT / "artifacts"


def calibration_sets(real_windows: np.ndarray, generator: np.random.Generator) -> list[tuple[str, np.ndarray, dict]]:
    """The calibration sets to compare, and the claim each one tests."""
    shape = (512, DEPLOY_WINDOW, DEPLOY_FEATURES)
    return [
        ("real, n=32", real_windows[:32], {"percentile": 99.99}),
        ("real, n=128", real_windows[:128], {"percentile": 99.99}),
        ("real, n=512", real_windows[:512], {"percentile": 99.99}),
        ("real, n=512, MinMax", real_windows[:512], {"percentile": None}),
        ("real, n=512, pct 99.9", real_windows[:512], {"percentile": 99.9}),
        ("real, n=512, pct 99.0", real_windows[:512], {"percentile": 99.0}),
        ("gaussian noise", generator.standard_normal(shape).astype(np.float32), {"percentile": None}),
        ("noise x8, too wide", (generator.standard_normal(shape) * 8).astype(np.float32), {"percentile": None}),
        ("noise x0.05, too narrow", (generator.standard_normal(shape) * 0.05).astype(np.float32), {"percentile": None}),
    ]


def main() -> int:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--teacher-run", default=None)
    args = parser.parse_args()

    run_path = Path(args.teacher_run or sorted(glob.glob(str(BENCHMARK_DIR / "train_ours_*.json")))[-1])
    teacher_run = json.loads(run_path.read_text(encoding="utf-8"))
    model = TickToSignalNet(ModelConfig())
    model.load_state_dict(torch.load(REPO_ROOT / teacher_run["artefacts"]["checkpoint"], map_location="cpu"))
    model.eval()

    sessions = load_sessions(sorted(glob.glob(teacher_run["data"]["tapes"])))
    sample_index = build_sample_index(sessions)
    fold = teacher_run["data"]["fold"]
    train_positions = np.arange(fold["train_start"], fold["train_end"] + 1)
    test_positions = np.arange(fold["test_start"], fold["test_end"] + 1)

    generator = np.random.default_rng(0)
    # Calibration comes from the TRAIN split; test windows would leak the
    # evaluation distribution into the deployed artefact.
    real = gather_windows(
        sessions, sample_index, generator.choice(train_positions, size=1024, replace=False)
    )
    windows = gather_windows(sessions, sample_index, test_positions)
    truth = sample_index.labels[test_positions]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    fp32_path = export_to_onnx(model, ARTIFACT_DIR / "calibration_study_fp32.onnx")
    with torch.no_grad():
        reference = model(torch.from_numpy(windows)).numpy().argmax(axis=1)
    baseline = macro_f1(truth, reference)
    print(f"fp32 PyTorch macro-F1 on {len(windows):,} held-out windows: {baseline:.4f}")
    print(f"real activation percentiles: 99.9%={np.percentile(np.abs(real), 99.9):.2f} "
          f"99.99%={np.percentile(np.abs(real), 99.99):.2f} max={np.abs(real).max():.2f}\n")

    print(f"{'calibration set':<26} {'macro-F1':>9} {'vs fp32':>9} {'agreement':>10}")
    rows = []
    for name, calibration, options in calibration_sets(real, generator):
        path = ARTIFACT_DIR / "calibration_study_int8.onnx"
        quantize_to_int8(fp32_path, path, calibration, percentile=options["percentile"])
        predictions = run_onnx(make_session(path), windows).argmax(axis=1)
        score = macro_f1(truth, predictions)
        agreement = float(np.mean(predictions == reference))
        rows.append(
            {
                "calibration": name,
                "samples": int(len(calibration)),
                "percentile": options["percentile"],
                "macro_f1": score,
                "delta_vs_fp32": score - baseline,
                "argmax_agreement": agreement,
            }
        )
        print(f"{name:<26} {score:>9.4f} {score - baseline:>+9.4f} {agreement:>10.4f}")
        path.unlink(missing_ok=True)

    payload = {
        "benchmark": "calibration",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine": {"platform": platform.platform(), "python": platform.python_version()},
        "teacher_run": run_path.name,
        "test_samples": int(len(windows)),
        "fp32_macro_f1": baseline,
        "activation_percentiles": {
            "p99_9": float(np.percentile(np.abs(real), 99.9)),
            "p99_99": float(np.percentile(np.abs(real), 99.99)),
            "max": float(np.abs(real).max()),
            "std": float(real.std()),
        },
        "rows": rows,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = BENCHMARK_DIR / f"calibration_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fp32_path.unlink(missing_ok=True)
    print(f"\nsaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
