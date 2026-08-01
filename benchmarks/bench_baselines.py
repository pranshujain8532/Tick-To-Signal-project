"""Score the baselines the deep model has to beat, on exactly the same split.

Run after a capture, from the repo root:

    python benchmarks/bench_baselines.py --tapes "data/stage3/*.tape"

Writes a timestamped JSON under `benchmarks/` and prints a summary. The
numbers here are what the README's model row is compared against, so the
splitting must be identical to `ml/train.py`'s — it is imported from there
rather than reimplemented, because two copies of a splitting rule is exactly
how a comparison quietly stops being fair.

Three baselines, in increasing order of how much they should worry us:

  1. **Majority class.** Always predict whichever class was most common in the
     training block. Macro-F1 around 0.17 on a balanced three-class problem.
     Any result that does not clear this is not a result.
  2. **Queue imbalance with a dead zone.** Two lines of arithmetic, no fitted
     parameters beyond a threshold chosen on the training block. This is the
     classic cheap microstructure signal and it is the honest floor: if the
     network cannot beat it, 320,000 parameters bought nothing.
  3. **Multinomial logistic regression on the last snapshot.** 40 features,
     123 parameters, no history at all. The gap between this and the deep
     model is precisely what the 100-step window and the convolutions are
     worth — which is the number Stage 6 and Stage 7 are about to spend weeks
     compressing, so it is worth knowing before starting.
"""

from __future__ import annotations

import argparse
import glob
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.baseline import evaluate_imbalance_rule, fit_logistic_baseline
from ml.dataset import build_sample_index, describe_sessions, gather_last_rows, load_sessions
from ml.labels import DEFAULT_SMOOTHING_K
from ml.metrics import format_report, majority_class_baseline
from ml.splits import required_embargo, walk_forward_splits
from ml.train import carve_validation

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tapes", default="data/stage3/*.tape")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    tape_paths = sorted(glob.glob(args.tapes))
    if not tape_paths:
        raise SystemExit(f"no tapes matched {args.tapes!r}")
    sessions = load_sessions(tape_paths)
    sample_index = build_sample_index(sessions)
    print(describe_sessions(sessions, sample_index))

    embargo = required_embargo(100, DEFAULT_SMOOTHING_K)
    split = walk_forward_splits(len(sample_index), args.folds, embargo)[-1]
    # The deep model fits on this same subset — the validation tail is carved
    # off for early stopping and must not be part of the baseline's training
    # data either, or the two models would see different amounts of history.
    fit_positions, _validation_positions = carve_validation(split, embargo, args.validation_fraction)
    test_positions = split.test_indices()

    fit_features = gather_last_rows(sessions, sample_index, fit_positions)
    test_features = gather_last_rows(sessions, sample_index, test_positions)
    fit_labels = sample_index.labels[fit_positions]
    test_labels = sample_index.labels[test_positions]
    print(f"\nfit {len(fit_positions):,} | test {len(test_positions):,} | embargo {embargo}")

    majority = majority_class_baseline(fit_labels, test_labels)
    print(f"\nmajority class        : macro-F1 {majority['macro_f1']:.4f}  acc {majority['accuracy']:.4f}")

    imbalance = evaluate_imbalance_rule(fit_features, fit_labels, test_features, test_labels)
    print(
        f"queue imbalance rule  : macro-F1 {imbalance['macro_f1']:.4f}  acc {imbalance['accuracy']:.4f}  "
        f"(dead zone {imbalance['dead_zone']:.4f} chosen on train)"
    )

    started = time.perf_counter()
    logistic = fit_logistic_baseline(fit_features, fit_labels, test_features, test_labels, seed=args.seed)
    fit_seconds = time.perf_counter() - started
    print(
        f"logistic regression   : macro-F1 {logistic.macro_f1:.4f}  acc {logistic.accuracy:.4f}  "
        f"({logistic.coefficient_count} parameters, fitted in {fit_seconds:.1f}s)"
    )
    print("\nlogistic regression, per class:")
    print(format_report(test_labels, logistic.predictions))

    payload = {
        "benchmark": "baselines",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine": {"platform": platform.platform(), "python": platform.python_version()},
        "data": {
            "tapes": args.tapes,
            "sessions": [session.name for session in sessions],
            "samples": len(sample_index),
            "fold": asdict(split),
            "embargo": embargo,
            "fit_samples": len(fit_positions),
            "test_samples": len(test_positions),
        },
        "results": {
            "majority_class": majority,
            "queue_imbalance_rule": imbalance,
            "logistic_regression": {
                "macro_f1": logistic.macro_f1,
                "accuracy": logistic.accuracy,
                "parameters": logistic.coefficient_count,
                "fit_seconds": fit_seconds,
            },
        },
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = REPO_ROOT / "benchmarks" / f"baselines_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nsaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
