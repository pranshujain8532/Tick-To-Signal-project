"""Run the Stage 5 signal characterisation and save every artefact it produces.

    python benchmarks/bench_evaluation.py --run benchmarks/train_ours_btcusdt_*.json

Loads a saved checkpoint, replays it over the *held-out test fold only*, and
writes a timestamped JSON plus the decay and PnL charts that the README and
notebook 05 both read. Nothing in this file is retyped anywhere; the notebook
loads the JSON.

METHOD
    The fold is reconstructed from the training run's own record rather than
    recomputed, so the test block evaluated here is byte-identical to the one
    the model never saw. Reconstructing it independently would risk an
    off-by-one that quietly turns a held-out score into a training score.
"""

from __future__ import annotations

import argparse
import glob
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.dataset import BatchedWindowLoader, build_sample_index, load_sessions  # noqa: E402
from ml.eval import (  # noqa: E402
    BINANCE_TAKER_FEE_BPS,
    breakeven_fee_bps,
    block_shuffle_null,
    falsify,
    ic_distribution,
    pooled_decay_curve,
    pooled_forward_returns,
    resolve_prices_per_session,
    signal_from_probabilities,
    simulate_trades_from_prices,
)
from ml.labels import DEFAULT_SMOOTHING_K  # noqa: E402
from ml.metrics import confusion_matrix, macro_f1, per_class_scores  # noqa: E402
from ml.model import ModelConfig, TickToSignalNet  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = REPO_ROOT / "benchmarks"

# Wall-clock horizons for the decay curve. Spread over three decades because a
# half-life is only meaningful if the curve is measured on both sides of it.
DECAY_HORIZONS_MS = (100.0, 250.0, 500.0, 1_000.0, 2_000.0, 3_500.0, 5_000.0, 10_000.0, 20_000.0, 30_000.0)
FEE_LEVELS_BPS = (0.0, 1.0, 2.0, 5.0, 7.5, 10.0)


def predict_probabilities(
    checkpoint_path: Path,
    sessions: list,
    sample_index,
    positions: np.ndarray,
    device: str,
) -> np.ndarray:
    """Softmax outputs for every test sample, in index order."""
    model = TickToSignalNet(ModelConfig())
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()

    loader = BatchedWindowLoader(sessions, sample_index, positions, batch_size=512, shuffle=False)
    outputs = []
    with torch.no_grad():
        for windows, _labels in loader:
            batch = torch.from_numpy(windows).to(device, dtype=torch.float32)
            outputs.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(outputs)


def label_horizon_ms(sessions: list, session_ids: np.ndarray, rows: np.ndarray, horizon_rows: int) -> float:
    """How long `horizon_rows` snapshots actually take, in milliseconds.

    Measured directly as the median of `timestamp[row + k] - timestamp[row]`
    rather than as `k * median_gap`. The naive version returns **zero** on our
    tapes, and the reason is worth knowing: `local_ts_ns` is stamped once per
    *websocket frame*, and one frame carrying 39 level updates produces about
    four snapshots that all share it. So the median gap between adjacent
    snapshots is genuinely 0 ns, and multiplying it by anything stays 0.
    Spanning the whole k-row window sidesteps the ties entirely.
    """
    spans = []
    for session_id in np.unique(session_ids):
        selected = session_ids == session_id
        session = sessions[int(session_id)]
        start_rows = rows[selected]
        end_rows = start_rows + horizon_rows
        inside = end_rows < len(session.timestamps_ns)
        if inside.any():
            spans.append(
                session.timestamps_ns[end_rows[inside]] - session.timestamps_ns[start_rows[inside]]
            )
    if not spans:
        return 0.0
    return float(np.median(np.concatenate(spans))) / 1e6


def build_report(run_record: dict, device: str) -> dict:
    tape_glob = run_record["data"]["tapes"]
    sessions = load_sessions(sorted(glob.glob(tape_glob)))
    sample_index = build_sample_index(sessions)

    fold = run_record["data"]["fold"]
    positions = np.arange(fold["test_start"], fold["test_end"] + 1)
    checkpoint = REPO_ROOT / run_record["artefacts"]["checkpoint"]

    probabilities = predict_probabilities(checkpoint, sessions, sample_index, positions, device)
    signal = signal_from_probabilities(probabilities)
    predictions = probabilities.argmax(axis=1)
    truth = sample_index.labels[positions]

    session_ids = sample_index.session_of_sample[positions]
    rows = sample_index.end_row_of_sample[positions]
    horizon_ms = label_horizon_ms(sessions, session_ids, rows, DEFAULT_SMOOTHING_K)
    label_horizon_return = pooled_forward_returns(sessions, session_ids, rows, horizon_ms)

    scores = per_class_scores(truth, predictions)
    distribution = ic_distribution(signal, label_horizon_return, block_size=500)
    curve = pooled_decay_curve(signal, sessions, session_ids, rows, DECAY_HORIZONS_MS)

    entry_bid, entry_ask, exit_bid, exit_ask, tradeable = resolve_prices_per_session(
        sessions, session_ids, rows, DEFAULT_SMOOTHING_K
    )
    confidence = float(np.quantile(np.abs(signal), 0.70))
    sweep = [
        simulate_trades_from_prices(
            signal, entry_bid, entry_ask, exit_bid, exit_ask, tradeable, confidence, fee
        )
        for fee in FEE_LEVELS_BPS
    ]
    gross = sweep[0]
    null = block_shuffle_null(signal, label_horizon_return, block_size=500, trials=200, seed=0)
    # Sweep the shift well past the signal's measured decay: persistence fades
    # with distance, a leak would not.
    controls = falsify(
        signal,
        label_horizon_return,
        shift=DEFAULT_SMOOTHING_K,
        seed=0,
        extra_shifts=(300, 900, 2_000, 4_000),
    )
    spreads = (entry_ask - entry_bid) / ((entry_ask + entry_bid) / 2.0) * 1e4

    return {
        "benchmark": "evaluation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine": {"platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__},
        "source_run": Path(run_record.get("_path", "")).name,
        "checkpoint": run_record["artefacts"]["checkpoint"],
        "test_fold": fold,
        "samples": int(len(positions)),
        "sessions_in_fold": [sessions[int(i)].name for i in np.unique(session_ids)],
        "label_horizon_ms": horizon_ms,
        "classification": {
            "macro_f1": macro_f1(truth, predictions),
            "accuracy": float(np.mean(truth == predictions)),
            "per_class_precision": scores["precision"].tolist(),
            "per_class_recall": scores["recall"].tolist(),
            "per_class_f1": scores["f1"].tolist(),
            "support": scores["support"].tolist(),
            "confusion_matrix": confusion_matrix(truth, predictions).tolist(),
            "base_rates": (np.bincount(truth, minlength=3) / len(truth)).tolist(),
        },
        "information_coefficient": {
            "pooled": distribution.pooled_ic,
            "block_size": distribution.block_size,
            "block_ics": distribution.block_ics.tolist(),
            "mean": distribution.mean,
            "median": distribution.median,
            "std": distribution.standard_deviation,
            "fraction_positive": distribution.fraction_positive,
            "information_ratio": distribution.information_ratio,
        },
        "decay": {
            "rows": curve.as_rows(),
            "half_life_ms": curve.half_life_ms,
            "decay_rate_per_ms": curve.decay_rate_per_ms,
            "fit_r_squared": curve.fit_quality,
            "peak_horizon_ms": curve.peak_horizon_ms,
            "peak_ic": curve.peak_ic,
        },
        "trading": {
            "confidence_threshold": confidence,
            "trade_count": gross.trade_count,
            "long_count": gross.long_count,
            "short_count": gross.short_count,
            "gross_bps_per_trade": gross.gross_bps_per_trade,
            "breakeven_fee_bps": breakeven_fee_bps(gross.gross_bps_per_trade),
            "median_spread_bps": float(np.nanmedian(spreads)),
            "fee_sweep": [
                {
                    "fee_bps": s.fee_bps,
                    "net_bps_per_trade": s.net_bps_per_trade,
                    "total_net_bps": s.total_net_bps,
                    "win_rate": s.win_rate,
                }
                for s in sweep
            ],
            "binance_taker_tiers_bps": BINANCE_TAKER_FEE_BPS,
        },
        "falsification": {
            "true_ic": controls.true_ic,
            "shifted_ics": {str(k): v for k, v in controls.shifted_ics.items()},
            "block_shuffled_ic": controls.block_shuffled_ic,
            "shuffled_signal_ic": controls.shuffled_signal_ic,
            "reversed_signal_ic": controls.reversed_signal_ic,
            "null_distribution": {
                "trials": int(len(null.ics)),
                "block_size": 500,
                "mean": null.mean,
                "std": null.standard_deviation,
                "z_score": null.z_score,
                "exceedance_rate": null.exceedance_rate,
                "ics": null.ics.tolist(),
            },
        },
        "_cumulative": {str(s.fee_bps): s.cumulative_net_bps.tolist() for s in sweep},
    }


def save_charts(report: dict, stem: str) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    decay_path = BENCHMARK_DIR / f"{stem}_decay.png"
    horizons = [row["horizon_ms"] for row in report["decay"]["rows"]]
    ics = [row["ic"] for row in report["decay"]["rows"]]
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(horizons, ics, marker="o", color="#264653", label="measured IC")
    half_life = report["decay"]["half_life_ms"]
    if np.isfinite(half_life) and ics[0] > 0:
        fitted = ics[0] * np.exp(-np.log(2) * (np.array(horizons) - horizons[0]) / half_life)
        axis.plot(horizons, fitted, "--", color="#e76f51", label=f"fit, half-life {half_life:,.0f} ms")
    axis.axhline(0, color="grey", linewidth=0.8)
    axis.set_xscale("log")
    axis.set_xlabel("evaluation horizon (ms, log scale)")
    axis.set_ylabel("Spearman IC")
    axis.set_title("Signal decay: information coefficient vs horizon")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(decay_path, dpi=130)
    plt.close(figure)

    pnl_path = BENCHMARK_DIR / f"{stem}_pnl.png"
    figure, axis = plt.subplots(figsize=(9, 4.5))
    colours = ["#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51", "#9b2226"]
    for (fee, curve), colour in zip(report["_cumulative"].items(), colours):
        axis.plot(np.array(curve), label=f"{float(fee):.1f} bps", color=colour, linewidth=1.2)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("trade number")
    axis.set_ylabel("cumulative PnL (bps of notional)")
    axis.set_title("Cumulative PnL at touch prices, by taker fee")
    axis.grid(alpha=0.25)
    axis.legend(title="taker fee / side", ncol=3, fontsize=9)
    figure.tight_layout()
    figure.savefig(pnl_path, dpi=130)
    plt.close(figure)

    ic_path = BENCHMARK_DIR / f"{stem}_ic_blocks.png"
    figure, axis = plt.subplots(figsize=(9, 4))
    blocks = np.array(report["information_coefficient"]["block_ics"], dtype=float)
    axis.bar(np.arange(len(blocks)), blocks, color=np.where(blocks > 0, "#2a9d8f", "#e76f51"))
    axis.axhline(0, color="black", linewidth=0.8)
    axis.axhline(report["information_coefficient"]["mean"], color="#264653", linestyle="--", label="mean")
    axis.set_xlabel(f"test block (of {report['information_coefficient']['block_size']} samples)")
    axis.set_ylabel("Spearman IC")
    axis.set_title("IC stability across the held-out period")
    axis.legend()
    figure.tight_layout()
    figure.savefig(ic_path, dpi=130)
    plt.close(figure)

    return {"decay": decay_path.name, "pnl": pnl_path.name, "ic_blocks": ic_path.name}


def print_summary(report: dict) -> None:
    classification = report["classification"]
    ic = report["information_coefficient"]
    trading = report["trading"]
    print(f"\nheld-out samples: {report['samples']:,} from {len(report['sessions_in_fold'])} session(s)")
    print(f"label horizon: {report['label_horizon_ms']:,.0f} ms")
    print(f"macro-F1 {classification['macro_f1']:.4f}  accuracy {classification['accuracy']:.4f}")
    print(f"base rates: {[round(r, 3) for r in classification['base_rates']]}")
    print(f"\npooled IC {ic['pooled']:+.4f}   mean {ic['mean']:+.4f}   std {ic['std']:.4f}")
    print(f"fraction of blocks positive {ic['fraction_positive']:.2f}   information ratio {ic['information_ratio']:+.3f}")
    print(
        f"\npeak IC {report['decay']['peak_ic']:+.4f} at {report['decay']['peak_horizon_ms']:,.0f} ms; "
        f"half-life from the peak {report['decay']['half_life_ms']:,.0f} ms "
        f"(fit R^2 {report['decay']['fit_r_squared']:.3f})"
    )
    for row in report["decay"]["rows"]:
        print(f"  {row['horizon_ms']:>8,.0f} ms   IC {row['ic']:+.4f}   n={row['samples']:,}")
    print(f"\ntrades {trading['trade_count']:,}  gross {trading['gross_bps_per_trade']:+.4f} bps/trade")
    print(f"median spread {trading['median_spread_bps']:.4f} bps")
    print(f"BREAKEVEN FEE {trading['breakeven_fee_bps']:+.4f} bps/side")
    for tier, fee in trading["binance_taker_tiers_bps"].items():
        verdict = "survives" if trading["breakeven_fee_bps"] >= fee else "does NOT survive"
        print(f"  vs {tier:<12} {fee:>5.2f} bps -> {verdict}")
    falsification = report["falsification"]
    print(f"\nfalsification — true IC {falsification['true_ic']:+.4f}")
    print("  shifted (must DECAY with distance, not vanish at the first step):")
    for shift, value in sorted(falsification["shifted_ics"].items(), key=lambda kv: int(kv[0])):
        print(f"    shift {int(shift):>5} rows   IC {value:+.4f}")
    null = falsification["null_distribution"]
    print(
        f"  block-shuffle null ({null['trials']} draws): mean {null['mean']:+.4f} "
        f"std {null['std']:.4f}  ->  true IC is {null['z_score']:+.1f} sigma, "
        f"exceeded by {null['exceedance_rate'] * 100:.1f}% of draws"
    )
    print(f"  element-shuffled (must be ~0) {falsification['shuffled_signal_ic']:+.4f}")
    print(f"  reversed (must mirror true)   {falsification['reversed_signal_ic']:+.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default=None, help="path to a train_*.json record; default is the newest ours_* run")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_path = Path(args.run) if args.run else Path(sorted(glob.glob(str(BENCHMARK_DIR / "train_ours_*.json")))[-1])
    run_record = json.loads(run_path.read_text(encoding="utf-8"))
    run_record["_path"] = str(run_path)
    print(f"evaluating {run_path.name} on {args.device}")

    report = build_report(run_record, args.device)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"evaluation_{stamp}"
    report["artefacts"] = save_charts(report, stem)
    print_summary(report)

    out_path = BENCHMARK_DIR / f"{stem}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nsaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
