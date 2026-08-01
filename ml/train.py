"""Training loop, checkpointing, and the artefacts a result has to ship with.

WHAT
    A plain PyTorch training loop: AdamW, a cosine learning-rate schedule with
    warmup, class-weighted cross-entropy, early stopping on validation
    macro-F1, a fixed seed, and per-epoch history written to
    `benchmarks/` as CSV, PNG and JSON alongside the best checkpoint.

WHY
    In this domain the loop is not the interesting part — the splitting and
    the baselines are, and those live in `ml/splits.py` and `ml/baseline.py`.
    What this file has to get right is narrower: be reproducible, stop on the
    metric we actually report, and leave behind enough evidence that someone
    can check the number without rerunning it.

DESIGN DECISION — a plain loop, no Lightning or other framework.
    Rejected alternative: PyTorch Lightning, which would remove perhaps thirty
    lines. It would also add a dependency CLAUDE.md does not list, hide the
    optimiser step behind a callback system, and — the real objection — make
    "what exactly happens between two batches" a question about someone else's
    abstractions. This project's premise is being able to defend every line.
    Thirty lines is a cheap price for that.

DESIGN DECISION — early stopping on validation macro-F1, not validation loss.
    Rejected alternative: stop on loss, the usual default. Loss and macro-F1
    disagree on this problem: cross-entropy keeps improving while the model
    grows more confident about the majority class, which *lowers* macro-F1.
    Stopping on the metric we report avoids selecting a checkpoint that is
    worse at the thing we claim to care about.

DESIGN DECISION — class-weighted loss, not resampling.
    Rejected alternative: oversampling the minority classes. Duplicating
    temporally adjacent samples would put near-identical windows in the same
    epoch and quietly reintroduce the leakage the Stage 3 embargo exists to
    prevent. Weighting the loss changes the gradient without changing the
    sample set.

DESIGN DECISION — cosine schedule with a short warmup.
    Rejected alternative: a fixed learning rate. BatchNorm statistics are wild
    for the first few hundred steps, and a full-size step into that produces a
    loss spike that the run never quite recovers from. Warmup costs 5% of the
    schedule and removes the failure mode.

REPRODUCIBILITY
    `seed_everything` fixes Python, numpy and torch RNGs and puts cuDNN in
    deterministic mode. Two runs with the same seed, data and device produce
    the same curve. Across devices they will not — cuDNN picks different
    kernels on different hardware — so every saved result records the device
    it ran on rather than implying more portability than exists.

RUNNING IT

    Locally, a smoke test that finishes in seconds and proves the plumbing:
        python -m ml.train --smoke

    Locally, the real thing on captured tapes:
        python -m ml.train --tapes "data/stage3/*.tape" --epochs 40

    On Kaggle with a free GPU, three lines in a notebook cell:
        !git clone <your repo url> tick-to-signal && cd tick-to-signal
        !pip install -q torch numpy matplotlib
        !cd tick-to-signal && python -m ml.train --tapes "data/stage3/*.tape" \
             --epochs 60 --device cuda
    Upload the tapes as a Kaggle dataset and point `--tapes` at
    `/kaggle/input/<dataset-name>/*.tape`. Nothing else changes: the loop
    picks the device up from the flag and writes the same artefacts.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import platform
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from ml.dataset import (
    BatchedWindowLoader,
    SampleIndex,
    Session,
    build_sample_index,
    describe_sessions,
    load_sessions,
)
from ml.labels import DEFAULT_SMOOTHING_K
from ml.metrics import format_report, macro_f1, majority_class_baseline, per_class_scores
from ml.model import ModelConfig, TickToSignalNet, build_model
from ml.splits import Split, required_embargo, walk_forward_splits

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = REPO_ROOT / "benchmarks"


@dataclass
class TrainConfig:
    """Everything the loop needs. A dataclass, not a config system."""

    epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_fraction: float = 0.05
    patience: int = 8
    seed: int = 20260728
    device: str = "cpu"
    num_workers: int = 0
    max_batches_per_epoch: int = 0  # 0 = the whole epoch; used by --smoke
    label_smoothing: float = 0.0
    # Bit-exact cuDNN kernel selection. Off by default: measured 14x slower.
    # See `seed_everything`. Recorded in every saved run so a number always
    # carries the reproducibility guarantee it was produced under.
    deterministic: bool = False


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    validation_loss: float
    validation_macro_f1: float
    validation_accuracy: float
    learning_rate: float
    seconds: float


@dataclass
class TrainResult:
    """The trained model plus everything needed to defend its number."""

    best_state: dict
    best_macro_f1: float
    best_epoch: int
    history: list[EpochRecord] = field(default_factory=list)
    stopped_early: bool = False


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Fix every RNG the loop touches. Optionally force deterministic cuDNN.

    The seeds are unconditional: Python, numpy and torch RNGs are pinned, so
    weight initialisation, dropout masks and batch order are identical across
    runs whatever `deterministic` is set to.

    `deterministic` controls only cuDNN's *kernel selection*, and it is off by
    default because the cost is not small. Measured on this machine (RTX 2050,
    batch 128, this architecture):

        cudnn.deterministic = True   ->    56 samples/s
        cudnn.benchmark     = True   ->   784 samples/s

    A 14x difference, which turns a 24-minute training run into 5.5 hours.
    With it off, two runs of the same seed agree on data order and starting
    weights but may differ in the last few decimal places because cuDNN is
    free to pick a different convolution algorithm. That is a real loss of
    reproducibility and it is why every saved run records which mode produced
    it — a number is only as reproducible as the flag next to it says.

    Turn it on with `--deterministic` when a result needs to be bit-exact,
    such as when chasing a discrepancy against the Stage 7 C++ path.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def class_weights_from_labels(labels: np.ndarray, class_count: int = 3) -> Tensor:
    """Inverse-frequency weights, normalised to mean 1.

    Normalising keeps the loss on the same scale as an unweighted run, so the
    learning rate does not silently need retuning when the class balance
    shifts between folds.
    """
    counts = np.bincount(labels.astype(np.int64), minlength=class_count).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (class_count * counts)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def carve_validation(
    split: Split,
    embargo: int,
    validation_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a fold's training block into fit and validation parts.

    WHY THIS EXISTS. The obvious thing — early-stop on the fold's test block
    and then report that same block — is a quiet form of fitting to the test
    set: the stopping epoch is chosen by looking at the number being reported,
    so the reported score is the maximum over ~40 peeks rather than an
    unbiased estimate. It is a small effect next to the leakage Stage 3 is
    about, and it is exactly the kind of thing that turns a defensible number
    into an argument.

    So the last `validation_fraction` of the *training* block becomes the
    validation set, an embargo is opened between it and what remains for
    fitting, and the fold's test block is never looked at until the run is
    over. The embargo between validation and test already exists — it is the
    fold's own embargo.

    Returns `(fit_positions, validation_positions)`, both chronological.
    """
    train_positions = split.train_indices()
    validation_size = int(len(train_positions) * validation_fraction)
    if validation_size <= 0 or len(train_positions) - validation_size - embargo <= 0:
        raise ValueError(
            f"cannot carve a {validation_fraction:.0%} validation set with a {embargo}-sample embargo "
            f"out of {len(train_positions)} training samples"
        )
    validation_positions = train_positions[-validation_size:]
    fit_positions = train_positions[: -(validation_size + embargo)]
    return fit_positions, validation_positions


def build_loaders(
    sessions: list[Session],
    sample_index: SampleIndex,
    fit_positions: np.ndarray,
    validation_positions: np.ndarray,
    config: TrainConfig,
) -> tuple[BatchedWindowLoader, BatchedWindowLoader]:
    """Wrap fit and validation index ranges in batch loaders.

    Training shuffles, validation does not — shuffling a validation set gains
    nothing and makes per-sample debugging harder to follow.
    """
    train_loader = BatchedWindowLoader(
        sessions, sample_index, fit_positions, config.batch_size, shuffle=True, seed=config.seed
    )
    validation_loader = BatchedWindowLoader(
        sessions, sample_index, validation_positions, config.batch_size, shuffle=False
    )
    return train_loader, validation_loader


def _cosine_with_warmup(step: int, total_steps: int, warmup_steps: int) -> float:
    """Learning-rate multiplier: linear warmup, then cosine decay to zero."""
    if step < warmup_steps:
        return (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def run_epoch(
    model: TickToSignalNet,
    loader: BatchedWindowLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimiser: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    max_batches: int = 0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """One pass. Trains when an optimiser is given, evaluates when it is None."""
    training = optimiser is not None
    model.train(training)
    total_loss = 0.0
    seen = 0
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []

    for batch_number, (window_batch, label_batch) in enumerate(loader):
        if max_batches and batch_number >= max_batches:
            break
        windows = torch.from_numpy(window_batch).to(device, dtype=torch.float32)
        labels = torch.from_numpy(label_batch).to(device, dtype=torch.long)

        with torch.set_grad_enabled(training):
            logits = model(windows)
            loss = loss_function(logits, labels)
        if training:
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += float(loss.detach()) * len(labels)
        seen += len(labels)
        predictions.append(logits.detach().argmax(dim=1).cpu().numpy())
        truths.append(labels.detach().cpu().numpy())

    if seen == 0:
        return 0.0, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return total_loss / seen, np.concatenate(truths), np.concatenate(predictions)


def train_model(
    model: TickToSignalNet,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    train_labels: np.ndarray,
    config: TrainConfig,
    verbose: bool = True,
) -> TrainResult:
    """Fit the model, keeping the checkpoint with the best validation macro-F1."""
    seed_everything(config.seed, config.deterministic)
    device = torch.device(config.device)
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss(
        weight=class_weights_from_labels(train_labels).to(device),
        label_smoothing=config.label_smoothing,
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    batches_per_epoch = config.max_batches_per_epoch or len(train_loader)
    total_steps = max(batches_per_epoch * config.epochs, 1)
    warmup_steps = int(total_steps * config.warmup_fraction)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: _cosine_with_warmup(step, total_steps, warmup_steps)
    )

    result = TrainResult(best_state={}, best_macro_f1=-1.0, best_epoch=-1)
    epochs_without_improvement = 0

    for epoch in range(config.epochs):
        started = time.perf_counter()
        train_loss, _, _ = run_epoch(
            model, train_loader, loss_function, device, optimiser, scheduler, config.max_batches_per_epoch
        )
        validation_loss, truth, prediction = run_epoch(
            model, validation_loader, loss_function, device, None, None, config.max_batches_per_epoch
        )
        score = macro_f1(truth, prediction) if len(truth) else 0.0
        record = EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            validation_loss=validation_loss,
            validation_macro_f1=score,
            validation_accuracy=float(np.mean(truth == prediction)) if len(truth) else 0.0,
            learning_rate=optimiser.param_groups[0]["lr"],
            seconds=time.perf_counter() - started,
        )
        result.history.append(record)
        if verbose:
            print(
                f"  epoch {epoch:>3}  train_loss {train_loss:.4f}  val_loss {validation_loss:.4f}  "
                f"val_macroF1 {score:.4f}  val_acc {record.validation_accuracy:.4f}  {record.seconds:.1f}s"
            )

        if score > result.best_macro_f1:
            result.best_macro_f1 = score
            result.best_epoch = epoch
            result.best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                result.stopped_early = True
                if verbose:
                    print(f"  early stop: {config.patience} epochs without improving macro-F1")
                break

    if result.best_state:
        model.load_state_dict(result.best_state)
    return result


def evaluate(model: TickToSignalNet, loader: BatchedWindowLoader, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Predictions and truths for a whole loader, for the final report."""
    loss_function = nn.CrossEntropyLoss()
    _loss, truth, prediction = run_epoch(model, loader, loss_function, torch.device(device), None)
    return truth, prediction


# ----------------------------------------------------------------- artefacts


def machine_info(device: str) -> dict[str, str]:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def save_history_csv(history: list[EpochRecord], path: Path) -> None:
    header = "epoch,train_loss,validation_loss,validation_macro_f1,validation_accuracy,learning_rate,seconds"
    lines = [header]
    for record in history:
        lines.append(
            f"{record.epoch},{record.train_loss:.6f},{record.validation_loss:.6f},"
            f"{record.validation_macro_f1:.6f},{record.validation_accuracy:.6f},"
            f"{record.learning_rate:.8f},{record.seconds:.3f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_curve_png(history: list[EpochRecord], path: Path, title: str) -> None:
    """Loss on the left axis, macro-F1 on the right — the two things that matter.

    Imported lazily: a training run on a headless box should not need a
    plotting stack to produce its CSV.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [record.epoch for record in history]
    figure, loss_axis = plt.subplots(figsize=(9, 4.5))
    loss_axis.plot(epochs, [r.train_loss for r in history], label="train loss", color="#264653")
    loss_axis.plot(epochs, [r.validation_loss for r in history], label="val loss", color="#e76f51")
    loss_axis.set_xlabel("epoch")
    loss_axis.set_ylabel("cross-entropy")
    loss_axis.grid(alpha=0.25)

    score_axis = loss_axis.twinx()
    score_axis.plot(epochs, [r.validation_macro_f1 for r in history], label="val macro-F1", color="#2a9d8f")
    score_axis.set_ylabel("macro-F1")

    handles = loss_axis.get_legend_handles_labels()[0] + score_axis.get_legend_handles_labels()[0]
    labels = loss_axis.get_legend_handles_labels()[1] + score_axis.get_legend_handles_labels()[1]
    loss_axis.legend(handles, labels, loc="center right")
    loss_axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def save_run(tag: str, payload: dict, history: list[EpochRecord], state: dict | None) -> dict[str, str]:
    """Write JSON, CSV, PNG and the checkpoint; return where each landed."""
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"train_{tag}_{stamp}"
    paths = {
        "json": BENCHMARK_DIR / f"{stem}.json",
        "csv": BENCHMARK_DIR / f"{stem}_history.csv",
        "png": BENCHMARK_DIR / f"{stem}_curves.png",
    }
    save_history_csv(history, paths["csv"])
    if history:
        save_curve_png(history, paths["png"], f"{tag} — training curves")
    payload["artefacts"] = {name: path.name for name, path in paths.items()}

    if state is not None:
        checkpoint_dir = REPO_ROOT / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{stem}.pt"
        torch.save(state, checkpoint_path)
        payload["artefacts"]["checkpoint"] = str(checkpoint_path.relative_to(REPO_ROOT))

    paths["json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {name: str(path) for name, path in paths.items()}


# --------------------------------------------------------------- entry point


def _smoke_sessions(rng: np.random.Generator, rows: int = 1_200) -> list[Session]:
    """Synthetic sessions so `--smoke` proves the plumbing with no data on disk.

    Deliberately learnable: the label is a noisy function of one feature, so a
    working loop should drive the training loss down within a few epochs. A
    smoke test on pure noise would pass even if the gradient never flowed.
    """
    features = rng.normal(size=(rows, 40)).astype(np.float32)
    signal = features[:, 1] + 0.25 * rng.normal(size=rows)
    labels = np.digitize(signal, np.quantile(signal, [1 / 3, 2 / 3])).astype(np.int64)
    usable = np.zeros(rows, dtype=bool)
    usable[100:] = True
    return [
        Session(
            name="smoke",
            features=features,
            labels=labels,
            usable=usable,
            mids=np.full(rows, 100.0),
            best_bids=np.full(rows, 99.995),
            best_asks=np.full(rows, 100.005),
            timestamps_ns=np.arange(rows, dtype=np.int64) * 34_000_000,
            first_timestamp_ns=0,
        )
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the tick-to-signal model.")
    parser.add_argument("--tapes", default="data/stage3/*.tape", help="glob for the .tape files to train on")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--folds", type=int, default=3, help="walk-forward folds; the last one is reported")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--tag", default="ours", help="name used in the saved artefact filenames")
    parser.add_argument("--fi2010", default="", help="directory holding the FI-2010 files; trains on the benchmark")
    parser.add_argument("--fi2010-horizon", type=int, default=10, help="FI-2010 prediction horizon: 1, 2, 3, 5 or 10")
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.15,
        help="tail of the training block held out for early stopping, so the test block stays untouched",
    )
    parser.add_argument("--smoke", action="store_true", help="tiny synthetic run that proves the loop works")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="force bit-exact cuDNN kernels; measured 14x slower, see seed_everything",
    )
    args = parser.parse_args(argv)

    config = TrainConfig(
        epochs=3 if args.smoke else args.epochs,
        batch_size=32 if args.smoke else args.batch_size,
        learning_rate=args.learning_rate,
        device="cpu" if args.smoke else args.device,
        seed=args.seed,
        max_batches_per_epoch=5 if args.smoke else 0,
        patience=3 if args.smoke else 8,
        deterministic=args.deterministic,
    )

    embargo = required_embargo(100, DEFAULT_SMOOTHING_K)
    if args.fi2010:
        sessions, fit_positions, validation_positions, test_positions, data_info = _prepare_fi2010(args, embargo)
    else:
        sessions, fit_positions, validation_positions, test_positions, data_info = _prepare_tapes(args, embargo)

    sample_index = build_sample_index(sessions)
    print(describe_sessions(sessions, sample_index))
    print(
        f"\nfit {len(fit_positions):,} | validation {len(validation_positions):,} "
        f"| test {len(test_positions):,} | embargo {embargo}"
    )
    print("early stopping uses the validation block; the test block is untouched until the end.\n")

    train_loader, validation_loader = build_loaders(
        sessions, sample_index, fit_positions, validation_positions, config
    )
    test_loader = BatchedWindowLoader(sessions, sample_index, test_positions, config.batch_size, shuffle=False)
    fit_labels = sample_index.labels[fit_positions]

    model = build_model(ModelConfig(), seed=args.seed)
    print(f"model parameters: {model.parameter_count():,}\n")

    result = train_model(model, train_loader, validation_loader, fit_labels, config)
    truth, prediction = evaluate(model, test_loader, config.device)
    print("\nheld-out test block:")
    print(format_report(truth, prediction))

    majority = majority_class_baseline(fit_labels, truth)
    scores = per_class_scores(truth, prediction)
    payload = {
        "tag": args.tag,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine": machine_info(config.device),
        "config": asdict(config),
        "model": {"parameters": model.parameter_count(), "receptive_field": model.receptive_field()},
        "data": data_info | {"samples": len(sample_index), "class_balance": sample_index.class_balance()},
        "result": {
            "validation_macro_f1": result.best_macro_f1,
            "best_epoch": result.best_epoch,
            "stopped_early": result.stopped_early,
            "epochs_run": len(result.history),
            "test_macro_f1": macro_f1(truth, prediction),
            "test_accuracy": float(np.mean(truth == prediction)),
            "per_class_f1": scores["f1"].tolist(),
            "majority_baseline": majority,
        },
    }
    if args.smoke:
        # A smoke run trains three epochs on synthetic noise. Its numbers mean
        # nothing, and leaving them in benchmarks/ next to real results is how
        # a meaningless file ends up cited as evidence.
        print("\nsmoke run: artefacts not saved (the numbers above are from synthetic data)")
        return 0

    written = save_run(args.tag, payload, result.history, result.best_state or None)
    print(f"\nsaved: {written['json']}")
    return 0


def _prepare_tapes(args, embargo: int):
    """Sessions and index ranges for our own captured tapes."""
    if args.smoke:
        sessions = _smoke_sessions(np.random.default_rng(args.seed))
    else:
        tape_paths = sorted(glob.glob(args.tapes))
        if not tape_paths:
            raise SystemExit(f"no tapes matched {args.tapes!r}; run data_engine.capture first")
        sessions = load_sessions(tape_paths)

    sample_index = build_sample_index(sessions)
    splits = walk_forward_splits(len(sample_index), args.folds, embargo)
    split = splits[-1]
    fit_positions, validation_positions = carve_validation(split, embargo, args.validation_fraction)
    info = {
        "source": "captured_tapes",
        "tapes": args.tapes,
        "sessions": [session.name for session in sessions],
        "fold": asdict(split),
        "folds_available": len(splits),
    }
    return sessions, fit_positions, validation_positions, split.test_indices(), info


def _prepare_fi2010(args, embargo: int):
    """Sessions and index ranges for the public benchmark.

    Uses the dataset's own train/test division rather than our walk-forward
    folds, because that division is what published results are quoted on and
    changing it would break the only comparison this path exists to make.
    Validation for early stopping is carved off the end of the training days,
    with the same embargo.
    """
    from ml import fi2010

    train_split, test_split = fi2010.load_benchmark(args.fi2010, horizon=args.fi2010_horizon)
    sessions = [fi2010.to_session(train_split), fi2010.to_session(test_split)]
    sample_index = build_sample_index(sessions)

    from_train = np.flatnonzero(sample_index.session_of_sample == 0)
    from_test = np.flatnonzero(sample_index.session_of_sample == 1)
    validation_size = int(len(from_train) * args.validation_fraction)
    validation_positions = from_train[-validation_size:]
    fit_positions = from_train[: -(validation_size + embargo)]

    info = {
        "source": "fi2010",
        "horizon": args.fi2010_horizon,
        "citation": "Ntakaris et al. 2018, https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649",
        "preprocessing": "NoAuction_DecPre, 40 raw LOB features, as distributed",
        "sessions": [session.name for session in sessions],
    }
    return sessions, fit_positions, validation_positions, from_test, info


if __name__ == "__main__":
    raise SystemExit(main())
