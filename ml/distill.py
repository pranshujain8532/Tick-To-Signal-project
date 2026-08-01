"""Knowledge distillation: compress the teacher into a ~32K-parameter student.

WHAT
    Defines the student architecture (same blocks as the teacher, narrower),
    the distillation loss, and a training loop that fits the student either
    from the teacher's soft outputs or from hard labels alone — so the two can
    be compared directly.

WHY
    Stage 7 hand-writes this forward pass in C++. Every parameter is a weight
    someone has to load, lay out, and multiply, so a 10x smaller model is a
    materially easier thing to build and a materially faster thing to run. The
    question is what that costs in accuracy, and the only way to answer it is
    to train the same student twice — once with the teacher, once without —
    and report the difference.

    **The delta between those two runs is the entire evidence that
    distillation does anything.** A distilled student reported on its own says
    nothing: it might be scoring well because the architecture is adequate, not
    because the teacher taught it. That control run is the point of this file.

DESIGN DECISION — the student keeps the teacher's block structure and its
receptive field.
    `conv_channels` 32 -> 8, `inception_channels` 64 -> 16, `tcn_channels`
    96 -> 32, but the same four dilations (1, 2, 4, 8) and therefore the same
    83-timestep receptive field. Rejected alternative: drop a dilation to save
    more parameters. That would change what the student can *see*, so a drop in
    accuracy could no longer be attributed to capacity — it would be confounded
    with context. Narrowing while holding the receptive field fixed keeps the
    comparison clean.

DESIGN DECISION — soft-target distillation, with the hard labels kept.
    Rejected alternative: train the small architecture on hard labels only.
    That is the correct control experiment and it is run as one, but it is not
    the method: a one-hot label says "this window was up" while the teacher's
    full distribution says "this window was 60% up, 35% flat, 5% down", and the
    second carries information about *how ambiguous the moment was*. On this
    problem most moments are genuinely ambiguous — Stage 3 measured the mid as
    unchanged 99% of the time — so that extra signal is a larger fraction of
    what there is to learn than it would be on a clean task. This is the
    "dark knowledge" argument, and it is why the loss keeps both terms.

INFORMATION HORIZON
    The teacher's logits are a function of the same window the student sees,
    so distillation adds no new information about the future. The teacher was
    trained on the training fold only, and its logits are precomputed over
    that same fold, so no test-fold information reaches the student.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor, nn

from ml.dataset import BatchedWindowLoader, SampleIndex, Session
from ml.metrics import macro_f1
from ml.model import ModelConfig, TickToSignalNet, build_model
from ml.train import TrainConfig, _cosine_with_warmup, class_weights_from_labels, seed_everything

# Same blocks and the same 83-step receptive field as the teacher, roughly one
# tenth the parameters (32,155 against 319,715).
STUDENT_CONFIG = ModelConfig(
    conv_channels=8,
    inception_channels=16,
    tcn_channels=32,
    tcn_dilations=(1, 2, 4, 8),
)


@dataclass
class DistillConfig:
    """Knobs specific to distillation. Training knobs live in `TrainConfig`."""

    # Weight on the hard-label term. 0.3 leaves most of the gradient coming
    # from the teacher, which is the usual setting when the teacher is
    # substantially better than the student can be on its own.
    alpha: float = 0.3
    # Softening temperature. Above 1 this flattens both distributions, which
    # surfaces the small probabilities — the relative ordering of the two
    # classes the teacher did *not* pick is most of the "dark knowledge".
    temperature: float = 4.0


@dataclass
class StudentResult:
    """One student training run, and enough context to compare two of them."""

    label: str
    best_macro_f1: float
    best_epoch: int
    epochs_run: int
    parameters: int
    history: list[dict] = field(default_factory=list)
    state: dict = field(default_factory=dict, repr=False)


def build_student(seed: int | None = None) -> TickToSignalNet:
    """The compressed model. Same blocks as the teacher, ~10x fewer weights."""
    return build_model(STUDENT_CONFIG, seed=seed)


def distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    alpha: float,
    temperature: float,
    class_weight: Tensor | None = None,
) -> Tensor:
    """`alpha * CE(hard) + (1 - alpha) * T^2 * KL(student/T || teacher/T)`.

    WHY THE T^2 FACTOR — this is the detail interviewers ask about.

    Softening the logits by `T` before the softmax shrinks the gradients of the
    KL term by roughly `1/T^2`. The intuition: dividing the logits by `T`
    flattens the distribution, and the derivative of the softened softmax with
    respect to a logit carries a factor of `1/T`; that factor appears once from
    the student's own softening and once through the size of the residual
    `(p_student - p_teacher)`, so the gradient scales as `1/T^2`.

    Without a correction, raising the temperature would therefore quietly
    shrink the distillation term relative to the hard-label term, and `alpha`
    would no longer mean what it says — changing `T` would silently change the
    balance between the two objectives. Multiplying by `T^2` cancels it, so the
    two terms stay comparable and `T` controls only *how soft* the targets are,
    which is the one thing it should control. This is Hinton et al.'s original
    argument and the reason the factor is there rather than folded into a
    tuned learning rate.

    Both distributions are softened by the same `T`; the KL is
    `KL(teacher || student)` in the sense that the teacher supplies the target
    probabilities, which is what `F.kl_div` expects as its second argument.
    """
    hard_loss = nn.functional.cross_entropy(student_logits, labels, weight=class_weight)

    student_log_probabilities = nn.functional.log_softmax(student_logits / temperature, dim=1)
    teacher_probabilities = nn.functional.softmax(teacher_logits / temperature, dim=1)
    soft_loss = nn.functional.kl_div(
        student_log_probabilities, teacher_probabilities, reduction="batchmean"
    )

    return alpha * hard_loss + (1.0 - alpha) * (temperature ** 2) * soft_loss


def precompute_teacher_logits(
    teacher: TickToSignalNet,
    sessions: list[Session],
    sample_index: SampleIndex,
    positions: np.ndarray,
    device: str = "cpu",
    batch_size: int = 512,
) -> np.ndarray:
    """Teacher logits for every sample, computed once.

    The teacher is frozen, so running it inside the student's training loop
    would recompute the same numbers every epoch for no reason — at 320K
    parameters over 27,674 samples that is most of the wall clock. Precomputing
    turns it into one pass, and it is stored aligned to `positions` so the
    training loop can index it exactly like the labels.
    """
    teacher = teacher.to(device).eval()
    loader = BatchedWindowLoader(sessions, sample_index, positions, batch_size, shuffle=False)
    logits = np.empty((len(positions), 3), dtype=np.float32)
    written = 0
    with torch.no_grad():
        for windows, _labels in loader:
            batch = torch.from_numpy(windows).to(device, dtype=torch.float32)
            output = teacher(batch).cpu().numpy()
            logits[written : written + len(output)] = output
            written += len(output)
    return logits


def _iterate_batches(
    sessions: list[Session],
    sample_index: SampleIndex,
    positions: np.ndarray,
    teacher_logits: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    """Yield `(windows, labels, teacher_logits_or_None)` aligned by position.

    `BatchedWindowLoader` permutes an index into `positions`, so the teacher
    logits — stored in the same order — can be gathered with the same
    permutation. Doing the alignment here rather than inside the loader keeps
    the loader free of any knowledge of distillation.
    """
    from ml.dataset import gather_windows

    generator = np.random.default_rng(seed)
    order = generator.permutation(len(positions)) if shuffle else np.arange(len(positions))
    for start in range(0, len(order), batch_size):
        slots = order[start : start + batch_size]
        chosen = positions[slots]
        windows = gather_windows(sessions, sample_index, chosen)
        labels = sample_index.labels[chosen]
        soft = teacher_logits[slots] if teacher_logits is not None else None
        yield windows, labels, soft


def train_student(
    student: TickToSignalNet,
    sessions: list[Session],
    sample_index: SampleIndex,
    train_positions: np.ndarray,
    validation_positions: np.ndarray,
    config: TrainConfig,
    distill: DistillConfig | None,
    teacher_logits: np.ndarray | None,
    label: str,
    verbose: bool = True,
) -> StudentResult:
    """Fit the student, with the teacher's soft targets when `distill` is given.

    Passing `distill=None` runs the control: the identical architecture, seed,
    schedule and data, trained on hard labels alone. Everything except the loss
    is held fixed, which is what makes the difference in scores attributable to
    distillation rather than to any of the dozen other things that differ
    between two training runs.
    """
    seed_everything(config.seed)
    device = torch.device(config.device)
    student = student.to(device)

    train_labels = sample_index.labels[train_positions]
    class_weight = class_weights_from_labels(train_labels).to(device)
    optimiser = torch.optim.AdamW(student.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    batches_per_epoch = max(1, int(np.ceil(len(train_positions) / config.batch_size)))
    total_steps = batches_per_epoch * config.epochs
    warmup_steps = int(total_steps * config.warmup_fraction)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: _cosine_with_warmup(step, total_steps, warmup_steps)
    )

    result = StudentResult(label=label, best_macro_f1=-1.0, best_epoch=-1, epochs_run=0,
                           parameters=student.parameter_count())
    epochs_without_improvement = 0

    for epoch in range(config.epochs):
        started = time.perf_counter()
        train_loss = _train_one_epoch(
            student, sessions, sample_index, train_positions, teacher_logits,
            optimiser, scheduler, device, config, distill, class_weight,
        )
        score = _validate(student, sessions, sample_index, validation_positions, device, config.batch_size)
        result.history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_macro_f1": score,
             "seconds": time.perf_counter() - started}
        )
        result.epochs_run = epoch + 1
        if verbose:
            print(f"  [{label}] epoch {epoch:>3}  loss {train_loss:.4f}  val_macroF1 {score:.4f}"
                  f"  {result.history[-1]['seconds']:.1f}s")

        if score > result.best_macro_f1:
            result.best_macro_f1 = score
            result.best_epoch = epoch
            result.state = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                if verbose:
                    print(f"  [{label}] early stop after {config.patience} epochs without improvement")
                break

    if result.state:
        student.load_state_dict(result.state)
    return result


def _train_one_epoch(
    student, sessions, sample_index, positions, teacher_logits,
    optimiser, scheduler, device, config, distill, class_weight,
) -> float:
    """One optimisation pass. Uses the distillation loss only when asked to."""
    student.train()
    total_loss = 0.0
    seen = 0
    for windows, labels, soft in _iterate_batches(
        sessions, sample_index, positions, teacher_logits, config.batch_size, True, config.seed
    ):
        batch = torch.from_numpy(windows).to(device, dtype=torch.float32)
        targets = torch.from_numpy(labels).to(device, dtype=torch.long)
        logits = student(batch)

        if distill is not None and soft is not None:
            teacher = torch.from_numpy(soft).to(device, dtype=torch.float32)
            loss = distillation_loss(logits, teacher, targets, distill.alpha, distill.temperature, class_weight)
        else:
            loss = nn.functional.cross_entropy(logits, targets, weight=class_weight)

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        scheduler.step()
        total_loss += float(loss.detach()) * len(labels)
        seen += len(labels)
    return total_loss / max(seen, 1)


def _validate(student, sessions, sample_index, positions, device, batch_size: int) -> float:
    """Macro-F1 on the validation block. No teacher involved."""
    student.eval()
    predictions = []
    truths = []
    with torch.no_grad():
        for windows, labels, _soft in _iterate_batches(
            sessions, sample_index, positions, None, batch_size, False, 0
        ):
            batch = torch.from_numpy(windows).to(device, dtype=torch.float32)
            predictions.append(student(batch).argmax(dim=1).cpu().numpy())
            truths.append(labels)
    if not predictions:
        return 0.0
    return macro_f1(np.concatenate(truths), np.concatenate(predictions))


def evaluate_on(
    model: TickToSignalNet,
    sessions: list[Session],
    sample_index: SampleIndex,
    positions: np.ndarray,
    device: str = "cpu",
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Predictions and truths on a held-out block, for the final report."""
    model = model.to(device).eval()
    predictions = []
    truths = []
    with torch.no_grad():
        for windows, labels, _soft in _iterate_batches(
            sessions, sample_index, positions, None, batch_size, False, 0
        ):
            batch = torch.from_numpy(windows).to(device, dtype=torch.float32)
            predictions.append(model(batch).argmax(dim=1).cpu().numpy())
            truths.append(labels)
    return np.concatenate(truths), np.concatenate(predictions)


# --------------------------------------------------------------- entry point


def main(argv: list[str] | None = None) -> int:
    """Train the student twice — with and without the teacher — and compare.

    Both runs share the seed, schedule, data, split and architecture. Only the
    loss differs, which is what makes the gap between them attributable to
    distillation rather than to any of the other things that usually differ
    between two training runs.
    """
    import argparse
    import glob
    import json
    from dataclasses import asdict
    from datetime import datetime, timezone
    from pathlib import Path

    from ml.dataset import build_sample_index, load_sessions
    from ml.labels import DEFAULT_SMOOTHING_K
    from ml.metrics import format_report
    from ml.model import ModelConfig
    from ml.splits import required_embargo, walk_forward_splits
    from ml.train import BENCHMARK_DIR, REPO_ROOT, carve_validation, machine_info

    parser = argparse.ArgumentParser(description="Distil the teacher into a small student.")
    parser.add_argument("--teacher-run", default=None, help="train_ours_*.json record of the teacher")
    parser.add_argument("--tapes", default=None, help="override the tape glob from the teacher run")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--alpha", type=float, default=DistillConfig.alpha)
    parser.add_argument("--temperature", type=float, default=DistillConfig.temperature)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help=(
            "repeat the paired comparison with this many seeds. One seed cannot "
            "distinguish a small distillation delta from run-to-run noise, so a "
            "claim either way needs an error bar"
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    run_path = Path(args.teacher_run or sorted(glob.glob(str(BENCHMARK_DIR / "train_ours_*.json")))[-1])
    teacher_run = json.loads(run_path.read_text(encoding="utf-8"))
    tape_glob = args.tapes or teacher_run["data"]["tapes"]

    sessions = load_sessions(sorted(glob.glob(tape_glob)))
    sample_index = build_sample_index(sessions)
    embargo = required_embargo(100, DEFAULT_SMOOTHING_K)
    split = walk_forward_splits(len(sample_index), args.folds, embargo)[-1]
    fit_positions, validation_positions = carve_validation(split, embargo)
    test_positions = split.test_indices()

    teacher = build_model(ModelConfig())
    teacher.load_state_dict(torch.load(REPO_ROOT / teacher_run["artefacts"]["checkpoint"], map_location="cpu"))
    print(f"teacher {teacher.parameter_count():,} params from {run_path.name}")
    print(f"fit {len(fit_positions):,} | validation {len(validation_positions):,} | test {len(test_positions):,}\n")

    print("precomputing teacher logits over the fit block (the teacher is frozen)...")
    teacher_logits = precompute_teacher_logits(teacher, sessions, sample_index, fit_positions, args.device)

    distill = DistillConfig(alpha=args.alpha, temperature=args.temperature)
    seeds = [args.seed + offset for offset in range(args.seeds)]
    per_seed = []
    best_distilled_state = None
    best_distilled_score = -1.0

    for seed in seeds:
        config = TrainConfig(
            epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
            seed=seed, device=args.device,
        )
        print(f"\n=== seed {seed}: student WITH distillation (alpha={distill.alpha}, T={distill.temperature}) ===")
        distilled = train_student(
            build_student(seed=seed), sessions, sample_index, fit_positions, validation_positions,
            config, distill, teacher_logits, label="distilled",
        )
        print(f"\n=== seed {seed}: student WITHOUT distillation (control: same everything, hard labels) ===")
        scratch = train_student(
            build_student(seed=seed), sessions, sample_index, fit_positions, validation_positions,
            config, None, None, label="scratch",
        )

        entry = {"seed": seed}
        for name, result in (("distilled", distilled), ("scratch", scratch)):
            model = build_student()
            model.load_state_dict(result.state)
            truth, prediction = evaluate_on(model, sessions, sample_index, test_positions, args.device)
            entry[name] = {
                "validation_macro_f1": result.best_macro_f1,
                "test_macro_f1": macro_f1(truth, prediction),
                "best_epoch": result.best_epoch,
                "epochs_run": result.epochs_run,
                "parameters": result.parameters,
                "history": result.history,
            }
            print(f"\n[seed {seed} / {name}] held-out test block:")
            print(format_report(truth, prediction))
            if name == "distilled" and result.best_macro_f1 > best_distilled_score:
                best_distilled_score = result.best_macro_f1
                best_distilled_state = result.state
        entry["delta"] = entry["distilled"]["test_macro_f1"] - entry["scratch"]["test_macro_f1"]
        per_seed.append(entry)
        print(f"\nseed {seed} delta: {entry['delta']:+.4f} macro-F1")

    deltas = np.array([entry["delta"] for entry in per_seed])
    distilled_scores = np.array([entry["distilled"]["test_macro_f1"] for entry in per_seed])
    scratch_scores = np.array([entry["scratch"]["test_macro_f1"] for entry in per_seed])
    delta = float(deltas.mean())

    checkpoint_dir = REPO_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_distilled_state, checkpoint_dir / "student_distilled.pt")

    print(f"\n{'=' * 62}")
    print(f"distilled test macro-F1 : {distilled_scores.mean():.4f} +/- {distilled_scores.std():.4f}  (n={len(seeds)})")
    print(f"scratch   test macro-F1 : {scratch_scores.mean():.4f} +/- {scratch_scores.std():.4f}")
    print(f"DISTILLATION DELTA      : {delta:+.4f} +/- {deltas.std():.4f}")
    if len(seeds) > 1 and abs(delta) < deltas.std():
        print("-> the delta is smaller than its own spread across seeds: no measurable effect")
    print("(the delta is the evidence distillation did anything; the distilled score alone is not)")

    results = {
        "distilled": {
            "test_macro_f1_mean": float(distilled_scores.mean()),
            "test_macro_f1_std": float(distilled_scores.std()),
            "parameters": per_seed[0]["distilled"]["parameters"],
        },
        "scratch": {
            "test_macro_f1_mean": float(scratch_scores.mean()),
            "test_macro_f1_std": float(scratch_scores.std()),
            "parameters": per_seed[0]["scratch"]["parameters"],
        },
        "per_seed": per_seed,
    }

    payload = {
        "benchmark": "distillation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine": machine_info(args.device),
        "teacher": {
            "run": run_path.name,
            "parameters": teacher.parameter_count(),
            "test_macro_f1": teacher_run["result"]["test_macro_f1"],
        },
        "student_config": asdict(STUDENT_CONFIG),
        "distill_config": asdict(distill),
        "train_config": asdict(TrainConfig(
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, seed=args.seed, device=args.device)),
        "compression_ratio": teacher.parameter_count() / results["distilled"]["parameters"],
        "seeds": seeds,
        "results": results,
        "distillation_delta_macro_f1": delta,
        "distillation_delta_std": float(deltas.std()),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = BENCHMARK_DIR / f"distillation_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
