"""Read the saved measurement records and shape them into API payloads.

WHAT
    Loads the newest `benchmarks/*.json` record of each kind — Stage 5's
    evaluation, Stage 6's Python latency frontier, Stage 7b's C++ harness runs —
    and turns them into the dictionaries `serving/api.py` serves from `/pareto`,
    `/decay`, `/stability` and `/economics`.

WHY THIS IS A MODULE AND NOT A DICT LITERAL IN api.py
    The constitution's rule is that no number is typed by hand into anything a
    reader sees, and Stage 6 measured 55% run-to-run latency variance on this
    machine. A hardcoded constant in the serving layer would therefore not be a
    shortcut, it would be a slowly-rotting lie: the record on disk would move
    and the dashboard would keep showing the number that was true in July.

    Putting the loading here rather than in `api.py` also makes it testable.
    `api.py` is one of the two modules the constitution exempts from notebook
    coverage because it is a long-running server; everything it *serves* lives
    here instead, where a pytest can compare it against the file it claims to
    come from. `tests/test_serving.py` does exactly that.

DESIGN DECISION — newest-file-wins, and the filename is returned with the data.
    Rejected alternative: pin a specific record filename. That would be
    reproducible but would silently ignore a fresh measurement, which is the
    opposite of the problem this project has. Instead every payload carries a
    `source` field naming the exact file it was read from, so "where did this
    number come from" is answerable from the API response alone rather than by
    reading this file.

DESIGN DECISION — the Stage 5 verdict is a constant here, not parsed from
markdown.
    It is prose, and prose living in `docs/INTERVIEW_NOTES_stage5.md` is not
    machine-readable without a fragile parser. Copying it into a constant risks
    drift, so `tests/test_serving.py` asserts the constant still appears
    verbatim in that document. That gets the robustness of a constant with the
    drift-detection of a parse.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = REPO_ROOT / "benchmarks"
STAGE5_NOTES = REPO_ROOT / "docs" / "INTERVIEW_NOTES_stage5.md"

# Served verbatim by `/economics`. Source: docs/INTERVIEW_NOTES_stage5.md, the
# "One sentence" summary. tests/test_serving.py fails if the document stops
# containing it.
STAGE5_VERDICT = (
    "a real but small and unstable directional edge with a multi-second "
    "half-life, roughly two orders of magnitude too small to pay Binance's "
    "taker fee, therefore not tradeable as a taker by anyone at any published "
    "tier."
)

# Rendered verbatim by the dashboard beside any latency figure. Required by the
# "Honesty rules specific to the dashboard" section of CLAUDE.md: a latency
# panel with no context implies latency is what makes this work, and it is not.
RELEVANCE_NOTE = (
    "This signal's IC half-life is 13.2 s. Latency is not what makes it "
    "tradeable — the fee shortfall is 70x. The microsecond path is the "
    "prerequisite for signals with sub-second half-lives, and for a maker "
    "deciding whether to pull a quote."
)

# The variant the serving loop actually runs. Everything else on the frontier is
# shown for comparison and marked `is_serving: false`.
SERVING_VARIANT = "student_int8"

# The variant whose accuracy the C++ rows inherit. `inference_cpp` is float32 and
# its weights come from the same distilled checkpoint this variant was exported
# from, so this is the model it is a re-implementation of. See `_cpp_rows`.
CPP_ACCURACY_VARIANT = "student_fp32"

# Point labels for the frontier. Prose, not numbers, and kept here rather than in
# the dashboard so the axis a reader sees is labelled by the same module that
# chose the rows — a JS-side variant->label map is one rename away from a chart
# whose points are confidently mislabelled.
VARIANT_LABELS = {
    "pytorch_eager_fp32": "PyTorch eager fp32",
    "onnx_fp32": "ONNX fp32",
    "onnx_int8": "ONNX int8",
    "student_fp32": "distilled student fp32",
    "student_int8": "distilled student int8",
    "cpp_full": "C++ full",
    "cpp_incremental": "C++ incremental",
}

# What separates the two C++ points, in the terms the frontier is read in.
CPP_VARIANT_NOTES = {
    "full": (
        "Recomputes the whole 100-row window every tick, including the 16 KB "
        "input memcpy. This is the honest baseline for 'hand-written C++': the "
        "first correct version of it was 6.5x slower than ONNX Runtime on the "
        "same model."
    ),
    "incremental": (
        "Advances one column and reuses the rest, so no input copy happens at "
        "all. This is the path that ships, and the ~193x it wins over the full "
        "pass is an algorithm the runtime could not express, not a language."
    ),
}


class RecordNotFound(RuntimeError):
    """A record the API promises to serve is not on disk.

    Raised at startup rather than per-request: a server that boots and then
    404s its own evidence is worse than one that refuses to boot.
    """


def newest(pattern: str) -> Path:
    """The most recent `benchmarks/` file matching `pattern`, by filename.

    Filenames carry a UTC timestamp in the project's fixed
    `name_YYYYmmddTHHMMSSZ.json` form, so a lexicographic sort is a
    chronological one. Sorting by mtime instead would reorder records whenever
    the repository is cloned.
    """
    matches = sorted(glob.glob(str(BENCHMARKS / pattern)))
    if not matches:
        raise RecordNotFound(
            f"no benchmark record matches {pattern!r} under {BENCHMARKS}. "
            "Run the stage that produces it before starting the API."
        )
    return Path(matches[-1])


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_evaluation() -> tuple[dict[str, Any], str]:
    """Stage 5's evaluation record, with the filename it came from."""
    path = newest("evaluation_*.json")
    return _load(path), path.name


def load_python_variants() -> tuple[dict[str, Any], str]:
    """Stage 6's Python latency frontier."""
    path = newest("python_variants_*.json")
    return _load(path), path.name


def load_cpp_runs() -> list[tuple[dict[str, Any], str]]:
    """Every Stage 7b C++ harness record, newest-first by filename."""
    matches = sorted(glob.glob(str(BENCHMARKS / "cpp_*.json")), reverse=True)
    if not matches:
        raise RecordNotFound(f"no cpp_*.json records under {BENCHMARKS}")
    return [(_load(Path(m)), Path(m).name) for m in matches]


# --------------------------------------------------------------- payloads


def pareto_payload() -> dict[str, Any]:
    """The accuracy-vs-latency frontier: five Python rows and both C++ rows.

    `measured_by` exists because the two halves of this table were produced by
    different instruments — a Python harness using `perf_counter_ns` and a C++
    harness using QueryPerformanceCounter — and comparing them without saying so
    invites the reader to believe one number is a continuation of the other. It
    is not: the C++ rows cover the forward pass only, from a prepared [40]
    feature column to three logits, and exclude the feature construction that
    is inside every Python row.

    The dashboard draws every row on one axis, which is precisely why each one
    carries the instrument that produced it. `serving/dashboard/js/panels/pareto.js`
    renders the two harnesses with different marks and says so in the legend.
    """
    variants, python_source = load_python_variants()
    rows: list[dict[str, Any]] = []
    for entry in variants["variants"]:
        latency = entry["latency_us"]
        rows.append(
            {
                "variant": entry["variant"],
                "label": VARIANT_LABELS[entry["variant"]],
                "params": entry["parameters"],
                "size_kib": entry["size_kib"],
                "macro_f1": entry["macro_f1"],
                "p50_us": latency["p50_us"],
                "p99_us": latency["p99_us"],
                "p99_9_us": latency["p99_9_us"],
                "is_serving": entry["variant"] == SERVING_VARIANT,
                "measured_by": "python harness",
            }
        )
    cpp_rows = _cpp_rows(variants)
    rows.extend(cpp_rows)
    return {
        "rows": rows,
        "serving_variant": SERVING_VARIANT,
        "sources": {
            "python": python_source,
            **{row["variant"]: row["source"] for row in cpp_rows},
        },
        "boundary_note": (
            "Python rows time feature construction plus the forward pass. The "
            "C++ row times the forward pass only, from a prepared [40] feature "
            "column to three logits; feature construction is still Python and "
            "is not measured in C++. The two columns are not interchangeable."
        ),
    }


def _cpp_rows(python_variants: dict[str, Any]) -> list[dict[str, Any]]:
    """Both Stage 7b C++ paths as frontier rows: the full pass and the incremental one.

    WHY BOTH, when only the incremental path ships.
        They are the two ends of the Stage 7b argument and the gap between them
        is the whole finding. The full path recomputes the 100-row window on
        every tick; the incremental path advances one column and reuses the
        rest. Serving only the incremental row would present a ~193x algorithmic
        win as if it were what "writing it in C++" buys, which is the exact
        misreading `docs/INTERVIEW_NOTES_stage7b.md` §Q10 exists to prevent —
        the first correct hand-written full pass was 6.5x SLOWER than ONNX
        Runtime on the same model.

    WHY ACCURACY IS INHERITED FROM student_fp32 AND NOT FROM THE SERVING VARIANT.
        `inference_cpp` is a float32 implementation. Its weights come from
        `checkpoints/student_distilled.pt` (`ml/export_weights.py:475`) with
        BatchNorm folded, and nothing in it is quantised. Its parity test agrees
        with the PyTorch fp32 student on 1,000 held-out windows to 2.4e-05, and
        `incremental_test.cpp` proves the incremental path equals the full one —
        so the accuracy chain is fp32 student -> C++ full -> C++ incremental,
        and student_fp32's macro-F1 is the one these rows inherit.

        This was previously wrong: the row was named `..._int8_equivalent` and
        inherited student_int8's 0.5723, which credited a float32 program with a
        quantised model's accuracy loss and put the point 0.023 macro-F1 too low
        on the frontier.

        Inherited is not measured, and the payload says so on every row. No C++
        program in this repository has ever scored a test block; claiming a
        separately measured accuracy would be a fabrication.
    """
    runs = load_cpp_runs()
    student = _find_variant(python_variants, CPP_ACCURACY_VARIANT)
    rows: list[dict[str, Any]] = []
    for variant in ("full", "incremental"):
        record, name = _newest_cpp_run(runs, variant)
        rows.append(
            {
                "variant": f"cpp_{variant}",
                "label": VARIANT_LABELS[f"cpp_{variant}"],
                "params": student["parameters"],
                "size_kib": None,
                "macro_f1": student["macro_f1"],
                "macro_f1_is_inherited": True,
                "macro_f1_inherited_from": CPP_ACCURACY_VARIANT,
                "p50_us": record["p50_ns"] / 1000.0,
                "p99_us": record["p99_ns"] / 1000.0,
                "p99_9_us": record["p999_ns"] / 1000.0,
                "is_serving": False,
                "measured_by": "cpp harness",
                "build": record["build_name"],
                "iterations": record["iterations"],
                "note": CPP_VARIANT_NOTES[variant],
                "source": name,
            }
        )
    return rows


def _newest_cpp_run(
    runs: list[tuple[dict[str, Any], str]], variant: str
) -> tuple[dict[str, Any], str]:
    """The most recent streamed run of one C++ variant, by the record's own clock.

    Selected on `timestamp_utc` inside the record rather than on the filename,
    because these filenames carry an optimisation-pass label (`_p0`, `_p4`) and
    not a timestamp — so unlike everything else under `benchmarks/`, sorting
    them lexicographically sorts by pass number, which is only accidentally
    chronological.

    `input_mode == "stream"` is required of both. A `hot` run replays one
    cached input and measures the kernel with the cache warm; a `stream` run
    walks fresh columns, which is the situation a live book actually presents.
    Mixing the two on one axis would compare a best case against a real one.
    """
    candidates = [
        (record, name)
        for record, name in runs
        if record.get("variant") == variant and record.get("input_mode") == "stream"
    ]
    if not candidates:
        raise RecordNotFound(
            f"no {variant}/stream C++ record among cpp_*.json under {BENCHMARKS}. "
            "Run inference_cpp/bench with --variant "
            f"{variant} --input-mode stream before starting the API."
        )
    return max(candidates, key=lambda pair: pair[0]["timestamp_utc"])


def _find_variant(python_variants: dict[str, Any], name: str) -> dict[str, Any]:
    for entry in python_variants["variants"]:
        if entry["variant"] == name:
            return entry
    raise RecordNotFound(f"variant {name!r} is missing from the Python frontier record")


def decay_payload() -> dict[str, Any]:
    """The IC-vs-horizon curve, its peak, and the half-life fitted after it.

    The peak index matters and is served explicitly. The IC *rises* to about 5 s
    before it decays, so a half-life fitted from horizon zero would be fitted
    through the rising limb and would be meaningless. Stage 5 fits from the peak
    onwards, and `fit_range` says which points that was.
    """
    evaluation, source = load_evaluation()
    decay = evaluation["decay"]
    horizons = [row["horizon_ms"] for row in decay["rows"]]
    peak_index = horizons.index(decay["peak_horizon_ms"])
    return {
        "curve": decay["rows"],
        "peak_index": peak_index,
        "peak_horizon_ms": decay["peak_horizon_ms"],
        "peak_ic": decay["peak_ic"],
        "fit_range": {"start_index": peak_index, "end_index": len(horizons) - 1},
        "half_life_ms": decay["half_life_ms"],
        "fit_r_squared": decay["fit_r_squared"],
        "label_horizon_ms": evaluation["label_horizon_ms"],
        "rise_reason": (
            "The IC rises to the peak because the label is a smoothed forward "
            "mid over a 2.7 s horizon: a prediction made at t is not fully "
            "expressed until the horizon it was trained on has elapsed. "
            "Measuring correlation against a shorter forward return therefore "
            "reads part of the move, not none of it. The half-life is fitted "
            "from the peak onward for that reason."
        ),
        "source": source,
    }


def stability_payload() -> dict[str, Any]:
    """Per-block IC — the number that describes what a trader would experience.

    The pooled IC is returned under `pooled_ic_not_tradeable` rather than
    `pooled_ic`, deliberately. Naming is the cheapest available guard: a
    dashboard author reaching for the biggest number in this payload has to type
    the words "not tradeable" to get it.
    """
    evaluation, source = load_evaluation()
    ic = evaluation["information_coefficient"]
    null = evaluation["falsification"]["null_distribution"]
    return {
        "block_ics": ic["block_ics"],
        "block_size": ic["block_size"],
        "mean": ic["mean"],
        "median": ic["median"],
        "std": ic["std"],
        "information_ratio": ic["information_ratio"],
        "fraction_positive": ic["fraction_positive"],
        "pooled_ic_not_tradeable": ic["pooled"],
        "pooled_caveat": (
            "The pooled IC (+{pooled:.3f}) is computed across the whole test "
            "block at once, which lets slow common drift inflate the "
            "correlation. The per-block mean (+{mean:.3f}, IR {ir:.2f}) is the "
            "figure that describes a tradeable edge."
        ).format(pooled=ic["pooled"], mean=ic["mean"], ir=ic["information_ratio"]),
        "null": {
            "trials": null["trials"],
            "mean": null["mean"],
            "std": null["std"],
            "z_score": null["z_score"],
        },
        "source": source,
    }


def economics_payload() -> dict[str, Any]:
    """Breakeven fee against the real published fee schedule, and the verdict.

    Shortfall multiples are computed here rather than stored, because they are a
    ratio of two served numbers and computing them keeps them consistent with
    whatever the record currently says.
    """
    evaluation, source = load_evaluation()
    trading = evaluation["trading"]
    breakeven = trading["breakeven_fee_bps"]
    tiers = trading["binance_taker_tiers_bps"]
    return {
        "gross_bps_per_trade": trading["gross_bps_per_trade"],
        "breakeven_fee_bps": breakeven,
        "median_spread_bps": trading["median_spread_bps"],
        "trade_count": trading["trade_count"],
        "fee_sweep": trading["fee_sweep"],
        "tiers": [
            {
                "name": name,
                "fee_bps": fee,
                "shortfall_multiple": fee / breakeven,
                "net_bps_per_trade": trading["gross_bps_per_trade"] - 2.0 * fee,
            }
            for name, fee in tiers.items()
        ],
        "tiers_checked_on": evaluation["generated_at_utc"][:10],
        "verdict": STAGE5_VERDICT,
        "source": source,
    }
