"""Measure the binary tape against the raw JSON archive it replaces.

Run against a capture session that produced *both* artefacts, so the two
formats are describing exactly the same market data:

    python benchmarks/bench_binfmt.py --data-dir data/ --repeats 5

Writes a timestamped JSON result under `benchmarks/` and prints a summary.
Nothing here is hand-typed into the README; the README quotes this file.

WHAT IS MEASURED, AND WHY IN THIS SHAPE

  1. Size. Four numbers, not one: binary and JSON, each raw and gzipped.
     Quoting only "binary vs raw JSON" would be choosing the flattering
     comparison — gzip is available to both formats and costs one line of
     code, so a size claim that ignores it is not a real claim.

  2. Sequential replay throughput. Reconstruct the book event by event from
     each format and count level updates per second. This is apples to apples:
     both paths do the same work and end in the same state.

  3. Vectorised extraction. Pull a spread time series out of the tape with
     `load_snapshots`, versus the only way to get the same series from JSON,
     which is a full replay. This is the comparison the format was designed
     to win, and it is reported separately from (2) rather than blended in.

METHOD
    `time.perf_counter`, the highest-resolution clock available. One untimed
    warmup pass per path so the file is in the OS page cache and the branch
    predictors have seen the loop — otherwise the first measurement is a disk
    benchmark wearing a parser's clothes. Then `--repeats` timed passes, of
    which we report the median (robust to a scheduler hiccup) and the minimum
    (the closest thing to the machine's real capability). Never the mean.

    No core pinning and no thermal control: this is a laptop running a desktop
    OS, and pretending otherwise would be the kind of benchmark theatre
    `docs/benchmark_methodology.md` exists to prevent. These are throughput
    numbers spanning seconds, where scheduling noise averages out; the
    microsecond latency work in Stage 7 needs, and will get, much more care.
"""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

# Benchmarks are run as scripts from the repo root, not as an installed
# package, so the root has to be on the path before `data_engine` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_engine.binfmt import EVENT_DIFF, EVENT_PADDING
from data_engine.book import PRICE_SCALE, QTY_SCALE, OrderBook, to_fixed
from data_engine.replay import TapeReader, snapshot_spreads

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TimingResult:
    """One timed path: what it did, how fast, and how much it varied."""

    label: str
    units_processed: int
    unit_name: str
    median_seconds: float
    min_seconds: float
    samples: list[float]

    @property
    def median_units_per_second(self) -> float:
        return self.units_processed / self.median_seconds


def time_path(label: str, work: Callable[[], int], unit_name: str, repeats: int) -> TimingResult:
    """Warm up once untimed, then time `repeats` passes and keep them all."""
    units = work()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        units = work()
        samples.append(time.perf_counter() - started)
    return TimingResult(
        label=label,
        units_processed=units,
        unit_name=unit_name,
        median_seconds=statistics.median(samples),
        min_seconds=min(samples),
        samples=samples,
    )


# ------------------------------------------------------------------- the work


def replay_binary(tape_path: Path) -> int:
    """Full sequential replay from the tape. Returns level updates applied."""
    applied = 0
    with TapeReader(tape_path) as reader:
        for _timestamp, book in reader.iter_books():
            applied += 1
        # Touch the book so the loop cannot be optimised into nothing and so
        # we prove it ended somewhere sane rather than empty.
        assert book.best_bid() is not None
    return applied


def replay_json(archive_path: Path) -> int:
    """Full sequential replay from the raw archive. Returns records handled.

    Deliberately the *fair* JSON implementation, not a strawman: one
    `json.loads` per line, one per embedded frame, and the same `OrderBook`
    the binary path uses. The cost being measured is parsing, not a
    handicapped algorithm.

    Trades are counted as handled records even though they do not touch the
    book, because the binary path yields them too. Comparing a loop that
    processes trades against one that skips them would inflate the binary
    side's throughput by whatever fraction of the feed is trades — which here
    is most of it.
    """
    book = OrderBook("BTCUSDT")
    handled = 0
    with gzip.open(archive_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["source"] != "websocket":
                continue
            payload = json.loads(record["raw"])["data"]
            if payload.get("e") == "trade":
                to_fixed(payload["p"], PRICE_SCALE)
                handled += 1
                continue
            if payload.get("e") != "depthUpdate":
                continue
            for side_key, is_bid in (("b", True), ("a", False)):
                for price_string, quantity_string in payload[side_key]:
                    book.apply_level_update(
                        is_bid, to_fixed(price_string, PRICE_SCALE), to_fixed(quantity_string, QTY_SCALE)
                    )
                    handled += 1
    return handled


def spreads_from_binary(tape_path: Path) -> int:
    """Vectorised spread series straight out of the mapped snapshots."""
    with TapeReader(tape_path) as reader:
        spreads = snapshot_spreads(reader.load_snapshots(), reader.header.price_scale)
        finite = int(np.count_nonzero(np.isfinite(spreads)))
    return finite


def spreads_from_json(archive_path: Path, sample_every: int) -> int:
    """Same series from JSON, which means replaying the whole archive."""
    book = OrderBook("BTCUSDT")
    applied = 0
    sampled = 0
    with gzip.open(archive_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["source"] != "websocket":
                continue
            payload = json.loads(record["raw"])["data"]
            if payload.get("e") != "depthUpdate":
                continue
            for side_key, is_bid in (("b", True), ("a", False)):
                for price_string, quantity_string in payload[side_key]:
                    book.apply_level_update(
                        is_bid, to_fixed(price_string, PRICE_SCALE), to_fixed(quantity_string, QTY_SCALE)
                    )
                    applied += 1
                    if applied % sample_every == 0 and book.spread() is not None:
                        sampled += 1
    return sampled


# -------------------------------------------------------------------- sizing


def measure_sizes(tape_path: Path, archive_path: Path) -> dict[str, int]:
    """On-disk and gzipped sizes of both artefacts, plus their record counts."""
    archive_bytes = archive_path.read_bytes()
    plain_json = gzip.decompress(archive_bytes)
    tape_bytes = tape_path.read_bytes()
    return {
        "json_gzip_bytes": len(archive_bytes),
        "json_plain_bytes": len(plain_json),
        "tape_bytes": len(tape_bytes),
        "tape_gzip_bytes": len(gzip.compress(tape_bytes, 6)),
    }


def count_tape_records(tape_path: Path) -> dict[str, int]:
    """Real (non-padding) events and snapshots actually stored on the tape."""
    with TapeReader(tape_path) as reader:
        events = reader.load_events()
        event_types = events["event_type"]
        return {
            "tape_blocks": int(reader.block_count),
            "tape_snapshots": int(reader.block_count),
            "tape_events_real": int(np.count_nonzero(event_types != EVENT_PADDING)),
            "tape_events_diff": int(np.count_nonzero(event_types == EVENT_DIFF)),
            "tape_events_padding": int(np.count_nonzero(event_types == EVENT_PADDING)),
            "snapshot_interval": int(reader.header.snapshot_interval),
            "depth_levels": int(reader.header.depth_levels),
        }


def machine_info() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


# ---------------------------------------------------------------------- main


def find_inputs(data_dir: Path) -> tuple[Path, Path]:
    """Pick the newest tape and the newest raw archive from a capture run."""
    tapes = sorted(data_dir.glob("*.tape"))
    archives = sorted(data_dir.glob("*.jsonl.gz"))
    if not tapes or not archives:
        raise SystemExit(
            f"need both a .tape and a .jsonl.gz in {data_dir}; run "
            "`python -m data_engine.capture --symbol btcusdt --out data/ --max-messages 12000` first"
        )
    return tapes[-1], archives[-1]


def build_report(tape_path: Path, archive_path: Path, repeats: int) -> dict:
    sizes = measure_sizes(tape_path, archive_path)
    counts = count_tape_records(tape_path)

    sequential_binary = time_path("binary sequential replay", lambda: replay_binary(tape_path), "records", repeats)
    sequential_json = time_path("json sequential replay", lambda: replay_json(archive_path), "records", repeats)
    sample_every = counts["snapshot_interval"]
    vector_binary = time_path("binary vectorised spreads", lambda: spreads_from_binary(tape_path), "snapshots", repeats)
    vector_json = time_path(
        "json replay to same spreads", lambda: spreads_from_json(archive_path, sample_every), "snapshots", repeats
    )

    events = counts["tape_events_real"]
    return {
        "benchmark": "binfmt",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine": machine_info(),
        "inputs": {"tape": tape_path.name, "archive": archive_path.name, "repeats": repeats},
        "counts": counts,
        "sizes": sizes,
        "bytes_per_event": {
            "binary": sizes["tape_bytes"] / events,
            "binary_gzip": sizes["tape_gzip_bytes"] / events,
            "json_plain": sizes["json_plain_bytes"] / events,
            "json_gzip": sizes["json_gzip_bytes"] / events,
        },
        "size_ratios": {
            "binary_vs_json_plain": sizes["json_plain_bytes"] / sizes["tape_bytes"],
            "binary_vs_json_gzip": sizes["json_gzip_bytes"] / sizes["tape_bytes"],
            "binary_gzip_vs_json_gzip": sizes["json_gzip_bytes"] / sizes["tape_gzip_bytes"],
        },
        "timings": {
            result.label: asdict(result) | {"median_units_per_second": result.median_units_per_second}
            for result in (sequential_binary, sequential_json, vector_binary, vector_json)
        },
        "speedups": {
            "sequential_binary_vs_json": sequential_json.median_seconds / sequential_binary.median_seconds,
            "vectorised_binary_vs_json_replay": vector_json.median_seconds / vector_binary.median_seconds,
        },
        "caveats": [
            "The binary record count includes one block anchor per snapshot_interval events "
            "(~1% of records); the JSON count has no equivalent. Wall-clock ratios are the "
            "honest comparison, not the per-record rates.",
            "json_plain_bytes includes the REST depth snapshots the daemon archives for "
            "offline cross-checking. The tape does not store those, so JSON carries a small "
            "amount of data the binary format does not.",
            "No core pinning, no thermal control, laptop running a desktop OS.",
        ],
    }


def print_summary(report: dict) -> None:
    sizes, ratios = report["sizes"], report["size_ratios"]
    print("\n--- size ---")
    for name, value in sizes.items():
        print(f"  {name:24s} {value:>12,} bytes")
    print(f"  binary is {ratios['binary_vs_json_plain']:.2f}x smaller than raw JSON")
    print(f"  binary is {1 / ratios['binary_vs_json_gzip']:.2f}x LARGER than gzipped JSON")
    print(f"  gzipped binary is {1 / ratios['binary_gzip_vs_json_gzip']:.2f}x larger than gzipped JSON")

    print("\n--- throughput (median of timed passes) ---")
    for label, timing in report["timings"].items():
        rate = timing["median_units_per_second"]
        print(f"  {label:32s} {timing['median_seconds']:7.3f}s  {rate:>14,.0f} {timing['unit_name']}/s")
    print(f"\n  sequential replay speedup : {report['speedups']['sequential_binary_vs_json']:.2f}x")
    print(f"  vectorised scan speedup   : {report['speedups']['vectorised_binary_vs_json_replay']:.1f}x")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data/", help="directory holding a .tape and a .jsonl.gz")
    parser.add_argument("--repeats", type=int, default=5, help="timed passes per path, after one warmup")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "benchmarks"), help="where to save the JSON result")
    args = parser.parse_args()

    tape_path, archive_path = find_inputs(Path(args.data_dir))
    print(f"tape    : {tape_path}")
    print(f"archive : {archive_path}")

    report = build_report(tape_path, archive_path, args.repeats)
    print_summary(report)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.out_dir) / f"binfmt_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nsaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
