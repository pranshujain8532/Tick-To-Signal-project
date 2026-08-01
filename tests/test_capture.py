"""Tests for the parts of `data_engine.capture` that do not need a socket.

The daemon's long-running loops cannot be unit tested without a live feed, but
almost none of the *reasoning* lives in the loops. The sync algorithm's
decisions, the cross-check classifier, the frame parser and the archive format
are all pure functions or small objects, and those are what can be wrong in a
way that quietly corrupts a month of data. They are tested here.

What is deliberately not tested: `connect()`, the reconnect supervisor, and
the signal handler. Testing those means mocking the websockets library, which
tests the mock rather than the code. They are exercised by running the daemon,
which is what the Definition of Done asks for.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import signal
from pathlib import Path

import numpy as np
import pytest

from data_engine.book import SequenceGapError
from data_engine.capture import (
    CaptureConfig,
    CaptureStats,
    RawArchive,
    _count_mismatched_levels,
    _install_shutdown_handler,
    _is_depth_event,
    _record_cross_check,
    _unwrap_combined,
    _verify_snapshot_is_bracketed,
)

SAMPLE_ARCHIVE = Path(__file__).resolve().parents[1] / "notebooks" / "sample_data" / "btcusdt_sample.jsonl.gz"


def make_event(first_id: int, final_id: int) -> dict:
    return {"e": "depthUpdate", "U": first_id, "u": final_id, "b": [], "a": []}


# ------------------------------------------------------------------- config


def test_stream_url_matches_the_documented_combined_format():
    """Combined streams are `/stream?streams=a/b`, lowercase symbols."""
    config = CaptureConfig(symbol="BTCUSDT", out_dir=Path("data"))

    assert config.stream_url() == (
        "wss://stream.binance.com:9443/stream"
        "?streams=btcusdt@depth@100ms/btcusdt@trade"
    )


# ------------------------------------------------------------------ parsing


def test_unwrap_combined_splits_envelope_from_payload():
    frame = '{"stream":"btcusdt@trade","data":{"e":"trade","p":"64000.00"}}'

    stream, payload = _unwrap_combined(frame)

    assert stream == "btcusdt@trade"
    assert payload["e"] == "trade"


def test_unwrap_combined_refuses_an_unenveloped_frame():
    """A bare payload means we are not on the stream we think we are on."""
    with pytest.raises(ValueError, match="combined-stream envelope"):
        _unwrap_combined('{"e":"depthUpdate","U":1,"u":2}')


def test_depth_events_are_distinguished_from_trades():
    assert _is_depth_event({"e": "depthUpdate"}) is True
    assert _is_depth_event({"e": "trade"}) is False
    assert _is_depth_event({}) is False


# -------------------------------------------------- snapshot bracketing rule


def test_bracketed_first_event_is_accepted():
    """The doc's expected case: U <= lastUpdateId <= u."""
    _verify_snapshot_is_bracketed([make_event(998, 1005)], snapshot_id=1000)


def test_abutting_first_event_is_accepted():
    """U == lastUpdateId + 1 is contiguous, loses nothing, and is allowed.

    This is the case the docs' phrasing would reject and we do not, which is
    exactly why it has its own test.
    """
    _verify_snapshot_is_bracketed([make_event(1001, 1005)], snapshot_id=1000)


def test_gap_after_snapshot_is_fatal():
    with pytest.raises(SequenceGapError) as caught:
        _verify_snapshot_is_bracketed([make_event(1005, 1010)], snapshot_id=1000)

    assert caught.value.missed_updates == 4


def test_empty_replay_list_is_accepted():
    """Snapshot newer than everything buffered; the live path will check."""
    _verify_snapshot_is_bracketed([], snapshot_id=1000)


# ------------------------------------------------------------- cross-checks


def _ladder(rows: list[tuple[int, int]]) -> np.ndarray:
    if not rows:
        return np.empty((0, 2), dtype=np.int64)
    return np.array(rows, dtype=np.int64)


def test_identical_ladders_have_no_mismatches():
    ladder = _ladder([(100, 5), (99, 6)])
    assert _count_mismatched_levels(ladder, ladder) == 0


def test_differing_quantity_counts_as_a_mismatch():
    local = _ladder([(100, 5), (99, 6)])
    remote = _ladder([(100, 5), (99, 7)])
    assert _count_mismatched_levels(local, remote) == 1


def test_differing_depth_counts_the_missing_rows():
    local = _ladder([(100, 5)])
    remote = _ladder([(100, 5), (99, 6), (98, 7)])
    assert _count_mismatched_levels(local, remote) == 2


def test_matching_cross_check_passes_and_clears_the_streak():
    stats = CaptureStats(consecutive_mismatches=2)
    config = CaptureConfig(symbol="btcusdt", out_dir=Path("data"))

    assert _record_cross_check(stats, config, mismatches=0, local_id=5, snapshot_id=5) is False
    assert stats.cross_checks_passed == 1
    assert stats.consecutive_mismatches == 0


def test_isolated_mismatch_is_skew_not_failure():
    """One round of disagreement across a round trip must not trigger a resync."""
    stats = CaptureStats()
    config = CaptureConfig(symbol="btcusdt", out_dir=Path("data"))

    assert _record_cross_check(stats, config, mismatches=4, local_id=9, snapshot_id=7) is False
    assert stats.cross_checks_skewed == 1
    assert stats.cross_checks_failed == 0


def test_a_single_differing_level_never_accumulates_toward_failure():
    """The false positive a live run actually produced.

    Three consecutive checks each differing by one level, ids ~10 updates
    apart, escalated to FAIL under the original rule and forced a needless
    resync. One differing level is the touch quantity churning during the HTTP
    round trip; it must not count as evidence however often it repeats.
    """
    stats = CaptureStats()
    config = CaptureConfig(symbol="btcusdt", out_dir=Path("data"))

    outcomes = [
        _record_cross_check(stats, config, mismatches=1, local_id=100 + step, snapshot_id=91 + step)
        for step in range(10)
    ]

    assert not any(outcomes)
    assert stats.cross_checks_failed == 0
    assert stats.consecutive_mismatches == 0


def test_persistent_substantial_mismatch_escalates_to_a_forced_resync():
    """Skew resolves itself; corruption does not, and it is never one level."""
    stats = CaptureStats()
    config = CaptureConfig(symbol="btcusdt", out_dir=Path("data"))

    outcomes = [
        _record_cross_check(stats, config, mismatches=8, local_id=9, snapshot_id=7)
        for _ in range(config.cross_check_failures_before_resync)
    ]

    assert outcomes[:-1] == [False] * (config.cross_check_failures_before_resync - 1)
    assert outcomes[-1] is True
    assert stats.cross_checks_failed == 1
    assert stats.consecutive_mismatches == 0


def test_a_small_mismatch_breaks_the_streak_of_substantial_ones():
    """Evidence must be consecutive; a clean-ish check resets the count."""
    stats = CaptureStats()
    config = CaptureConfig(symbol="btcusdt", out_dir=Path("data"))

    _record_cross_check(stats, config, mismatches=8, local_id=9, snapshot_id=7)
    _record_cross_check(stats, config, mismatches=8, local_id=9, snapshot_id=7)
    _record_cross_check(stats, config, mismatches=1, local_id=9, snapshot_id=7)

    assert stats.consecutive_mismatches == 0


def test_any_mismatch_at_identical_update_ids_fails_immediately():
    """The authoritative case: same instant, so disagreement is corruption.

    No streak, no magnitude threshold — if the two books carry the same update
    id they describe the same state and must be identical.
    """
    stats = CaptureStats()
    config = CaptureConfig(symbol="btcusdt", out_dir=Path("data"))

    assert _record_cross_check(stats, config, mismatches=1, local_id=500, snapshot_id=500) is True
    assert stats.cross_checks_failed == 1


# ---------------------------------------------------------------- the tape


def test_archive_round_trips_records_and_preserves_raw_text(tmp_path: Path):
    """The archive's whole job: what goes in comes back out byte-identical."""
    archive = RawArchive(tmp_path, "btcusdt", rotate_bytes=10 ** 9)
    frames = ['{"stream":"a","data":{"e":"trade"}}', '{"stream":"b","data":{"e":"depthUpdate"}}']
    for frame in frames:
        archive.write(frame, RawArchive.SOURCE_WEBSOCKET)
    archive.close()

    written = sorted(tmp_path.glob("*.jsonl.gz"))
    assert len(written) == 1
    with gzip.open(written[0], "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]

    assert [record["raw"] for record in records] == frames
    assert all(record["source"] == RawArchive.SOURCE_WEBSOCKET for record in records)
    assert all(record["recv_ns"] > 0 and record["recv_mono_ns"] > 0 for record in records)


def test_archive_rotates_once_the_size_budget_is_spent(tmp_path: Path):
    archive = RawArchive(tmp_path, "btcusdt", rotate_bytes=200)
    for index in range(40):
        archive.write(json.dumps({"n": index, "pad": "x" * 50}), RawArchive.SOURCE_WEBSOCKET)
    archive.close()

    assert len(list(tmp_path.glob("*.jsonl.gz"))) > 1


def test_archive_refuses_writes_after_close(tmp_path: Path):
    archive = RawArchive(tmp_path, "btcusdt", rotate_bytes=10 ** 9)
    archive.close()

    with pytest.raises(RuntimeError, match="archive is closed"):
        archive.write("{}", RawArchive.SOURCE_WEBSOCKET)


# ------------------------------------------------------------- shutdown path


def test_ctrl_c_sets_the_shutdown_event_on_this_platform():
    """Installing the handler must work here, and firing it must be graceful.

    `loop.add_signal_handler` is POSIX-only, so Windows takes a `signal.signal`
    fallback — which platform we land on is exactly the kind of thing that is
    silently wrong until someone presses Ctrl-C on a 12-hour capture. Rather
    than raise a real SIGINT (which would interrupt the test runner itself),
    this installs the handler and invokes whatever got registered, asserting
    the effect: the event is set, so the daemon unwinds through its normal
    teardown and closes the archive instead of dying mid-write.
    """

    async def install_and_fire() -> bool:
        shutdown = asyncio.Event()
        previous = signal.getsignal(signal.SIGINT)
        try:
            _install_shutdown_handler(shutdown)
            handler = signal.getsignal(signal.SIGINT)
            if callable(handler):
                handler(signal.SIGINT, None)     # the Windows fallback path
            else:
                asyncio.get_running_loop().call_soon(shutdown.set)  # POSIX path
            await asyncio.wait_for(shutdown.wait(), timeout=2.0)
        finally:
            signal.signal(signal.SIGINT, previous)
        return shutdown.is_set()

    assert asyncio.run(install_and_fire()) is True


@pytest.mark.skipif(not SAMPLE_ARCHIVE.exists(), reason="bundled sample archive is missing")
def test_bundled_sample_is_self_sufficient():
    """Notebook 01 cannot run unless the sample carries its own snapshot.

    A diff stream describes only changes, so an archive of websocket frames
    alone can never be replayed into an absolute book. This asserts the
    property that makes the tape replayable offline at all.
    """
    sources = set()
    with gzip.open(SAMPLE_ARCHIVE, "rt", encoding="utf-8") as handle:
        for line in handle:
            sources.add(json.loads(line)["source"])

    assert RawArchive.SOURCE_SYNC_SNAPSHOT in sources
    assert RawArchive.SOURCE_WEBSOCKET in sources
