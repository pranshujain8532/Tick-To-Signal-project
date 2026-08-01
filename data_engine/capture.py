"""Live order-book capture daemon for Binance spot L2 depth and trades.

WHAT
    A long-running asyncio process that subscribes to the Binance combined
    websocket stream for `<symbol>@depth@100ms` and `<symbol>@trade`, keeps a
    correct local order book using the venue's documented snapshot-sync
    algorithm, archives every raw message to rotating gzipped JSONL, proves
    its own correctness against a fresh REST snapshot every 60 seconds, and
    prints a heartbeat so you can watch it work.

WHY
    Capture is the one stage that cannot be redone. A bug in book building or
    feature engineering is fixable by replaying the tape; a gap in the tape is
    permanent. So the capture path stays short and boring, every interpretive
    decision is deferred to modules that can be re-run offline, and anything
    that smells like data loss is made loud and countable rather than handled.

DESIGN DECISION — archive the raw frames verbatim, not the parsed book.
    Rejected alternative: reconstruct the book live and write periodic
    snapshots, which is far less disk. It bakes today's reconstruction logic
    into the archive forever: any bug in the sequencing rules silently
    corrupts data we can never recover, and Stage 2's binary format would then
    have nothing independent to validate against. The raw frames are the
    ground truth that lets us re-derive everything and diff old parsers
    against new ones. Bytes are the cheapest thing in this system.
    For the same reason the archive stores the websocket text frame as an
    opaque string rather than a re-serialised parse — re-serialising would
    bake *this* module's json handling into the archive too.

DESIGN DECISION — a sequence gap tears down the whole session and resyncs.
    Rejected alternative: patch the hole in place by refetching a snapshot and
    merging while the socket stays up. Binance's own procedure says to discard
    the book and restart from the beginning, and the simpler control flow is
    worth more than the second of data it costs: there is exactly one code
    path that produces a valid book, so there is exactly one path to get
    right. Resyncs are counted and logged; a rising count is a signal about
    the network, not something to hide.

DESIGN DECISION — REST snapshots fetched with stdlib urllib on a thread.
    Rejected alternative: `aiohttp` or `requests`. Neither is in the
    dependency list in CLAUDE.md, and adding an HTTP stack for two GET
    requests a minute is not a trade worth making. `urllib.request` inside
    `asyncio.to_thread` is blocking-but-off-the-event-loop, which is all this
    needs at two requests per minute.

DESIGN DECISION — the 60-second cross-check distinguishes SKEW from FAIL.
    A REST snapshot and the live book describe different instants: the book
    keeps moving during the HTTP round trip. Comparing them naively produces
    false alarms in fast markets, and an alarm that cries wolf is worse than
    no alarm. So a mismatch is reported as SKEW (advisory) unless it persists
    across several consecutive checks, which is the signature of real
    corruption rather than of latency — transient skew resolves, corruption
    does not. See `_cross_check_loop`.

NOTE ON NOTEBOOK COVERAGE
    Per CLAUDE.md this is one of two modules whose logic cannot literally run
    in a notebook (an unbounded async loop against a live socket). Everything
    it knows about book semantics lives in `data_engine/book.py`, which is
    importable and is rebuilt step by step in
    `notebooks/01_orderbook_capture_walkthrough.ipynb` from an archive file
    this module produced.

PROTOCOL REFERENCES (checked 2026-07-27)
    Streams, payload fields, and the "How to manage a local order book
    correctly" procedure implemented in `_synchronise_book`:
        https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams
    REST depth snapshot endpoint, `limit` values and request weights:
        https://developers.binance.com/docs/binance-spot-api-docs/rest-api

Usage:
    python -m data_engine.capture --symbol btcusdt --out data/
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import logging
import signal
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from data_engine.binfmt import (
    DEFAULT_DEPTH_LEVELS,
    DEFAULT_SNAPSHOT_INTERVAL,
    EVENT_DIFF,
    EVENT_TRADE,
    SIDE_ASK,
    SIDE_BID,
    BinaryWriter,
    TapeHeader,
)
from data_engine.book import PRICE_SCALE, QTY_SCALE, OrderBook, SequenceGapError, to_fixed

LOGGER = logging.getLogger("capture")

WEBSOCKET_BASE_URL = "wss://stream.binance.com:9443/stream"
REST_DEPTH_URL = "https://api.binance.com/api/v3/depth"

# Binance closes any single connection at the 24 hour mark and pings every 20
# seconds. Both are normal operating conditions, not errors, so the reconnect
# path logs them at INFO.
_EXPECTED_DISCONNECT_HINT = "server closed the connection (expected at least every 24h)"


class CrossCheckFailure(Exception):
    """Raised when the periodic cross-check proves the local book is wrong.

    Exists so a cross-check-forced teardown reaches the supervisor the same
    way a sequence gap does — as an exception. The first version had
    `_cross_check_loop` simply return, which ended the session correctly but
    left `resync_count` untouched, so a 10-minute live run reconnected without
    the counter ever recording it. A resync the operator cannot see is exactly
    the kind of silence this project keeps refusing to accept.
    """


@dataclass
class CaptureConfig:
    """Everything the daemon needs to run. A dataclass, not a config system."""

    symbol: str
    out_dir: Path
    depth_update_speed: str = "100ms"
    snapshot_limit: int = 1000
    heartbeat_seconds: float = 10.0
    cross_check_seconds: float = 60.0
    cross_check_levels: int = 10
    # Escalation rule for cross-check mismatches when the two books describe
    # different instants. A mismatch must be substantial (this many levels of
    # the compared 2 x cross_check_levels) to count as evidence at all, and
    # then several must occur consecutively. Both numbers were raised after a
    # live run escalated on three single-level mismatches, which were skew.
    # A mismatch at *identical* update ids bypasses both and fails at once.
    cross_check_minimum_mismatch: int = 3
    cross_check_failures_before_resync: int = 5
    rotate_bytes: int = 64 * 1024 * 1024
    reconnect_backoff_seconds: float = 2.0
    max_messages: int = 0  # 0 means run until interrupted
    # The binary tape written alongside the raw archive. The raw .jsonl.gz
    # stays: it is the ground truth the binary format is validated against, and
    # discarding it would leave the format with nothing to be checked by.
    write_binary_tape: bool = True
    snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL
    depth_levels: int = DEFAULT_DEPTH_LEVELS

    def stream_url(self) -> str:
        """Build the combined-stream URL documented under 'Combined streams'."""
        symbol = self.symbol.lower()
        streams = f"{symbol}@depth@{self.depth_update_speed}/{symbol}@trade"
        return f"{WEBSOCKET_BASE_URL}?streams={streams}"


@dataclass
class CaptureStats:
    """Mutable counters shared by the reader, heartbeat and cross-check tasks."""

    started_monotonic: float = field(default_factory=time.monotonic)
    messages_total: int = 0
    depth_events_applied: int = 0
    resync_count: int = 0
    cross_checks_passed: int = 0
    cross_checks_skewed: int = 0
    cross_checks_failed: int = 0
    consecutive_mismatches: int = 0
    tape_events_written: int = 0
    tape_snapshots_written: int = 0
    _messages_at_last_heartbeat: int = 0
    _monotonic_at_last_heartbeat: float = field(default_factory=time.monotonic)

    def uptime_seconds(self) -> float:
        return time.monotonic() - self.started_monotonic

    def messages_per_second_since_last_heartbeat(self) -> float:
        """Rate over the interval since the previous heartbeat, then reset.

        Deliberately an interval rate rather than a lifetime average: a
        lifetime average hides a feed that has gone quiet, which is the exact
        failure this heartbeat exists to make visible.
        """
        now = time.monotonic()
        elapsed = now - self._monotonic_at_last_heartbeat
        delta = self.messages_total - self._messages_at_last_heartbeat
        self._messages_at_last_heartbeat = self.messages_total
        self._monotonic_at_last_heartbeat = now
        if elapsed <= 0:
            return 0.0
        return delta / elapsed


class RawArchive:
    """Append-only rotating writer for raw feed data.

    One line per record:
    `{"recv_ns": ..., "recv_mono_ns": ..., "source": ..., "raw": "..."}`.

    Two timestamps on purpose. `recv_ns` is the wall clock, needed to align
    this tape with anything else in the world. `recv_mono_ns` is a monotonic
    counter, which is the only one of the two safe for measuring gaps between
    messages — the wall clock can step backwards under NTP, and a negative
    inter-arrival time would quietly become a negative feature later.

    `source` distinguishes websocket frames from the REST snapshots this
    daemon fetches. Both go on the tape, which is what makes the tape
    *self-sufficient*: a diff stream alone can only ever describe changes, so
    an archive without snapshots cannot be replayed into an absolute book
    offline. Recording the sync snapshot means Stage 2's replay needs nothing
    but the file, and recording the 60-second cross-check snapshots means the
    tape also carries its own correctness proof — a later validator can
    re-run the comparison without a network connection.
    """

    SOURCE_WEBSOCKET = "websocket"
    SOURCE_SYNC_SNAPSHOT = "rest_depth_sync"
    SOURCE_CROSS_CHECK_SNAPSHOT = "rest_depth_crosscheck"

    def __init__(self, out_dir: Path, symbol: str, rotate_bytes: int) -> None:
        self.out_dir = out_dir
        self.symbol = symbol.lower()
        self.rotate_bytes = rotate_bytes
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path: Path | None = None
        self._handle: gzip.GzipFile | None = None
        self._bytes_written = 0
        self._file_sequence = 0
        self._rotate()

    def _rotate(self) -> None:
        """Close the current member and open a new one.

        Rotation is by size rather than by time so that file count and file
        size stay predictable whatever the market is doing. A gzip stream is
        not seekable, so bounded members also bound the cost of reading any
        one slice of the day, and bound how much a truncated tail can cost.

        The filename carries microseconds *and* a per-process counter because
        a second-resolution stamp is not unique: back-to-back rotations land
        in the same second, and the two files then collide. The counter rules
        that out inside one process and the microseconds rule it out across a
        restart. Opening with "x" rather than "a" makes any residual collision
        fail loudly — appending would silently concatenate two members into
        one file, which stays readable and therefore hides the fact that the
        size bound stopped being enforced. That is exactly the kind of quiet
        wrongness this project refuses.
        """
        self.close()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.path = self.out_dir / f"{self.symbol}_{stamp}_{self._file_sequence:04d}.jsonl.gz"
        self._handle = gzip.open(self.path, "xt", encoding="utf-8")
        self._bytes_written = 0
        self._file_sequence += 1
        LOGGER.info("archive: writing to %s", self.path)

    def write(self, raw_message: str, source: str) -> None:
        """Timestamp and append one record. Timestamps are taken here, late.

        Taking them at write time rather than accepting them from the caller
        keeps the two clocks read at one place in the code, so every record on
        the tape is stamped the same way.
        """
        if self._handle is None:
            raise RuntimeError("archive is closed")
        record = {
            "recv_ns": time.time_ns(),
            "recv_mono_ns": time.perf_counter_ns(),
            "source": source,
            "raw": raw_message,
        }
        line = json.dumps(record, separators=(",", ":")) + "\n"
        self._handle.write(line)
        self._bytes_written += len(line)
        if self._bytes_written >= self.rotate_bytes:
            self._rotate()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


# --------------------------------------------------------------------- REST


def _http_get_text(url: str, timeout: float) -> str:
    """Blocking GET returning the response body. Called via `asyncio.to_thread`."""
    request = urllib.request.Request(url, headers={"User-Agent": "tick-to-signal/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


async def fetch_depth_snapshot(
    symbol: str,
    limit: int,
    timeout: float = 10.0,
) -> tuple[str, dict[str, Any]]:
    """Fetch `GET /api/v3/depth`. Returns the response body and its parse.

    The undecoded body comes back alongside the parsed dict so the caller can
    archive exactly what the venue sent, for the same reason websocket frames
    are archived verbatim: re-serialising our own parse would put this
    module's json handling into the permanent record.

    `limit=1000` costs 50 request weight against a 6000/minute budget, so the
    60-second cross-check plus occasional resyncs use well under 1% of it.
    Levels beyond the limit are unknown to us until they next change — a real
    limitation of any depth-diff reconstruction, and the reason `top_n` is
    only trusted near the touch.
    """
    url = f"{REST_DEPTH_URL}?symbol={symbol.upper()}&limit={limit}"
    body = await asyncio.to_thread(_http_get_text, url, timeout)
    return body, json.loads(body)


# ------------------------------------------------------------------ parsing


def _unwrap_combined(raw_message: str) -> tuple[str, dict[str, Any]]:
    """Split a combined-stream frame into (stream name, payload).

    Combined streams wrap every payload as `{"stream": ..., "data": ...}`;
    a single-stream connection would not. We only ever open combined streams,
    so a frame without the envelope is a protocol surprise worth failing on
    rather than guessing about.
    """
    message = json.loads(raw_message)
    if "stream" not in message or "data" not in message:
        raise ValueError(f"unexpected frame without combined-stream envelope: {raw_message[:200]}")
    return message["stream"], message["data"]


def _is_depth_event(payload: dict[str, Any]) -> bool:
    return payload.get("e") == "depthUpdate"


# ------------------------------------------------------------------ syncing


async def _synchronise_book(
    websocket: Any,
    book: OrderBook,
    config: CaptureConfig,
    stats: CaptureStats,
    archive: RawArchive,
) -> None:
    """Seed `book` using Binance's documented local-order-book procedure.

    The steps below are the venue's, numbered as they are in the docs at
    https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams

      1. Open a websocket connection.                     (done by the caller)
      2. Buffer the events. Note the `U` of the first one.
      3. Get a depth snapshot.
      4. If the snapshot's `lastUpdateId` < that first `U`, go back to step 3.
      5. Discard buffered events whose `u` <= the snapshot's `lastUpdateId`.
      6. Set the local book to the snapshot.
      7. Apply the remaining buffered events, then all subsequent ones.

    HOW THIS AVOIDS BOTH MISSING AND DOUBLE-APPLYING A DIFF:
    the snapshot is a complete state at id L, and the diff stream is a
    contiguous run of id ranges. Step 4 rules out the snapshot being *older*
    than our buffer, which would leave a hole between the end of the snapshot
    and the start of what we saw. Step 5 rules out replaying history the
    snapshot already contains — every event ending at or before L is fully
    baked into it, so applying one would re-add a level the snapshot has
    already superseded. What survives both filters is exactly the events that
    overlap or immediately follow L, and `apply_diff` then enforces
    contiguity from there on. Missing and double-applying are the two failure
    modes, and steps 4 and 5 are each aimed at precisely one of them.
    """
    buffered = await _buffer_until_first_depth_event(websocket, stats, archive)
    first_event_id = int(buffered[0]["U"])

    snapshot_body, snapshot = await _fetch_snapshot_not_older_than(config, first_event_id)
    archive.write(snapshot_body, RawArchive.SOURCE_SYNC_SNAPSHOT)
    snapshot_id = int(snapshot["lastUpdateId"])

    # Step 5.
    fresh_events = [event for event in buffered if int(event["u"]) > snapshot_id]
    _verify_snapshot_is_bracketed(fresh_events, snapshot_id)

    # Steps 6 and 7. Anything still queued in the websocket buffer is read by
    # the live loop afterwards, in order, so it needs no special handling.
    book.apply_snapshot(snapshot)
    book.assert_valid()
    for event in fresh_events:
        if book.apply_diff(event):
            stats.depth_events_applied += 1

    LOGGER.info(
        "synchronised: snapshot_id=%d buffered=%d applied=%d depth=%s",
        snapshot_id,
        len(buffered),
        len(fresh_events),
        book.depth(),
    )


async def _buffer_until_first_depth_event(
    websocket: Any,
    stats: CaptureStats,
    archive: RawArchive,
) -> list[dict[str, Any]]:
    """Step 2: read frames until at least one depth event has been buffered.

    Trade frames arriving in the meantime are archived but not buffered — they
    carry no book state, so they are irrelevant to synchronisation and must
    still reach the tape.
    """
    buffered: list[dict[str, Any]] = []
    while not buffered:
        raw_message = await websocket.recv()
        _archive_message(raw_message, stats, archive)
        _stream_name, payload = _unwrap_combined(raw_message)
        if _is_depth_event(payload):
            buffered.append(payload)
    return buffered


async def _fetch_snapshot_not_older_than(
    config: CaptureConfig,
    first_event_id: int,
) -> tuple[str, dict[str, Any]]:
    """Steps 3 and 4: fetch snapshots until one is new enough to bridge to.

    A snapshot older than our first buffered event leaves an unobservable hole
    between the two, so it is retried rather than used. In practice this
    almost never loops, because the REST snapshot is normally well ahead of a
    stream we only just opened.
    """
    while True:
        body, snapshot = await fetch_depth_snapshot(config.symbol, config.snapshot_limit)
        snapshot_id = int(snapshot["lastUpdateId"])
        if snapshot_id >= first_event_id:
            return body, snapshot
        LOGGER.warning(
            "snapshot lastUpdateId=%d predates first buffered event U=%d; refetching",
            snapshot_id,
            first_event_id,
        )
        await asyncio.sleep(1.0)


def _verify_snapshot_is_bracketed(fresh_events: list[dict[str, Any]], snapshot_id: int) -> None:
    """Check the first event we are about to apply joins onto the snapshot.

    The docs describe the expected state as "the first buffered event should
    now have lastUpdateId within its [U;u] range", i.e. U <= L <= u. That is
    the usual case but it is not the only correct one: an event with
    U == L + 1 starts exactly where the snapshot ends, which is perfectly
    contiguous and loses nothing, yet falls outside the doc's bracket. So the
    hard requirement enforced here is the weaker, sufficient one — U <= L + 1,
    no hole — and the stricter doc-shaped bracket is only logged. Failing the
    strict form is interesting; failing the weak form is fatal.

    An empty list is fine and needs no check: it means the snapshot is newer
    than everything we buffered, and since the stream is contiguous the next
    live event must start at or before L + 1, where `apply_diff` will catch it.
    """
    if not fresh_events:
        LOGGER.info("snapshot is newer than every buffered event; nothing to replay")
        return

    first_id = int(fresh_events[0]["U"])
    final_id = int(fresh_events[0]["u"])
    if first_id > snapshot_id + 1:
        raise SequenceGapError(snapshot_id + 1, first_id, final_id)
    if not first_id <= snapshot_id <= final_id:
        LOGGER.info(
            "first replayed event [%d;%d] abuts rather than brackets snapshot_id=%d (contiguous, accepted)",
            first_id,
            final_id,
            snapshot_id,
        )


# ------------------------------------------------------------- live streaming


def _archive_message(raw_message: str, stats: CaptureStats, archive: RawArchive) -> None:
    """Persist one websocket frame. Called before any interpretation."""
    archive.write(raw_message, RawArchive.SOURCE_WEBSOCKET)
    stats.messages_total += 1


def _open_tape(config: CaptureConfig) -> BinaryWriter | None:
    """Start a binary tape for one session, or None if tapes are disabled.

    One tape per websocket session rather than per size budget: a session ends
    on a gap, a disconnect, or the venue's 24-hour cutoff, so files are
    naturally bounded, and a file boundary always coincides with a resync.
    That means no tape ever spans a resync, which is what lets a reader trust
    that the events inside one file are a single contiguous run.
    """
    if not config.write_binary_tape:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = config.out_dir / f"{config.symbol.lower()}_{stamp}.tape"
    header = TapeHeader(
        symbol=config.symbol,
        price_scale=PRICE_SCALE,
        qty_scale=QTY_SCALE,
        capture_start_ns=time.time_ns(),
        snapshot_interval=config.snapshot_interval,
        depth_levels=config.depth_levels,
    )
    LOGGER.info("tape: writing to %s", path)
    return BinaryWriter(path, header)


def _write_anchor_if_due(writer: BinaryWriter, book: OrderBook, local_ts_ns: int, exchange_ts_ns: int) -> None:
    """Emit a block anchor whenever the writer's cadence calls for one.

    The snapshot is taken from the book *after* the whole current diff has
    been applied, even when the cadence falls partway through emitting that
    diff's level records. That is safe because the remaining records of the
    same diff are already reflected in the snapshot, and re-applying a level
    update is idempotent — it sets the same price to the same quantity. The
    alternative, only snapshotting on diff boundaries, would break the fixed
    block cadence that the reader's O(1) seek and zero-copy view depend on.
    """
    if not writer.needs_snapshot:
        return
    bids, asks = book.top_n(writer.header.depth_levels)
    writer.write_snapshot(local_ts_ns, exchange_ts_ns, bids, asks)


def _write_depth_diff_to_tape(writer: BinaryWriter, book: OrderBook, local_ts_ns: int, payload: dict[str, Any]) -> None:
    """Decompose one depthUpdate into one fixed-width record per changed level."""
    exchange_ts_ns = int(payload.get("E", 0)) * 1_000_000  # venue sends milliseconds
    for side, key in ((SIDE_BID, "b"), (SIDE_ASK, "a")):
        for price_string, quantity_string in payload[key]:
            _write_anchor_if_due(writer, book, local_ts_ns, exchange_ts_ns)
            writer.write_event(
                local_ts_ns,
                exchange_ts_ns,
                EVENT_DIFF,
                side,
                to_fixed(price_string, PRICE_SCALE),
                to_fixed(quantity_string, QTY_SCALE),
            )


def _write_trade_to_tape(writer: BinaryWriter, book: OrderBook, local_ts_ns: int, payload: dict[str, Any]) -> None:
    """Record one trade, tagged with the side of the book it consumed.

    Binance's `m` flag says whether the *buyer* was the market maker. If it
    was, the resting order was a bid and the trade consumed the bid side;
    otherwise the taker bought and consumed the ask. Storing the consumed side
    rather than the raw flag means the tape records what happened to the book,
    which is what a feature actually wants.
    """
    exchange_ts_ns = int(payload.get("T", payload.get("E", 0))) * 1_000_000
    side = SIDE_BID if payload.get("m") else SIDE_ASK
    _write_anchor_if_due(writer, book, local_ts_ns, exchange_ts_ns)
    writer.write_event(
        local_ts_ns,
        exchange_ts_ns,
        EVENT_TRADE,
        side,
        to_fixed(payload["p"], PRICE_SCALE),
        to_fixed(payload["q"], QTY_SCALE),
    )


async def _stream_events(
    websocket: Any,
    book: OrderBook,
    config: CaptureConfig,
    stats: CaptureStats,
    archive: RawArchive,
    writer: BinaryWriter | None,
) -> None:
    """Read frames forever: archive first, then apply depth diffs to the book.

    Archiving happens before parsing so that a message we cannot interpret is
    still on the tape — the whole point of a raw archive is that it survives
    our misunderstandings. The binary tape is written only *after* the book
    has accepted the event, so a diff rejected as stale or fatal never reaches
    it: the binary tape holds validated data by construction.
    """
    async for raw_message in websocket:
        local_ts_ns = time.time_ns()
        _archive_message(raw_message, stats, archive)
        _stream_name, payload = _unwrap_combined(raw_message)

        if _is_depth_event(payload):
            if book.apply_diff(payload):
                stats.depth_events_applied += 1
                if writer is not None:
                    _write_depth_diff_to_tape(writer, book, local_ts_ns, payload)
        elif writer is not None and payload.get("e") == "trade" and book.best_bid() is not None:
            _write_trade_to_tape(writer, book, local_ts_ns, payload)

        if config.max_messages and stats.messages_total >= config.max_messages:
            LOGGER.info("reached --max-messages=%d; stopping", config.max_messages)
            return


async def _heartbeat_loop(book: OrderBook, config: CaptureConfig, stats: CaptureStats) -> None:
    """Log one status line every `heartbeat_seconds` until cancelled."""
    while True:
        await asyncio.sleep(config.heartbeat_seconds)
        bids, asks = book.depth()
        spread = book.spread()
        mid = book.mid_price()
        LOGGER.info(
            "heartbeat msgs/s=%.1f total=%d depth=%d/%d mid=%s spread=%s "
            "resyncs=%d xcheck=%d/%d/%d uptime=%.0fs",
            stats.messages_per_second_since_last_heartbeat(),
            stats.messages_total,
            bids,
            asks,
            f"{mid:.2f}" if mid is not None else "n/a",
            f"{spread:.2f}" if spread is not None else "n/a",
            stats.resync_count,
            stats.cross_checks_passed,
            stats.cross_checks_skewed,
            stats.cross_checks_failed,
            stats.uptime_seconds(),
        )


# ----------------------------------------------------------- cross-checking


def _count_mismatched_levels(local: np.ndarray, remote: np.ndarray) -> int:
    """Number of top-of-book rows that differ between two [price, qty] ladders."""
    compared = min(len(local), len(remote))
    if compared == 0:
        return max(len(local), len(remote))
    differing = int(np.sum(np.any(local[:compared] != remote[:compared], axis=1)))
    return differing + abs(len(local) - len(remote))


async def _cross_check_loop(
    book: OrderBook,
    config: CaptureConfig,
    stats: CaptureStats,
    archive: RawArchive,
) -> None:
    """Every 60s, compare our top-N against a fresh REST snapshot.

    This is the online correctness proof: reconstruction from diffs is a
    long chain of small state mutations, and the only way to know the chain
    has not drifted is to periodically ask the exchange what the answer
    should be. The snapshot goes on the tape as well as into the comparison,
    so the proof can be re-run offline against the archive.

    Raises `CrossCheckFailure` on a hard FAIL, which tears the session down
    and is counted as a resync by the supervisor. Raising rather than
    returning is deliberate: a bare return also ends the session, but it looks
    identical to a clean shutdown, so the resync goes uncounted.
    """
    while True:
        await asyncio.sleep(config.cross_check_seconds)
        snapshot_body, snapshot = await fetch_depth_snapshot(config.symbol, config.snapshot_limit)
        archive.write(snapshot_body, RawArchive.SOURCE_CROSS_CHECK_SNAPSHOT)
        local_bids, local_asks = book.top_n(config.cross_check_levels)
        local_id = book.last_update_id

        reference = OrderBook(book.symbol)
        reference.apply_snapshot(snapshot)
        remote_bids, remote_asks = reference.top_n(config.cross_check_levels)

        mismatches = _count_mismatched_levels(local_bids, remote_bids)
        mismatches += _count_mismatched_levels(local_asks, remote_asks)
        if _record_cross_check(stats, config, mismatches, local_id, reference.last_update_id):
            raise CrossCheckFailure(
                f"local book disagrees with the exchange on {mismatches} of the top "
                f"{2 * config.cross_check_levels} levels"
            )


def _record_cross_check(
    stats: CaptureStats,
    config: CaptureConfig,
    mismatches: int,
    local_id: int,
    snapshot_id: int,
) -> bool:
    """Classify one cross-check result and log it. Returns True to force a resync.

    Three outcomes, and which one applies turns on whether the two books
    describe the same instant.

    **Equal update ids — authoritative.** The snapshot and the local book are
    the same state, so they must be identical. Any mismatch at all is proof of
    corruption and fails immediately, with no streak to accumulate. This is
    the strongest evidence the daemon can obtain and it is acted on at once.

    **Different update ids — advisory.** The book moved during the HTTP round
    trip, so some disagreement is expected and is reported as SKEW. Escalation
    needs the mismatch to be both *persistent* and *substantial*: a run of
    consecutive checks each differing by at least
    `cross_check_minimum_mismatch` levels.

    The magnitude condition was added after a 10-minute live run escalated to
    FAIL on three consecutive checks that each differed by a single level with
    the ids only ~10 updates apart — unambiguously skew. The original rule
    counted "1 level differs" and "10 levels differ" as the same evidence,
    which they are not: one differing level is the touch quantity churning,
    while a real reconstruction error puts the same wrong level in every
    comparison. Counting only substantial mismatches, and requiring more of
    them in a row, makes the statistic match the thing it is trying to detect.
    """
    if mismatches == 0:
        stats.cross_checks_passed += 1
        stats.consecutive_mismatches = 0
        LOGGER.info("cross-check PASS local_id=%d snapshot_id=%d", local_id, snapshot_id)
        return False

    if local_id == snapshot_id:
        stats.cross_checks_failed += 1
        stats.consecutive_mismatches = 0
        LOGGER.error(
            "cross-check FAIL levels_differing=%d at identical update_id=%d; "
            "same instant, so this is corruption, not skew. Forcing resync",
            mismatches,
            local_id,
        )
        return True

    if mismatches < config.cross_check_minimum_mismatch:
        stats.cross_checks_skewed += 1
        stats.consecutive_mismatches = 0
        LOGGER.info(
            "cross-check SKEW levels_differing=%d local_id=%d snapshot_id=%d (ids differ by %d; "
            "too small to be evidence)",
            mismatches,
            local_id,
            snapshot_id,
            local_id - snapshot_id,
        )
        return False

    stats.consecutive_mismatches += 1
    if stats.consecutive_mismatches < config.cross_check_failures_before_resync:
        stats.cross_checks_skewed += 1
        LOGGER.warning(
            "cross-check SKEW levels_differing=%d local_id=%d snapshot_id=%d (%d substantial in a row)",
            mismatches,
            local_id,
            snapshot_id,
            stats.consecutive_mismatches,
        )
        return False

    stats.cross_checks_failed += 1
    stats.consecutive_mismatches = 0
    LOGGER.error(
        "cross-check FAIL levels_differing=%d local_id=%d snapshot_id=%d "
        "after %d consecutive substantial mismatches; forcing resync",
        mismatches,
        local_id,
        snapshot_id,
        config.cross_check_failures_before_resync,
    )
    return True


# ----------------------------------------------------------------- lifecycle


async def _run_session(
    config: CaptureConfig,
    stats: CaptureStats,
    archive: RawArchive,
    shutdown: asyncio.Event,
) -> None:
    """One connection: sync, then run reader/heartbeat/cross-check until one ends.

    Whichever task finishes first ends the session. That is the point: the
    reader finishing means the socket closed or a gap was found, the
    cross-check finishing means it demanded a resync, and shutdown finishing
    means the user pressed Ctrl-C. All three want the same teardown.
    """
    async with connect(config.stream_url(), ping_interval=20, ping_timeout=60, max_queue=4096) as websocket:
        LOGGER.info("connected: %s", config.stream_url())
        book = OrderBook(config.symbol)
        await _synchronise_book(websocket, book, config, stats, archive)
        writer = _open_tape(config)

        tasks = [
            asyncio.create_task(_stream_events(websocket, book, config, stats, archive, writer), name="reader"),
            asyncio.create_task(_heartbeat_loop(book, config, stats), name="heartbeat"),
            asyncio.create_task(_cross_check_loop(book, config, stats, archive), name="cross-check"),
            asyncio.create_task(shutdown.wait(), name="shutdown"),
        ]
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            # Closing pads the final block, so a tape is only readable as a
            # whole number of blocks if this runs. It is in `finally` because
            # a session ends far more often through a gap or a disconnect than
            # through the happy path.
            if writer is not None:
                writer.close()
                stats.tape_events_written += writer.events_written
                stats.tape_snapshots_written += writer.snapshots_written

        # Re-raise whatever the finished task raised, so the supervisor sees
        # a SequenceGapError as a SequenceGapError and not as a silent return.
        for task in done:
            task.result()


async def run_capture(config: CaptureConfig) -> CaptureStats:
    """Supervisor: keep a session alive until interrupted, resyncing as needed."""
    stats = CaptureStats()
    archive = RawArchive(config.out_dir, config.symbol, config.rotate_bytes)
    shutdown = asyncio.Event()
    _install_shutdown_handler(shutdown)

    try:
        while not shutdown.is_set():
            try:
                await _run_session(config, stats, archive, shutdown)
                if config.max_messages and stats.messages_total >= config.max_messages:
                    break
            except SequenceGapError as error:
                stats.resync_count += 1
                LOGGER.error("resync #%d: %s", stats.resync_count, error)
            except CrossCheckFailure as error:
                stats.resync_count += 1
                LOGGER.error("resync #%d after cross-check failure: %s", stats.resync_count, error)
            except (WebSocketException, OSError, urllib.error.URLError) as error:
                stats.resync_count += 1
                LOGGER.info("resync #%d: %s (%s)", stats.resync_count, error, _EXPECTED_DISCONNECT_HINT)
            if not shutdown.is_set():
                await asyncio.sleep(config.reconnect_backoff_seconds)
    finally:
        archive.close()
        LOGGER.info(
            "shutdown: messages=%d depth_applied=%d tape_events=%d tape_snapshots=%d "
            "resyncs=%d uptime=%.0fs",
            stats.messages_total,
            stats.depth_events_applied,
            stats.tape_events_written,
            stats.tape_snapshots_written,
            stats.resync_count,
            stats.uptime_seconds(),
        )
    return stats


def _install_shutdown_handler(shutdown: asyncio.Event) -> None:
    """Make Ctrl-C set the shutdown event instead of tearing the loop down.

    `loop.add_signal_handler` is POSIX-only, so Windows falls back to
    `signal.signal`. The fallback only runs the handler when the interpreter
    next reaches a bytecode boundary in the main thread, which in practice
    means shutdown is observed on the next event — a depth frame every 100ms,
    or the heartbeat at worst.
    """
    loop = asyncio.get_running_loop()

    def request_shutdown(*_: Any) -> None:
        loop.call_soon_threadsafe(shutdown.set)

    try:
        loop.add_signal_handler(signal.SIGINT, request_shutdown)
        loop.add_signal_handler(signal.SIGTERM, request_shutdown)
    except NotImplementedError:
        signal.signal(signal.SIGINT, request_shutdown)


def _parse_args(argv: list[str] | None = None) -> CaptureConfig:
    parser = argparse.ArgumentParser(description="Capture Binance spot L2 depth and trades.")
    parser.add_argument("--symbol", default="btcusdt", help="trading pair, e.g. btcusdt")
    parser.add_argument("--out", default="data/", help="directory for rotating .jsonl.gz archives")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    parser.add_argument(
        "--max-messages",
        type=int,
        default=0,
        help="stop after N messages (0 = run until Ctrl-C); used to cut notebook samples",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=DEFAULT_SNAPSHOT_INTERVAL,
        help=(
            "tape events per block anchor. Smaller means a denser sample grid for the ML "
            "pipeline and faster seeks, at proportionally more snapshot overhead on disk"
        ),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)-7s %(message)s",
        stream=sys.stdout,
    )
    return CaptureConfig(
        symbol=args.symbol,
        out_dir=Path(args.out),
        max_messages=args.max_messages,
        snapshot_interval=args.snapshot_interval,
    )


def main(argv: list[str] | None = None) -> int:
    config = _parse_args(argv)
    try:
        asyncio.run(run_capture(config))
    except KeyboardInterrupt:
        # Only reachable if Ctrl-C lands between the signal fallback being
        # installed and the loop noticing; the archive is closed either way.
        LOGGER.info("interrupted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
