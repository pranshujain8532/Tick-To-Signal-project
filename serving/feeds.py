"""Two order-book feeds behind one shape: recorded replay, and live capture.

WHAT
    `ReplayFeed` paces a set of committed tapes at a speed multiplier and yields
    one event per tape anchor. `LiveFeed` tails the directory that a running
    `data_engine.capture` writes into and yields the same events. Both emit
    session boundaries as events in their own right.

WHY REPLAY IS THE DEFAULT AND LIVE IS THE OPTION
    Someone evaluating this project should get a working demo from
    `docker compose up` with no exchange connectivity, no API keys, no network
    at all, and the same output every time. A demo that only works when Binance
    is reachable and the machine has credentials is a demo that will be broken
    on the day it matters. Live mode exists, is real, and is opt-in.

DESIGN DECISION — a session boundary is an EVENT, not a gap to smooth over.
    `SessionBoundary` is a member of the yielded union, so a consumer has to
    handle it to compile in a typed sense and to make sense in an untyped one.
    Rejected alternative: concatenate the tapes and let the timestamps jump.

    The reason is Stage 5's, and it is a correctness reason rather than a
    cosmetic one. The capture daemon starts a new tape when it resyncs, and the
    9-second gap between two tapes contains price moves nobody observed. Joining
    them invents a return across the gap: `ml/eval.py:resolve_prices_per_session`
    exists precisely to stop that from reaching a PnL number, and a serving feed
    that quietly re-joined what the research code carefully split would put the
    error back in a place with less scrutiny.

    It also has a live consequence the dashboard needs: the rolling z-score in
    `ml/features.py` needs 500 rows of history and the model window needs 100
    more, so a boundary means 599 anchors with no signal at all. That blackout
    is real and must be shown, not hidden behind a stale last value.

DESIGN DECISION — the tapes are copied into memory at construction.
    The whole committed replay set is about 4 MB. Holding it makes `seek` exact
    and O(1) and keeps file I/O out of the pacing loop. Rejected alternative:
    keep the `TapeReader` mmap views, which is zero-copy but leaves three
    mappings open for the life of the process — `data_engine/replay.py:close`
    explains why a view outlives its reader. For 4 MB the copy is the simpler
    contract.

DESIGN DECISION — the feed never drops, the transport does.
    Every anchor in the tape is yielded, whatever the speed multiplier, so the
    event sequence is a pure function of (tapes, seek position) and a test can
    assert two runs are identical. Dropping is a property of one slow websocket
    client and belongs there — see `serving/api.py:ClientChannel`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Iterable, Protocol, Union

import numpy as np

from data_engine.replay import TapeReader

LOGGER = logging.getLogger("serving.feeds")

# Speeds the control channel accepts. "max" replays with no pacing at all, which
# is what a test wants and what a recruiter never does.
ALLOWED_SPEEDS = (1.0, 10.0, 100.0)


@dataclass(frozen=True)
class BookAnchor:
    """One tape anchor: a full top-of-book ladder at a known time.

    Prices and quantities stay in the tape's fixed-point integers. Converting to
    float is presentation, and `serving/api.py` does it once, at the edge.
    """

    local_ts_ns: int
    session_id: str
    sequence: int
    bids: np.ndarray  # [depth, 2] int64: price, quantity
    asks: np.ndarray


@dataclass(frozen=True)
class SessionBoundary:
    """A discontinuity the consumer must not replay across.

    `gap_ns` is the unobserved market time between the previous session's last
    anchor and this one's first — 0 for the first session of a run, and for a
    loop restart, where the discontinuity is an artefact of looping rather than
    a real resync. `reason` distinguishes the two.
    """

    session_id: str
    previous_session_id: str | None
    gap_ns: int
    reason: str  # "start" | "resync" | "loop"


FeedEvent = Union[BookAnchor, SessionBoundary]


class Feed(Protocol):
    """What `serving/api.py` needs from a feed.

    A Protocol rather than an abstract base class: there are now two
    implementations, so the constitution permits an interface, but neither
    implementation shares any code with the other and inheritance would buy
    nothing but a base class to keep in sync.
    """

    def describe(self) -> dict[str, object]: ...

    def events(self) -> AsyncIterator[FeedEvent]: ...


@dataclass(frozen=True)
class _Session:
    """One tape, loaded and measured."""

    name: str
    snapshots: np.ndarray
    timestamps: np.ndarray
    price_scale: int
    qty_scale: int
    symbol: str

    @property
    def extent_ns(self) -> int:
        return int(self.timestamps[-1] - self.timestamps[0])


def load_sessions(tape_paths: Iterable[Path | str]) -> list[_Session]:
    """Read each tape into memory, in the order given.

    Order is the caller's, and for the committed replay set it is the capture
    order, which is what makes the gaps between them real session boundaries
    rather than an arbitrary shuffle.
    """
    sessions: list[_Session] = []
    for path in tape_paths:
        with TapeReader(path) as reader:
            snapshots = np.array(reader.load_snapshots())  # copy; see module docstring
            sessions.append(
                _Session(
                    name=Path(path).stem,
                    snapshots=snapshots,
                    timestamps=snapshots["local_ts_ns"].astype(np.int64),
                    price_scale=reader.header.price_scale,
                    qty_scale=reader.header.qty_scale,
                    symbol=reader.header.symbol,
                )
            )
    if not sessions:
        raise ValueError("a feed needs at least one tape")
    return sessions


def _tick_from_snapshots(snapshots: np.ndarray) -> int:
    """Smallest positive gap between adjacent price levels, in fixed-point units."""
    smallest = 0
    for side in ("bids", "asks"):
        prices = snapshots[side][:, :, 0].astype(np.int64)
        gaps = np.abs(np.diff(prices, axis=1))
        positive = gaps[gaps > 0]
        if positive.size:
            candidate = int(positive.min())
            smallest = candidate if smallest == 0 else min(smallest, candidate)
    return smallest


def measure_tick_size(sessions: list[_Session]) -> int:
    """The venue's price increment, in fixed-point units, measured not assumed.

    Taken as the smallest positive gap between adjacent levels on either side,
    across every snapshot. On a book this dense that gap is the tick, and
    measuring it means `spread_ticks` cannot silently be wrong because someone
    hardcoded BTCUSDT's 0.01 and then pointed the demo at another symbol. The
    value is reported by `/meta` so it can be checked rather than trusted.
    """
    smallest = 0
    for session in sessions:
        candidate = _tick_from_snapshots(session.snapshots)
        if candidate:
            smallest = candidate if smallest == 0 else min(smallest, candidate)
    if smallest == 0:
        raise ValueError("could not measure a tick size: every book had one level or fewer")
    return smallest


class ReplayFeed:
    """Paced replay of committed tapes, looping, seekable, speed-controlled."""

    def __init__(
        self,
        tape_paths: Iterable[Path | str],
        speed: float | str = 10.0,
        loop: bool = True,
    ) -> None:
        self.sessions = load_sessions(tape_paths)
        self.tick_size = measure_tick_size(self.sessions)
        # Promoted to attributes so a frame builder can read them off any feed
        # without knowing which kind it has. LiveFeed fills the same three in
        # prime(); they are the only scale information a frame needs.
        self.price_scale = self.sessions[0].price_scale
        self.qty_scale = self.sessions[0].qty_scale
        self.symbol = self.sessions[0].symbol
        self.loop = loop
        self._speed: float | str = speed
        self._paused = False
        self._resume = asyncio.Event()
        self._resume.set()
        # Resolved by seek() into (session index, anchor index) and consumed by
        # events(), which is the only place allowed to change position - so a
        # seek can never land mid-frame.
        self._seek_target: tuple[int, int] | None = None
        # Bumped on every loop wrap AND every seek, and part of the session id.
        # That is what makes "session_id changed" mean "throw away all history",
        # uniformly, whatever caused the discontinuity.
        self._epoch = 0

    # ------------------------------------------------------------- controls

    def set_speed(self, speed: float | str) -> None:
        """Accept only the speeds the UI offers, plus "max" for tests.

        Rejecting anything else is deliberate: a control channel that accepts
        arbitrary floats accepts 0.0, and a feed paced at zero is a hang that
        looks like a bug in the model.
        """
        if speed != "max" and float(speed) not in ALLOWED_SPEEDS:
            raise ValueError(f"speed must be one of {ALLOWED_SPEEDS} or 'max', got {speed!r}")
        self._speed = speed if speed == "max" else float(speed)

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        if paused:
            self._resume.clear()
        else:
            self._resume.set()

    def seek(self, fraction: float) -> None:
        """Jump to `fraction` through the total replay, 0.0 to 1.0.

        Resolved to a position immediately — `_locate` is pure — but applied by
        `events()`, which emits a boundary first. A seek always produces a
        boundary even when it lands in the same capture session, because it
        invalidates the consumer's 599 rows of normalisation history just as
        surely as a resync does.
        """
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"seek fraction must be in [0, 1], got {fraction}")
        self._seek_target = self._locate(fraction)

    # ------------------------------------------------------------ metadata

    @property
    def total_extent_ns(self) -> int:
        return sum(session.extent_ns for session in self.sessions)

    def describe(self) -> dict[str, object]:
        return {
            "mode": "replay",
            "symbol": self.sessions[0].symbol,
            "session_count": len(self.sessions),
            "sessions": [
                {
                    "name": session.name,
                    "anchors": len(session.timestamps),
                    "extent_s": session.extent_ns / 1e9,
                    "anchor_rate_hz": len(session.timestamps) / (session.extent_ns / 1e9),
                }
                for session in self.sessions
            ],
            "total_extent_s": self.total_extent_ns / 1e9,
            "tick_size_fixed": self.tick_size,
            "price_scale": self.sessions[0].price_scale,
            "qty_scale": self.sessions[0].qty_scale,
            "speed": self._speed,
            "loop": self.loop,
        }

    # ---------------------------------------------------------------- pacing

    def _locate(self, fraction: float) -> tuple[int, int]:
        """Map a fraction of total replay time onto (session index, anchor index)."""
        target_ns = fraction * self.total_extent_ns
        elapsed = 0
        for index, session in enumerate(self.sessions):
            if elapsed + session.extent_ns >= target_ns or index == len(self.sessions) - 1:
                offset = target_ns - elapsed
                anchor = int(np.searchsorted(session.timestamps - session.timestamps[0], offset))
                return index, min(anchor, len(session.timestamps) - 1)
            elapsed += session.extent_ns
        return 0, 0

    async def _wait_until(self, tape_delta_ns: int) -> None:
        """Sleep for one inter-anchor interval, scaled by the speed multiplier.

        EVERY path awaits, including the zero-delay ones, and that is load-
        bearing rather than tidy. `asyncio.sleep(0)` is the yield-to-the-loop
        idiom; without it this coroutine can run without ever suspending, and
        then the websocket sender tasks sharing the loop never get scheduled.
        The first version returned early at "max" and the server hung solid.

        It is not only "max" that hits this. The committed tapes have a MEDIAN
        inter-anchor delta of 0 ms — a 100 ms depth frame carrying dozens of
        level updates produces several anchors stamped within the same
        microsecond — so a real 10x replay walks runs of zero-length sleeps too.

        Falling behind is not corrected for. If the consumer cannot keep up at
        100x the replay simply runs as fast as it can, which is honest: a demo
        that "caught up" by skipping anchors would draw a book that never
        existed.
        """
        if self._speed == "max":
            await asyncio.sleep(0)
            return
        seconds = tape_delta_ns / 1e9 / float(self._speed)
        await asyncio.sleep(max(seconds, 0.0))

    async def events(self) -> AsyncIterator[FeedEvent]:
        """Yield boundaries and anchors forever (or once through, if not looping).

        This is the only place that changes position, which is what keeps the
        rule "every discontinuity is preceded by a SessionBoundary" true for
        resyncs, loop wraps and seeks alike.
        """
        session_index, start_anchor = 0, 0
        previous_id: str | None = None
        reason = "start"

        while True:
            session = self.sessions[session_index]
            session_id = f"{session.name}#{self._epoch}"
            yield SessionBoundary(
                session_id=session_id,
                previous_session_id=previous_id,
                gap_ns=self._gap_before(session_index, reason),
                reason=reason,
            )
            previous_id = session_id

            async for anchor in self._anchors(session, session_id, start_anchor):
                yield anchor

            if self._seek_target is not None:
                session_index, start_anchor = self._seek_target
                self._seek_target = None
                self._epoch += 1
                reason = "seek"
                continue

            session_index += 1
            start_anchor = 0
            reason = "resync"
            if session_index >= len(self.sessions):
                if not self.loop:
                    return
                session_index = 0
                self._epoch += 1
                reason = "loop"

    def _gap_before(self, session_index: int, reason: str) -> int:
        """Unobserved market time in front of this session, if it is knowable.

        Only a resync has a knowable gap: it is the wall-clock distance between
        the previous tape's last anchor and this one's first, and it is real
        market nobody recorded. A loop wrap or a seek is an artefact of the demo,
        so reporting a duration for it would invent one.
        """
        if reason != "resync" or session_index == 0:
            return 0
        previous = self.sessions[session_index - 1]
        return int(self.sessions[session_index].timestamps[0] - previous.timestamps[-1])

    async def _anchors(
        self, session: _Session, session_id: str, start: int
    ) -> AsyncIterator[BookAnchor]:
        """Yield one session's anchors, honouring pause and speed.

        Returns as soon as a seek is pending so `events()` can apply it and emit
        the boundary. It deliberately does not consume `_seek_target` itself.
        """
        index = start
        while index < len(session.timestamps):
            await self._resume.wait()
            if self._seek_target is not None:
                return

            anchor_ts = int(session.timestamps[index])
            yield BookAnchor(
                local_ts_ns=anchor_ts,
                session_id=session_id,
                sequence=index,
                bids=session.snapshots["bids"][index],
                asks=session.snapshots["asks"][index],
            )
            index += 1
            if index < len(session.timestamps):
                await self._wait_until(int(session.timestamps[index]) - anchor_ts)


class LiveFeed:
    """Tail the tape directory that a running capture daemon is writing.

    DESIGN DECISION — read the capture daemon's output rather than open a second
    exchange connection.
        Rejected alternative: have the API hold its own websocket to Binance and
        maintain its own book. That would duplicate the snapshot-synchronisation
        algorithm in `data_engine/capture.py` — the single most delicate piece of
        code in the project, with a documented resync procedure and a REST
        cross-check — into a second place where it could drift. Reading the tape
        the daemon already writes means there is exactly one implementation of
        "what the book looked like", which is the same argument
        `data_engine/replay.py` makes for having one replay path.

        It also makes the container story honest: `docker compose --profile live
        up` runs the daemon and the API as separate services sharing a volume,
        which is how this would actually be deployed.

    A NEW TAPE FILE IS A NEW SESSION, by construction: the daemon starts one
    when it resyncs, so file boundaries and session boundaries are the same
    thing and no gap detection heuristic is needed.
    """

    def __init__(self, tape_directory: Path | str, poll_seconds: float = 0.25) -> None:
        self.directory = Path(tape_directory)
        self.poll_seconds = poll_seconds
        self._tick_size = 0
        self._delivered = 0
        # Filled by prime(), which must run before the first frame is built.
        self.price_scale = 0
        self.qty_scale = 0
        self.symbol = ""
        self.tick_size = 0

    def prime(self, timeout_s: float = 60.0, poll_s: float = 0.5) -> None:
        """Block until a tape exists, then read its header for the scales.

        The API cannot build a frame without `price_scale` and `qty_scale`, and
        under `--profile live` the capture daemon and the API start together, so
        for the first few seconds the directory is legitimately empty. Waiting is
        the honest response; booting with guessed scales and correcting later
        would emit a burst of wrong prices first.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            newest = self._newest_tape()
            if newest is not None:
                with TapeReader(newest) as reader:
                    self.price_scale = reader.header.price_scale
                    self.qty_scale = reader.header.qty_scale
                    self.symbol = reader.header.symbol
                    snapshots = reader.load_snapshots()
                    if len(snapshots):
                        self._tick_size = _tick_from_snapshots(snapshots)
                if self._tick_size:
                    self.tick_size = self._tick_size
                    return
            time.sleep(poll_s)
        raise TimeoutError(
            f"no tape with a readable book appeared under {self.directory} within "
            f"{timeout_s:.0f}s. Is the capture service running?"
        )

    def describe(self) -> dict[str, object]:
        tapes = sorted(self.directory.glob("*.tape"))
        return {
            "mode": "live",
            "symbol": self.symbol,
            "tape_directory": str(self.directory),
            "session_count": len(tapes),
            # Each tape IS a session, and a live feed has no end, so there is no
            # total extent to report. None rather than 0: the value is unknown,
            # not zero, and the difference matters to whatever renders it.
            "sessions": [{"name": tape.stem} for tape in tapes],
            "total_extent_s": None,
            "newest_tape": tapes[-1].name if tapes else None,
            "tick_size_fixed": self._tick_size,
            "price_scale": self.price_scale,
            "qty_scale": self.qty_scale,
            "speed": None,  # a live feed has no speed control; wall clock is the clock
        }

    async def events(self) -> AsyncIterator[FeedEvent]:
        """Follow the newest tape, switching when the daemon starts another.

        `self._delivered` rather than a local counts how many blocks of the
        current tape have been emitted, because the draining helper is an async
        generator and cannot hand a value back through `yield from`.
        """
        current: Path | None = None
        previous_id: str | None = None
        self._delivered = 0

        while True:
            newest = self._newest_tape()
            if newest is None:
                await asyncio.sleep(self.poll_seconds)
                continue

            if newest != current:
                current = newest
                self._delivered = 0
                session_id = newest.stem
                yield SessionBoundary(
                    session_id=session_id,
                    previous_session_id=previous_id,
                    # Unknown by construction: the daemon owns the resync and
                    # this reader only sees that a new file appeared. Reporting
                    # a guess would be worse than reporting nothing.
                    gap_ns=0,
                    reason="start" if previous_id is None else "resync",
                )
                previous_id = session_id

            async for anchor in self._drain_new_blocks(current, previous_id or current.stem):
                yield anchor
            await asyncio.sleep(self.poll_seconds)

    def _newest_tape(self) -> Path | None:
        tapes = sorted(self.directory.glob("*.tape"))
        return tapes[-1] if tapes else None

    async def _drain_new_blocks(self, path: Path, session_id: str) -> AsyncIterator[BookAnchor]:
        """Yield blocks that have appeared since the last poll.

        Re-opening the tape each poll is deliberate. The writer appends and the
        reader counts only whole blocks, so a partially written block is simply
        not there yet and will be picked up next time — the format's
        truncation-tolerance (see `TapeReader._count_whole_blocks`) is what makes
        tailing a live file safe without any locking.
        """
        with TapeReader(path) as reader:
            snapshots = reader.load_snapshots()
            available = len(snapshots)
            if self._tick_size == 0 and available:
                self._tick_size = _tick_from_snapshots(snapshots)
            for index in range(self._delivered, available):
                yield BookAnchor(
                    local_ts_ns=int(snapshots[index]["local_ts_ns"]),
                    session_id=session_id,
                    sequence=index,
                    bids=np.array(snapshots[index]["bids"]),
                    asks=np.array(snapshots[index]["asks"]),
                )
            self._delivered = available


def default_replay_tapes(repo_root: Path) -> list[Path]:
    """The committed replay set, in capture order.

    Three consecutive prefixes of the three real capture sessions from
    2026-07-27, so the demo loop crosses two genuine resync boundaries rather
    than a synthetic one. See notebooks/sample_data/README.md for the extents.
    """
    directory = repo_root / "notebooks" / "sample_data"
    tapes = sorted(directory.glob("btcusdt_replay_s*.tape"))
    if not tapes:
        raise FileNotFoundError(
            f"no btcusdt_replay_s*.tape under {directory}. These are committed; "
            "a checkout missing them is incomplete."
        )
    return tapes
