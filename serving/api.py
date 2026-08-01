"""FastAPI service: a push websocket for the dashboard, and the evidence behind it.

WHAT
    `/ws/stream` pushes one frame per tape anchor to every connected client,
    dropping stale frames rather than queueing them. Seven REST endpoints serve
    the project's measured results, every value read from a file under
    `benchmarks/` at request time.

WHY PUSH AND NOT POLL
    The book updates about 23-32 times a second in the committed replay, and the
    dashboard renders at 60 fps. A polling client has to guess that rate: too
    slow and it misses book states, too fast and it burns a round trip to be told
    nothing changed. Worse, polling puts the client in charge of freshness, so a
    client that stalls for 300 ms comes back and asks for "now" — and gets it,
    silently skipping everything between, with no way for anyone to know it
    happened. A push socket makes the server the authority on what changed, and
    makes the drops explicit and countable.

WHY DROP AND NEVER QUEUE — the single most important line in this file.
    `ClientChannel` holds exactly ONE pending frame. If a second arrives before
    the first has been written to the socket, the FIRST IS DESTROYED and counted.

    For a live market view a stale frame is not merely less useful than a fresh
    one, it is actively wrong: it draws a book that no longer exists, next to a
    signal about a price that has already moved. Delivering it late is worse than
    never delivering it, because the client cannot tell the difference.

    And an unbounded queue is how dashboards die. A client on a slow link, a
    laptop that sleeps, a browser tab throttled in the background — any of them
    turns a queue into a memory leak that grows at the anchor rate and delivers,
    on recovery, a burst of history the user did not ask for. Bounding the queue
    at one frame makes the failure mode "you missed some frames", which is
    survivable and visible, instead of "the server fell over", which is not.

    `dropped_frames` is per connection and is reported in `/health`. It is
    expected to be non-zero at 10x and above; a zero there would mean the client
    is keeping up with 300 frames a second, which no renderer needs to.

WHAT THE LATENCY NUMBERS HERE MEAN — see also serving/signal.py.
    `serving_infer_us` in each frame is the ONNX int8 forward pass in THIS
    process, and nothing else. It excludes feature construction, which is
    measured separately and reported by `/latency` — and which is not small: it
    was measured at roughly 60% of the forward pass on this machine.

    Stage 7b's C++ figure is a different instrument measuring a different thing:
    a hand-written forward pass from an already-prepared [40] feature column,
    pinned and warmed over a million iterations. `/pareto` carries both, each
    tagged with `measured_by`, and `/latency` carries the relevance note that
    CLAUDE.md requires beside any latency claim. Nothing in this file reports the
    C++ figure as the serving figure.

DESIGN DECISION — one producer task, many consumers.
    The feed is drained by a single background task which runs the model once
    per anchor and offers the resulting frame to every connected channel.
    Rejected alternative: one feed and one model per connection, which is simpler
    to write and would run N copies of a 1.8 ms pipeline for N viewers, making
    the reported latency a function of how many people are watching.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import platform
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from serving import records
from serving.feeds import (
    BookAnchor,
    LiveFeed,
    ReplayFeed,
    SessionBoundary,
    default_replay_tapes,
)
from serving.signal import HISTORY_ROWS, SignalEngine

LOGGER = logging.getLogger("serving.api")
REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = Path(__file__).resolve().parent / "dashboard"
DEFAULT_MODEL = REPO_ROOT / "notebooks" / "sample_data" / "student_int8.onnx"

# Rolling window for the serving latency percentiles. 20,000 samples at ~25
# anchors/s is about thirteen minutes of history at 1x, which is long enough for
# p99.9 to rest on twenty observations rather than one.
LATENCY_WINDOW = 20_000


class ClientChannel:
    """A one-frame mailbox for a single websocket. Newest wins, always.

    `offer` never blocks and never grows: it either fills an empty slot or
    overwrites a full one, counting the frame it destroyed.

    WHY ONE MAILBOX IS NOT ENOUGH ON ITS OWN — measured, not theorised.
        Bounding this queue at one frame does not bound what the CLIENT sees,
        because `websocket.send_text()` returns when the frame is handed to the
        transport, not when it is delivered. Under uvicorn there is a write
        buffer, and under that a socket buffer.

        Measured against the running container: a client reading one frame every
        250 ms, while the server produced ~244 frames/s, received frames with
        strictly consecutive sequence numbers — +1 per read — and the server had
        accepted 239 sends for 31 actual receives. Roughly 208 stale frames were
        sitting below this class, being delivered in order, minutes behind the
        market. That is precisely the failure this policy exists to prevent, and
        the mailbox alone did not prevent it.

    CREDIT MODE is the fix, and it is opt-in via `/ws/stream?flow=ack`.
        The server sends at most `credit` frames that the client has not yet
        acknowledged with `{"cmd": "ack"}`. With a credit of 1 there can be no
        buffer below this class, because a second frame is never handed to the
        transport until the client says it consumed the first. The dashboard
        acks once per rendered frame, which makes the flow render-driven.

        The default remains fire-and-forget so that a plain client — `wscat`,
        a browser console, anything exploratory — still works without
        implementing a protocol. That mode is honest about what it guarantees:
        the server-side queue is bounded, the transport's is not.
    """

    EVENT_CAPACITY = 32

    def __init__(self, credit: int | None = None) -> None:
        self._pending: str | None = None
        # Events are NOT state and must not be dropped. See `post`.
        self._events: deque[str] = deque()
        self._ready = asyncio.Event()
        # None means fire-and-forget. An integer is the number of frames that
        # may be in flight before the client must acknowledge.
        self._credit = credit
        self._has_credit = asyncio.Event()
        if credit is None or credit > 0:
            self._has_credit.set()
        self.dropped_frames = 0
        self.sent_frames = 0
        self.dropped_events = 0

    @property
    def flow_controlled(self) -> bool:
        return self._credit is not None

    def offer(self, frame: str) -> None:
        if self._pending is not None:
            # The client has not drained the previous frame. It is now stale, so
            # it dies here rather than being delivered late behind this one.
            self.dropped_frames += 1
        self._pending = frame
        self._ready.set()

    def post(self, event: str) -> None:
        """Queue an EVENT, which is never dropped in favour of a newer message.

        WHY THIS EXISTS — a real defect, found by the Stage 8b dashboard.
            Session boundaries were originally broadcast through `offer`, which
            meant they lived in the same one-frame slot as book state. The
            producer emits ~250 frames a second and a render-driven client takes
            60, so the anchor immediately following a boundary overwrote it
            essentially every time. Measured against this server: a client in
            credit mode received 1,126 frames and ZERO session boundaries in 30
            seconds, across roughly five real boundaries.

            The consequence was not a missing notification. The client never
            learned that its history had been invalidated, so it drew one
            continuous tape across a 422-second hole in the data — which is
            precisely the fabrication `serving/feeds.py` emits boundaries as
            events to prevent, undone one layer further out.

        THE DISTINCTION THAT WAS MISSING: state versus event.
            A book frame is STATE. A newer one makes an older one worthless, so
            dropping it is correct and the whole design rests on that. A session
            boundary is an EVENT: it is not superseded by anything, it says that
            everything before it must be discarded, and delivering it late is
            still useful while not delivering it at all is corrupting. So state
            gets a one-slot mailbox and events get a queue.

        The queue is bounded like everything else here, and an overflow is
        counted and reported by /health rather than silently discarded.
        """
        if len(self._events) >= self.EVENT_CAPACITY:
            self.dropped_events += 1
            return
        self._events.append(event)
        # Whatever frame was waiting predates this discontinuity, so it is stale
        # by definition and dies here rather than being delivered after the
        # boundary that invalidates it.
        if self._pending is not None:
            self.dropped_frames += 1
            self._pending = None
        self._ready.set()

    def grant(self, amount: int = 1) -> None:
        """The client confirmed it consumed a frame; allow another in flight."""
        if self._credit is None:
            return
        self._credit += amount
        if self._credit > 0:
            self._has_credit.set()

    async def take(self) -> str:
        # Credit first, then a message: waiting for credit before choosing what
        # to send is what makes the client receive the NEWEST frame at the
        # moment it becomes able to accept one, rather than the one that was
        # newest when it fell behind.
        #
        # Credit applies to events as well as frames, so a flow-controlled
        # client must acknowledge everything it receives on this channel, not
        # only the frames. Exempting events would be a second rule to get
        # wrong, and the client is ours.
        await self._has_credit.wait()
        await self._ready.wait()
        # Events first and in order: a boundary must reach the client before any
        # frame of the session it introduces.
        if self._events:
            message = self._events.popleft()
        else:
            message = self._pending
            self._pending = None
        if not self._events and self._pending is None:
            self._ready.clear()
        self.sent_frames += 1
        if self._credit is not None:
            self._credit -= 1
            if self._credit <= 0:
                self._has_credit.clear()
        assert message is not None  # only set together with _ready
        return message


@dataclass
class ServingState:
    """Everything the process holds between requests."""

    feed: ReplayFeed
    engine: SignalEngine
    mode: str
    channels: set[ClientChannel] = field(default_factory=set)
    infer_ns: deque = field(default_factory=lambda: deque(maxlen=LATENCY_WINDOW))
    feature_ns: deque = field(default_factory=lambda: deque(maxlen=LATENCY_WINDOW))
    frames_produced: int = 0
    boundaries_seen: int = 0
    # Cumulative across every connection ever opened. Per-channel counters die
    # with the socket, and "how many frames did this server drop today" is
    # exactly the question a dashboard's own health panel has to answer.
    dropped_frames_total: int = 0
    sent_frames_total: int = 0
    dropped_events_total: int = 0
    current_session: str | None = None
    started_at: float = field(default_factory=time.time)
    producer: asyncio.Task | None = None


STATE: ServingState | None = None


def state() -> ServingState:
    if STATE is None:
        raise RuntimeError("the serving state is built during startup; this ran too early")
    return STATE


# --------------------------------------------------------------- frame building


def _ladder(levels: np.ndarray, price_scale: int, qty_scale: int) -> list[list[float]]:
    """One side of the book as `[[price, quantity], ...]` in real units.

    A level whose price is zero is the writer's padding for a side shorter than
    `depth_levels`, not a real order at price zero, so it is dropped rather than
    drawn. `ml/features.py:count_absent_levels` exists because this is rare; the
    dashboard still must not render it.
    """
    rows: list[list[float]] = []
    for price, quantity in levels:
        if price > 0:
            rows.append([float(price) / price_scale, float(quantity) / qty_scale])
    return rows


def build_frame(anchor: BookAnchor, result: Any, feed: ReplayFeed) -> dict[str, Any]:
    """One websocket frame. `signal` is null until the engine is warm.

    A null signal is not an error state and is not hidden: after every session
    boundary the normalisation history is destroyed on purpose, and 599 anchors
    must pass before an input can be built. `warmup` carries the countdown so the
    dashboard can say why it is showing nothing, rather than freezing on a stale
    last value — which would be the exact dishonesty this project exists to
    avoid.
    """
    price_scale, qty_scale = feed.price_scale, feed.qty_scale
    best_bid = float(anchor.bids[0][0]) / price_scale
    best_ask = float(anchor.asks[0][0]) / price_scale
    tick = feed.tick_size / price_scale

    frame: dict[str, Any] = {
        "t_ns": anchor.local_ts_ns,
        "seq": anchor.sequence,
        "session_id": anchor.session_id,
        "bids": _ladder(anchor.bids, price_scale, qty_scale),
        "asks": _ladder(anchor.asks, price_scale, qty_scale),
        "mid": (best_bid + best_ask) / 2.0,
        "spread_ticks": round((best_ask - best_bid) / tick, 3),
        "signal": None,
        "serving_infer_us": None,
    }
    if result.ready:
        down, flat, up = result.probabilities
        frame["signal"] = {
            "p_down": down,
            "p_flat": flat,
            "p_up": up,
            # Computed server-side so every client shows the same number, and so
            # the definition lives next to the model that produced the inputs.
            "score": result.score,
        }
        frame["serving_infer_us"] = result.infer_ns / 1000.0
    else:
        frame["warmup"] = {
            "anchors_remaining": result.anchors_until_ready,
            "anchors_required": HISTORY_ROWS,
        }
    return frame


# ------------------------------------------------------------------- producer


async def _produce(app_state: ServingState) -> None:
    """Drain the feed forever, run the model, offer frames to every client."""
    async for event in app_state.feed.events():
        if isinstance(event, SessionBoundary):
            # A boundary destroys the history rather than smoothing across it.
            app_state.engine.reset()
            app_state.boundaries_seen += 1
            app_state.current_session = event.session_id
            # Posted, not offered: a boundary is an event and must survive the
            # next anchor. See ClientChannel.post.
            _broadcast_event(app_state, {"type": "session_boundary", **_boundary_payload(event)})
            continue

        result = app_state.engine.push(event.bids, event.asks)
        if result.ready:
            app_state.infer_ns.append(result.infer_ns)
            app_state.feature_ns.append(result.feature_ns)
        frame = build_frame(event, result, app_state.feed)
        frame["type"] = "frame"
        app_state.frames_produced += 1
        _broadcast(app_state, frame)


def _boundary_payload(event: SessionBoundary) -> dict[str, Any]:
    return {
        "session_id": event.session_id,
        "previous_session_id": event.previous_session_id,
        "gap_ns": event.gap_ns,
        "reason": event.reason,
        "note": (
            "History is discarded here. Prices either side of this boundary are "
            "not comparable and no return may be computed across it."
        ),
    }


def _broadcast(app_state: ServingState, payload: dict[str, Any]) -> None:
    """Offer one serialised payload to every channel. Never blocks, never queues."""
    encoded = json.dumps(payload, separators=(",", ":"))
    for channel in app_state.channels:
        channel.offer(encoded)


def _broadcast_event(app_state: ServingState, payload: dict[str, Any]) -> None:
    """Queue one event on every channel. Never blocks; bounded, and never dropped
    in favour of a newer message."""
    encoded = json.dumps(payload, separators=(",", ":"))
    for channel in app_state.channels:
        channel.post(encoded)


# ----------------------------------------------------------------------- app


def _build_feed(
    mode: str,
    tapes: Iterable[Path] | None,
    speed: float | str,
    tape_directory: Path | str,
) -> ReplayFeed | LiveFeed:
    """Demo replays committed tapes; live tails what the capture daemon writes."""
    if mode == "demo":
        return ReplayFeed(list(tapes) if tapes else default_replay_tapes(REPO_ROOT), speed=speed)
    if mode == "live":
        feed = LiveFeed(tape_directory)
        # Blocks until the capture daemon has written a readable book, because a
        # frame cannot be built without the tape header's scales.
        feed.prime()
        return feed
    raise ValueError(f"TTS_MODE must be 'demo' or 'live', got {mode!r}")


def create_app(
    tapes: Iterable[Path] | None = None,
    model_path: Path | str = DEFAULT_MODEL,
    speed: float | str = 10.0,
    mode: str = "demo",
    tape_directory: Path | str = "/data",
) -> FastAPI:
    """Build the application. Demo mode by default, with no network needed."""
    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        """Load the model and start the producer before the first request.

        The model is loaded exactly once, here, and warmed with a throwaway
        forward pass — the first call through any runtime is an outlier, and one
        outlier at second zero of a demo lands in the first histogram a viewer
        sees. The clock's real resolution is measured and logged before anything
        depends on it, for the reason Stage 7b documents.
        """
        global STATE
        feed = _build_feed(mode, tapes, speed, tape_directory)
        engine = SignalEngine(model_path, feed.price_scale, feed.qty_scale)
        LOGGER.info(
            "mode=%s clock resolution %d ns, read-pair overhead %d ns; "
            "%d anchors of warmup per session",
            mode,
            engine.clock.resolution_ns,
            engine.clock.overhead_ns,
            HISTORY_ROWS,
        )
        STATE = ServingState(feed=feed, engine=engine, mode=mode)
        STATE.producer = asyncio.create_task(_produce(STATE))
        try:
            yield
        finally:
            STATE.producer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await STATE.producer

    application = FastAPI(title="tick-to-signal serving", version="8a", lifespan=lifespan)
    _register_routes(application)
    if DASHBOARD_DIR.is_dir():
        application.mount("/", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")
    return application


def _register_routes(application: FastAPI) -> None:
    @application.websocket("/ws/stream")
    async def stream(websocket: WebSocket) -> None:
        await websocket.accept()
        app_state = state()
        # `?flow=ack` turns on credit-based flow control; anything else is
        # fire-and-forget so an exploratory client needs no protocol.
        credit = 1 if websocket.query_params.get("flow") == "ack" else None
        channel = ClientChannel(credit=credit)
        app_state.channels.add(channel)
        sender = asyncio.create_task(_send_loop(websocket, channel))
        try:
            await _control_loop(websocket, app_state, channel)
        except WebSocketDisconnect:
            pass
        finally:
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender
            app_state.channels.discard(channel)
            # Fold this connection's counters into the running totals before the
            # channel is garbage, or every disconnect silently resets the drops.
            app_state.dropped_frames_total += channel.dropped_frames
            app_state.sent_frames_total += channel.sent_frames
            app_state.dropped_events_total += channel.dropped_events

    @application.get("/meta")
    async def meta() -> JSONResponse:
        return JSONResponse(_meta_payload())

    @application.get("/pareto")
    async def pareto() -> JSONResponse:
        return JSONResponse(records.pareto_payload())

    @application.get("/decay")
    async def decay() -> JSONResponse:
        return JSONResponse(records.decay_payload())

    @application.get("/stability")
    async def stability() -> JSONResponse:
        return JSONResponse(records.stability_payload())

    @application.get("/economics")
    async def economics() -> JSONResponse:
        return JSONResponse(records.economics_payload())

    @application.get("/latency")
    async def latency() -> JSONResponse:
        return JSONResponse(_latency_payload())

    @application.get("/health")
    async def health() -> JSONResponse:
        app_state = state()
        return JSONResponse(
            {
                "status": "ok",
                "mode": app_state.mode,
                "uptime_s": time.time() - app_state.started_at,
                "frames_produced": app_state.frames_produced,
                "boundaries_seen": app_state.boundaries_seen,
                "current_session": app_state.current_session,
                "clients": len(app_state.channels),
                # Live connections plus everything that has already hung up. A
                # non-zero drop count at 10x and above is expected and healthy:
                # it means the transport is protecting the client from a rate no
                # renderer needs. Zero would mean nobody is connected, or the
                # client is somehow consuming 300 frames a second.
                "dropped_frames": app_state.dropped_frames_total
                + sum(c.dropped_frames for c in app_state.channels),
                "sent_frames": app_state.sent_frames_total
                + sum(c.sent_frames for c in app_state.channels),
                # Boundaries that could not be queued because a client was more
                # than 32 events behind. This should be zero forever; a non-zero
                # value means some client has been shown a discontinuity it was
                # never told about, which is a correctness problem rather than a
                # performance one.
                "dropped_events": app_state.dropped_events_total
                + sum(c.dropped_events for c in app_state.channels),
                "engine_ready": app_state.engine.ready,
                "anchors_until_ready": app_state.engine.anchors_until_ready,
            }
        )


async def _send_loop(websocket: WebSocket, channel: ClientChannel) -> None:
    """Write whatever the channel currently holds, forever."""
    while True:
        frame = await channel.take()
        await websocket.send_text(frame)


async def _control_loop(
    websocket: WebSocket, app_state: ServingState, channel: ClientChannel
) -> None:
    """Apply `speed`, `seek` and `pause`, and refuse anything else by name.

    Unknown commands are rejected explicitly rather than ignored. A silently
    dropped command looks to the client exactly like a server that is broken, and
    the client has no way to tell which.
    """
    while True:
        message = await websocket.receive_text()
        try:
            command = json.loads(message)
        except json.JSONDecodeError:
            await websocket.send_text(json.dumps({"type": "error", "detail": "not valid JSON"}))
            continue
        await _apply_command(websocket, app_state, channel, command)


async def _apply_command(
    websocket: WebSocket,
    app_state: ServingState,
    channel: ClientChannel,
    command: dict[str, Any],
) -> None:
    name = command.get("cmd")
    if name == "ack":
        # Deliberately silent and handled before everything else: an ack arrives
        # once per rendered frame, so replying to it would double the traffic it
        # exists to regulate. It is also valid in live mode, unlike the others.
        channel.grant()
        return
    if not isinstance(app_state.feed, ReplayFeed):
        # A live feed has no speed, no seek and no pause: the wall clock is the
        # clock and the exchange is not going to wait. Refusing by name beats
        # accepting the command and doing nothing.
        await websocket.send_text(
            json.dumps(
                {
                    "type": "error",
                    "detail": f"{name!r} is not available in live mode; "
                    "speed, seek and pause only exist for a replay feed",
                }
            )
        )
        return
    try:
        if name == "speed":
            app_state.feed.set_speed(command["value"])
        elif name == "seek":
            app_state.feed.seek(float(command["value"]))
        elif name == "pause":
            app_state.feed.set_paused(bool(command["value"]))
        else:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "detail": f"unknown command {name!r}; expected speed, seek, pause or ack",
                    }
                )
            )
            return
    except (KeyError, ValueError) as error:
        await websocket.send_text(json.dumps({"type": "error", "detail": str(error)}))
        return
    await websocket.send_text(json.dumps({"type": "ack", "cmd": name}))


# ------------------------------------------------------------------ payloads


def _meta_payload() -> dict[str, Any]:
    app_state = state()
    described = app_state.feed.describe()
    return {
        "symbol": described["symbol"],
        "serving_variant": (
            "ONNX int8 distilled student, 32,155 params, 126 KiB"
        ),
        "serving_variant_key": records.SERVING_VARIANT,
        "mode": app_state.mode,
        "is_demo": app_state.mode == "demo",
        "tape_source": [entry["name"] for entry in described["sessions"]],
        "session_count": described["session_count"],
        "total_extent_s": described["total_extent_s"],
        "sessions": described["sessions"],
        "tick_size": described["tick_size_fixed"] / described["price_scale"],
        "speed": described["speed"],
        "engine": app_state.engine.describe(),
        "build": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "onnxruntime": _onnxruntime_version(),
        },
    }


def _onnxruntime_version() -> str:
    import onnxruntime

    return onnxruntime.__version__


def _percentiles(samples: Iterable[int]) -> dict[str, float] | None:
    """Rolling percentiles as plain Python floats.

    Every value is cast explicitly. `np.percentile` returns `numpy.float64`,
    which the stdlib JSON encoder refuses, and the failure surfaces as a 500 from
    the endpoint rather than anywhere near the arithmetic that caused it.
    """
    values = np.asarray(list(samples), dtype=np.float64)
    if values.size == 0:
        return None
    p50, p99, p999 = np.percentile(values, [50, 99, 99.9])
    return {
        "p50_us": float(p50) / 1000.0,
        "p99_us": float(p99) / 1000.0,
        "p99_9_us": float(p999) / 1000.0,
        "min_us": float(values.min()) / 1000.0,
        "max_us": float(values.max()) / 1000.0,
        "samples": int(values.size),
    }


def _histogram(samples: Iterable[int], bins: int = 64) -> dict[str, Any] | None:
    """Bins spanning min..p99.9, with the overflow counted rather than dropped."""
    values = np.asarray(list(samples), dtype=np.float64)
    if values.size == 0:
        return None
    upper = float(np.percentile(values, 99.9))
    low = float(values.min())
    if upper <= low:
        upper = low + 1.0
    counts, edges = np.histogram(values, bins=bins, range=(low, upper))
    return {
        "edges_us": (edges / 1000.0).tolist(),
        "counts": counts.tolist(),
        "overflow": int((values > upper).sum()),
    }


def _latency_payload() -> dict[str, Any]:
    """Rolling serving latency, with the boundary and the caveat attached.

    If the clock cannot resolve the measurement, this says so and withholds the
    percentiles instead of publishing a grid of clock ticks. Stage 7b published
    that grid once before its own resolution check caught it.
    """
    app_state = state()
    inference = _percentiles(app_state.infer_ns)
    resolvable = inference is not None and app_state.engine.clock.can_resolve(
        inference["p50_us"] * 1000.0
    )
    return {
        "resolvable": resolvable,
        "clock": app_state.engine.clock.describe(),
        "inference": inference if resolvable else None,
        "inference_histogram": _histogram(app_state.infer_ns) if resolvable else None,
        "feature_construction": _percentiles(app_state.feature_ns),
        "iterations": len(app_state.infer_ns),
        "window": LATENCY_WINDOW,
        "measures": (
            "The ONNX int8 forward pass in this process, for one [1, 100, 40] "
            "input. Feature construction is reported separately and is NOT "
            "included; it is not small. Excluded entirely: websocket framing, "
            "JSON serialisation, book maintenance and network time."
        ),
        "not_the_cpp_figure": (
            "Stage 7b's ~11 us is a different program measured by a different "
            "harness: a hand-written C++ forward pass from an already-prepared "
            "[40] feature column, pinned to one core over 1,000,000 iterations. "
            "It is not what this server runs. See /pareto for both, tagged."
        ),
        "relevance_note": records.RELEVANCE_NOTE,
        "unresolvable_note": (
            None
            if resolvable
            else "The clock cannot resolve this duration to better than 1%, so "
            "percentiles are withheld rather than published as a grid."
        ),
    }


def app_from_environment() -> FastAPI:
    """The application docker-compose starts, configured by environment.

    Environment rather than CLI flags because the container's CMD is fixed and
    compose sets the environment — and because a misspelled variable falls back
    to a working demo instead of failing to boot.
    """
    mode = os.environ.get("TTS_MODE", "demo")
    raw_speed = os.environ.get("TTS_SPEED", "10")
    speed: float | str = "max" if raw_speed == "max" else float(raw_speed)
    return create_app(
        speed=speed,
        mode=mode,
        tape_directory=os.environ.get("TTS_TAPE_DIR", "/data"),
        model_path=os.environ.get("TTS_MODEL", str(DEFAULT_MODEL)),
    )


app = app_from_environment()
