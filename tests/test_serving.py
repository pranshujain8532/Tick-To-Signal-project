"""Tests for the serving layer.

Five things are checked, and each of them is a way this stage could lie:

  * the frame schema, because the dashboard is written against it;
  * backpressure, because "drop the older frame" is a claim about behaviour
    under load and claims about behaviour under load are the ones that turn out
    to be wrong;
  * replay determinism, because a demo that differs run to run cannot be
    debugged and cannot be trusted;
  * session boundaries, because smoothing over a resync is the specific error
    Stage 5 built machinery to prevent;
  * and that every served number matches the file it claims to come from,
    because a serving layer that drifts from its own evidence is worse than one
    with no numbers at all.

WHY THESE MOSTLY DO NOT SPEAK HTTP
    `ClientChannel`, `ReplayFeed` and `serving/records.py` are plain importable
    objects, so testing them directly is both faster and more precise than
    driving them through a socket — a failure points at the policy rather than
    at the transport. One end-to-end test exercises the real ASGI app so the
    wiring is covered too.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from serving import records
from serving.feeds import (
    BookAnchor,
    ReplayFeed,
    SessionBoundary,
    default_replay_tapes,
    measure_tick_size,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL = REPO_ROOT / "notebooks" / "sample_data" / "student_int8.onnx"


@pytest.fixture(scope="module")
def tapes() -> list[Path]:
    return default_replay_tapes(REPO_ROOT)


def collect(feed: ReplayFeed, limit: int) -> list:
    """Drain `limit` events from a feed, synchronously, for a test."""

    async def drain() -> list:
        events = []
        async for event in feed.events():
            events.append(event)
            if len(events) >= limit:
                break
        return events

    return asyncio.run(drain())


# ------------------------------------------------------------ the replay feed


def test_the_committed_replay_set_spans_several_capture_sessions(tapes):
    """A single tape cannot demonstrate a boundary, so the demo needs several.

    The capture daemon starts a new tape when it resyncs, so session boundaries
    are between files. A one-tape demo would have nothing to show.
    """
    assert len(tapes) >= 2, "the replay demo needs at least two capture sessions"


def test_replay_is_deterministic(tapes):
    """Two runs of the same feed must produce an identical event sequence.

    Pacing uses wall-clock sleeps, so the *timing* varies between runs. The
    content must not: every field comes from the tape, and a demo whose output
    depends on how busy the machine was is a demo nobody can debug.
    """
    first = collect(ReplayFeed(tapes, speed="max", loop=False), 800)
    second = collect(ReplayFeed(tapes, speed="max", loop=False), 800)

    assert len(first) == len(second)
    for left, right in zip(first, second):
        assert type(left) is type(right)
        if isinstance(left, BookAnchor):
            assert (left.local_ts_ns, left.session_id, left.sequence) == (
                right.local_ts_ns,
                right.session_id,
                right.sequence,
            )
            assert np.array_equal(left.bids, right.bids)
            assert np.array_equal(left.asks, right.asks)
        else:
            assert (left.session_id, left.reason, left.gap_ns) == (
                right.session_id,
                right.reason,
                right.gap_ns,
            )


def test_every_session_is_introduced_by_a_boundary(tapes):
    """No anchor may arrive under a session id that was never announced.

    This is the invariant the consumer relies on to know when to throw away its
    normalisation history. If an anchor could appear under a new id with no
    boundary in front of it, `SignalEngine` would normalise one session's book
    against another's statistics.
    """
    events = collect(ReplayFeed(tapes, speed="max", loop=True), 6000)
    announced: set[str] = set()

    for event in events:
        if isinstance(event, SessionBoundary):
            announced.add(event.session_id)
        else:
            assert event.session_id in announced, (
                f"anchor arrived under unannounced session {event.session_id!r}"
            )


def test_a_resync_boundary_reports_the_unobserved_gap(tapes):
    """The gap between two capture sessions is real market nobody recorded.

    Reporting it is what lets a consumer refuse to compute a return across it.
    Loop wraps and seeks report zero instead, because their discontinuity is an
    artefact of the demo and inventing a duration for it would be a fabrication.
    """
    events = collect(ReplayFeed(tapes, speed="max", loop=True), 6000)
    boundaries = [event for event in events if isinstance(event, SessionBoundary)]

    resyncs = [boundary for boundary in boundaries if boundary.reason == "resync"]
    assert resyncs, "the committed set should contain at least one resync boundary"
    for boundary in resyncs:
        assert boundary.gap_ns > 0
    for boundary in boundaries:
        if boundary.reason in ("loop", "start"):
            assert boundary.gap_ns == 0


def test_looping_changes_the_session_id(tapes):
    """A loop wrap is a discontinuity, so it must not reuse the previous id.

    "session_id changed" is the consumer's single signal to reset. If wrapping
    reused an id, the dashboard would draw a price jump inside one continuous
    series.
    """
    events = collect(ReplayFeed(tapes, speed="max", loop=True), 6000)
    loops = [e for e in events if isinstance(e, SessionBoundary) and e.reason == "loop"]

    assert loops, "6000 events should be enough to wrap the committed set"
    assert loops[0].session_id != loops[0].previous_session_id


def test_seek_emits_a_boundary_because_it_invalidates_history(tapes):
    """Seeking is as destructive to the rolling z-score as a resync is."""
    feed = ReplayFeed(tapes, speed="max", loop=False)

    async def drain() -> list:
        events = []
        async for event in feed.events():
            events.append(event)
            if len(events) == 50:
                feed.seek(0.9)
            if len(events) >= 400:
                break
        return events

    events = asyncio.run(drain())
    reasons = [e.reason for e in events if isinstance(e, SessionBoundary)]
    assert "seek" in reasons


def test_unknown_speeds_are_refused(tapes):
    """A control channel that accepts 0.0 accepts a hang that looks like a bug."""
    feed = ReplayFeed(tapes, speed="max", loop=False)
    for bad in (0.0, -1.0, 7.5, 1000.0):
        with pytest.raises(ValueError):
            feed.set_speed(bad)
    feed.set_speed(10.0)
    feed.set_speed("max")


def test_seek_fraction_is_bounded(tapes):
    feed = ReplayFeed(tapes, speed="max", loop=False)
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            feed.seek(bad)


def test_tick_size_is_measured_not_assumed(tapes):
    """BTCUSDT's tick is 0.01, and the code must derive that rather than know it."""
    feed = ReplayFeed(tapes, speed="max", loop=False)
    assert measure_tick_size(feed.sessions) == feed.tick_size
    assert feed.tick_size / feed.sessions[0].price_scale == pytest.approx(0.01)


# ------------------------------------------------------------- backpressure


def test_a_slow_consumer_drops_the_older_frame():
    """THE backpressure claim: newest wins, and the loser is counted.

    A stale book beside a moved price is not a degraded view, it is a wrong one.
    This asserts the older frame is destroyed rather than delivered late, and
    that the count of destroyed frames is exact — because `dropped_frames` is
    reported by /health and a wrong count would be its own small lie.
    """
    from serving.api import ClientChannel

    channel = ClientChannel()
    for index in range(5):
        channel.offer(f"frame-{index}")

    assert channel.dropped_frames == 4, "four frames were superseded before delivery"
    assert asyncio.run(channel.take()) == "frame-4", "the newest frame must win"
    assert channel.sent_frames == 1


def test_draining_between_offers_drops_nothing():
    """A consumer that keeps up must lose nothing, or the policy is too eager."""
    from serving.api import ClientChannel

    channel = ClientChannel()

    async def alternate() -> list[str]:
        taken = []
        for index in range(5):
            channel.offer(f"frame-{index}")
            taken.append(await channel.take())
        return taken

    taken = asyncio.run(alternate())
    assert taken == [f"frame-{i}" for i in range(5)]
    assert channel.dropped_frames == 0


def test_credit_mode_refuses_to_hand_a_second_frame_to_the_transport():
    """The bug the one-frame mailbox did NOT prevent, now covered.

    `websocket.send_text()` returns when the frame reaches the transport, not
    when it reaches the client, so bounding the server-side queue at one frame
    still let ~208 stale frames accumulate in uvicorn's write buffer and the
    socket below it. Measured against the running container: a client reading
    every 250 ms received strictly consecutive sequence numbers while the server
    produced ~244 frames/s.

    In credit mode the sender cannot take a second frame until the client acks,
    so there is nothing to buffer. This asserts the sender BLOCKS rather than
    racing ahead, which is the property the previous design lacked.
    """
    from serving.api import ClientChannel

    channel = ClientChannel(credit=1)

    async def scenario() -> tuple[str, str, bool]:
        channel.offer("frame-0")
        first = await channel.take()

        # Credit is spent. More frames arrive; the newest must win, and the
        # sender must not be able to take any of them yet.
        for index in range(1, 50):
            channel.offer(f"frame-{index}")
        blocked_task = asyncio.create_task(channel.take())
        await asyncio.sleep(0)
        still_blocked = not blocked_task.done()

        channel.grant()
        second = await blocked_task
        return first, second, still_blocked

    first, second, still_blocked = asyncio.run(scenario())
    assert first == "frame-0"
    assert still_blocked, "without an ack the sender must not take another frame"
    assert second == "frame-49", "on the ack, the client gets the NEWEST frame, not the next one"
    assert channel.dropped_frames == 48


def test_a_session_boundary_is_never_dropped_in_favour_of_a_newer_frame():
    """The bug the Stage 8b dashboard found, and the reason events are not state.

    Boundaries were originally broadcast through `offer`, so they shared the
    one-frame slot with book state. The producer emits ~250 frames a second and
    a render-driven client takes 60, so the anchor after a boundary overwrote it
    essentially always: measured against the running server, a credit-mode
    client received 1,126 frames and ZERO boundaries in 30 seconds.

    The client therefore never learned its history was invalid and drew one
    continuous tape across a 422-second hole in the data — the exact fabrication
    `serving/feeds.py` emits boundaries to prevent.
    """
    from serving.api import ClientChannel

    channel = ClientChannel()
    channel.post("boundary")
    for index in range(100):
        channel.offer(f"frame-{index}")

    async def scenario() -> list[str]:
        return [await channel.take(), await channel.take()]

    first, second = asyncio.run(scenario())
    assert first == "boundary", "the boundary must survive 100 newer frames"
    assert second == "frame-99", "and the frame after it is still the newest one"


def test_posting_an_event_discards_the_frame_that_preceded_it():
    """A frame waiting when a boundary arrives is stale by definition.

    It describes the book before a discontinuity that invalidates it, so
    delivering it after the boundary would attribute the old session's book to
    the new one. Dropping it is the same policy as everywhere else here: the
    newest state wins, and immediately after a boundary the newest state is
    "nothing yet".
    """
    from serving.api import ClientChannel

    channel = ClientChannel()
    channel.offer("stale-frame")
    channel.post("boundary")

    assert channel.dropped_frames == 1
    assert asyncio.run(channel.take()) == "boundary"


def test_events_are_delivered_in_order_and_the_queue_is_bounded():
    """Ordering matters and growth does not happen. Both, or neither is safe."""
    from serving.api import ClientChannel

    channel = ClientChannel()
    for index in range(ClientChannel.EVENT_CAPACITY + 5):
        channel.post(f"event-{index}")

    assert channel.dropped_events == 5, "overflow is counted, not silently discarded"

    async def drain() -> list[str]:
        return [await channel.take() for _ in range(ClientChannel.EVENT_CAPACITY)]

    events = asyncio.run(drain())
    assert events == [f"event-{i}" for i in range(ClientChannel.EVENT_CAPACITY)]


def test_credit_covers_events_too_so_a_client_must_ack_them():
    """Whatever consumes credit must be acknowledged, or the stream deadlocks.

    An earlier version of the dashboard acked only frames. The first boundary
    then spent the single credit and never returned it, and the stream stopped
    dead — which looks exactly like a hung server.
    """
    from serving.api import ClientChannel

    channel = ClientChannel(credit=1)
    channel.post("boundary")
    channel.offer("frame")

    async def scenario() -> tuple[str, bool, str]:
        first = await channel.take()
        blocked = asyncio.create_task(channel.take())
        await asyncio.sleep(0)
        still_blocked = not blocked.done()
        channel.grant()
        return first, still_blocked, await blocked

    first, still_blocked, second = asyncio.run(scenario())
    assert first == "boundary"
    assert still_blocked, "an unacknowledged event must hold the credit window shut"
    assert second == "frame"


def test_fire_and_forget_is_the_default_and_says_so():
    """A plain client needs no protocol; the guarantee is correspondingly weaker."""
    from serving.api import ClientChannel

    assert ClientChannel().flow_controlled is False
    assert ClientChannel(credit=1).flow_controlled is True
    # Granting credit to a fire-and-forget channel is a no-op, not an error: an
    # exploratory client may send acks nobody asked it for.
    channel = ClientChannel()
    channel.grant()
    channel.offer("only")
    assert asyncio.run(channel.take()) == "only"


def test_the_channel_never_grows():
    """Unbounded queues are how dashboards die; this one holds exactly one frame."""
    from serving.api import ClientChannel

    channel = ClientChannel()
    for index in range(10_000):
        channel.offer(f"frame-{index}")

    assert channel.dropped_frames == 9_999
    assert asyncio.run(channel.take()) == "frame-9999"
    # And the slot is empty again, so nothing accumulated behind it.
    assert channel._pending is None


# -------------------------------------------------------------- frame schema


def test_frame_schema(tapes):
    """The contract the dashboard is written against."""
    from serving.api import build_frame
    from serving.signal import SignalResult

    feed = ReplayFeed(tapes, speed="max", loop=False)
    anchor = next(e for e in collect(feed, 5) if isinstance(e, BookAnchor))

    warming = build_frame(anchor, SignalResult(False, None, None, 0, 0, 42), feed)
    assert set(warming) >= {
        "t_ns", "seq", "session_id", "bids", "asks", "mid", "spread_ticks",
        "signal", "serving_infer_us",
    }
    assert warming["signal"] is None, "no signal before the history is full"
    assert warming["serving_infer_us"] is None
    assert warming["warmup"]["anchors_remaining"] == 42

    ready = build_frame(
        anchor, SignalResult(True, (0.2, 0.3, 0.5), 0.3, 812_000, 400_000, 0), feed
    )
    assert ready["signal"] == {"p_down": 0.2, "p_flat": 0.3, "p_up": 0.5, "score": 0.3}
    assert ready["serving_infer_us"] == pytest.approx(812.0)
    # score is computed server-side so every client agrees on it.
    assert ready["signal"]["score"] == pytest.approx(
        ready["signal"]["p_up"] - ready["signal"]["p_down"]
    )
    # Whole frame must survive the encoder the websocket actually uses.
    json.dumps(ready)


def test_ladder_drops_the_writers_zero_padding(tapes):
    """A zero-price level is padding, not an order at price zero."""
    from serving.api import _ladder

    levels = np.array([[6_400_000_000_000, 100], [0, 0], [0, 0]], dtype=np.int64)
    rows = _ladder(levels, price_scale=100_000_000, qty_scale=100_000_000)
    assert len(rows) == 1
    assert rows[0][0] == pytest.approx(64_000.0)


# ------------------------------------------- served numbers vs their records


def test_pareto_matches_the_records_it_cites():
    payload = records.pareto_payload()
    source, _ = records.load_python_variants()
    by_name = {entry["variant"]: entry for entry in source["variants"]}

    python_rows = [row for row in payload["rows"] if row["measured_by"] == "python harness"]
    assert len(python_rows) == len(source["variants"])
    for row in python_rows:
        original = by_name[row["variant"]]
        assert row["macro_f1"] == original["macro_f1"]
        assert row["params"] == original["parameters"]
        assert row["p50_us"] == original["latency_us"]["p50_us"]
        assert row["p99_us"] == original["latency_us"]["p99_us"]
        assert row["p99_9_us"] == original["latency_us"]["p99_9_us"]

    serving = [row for row in payload["rows"] if row["is_serving"]]
    assert len(serving) == 1 and serving[0]["variant"] == records.SERVING_VARIANT


def test_the_cpp_rows_are_tagged_and_their_accuracy_is_declared_inherited():
    """No C++ program in this repository has ever scored a test block.

    The C++ path is bit-checked against PyTorch, so it computes the same
    function and therefore has the same macro-F1 — but presenting an inherited
    number as a measured one is exactly the fabrication the constitution bans.
    """
    payload = records.pareto_payload()
    cpp = {row["variant"]: row for row in payload["rows"] if row["measured_by"] == "cpp harness"}

    assert set(cpp) == {"cpp_full", "cpp_incremental"}
    for row in cpp.values():
        assert row["macro_f1_is_inherited"] is True
        assert row["is_serving"] is False, "the C++ path is not what the server runs"
        assert row["source"].startswith("cpp_")
        assert payload["sources"][row["variant"]] == row["source"]


def test_both_cpp_variants_are_served_because_the_gap_is_the_finding():
    """Panel 7 needs six labelled points, and two of them are the C++ pair.

    Serving only the incremental row would present a ~193x algorithmic win as
    though it were what writing the thing in C++ bought. The full pass is the
    honest baseline and it is slower than ONNX Runtime, which is the actual
    Stage 7b finding.
    """
    payload = records.pareto_payload()
    by_variant = {row["variant"]: row for row in payload["rows"]}

    full, incremental = by_variant["cpp_full"], by_variant["cpp_incremental"]
    assert full["label"] == "C++ full" and incremental["label"] == "C++ incremental"
    assert full["p50_us"] > incremental["p50_us"], "the full pass recomputes the window"
    # The one that ships must be faster than what the server actually runs, and
    # the one that does not ship must be slower than ONNX Runtime on the same
    # model — that pair of facts is the whole point of showing both.
    assert incremental["p50_us"] < by_variant["student_int8"]["p50_us"]
    assert full["p50_us"] > by_variant["onnx_int8"]["p50_us"]

    # Every row the dashboard plots must be labelled by the server, not by JS.
    assert all(row.get("label") for row in payload["rows"])


def test_the_cpp_rows_inherit_the_fp32_students_accuracy_not_the_int8_one():
    """`inference_cpp` is float32. Crediting it with int8's accuracy loss is wrong.

    Its weights come from the distilled checkpoint with BatchNorm folded and no
    quantisation anywhere (`ml/export_weights.py`), and its parity test agrees
    with the PyTorch fp32 student on 1,000 held-out windows. The chain is
    fp32 student -> C++ full -> C++ incremental, so fp32 is the number to
    inherit. This was served wrong until Stage 8b.
    """
    payload = records.pareto_payload()
    by_variant = {row["variant"]: row for row in payload["rows"]}

    for name in ("cpp_full", "cpp_incremental"):
        row = by_variant[name]
        assert row["macro_f1_inherited_from"] == "student_fp32"
        assert row["macro_f1"] == by_variant["student_fp32"]["macro_f1"]
        assert row["macro_f1"] != by_variant["student_int8"]["macro_f1"]


def test_cpp_runs_are_chosen_by_the_record_clock_and_only_from_streamed_runs():
    """These filenames carry a pass number, not a timestamp, so sorting them lies.

    A `hot` run replays one cached input; a `stream` run walks fresh columns,
    which is what a live book presents. Putting a hot number on the same axis as
    a streamed one would compare a best case against a real one.
    """
    runs = records.load_cpp_runs()
    for variant in ("full", "incremental"):
        record, name = records._newest_cpp_run(runs, variant)
        assert record["input_mode"] == "stream"
        streamed = [
            r for r, _ in runs if r["variant"] == variant and r["input_mode"] == "stream"
        ]
        assert record["timestamp_utc"] == max(r["timestamp_utc"] for r in streamed), name

    with pytest.raises(records.RecordNotFound, match="no nonesuch/stream"):
        records._newest_cpp_run(runs, "nonesuch")


def test_decay_matches_the_evaluation_record():
    payload = records.decay_payload()
    evaluation, name = records.load_evaluation()

    assert payload["source"] == name
    assert payload["half_life_ms"] == evaluation["decay"]["half_life_ms"]
    assert payload["fit_r_squared"] == evaluation["decay"]["fit_r_squared"]
    assert payload["curve"] == evaluation["decay"]["rows"]
    # The peak must be a real index into the curve, and the fit must start there.
    peak = payload["curve"][payload["peak_index"]]
    assert peak["horizon_ms"] == evaluation["decay"]["peak_horizon_ms"]
    assert payload["fit_range"]["start_index"] == payload["peak_index"]


def test_stability_serves_the_pooled_ic_under_a_name_that_warns():
    """Naming is the cheapest guard available against quoting the wrong number."""
    payload = records.stability_payload()
    evaluation, _ = records.load_evaluation()
    ic = evaluation["information_coefficient"]

    assert "pooled_ic" not in payload
    assert payload["pooled_ic_not_tradeable"] == ic["pooled"]
    assert payload["mean"] == ic["mean"]
    assert payload["information_ratio"] == ic["information_ratio"]
    assert payload["block_ics"] == ic["block_ics"]
    # The per-block mean is what a trader would experience, and it is far below
    # the pooled figure. If that ever stops being true the caveat needs rewriting.
    assert payload["mean"] < payload["pooled_ic_not_tradeable"]


def test_economics_matches_the_record_and_the_shortfall_arithmetic():
    payload = records.economics_payload()
    evaluation, _ = records.load_evaluation()
    trading = evaluation["trading"]

    assert payload["breakeven_fee_bps"] == trading["breakeven_fee_bps"]
    assert payload["gross_bps_per_trade"] == trading["gross_bps_per_trade"]

    served = {tier["name"]: tier for tier in payload["tiers"]}
    for name, fee in trading["binance_taker_tiers_bps"].items():
        assert served[name]["fee_bps"] == fee
        assert served[name]["shortfall_multiple"] == pytest.approx(
            fee / trading["breakeven_fee_bps"]
        )
    # The three tiers CLAUDE.md pins by name must all be present.
    for name in ("VIP 0", "VIP 0 + BNB", "VIP 9"):
        assert name in served
    # Every published tier must be unaffordable, or the Stage 5 verdict is stale.
    assert all(tier["net_bps_per_trade"] < 0 for tier in payload["tiers"])


def test_the_verdict_is_still_what_stage_5_wrote():
    """Guards the one piece of prose this layer copies out of a document."""
    notes = (REPO_ROOT / "docs" / "INTERVIEW_NOTES_stage5.md").read_text(encoding="utf-8")
    collapsed = " ".join(notes.split())

    assert records.STAGE5_VERDICT in collapsed
    assert records.economics_payload()["verdict"] == records.STAGE5_VERDICT


def test_the_relevance_note_states_all_three_forbidden_confusions():
    """CLAUDE.md requires this text beside any latency claim."""
    note = records.RELEVANCE_NOTE
    assert "13.2 s" in note
    assert "70x" in note
    assert "not what makes it" in note


# ------------------------------------------------------------------ the app


def test_the_app_serves_frames_and_its_own_evidence(tapes):
    """One end-to-end pass so the wiring is covered, not just the parts."""
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from serving.api import create_app

    application = create_app(tapes=tapes, model_path=MODEL, speed="max")
    with fastapi_testclient.TestClient(application) as client:
        for path in ("/health", "/meta", "/pareto", "/decay", "/stability",
                     "/economics", "/latency"):
            assert client.get(path).status_code == 200, path

        meta = client.get("/meta").json()
        assert meta["is_demo"] is True
        assert meta["session_count"] == len(tapes)
        assert "int8" in meta["serving_variant"]

        with client.websocket_connect("/ws/stream") as socket:
            frames = [socket.receive_json() for _ in range(20)]
            assert all("session_id" in frame for frame in frames if frame["type"] == "frame")

            socket.send_json({"cmd": "nonsense"})
            replies = [socket.receive_json() for _ in range(400)]
            errors = [reply for reply in replies if reply.get("type") == "error"]
            assert errors and "unknown command" in errors[0]["detail"]

        latency = client.get("/latency").json()
        assert latency["relevance_note"] == records.RELEVANCE_NOTE
        assert "not what this server runs" in latency["not_the_cpp_figure"]
        assert latency["clock"]["resolution_ns"] > 0


# ----------------------------------------------------------------- LiveFeed


def _drain(feed, limit: int, timeout: float = 20.0) -> list:
    """Take `limit` events from a feed that never ends, or give up.

    `LiveFeed.events` polls forever by design, so a test must bound it. The
    timeout is generous because it is a failure signal, not a pacing knob.
    """

    async def run() -> list:
        events = []

        async def pump() -> None:
            async for event in feed.events():
                events.append(event)
                if len(events) >= limit:
                    return

        await asyncio.wait_for(pump(), timeout=timeout)
        return events

    return asyncio.run(run())


def test_live_feed_waits_for_a_tape_then_reads_its_scales(tmp_path, tapes):
    """Under `--profile live` the API and the daemon start together.

    For the first seconds the directory is legitimately empty, and a frame
    cannot be built without the tape header's scales. Waiting is the honest
    response; booting with guessed scales would emit wrong prices first and
    correct them afterwards.
    """
    from serving.feeds import LiveFeed

    (tmp_path / tapes[0].name).write_bytes(tapes[0].read_bytes())
    feed = LiveFeed(tmp_path, poll_seconds=0.05)
    feed.prime(timeout_s=10.0)

    assert feed.price_scale == 100_000_000
    assert feed.symbol == "BTCUSDT"
    assert feed.tick_size / feed.price_scale == pytest.approx(0.01)
    assert feed.describe()["mode"] == "live"


def test_live_feed_gives_up_rather_than_guessing(tmp_path):
    """An empty directory forever is a broken deployment, not a slow one."""
    from serving.feeds import LiveFeed

    with pytest.raises(TimeoutError, match="capture service"):
        LiveFeed(tmp_path, poll_seconds=0.05).prime(timeout_s=0.3)


def test_a_new_tape_file_is_a_new_session(tmp_path, tapes):
    """The daemon starts a new tape when it resyncs, so files ARE sessions.

    That is why `LiveFeed` needs no gap-detection heuristic: the filesystem
    already carries the boundary, and inventing a threshold on timestamp deltas
    would be a second, worse answer to a question already answered.
    """
    from serving.feeds import LiveFeed

    (tmp_path / tapes[0].name).write_bytes(tapes[0].read_bytes())
    feed = LiveFeed(tmp_path, poll_seconds=0.02)
    feed.prime(timeout_s=10.0)

    first = _drain(feed, 40)
    assert isinstance(first[0], SessionBoundary)
    assert first[0].reason == "start"
    assert first[0].previous_session_id is None
    assert all(e.session_id == tapes[0].stem for e in first[1:] if isinstance(e, BookAnchor))

    # The daemon resyncs and starts a second tape.
    (tmp_path / tapes[1].name).write_bytes(tapes[1].read_bytes())
    later = _drain(LiveFeed(tmp_path, poll_seconds=0.02), 40)
    assert later[0].session_id == tapes[1].stem, "the newest tape is the live one"


def test_live_feed_does_not_redeliver_blocks_it_has_already_emitted(tmp_path, tapes):
    """Tailing re-opens the file every poll; it must not replay what it has sent.

    The format's truncation-tolerance means a partially written block is simply
    absent and arrives next poll, so tailing needs no locking — but it does need
    the reader to remember how far it got.
    """
    from serving.feeds import LiveFeed

    (tmp_path / tapes[0].name).write_bytes(tapes[0].read_bytes())
    feed = LiveFeed(tmp_path, poll_seconds=0.02)
    feed.prime(timeout_s=10.0)

    anchors = [e for e in _drain(feed, 120) if isinstance(e, BookAnchor)]
    sequences = [anchor.sequence for anchor in anchors]

    assert sequences == sorted(sequences), "blocks must arrive in tape order"
    assert len(sequences) == len(set(sequences)), "a block was delivered twice"
