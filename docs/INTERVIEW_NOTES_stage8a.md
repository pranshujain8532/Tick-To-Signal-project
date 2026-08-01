# Stage 8a — the serving layer

A push websocket that can feed a 60 fps dashboard, and seven REST endpoints that
serve this project's measured results rather than its flattering ones.

Everything here runs from `docker compose up` with no network, no exchange
credentials and no prior capture run.

---

## 1. The four decisions, and what was rejected

### 1.1 Push, not poll — `serving/api.py:stream`

The book updates 22–32 times a second in the committed replay and the dashboard
renders at 60 fps. A polling client has to guess that rate: too slow and it
misses book states, too fast and it burns a round trip to be told nothing
changed.

The deeper problem is authority. Under polling the *client* decides what
"current" means, so a client that stalls for 300 ms comes back, asks for now,
gets it, and silently skips everything in between — with no counter anywhere
recording that it happened. Push makes the server the authority and makes the
skipping explicit, countable and reported (`/health.dropped_frames`).

**Rejected:** long-polling and server-sent events. SSE would actually work for
the frame stream — it is one-directional and text — but the control channel
(`speed`, `seek`, `pause`) needs a client→server path, and running SSE downstream
plus POSTs upstream is two transports where one suffices.

### 1.2 Drop, never queue — `serving/api.py:ClientChannel`

> **Corrected in Stage 8b, and the correction is the interesting part.** This
> policy is right about *state* and was being applied to *events*. Session
> boundaries went through the same one-frame slot, so the anchor arriving
> immediately after a boundary overwrote it: a credit-mode client received
> **1,126 frames and zero boundaries in 30 seconds**. Everything below in this
> section still holds for frames; boundaries now go through a bounded FIFO that
> is drained first. See
> [INTERVIEW_NOTES_stage8b.md](INTERVIEW_NOTES_stage8b.md) §3.1.

`ClientChannel` holds exactly one pending frame. A second arrival destroys the
first and increments `dropped_frames`.

Two independent reasons, and the first is the one people miss:

**A stale frame is wrong, not merely late.** It draws a book that no longer
exists beside a signal about a price that has already moved. Delivered late it
is indistinguishable, to the viewer, from a current one. For a market view,
missing data is honest and late data is a lie.

**An unbounded queue is how dashboards die.** A slow link, a sleeping laptop, a
throttled background tab — each turns a queue into a memory leak growing at the
anchor rate, which then delivers a burst of history nobody asked for on recovery.
Bounding at one frame converts an unbounded failure into "you missed some
frames", which is survivable and visible.

A non-zero drop count at 10x is expected and healthy. Zero would mean either
nobody is connected or a renderer is consuming 300 frames a second, which no
renderer needs to.

**Rejected:** a bounded queue of, say, 8 frames. It only moves the problem — the
client still receives 7 stale frames before the fresh one, and now there is a
tuning parameter nobody can justify.

#### The mailbox alone did not work, and the container proved it

This is the most important thing in Stage 8a, and I got it wrong first.

Bounding the server-side queue at one frame does not bound what the CLIENT sees,
because `websocket.send_text()` returns when the frame reaches the *transport*,
not when it reaches the client. Under uvicorn there is a write buffer, and under
that a socket buffer.

Measured against the running container — a client reading one frame every 250 ms
while the server produced ~244 frames/s:

| | seq advance per 250 ms read | what that means |
|---|---|---|
| expected if fresh | ~60 | client sees current market |
| **mailbox only** | **1** | ~208 stale frames buffered below the mailbox |

The server had accepted 239 sends for 31 actual receives. The client was being
fed strictly consecutive frames, in order, minutes behind the market — precisely
the failure the policy exists to prevent. The unit test passed throughout,
because it tested the mailbox and the bug was underneath it.

**The fix is credit-based flow control**, opt-in via `/ws/stream?flow=ack`. The
server hands the transport at most one unacknowledged frame; the client acks
once per rendered frame. With a credit of 1 there is nothing to buffer, so the
frame the client receives is the newest at the moment it became able to accept
one. Re-measured in the same container:

| mode | seq advance per read (median / max) |
|---|---|
| default, fire-and-forget | 1 / 1 |
| `flow=ack` | **34 / 78** |

The default stays fire-and-forget so an exploratory client — `wscat`, a browser
console — needs no protocol, and the weaker guarantee is stated rather than
implied. The Stage 8b dashboard will ack on `requestAnimationFrame`, which makes
the flow render-driven and is the correct design for a 60 fps view.

`tests/test_serving.py:test_credit_mode_refuses_to_hand_a_second_frame_to_the_transport`
now asserts the sender *blocks* without an ack, which is the property the first
design lacked.

### 1.3 Measure the clock before trusting it — `serving/signal.py:MeasuredClock`

The clock's real resolution is measured at startup, logged, returned by
`/latency`, and used as a gate: if the serving path is not at least 100 clock
ticks long, `/latency` withholds the percentiles and says so instead of
publishing a grid.

This is not defensive programming for its own sake. Stage 7b's C++ harness found
`std::chrono::steady_clock` advertising a one-nanosecond period and resolving
about one millisecond, which recorded **19,518 of 20,000 forward passes as taking
exactly zero time** and produced a "p50 5,000 µs" that was a 1 ms grid rather
than a measurement. `perf_counter_ns` measures at 100 ns here and is fine — but
"expected to be fine" is precisely what the C++ run also assumed.

### 1.4 A session boundary is an event — `serving/feeds.py:SessionBoundary`

`SessionBoundary` is a member of the yielded union, so a consumer has to handle
it. Every session id is announced by a boundary before any anchor carries it,
and `tests/test_serving.py:test_every_session_is_introduced_by_a_boundary`
asserts that.

**Rejected:** concatenating the tapes and letting timestamps jump. The capture
daemon starts a new tape when it resyncs, so the ~9 s between two tapes contains
price moves nobody observed. `ml/eval.py:resolve_prices_per_session` exists to
keep that out of a PnL number; a serving feed that quietly re-joined what the
research code carefully split would put the error back somewhere with less
scrutiny.

---

## 2. The latency boundary, stated three times because it matters

| figure | what it covers | measured by | where it appears |
|---|---|---|---|
| `serving_infer_us` | ONNX int8 forward pass, this process, one `[1,100,40]` input | `serving/signal.py`, `perf_counter_ns` | every frame, `/latency.inference` |
| feature construction | mid-relative prices, `log1p` sizes, causal rolling z-score | same | `/latency.feature_construction` — **separate, never folded in** |
| ~11 µs C++ | hand-written forward pass from an already-prepared `[40]` column | `inference_cpp/bench/bench.cpp`, pinned, 1M iterations | `/pareto`, tagged `measured_by: cpp harness` |

**Feature construction is not small.** Measured on this machine at **p50 490 µs
against 663 µs of ONNX** — roughly 43% of the per-anchor cost. My own module
docstring originally guessed it would be negligible; the measurement said
otherwise and the guess was deleted. This makes the C++ comparison *more*
lopsided than it looks: the 11 µs figure excludes work the serving path cannot
avoid, and the C++ feature path does not exist at all (`TODO(measure)` in
`docs/benchmark_methodology.md`).

CLAUDE.md forbids three specific confusions. Each has a structural guard rather
than a comment:

1. **"low latency is what makes this tradeable"** → `records.RELEVANCE_NOTE` is
   returned by `/latency` and required by test to name the 13.2 s half-life and
   the 70× shortfall.
2. **"the microsecond figure is what this runs"** → `/latency.not_the_cpp_figure`
   says so in prose; `/pareto` tags every row with `measured_by`; the C++ row
   carries `is_serving: false`.
3. **"the pooled IC is the edge"** → `/stability` has no `pooled_ic` key at all.
   It is served as `pooled_ic_not_tradeable`, so a dashboard author reaching for
   the biggest number has to type the words "not tradeable" to get it.

---

## 3. The ten hardest questions

**Q1. Your drop policy loses data. How do you know you are not dropping the one
frame that mattered?**
I do not, and no policy can — the alternative is delivering it late, which is
worse. What I can do is make the loss visible and bounded: `dropped_frames` is
per connection and cumulative in `/health`, and the loss is always of the
*older* frame, so what survives is always the most recent state of the world.
For a market view that is the correct thing to preserve.
`serving/api.py:ClientChannel.offer`.

**Q2. Why one producer for all clients rather than one per connection?**
Because the model would otherwise run N times for N viewers and the reported
latency would become a function of how many people are watching. One producer
runs the ~1.15 ms pipeline once per anchor and offers the result to every
channel; `_broadcast` serialises once too. `serving/api.py:_produce`.

**Q3. Your feed emits every anchor even at 100x. Does the producer not fall
behind?**
Yes, and deliberately. `_wait_until` never corrects for lag: if the consumer
cannot keep up the replay simply runs slower than requested. The rejected
alternative — skipping anchors to hold the wall clock — would draw a book that
never existed. `serving/feeds.py:_wait_until`.

**Q4. There is a bug hiding in "yield to the event loop". Where was it?**
`_wait_until` originally returned early when the computed sleep was zero. At
`speed="max"` that made the producer a tight loop with no suspension point, so
the websocket sender tasks never got scheduled and the server hung solid. It is
not only `max` that hits it: the committed tapes have a **median inter-anchor
delta of 0 ms**, because one 100 ms depth frame carrying dozens of level updates
produces several anchors stamped within the same microsecond. Every path now
awaits, including `asyncio.sleep(0)`.

**Q5. Why does a seek emit a session boundary? It is the same session.**
Because "session_id changed" is the consumer's single signal to throw away
history, and a seek invalidates the 599 rows of normalisation history exactly as
a resync does. Making the invariant uniform — every discontinuity, whatever its
cause, gets a boundary and a new id — means the client needs one rule instead of
three. `serving/feeds.py:seek`, and `test_seek_emits_a_boundary_because_it_invalidates_history`.

**Q6. Your demo shows no signal for the first several seconds. Is that broken?**
No, and hiding it would be. `ml/features.py` normalises with a 500-row causal
rolling z-score and the model window is 100 rows, so **599 anchors — about 20 s
at 1x — must pass before an input can be built**, and that cost is paid again
after every boundary. The frame carries `signal: null` plus a `warmup` countdown
rather than repeating a stale last value. Freezing on the last good signal is the
exact dishonesty this project exists to avoid.

**Q7. Why is the C++ row's accuracy not measured?**
Because no C++ program in this repository has ever scored a test block. The C++
path is bit-checked against PyTorch on 1,000 real windows, so it computes the
same function and therefore has the same macro-F1 — but presenting an inherited
number as a measured one is a fabrication. The payload says
`macro_f1_is_inherited: true` and a test asserts it.
`serving/records.py:_cpp_row`.

**Q8. Why does the serving image not contain PyTorch?**
Because the deployment artefact is a 126 KiB ONNX graph and onnxruntime executes
it; torch would add ~2 GB to run a file smaller than the Dockerfile's build
context. That is the actual payoff of Stage 6's export, and it only holds if
nothing in the serving path imports a torch-importing module — which is why
`serving/signal.py:_open_session` duplicates six lines of
`ml/export.py:make_session` rather than importing it. The duplication is stated
in both places.

**Q9. You claim the tick size is 0.01 but you never wrote 0.01. How?**
`measure_tick_size` takes the smallest positive gap between adjacent price levels
across every snapshot. On a book this dense that gap is the venue tick. It
measures 1,000,000 fixed-point units = 0.01 USDT, is reported by `/meta` so it
can be checked, and means `spread_ticks` cannot be silently wrong if the demo is
pointed at another symbol.

**Q10. What in this stage is not verified?**
Live mode. `LiveFeed` is structurally complete and wired through `TTS_MODE=live`,
but it has never been run against a live exchange, and the `capture` compose
profile has never been started. The demo path is fully verified: the image builds
(463 MB), `docker compose up` reaches healthy, all seven endpoints answer, and
both flow modes have been driven with a real websocket client against the running
container.

**Q11 (the one worth asking). Your backpressure test passed while the bug was
live. Why?**
Because it tested `ClientChannel` and the bug was underneath it — in the
transport's write buffer, which no unit test of my code could see. It took
driving the actual container with a deliberately slow client to find that a
"drop stale frames" policy was delivering strictly consecutive frames minutes
late. The general lesson is the one this project keeps relearning: a test of the
component is not a test of the system, and claims about behaviour under load have
to be measured under load. See §1.2.

---

## 4. What is deliberately not here

- **The dashboard.** Stage 8b. `serving/dashboard/` is mounted if present and
  absent for now, so `/` currently 404s and the API is the whole deliverable.
- **A notebook.** The constitution exempts `serving/api.py` as a long-running
  server, but `feeds.py`, `signal.py` and `records.py` are importable and
  therefore owe a walkthrough notebook. It is not written. This is the one
  Definition-of-Done item Stage 8a does not satisfy, and it is called out rather
  than quietly skipped.
- **Authentication, rate limiting, TLS.** A read-only demo of public market data
  with no write path. Adding auth would be theatre.
- **Any claim about end-to-end latency under load.** This harness measures
  service time. See `docs/benchmark_methodology.md` §3 on coordinated omission.
