/* state.js — every buffer this dashboard owns, allocated once, at boot.
 *
 * WHAT
 *     One mutable object holding the current book, the current signal, and
 *     preallocated ring buffers for everything that has history. `stream.js`
 *     writes into it as messages arrive; `render.js` reads it sixty times a
 *     second. Nothing in this file draws, and nothing in it touches the DOM.
 *
 * WHY RINGS AND NOT ARRAYS THAT GROW
 *     This is the same decision as `serving/api.py:ClientChannel`, one layer
 *     up. An unbounded client-side history is a memory leak that grows at the
 *     frame rate, and a `push`/`shift` pair on a plain array reallocates and
 *     copies. Fixed typed arrays with a masked index have neither problem, and
 *     the mask is exact because every length here is a power of two.
 *
 * WHY IT MATTERS MORE THAN IT LOOKS — the measured reason, from Stage 7b.
 *     Hoisting an allocation out of the C++ hot loop moved p50 by 0% and
 *     p99.99 by 2.4x. Allocation cost is not small and constant, it is rare and
 *     enormous: the frame that triggers a collection is the frame that misses.
 *     A dashboard whose thesis is tail latency cannot have a p99.9 render frame
 *     caused by its own garbage. Same mechanism, same fix, one language up.
 *
 * DESIGN DECISION — one column per DELIVERED FRAME, not per anchor and not per
 * fixed slice of time.
 *     The socket runs in credit mode (`?flow=ack`, credit 1) and the render
 *     loop acks once per rendered frame, so the server hands over at most one
 *     frame per frame. A column is therefore exactly one delivered book state.
 *
 *     Rejected alternative: one column per anchor. The committed tapes have a
 *     median inter-anchor delta of 0 ms — a single 100 ms depth update produces
 *     several anchors stamped within the same microsecond — so an anchor axis
 *     would stretch bursts and compress quiet stretches, and the panel's whole
 *     subject is where liquidity rests over time.
 *
 *     What that costs, stated on the panel rather than hidden: anchors the
 *     server dropped to keep the client current are not drawn. They are counted
 *     from the `seq` gaps and reported in the header as anchors skipped per
 *     frame, which is this client's direct measurement of backpressure.
 *
 *     A frameless frame commits nothing. If the feed is paused, or the socket
 *     is down, or nothing arrived, the tape does not scroll — it freezes, and a
 *     frozen tape is honest in a way that a repeated last column would not be.
 */

/** Book depth the tape stores. The server sends 10 per side and so does the tape. */
export const DEPTH = 10;

/**
 * Columns of tape history. 1024 because it is the smallest power of two that
 * exceeds the widest this panel gets (~1080 CSS px at 1440, and one column is
 * one device pixel step) — and a power of two makes the ring index `i & 1023`
 * instead of a modulo. At the ~25-32 anchors/s of the committed tapes it is
 * roughly 35 s of delivered frames.
 *
 * This ring is NOT how the tape scrolls: scrolling is a blit of the canvas
 * itself. It exists so a resize or a monitor change can repaint history that
 * would otherwise be lost with the backing store, and so the y-axis can be
 * rescaled without leaving old pixels at a stale scale.
 */
export const TAPE_COLUMNS = 1024;
const TAPE_MASK = TAPE_COLUMNS - 1;

/** Score history for the signal sparkline: ~240 px wide, so 512 is 2/px. */
export const SIGNAL_HISTORY = 512;
const SIGNAL_MASK = SIGNAL_HISTORY - 1;

/**
 * Live serving-latency samples for the histogram. At roughly 60 delivered
 * frames a second this is about 68 s of inference times, which is long enough
 * for the p99 marker to rest on ~40 observations rather than one.
 */
export const LATENCY_SAMPLES = 4096;
const LATENCY_MASK = LATENCY_SAMPLES - 1;

/** Render durations behind the header meter: ~2 s at 60 fps. */
export const RENDER_SAMPLES = 128;
const RENDER_MASK = RENDER_SAMPLES - 1;

/** The widest a rendered gap may be, in columns. See `applyBoundary`. */
const GAP_COLUMN_CAP = 240;

/** How long the screen stays dimmed after a gap, in milliseconds. */
const GAP_DIM_MS = 1400;

/** Column flags. A column is drawn from these, not from ambient state. */
export const FLAG_GAP = 1; // unobserved market time: draw nothing at all
export const FLAG_WARMUP = 2; // book present, model not ready: no ribbon
export const FLAG_MARK = 4; // a boundary or speed change happened at this column

export function createState() {
  return {
    // ------------------------------------------------------- the live book
    bidPrice: new Float64Array(DEPTH), // float64: 64,973.21 needs more than
    askPrice: new Float64Array(DEPTH), // float32's ~7 significant digits
    bidSize: new Float32Array(DEPTH),
    askSize: new Float32Array(DEPTH),
    bidLevels: 0,
    askLevels: 0,
    // Previous sizes and the wall-clock time each level last changed, so the
    // ladder can flash a changed level and decay it over ~120 ms without
    // allocating an event object per change.
    prevBidSize: new Float32Array(DEPTH),
    prevAskSize: new Float32Array(DEPTH),
    bidFlashAt: new Float64Array(DEPTH),
    askFlashAt: new Float64Array(DEPTH),

    mid: NaN,
    spreadTicks: NaN,
    tickSize: 0.01, // replaced by /meta, which measures it rather than assuming
    tapeNs: 0,
    seq: -1,
    sessionId: null,

    // ---------------------------------------------------------- the signal
    hasSignal: false,
    pDown: 0,
    pFlat: 0,
    pUp: 0,
    score: 0,
    servingUs: NaN,
    warmupRemaining: 0,
    warmupRequired: 599,

    // ----------------------------------------------------- what the client
    // ------------------------------------------------------ measures itself
    framesReceived: 0,
    columnsCommitted: 0,
    /** seq gaps within one session: anchors the transport dropped for us. */
    anchorsSkipped: 0,
    skippedPerFrame: 0,
    lastSeq: -1,

    // ------------------------------------------------------------- history
    // Tape columns. Offsets are in ticks from mid and sizes are log1p(size),
    // both computed once here rather than per draw, because the draw happens
    // again on every repaint and the arithmetic does not change.
    colBidOffset: new Float32Array(TAPE_COLUMNS * DEPTH),
    colAskOffset: new Float32Array(TAPE_COLUMNS * DEPTH),
    colBidSize: new Float32Array(TAPE_COLUMNS * DEPTH),
    colAskSize: new Float32Array(TAPE_COLUMNS * DEPTH),
    colMid: new Float64Array(TAPE_COLUMNS),
    colScore: new Float32Array(TAPE_COLUMNS),
    colConfidence: new Float32Array(TAPE_COLUMNS),
    colFlags: new Uint8Array(TAPE_COLUMNS),
    colTapeNs: new Float64Array(TAPE_COLUMNS),
    colHead: -1,
    colCount: 0,

    scoreHistory: new Float32Array(SIGNAL_HISTORY),
    scoreHead: -1,
    scoreCount: 0,

    latency: new Float32Array(LATENCY_SAMPLES),
    latencyHead: -1,
    latencyCount: 0,
    /** Sorting scratch, so the percentile pass never allocates. */
    latencyScratch: new Float32Array(LATENCY_SAMPLES),

    renderMs: new Float32Array(RENDER_SAMPLES),
    renderHead: -1,
    renderCount: 0,
    renderScratch: new Float32Array(RENDER_SAMPLES),

    // ------------------------------------------------------ pending events
    /** True when a frame has arrived that no column has been written for yet. */
    hasPendingFrame: false,
    /** Blank columns still owed to a session gap, in columns. */
    gapColumnsPending: 0,
    /** Text for the banner while there is no data, or null. */
    gapMessage: null,
    /** Wall-clock deadline until which the panels are dimmed for a gap. */
    gapDimUntil: 0,
    /** A one-column marker (loop wrap, seek, speed change) owed to the tape. */
    markPending: null,
    /** Exponential mean of tape nanoseconds per column, for gap width. */
    nsPerColumn: 0,

    // ------------------------------------------------- fetched, once, at boot
    meta: null,
    pareto: null,
    stability: null,
    economics: null,
    latencyRecord: null,
    /** Endpoint -> the reason it is empty, so a panel can name what is missing. */
    failures: {},

    // ------------------------------------------------------- feed controls
    speed: 10,
    paused: false,
    connection: "connecting",
  };
}

/**
 * Fold one websocket frame into the live book.
 *
 * Reads only what the frame carries. A level whose price the server omitted is
 * not padded here either: `bidLevels` says how many are real, and the ladder
 * draws that many rows rather than ten rows some of which are zero.
 */
export function applyFrame(state, frame, nowMs) {
  const bids = frame.bids;
  const asks = frame.asks;

  state.bidLevels = Math.min(bids.length, DEPTH);
  state.askLevels = Math.min(asks.length, DEPTH);
  for (let level = 0; level < DEPTH; level += 1) {
    const bid = level < state.bidLevels ? bids[level] : null;
    const ask = level < state.askLevels ? asks[level] : null;
    const bidSize = bid === null ? 0 : bid[1];
    const askSize = ask === null ? 0 : ask[1];
    // A level flashes when its resting size changes, which is the event a
    // trader is watching for — not when its price changes, because a price
    // change is the whole ladder shifting and everything would flash at once.
    if (bidSize !== state.prevBidSize[level]) state.bidFlashAt[level] = nowMs;
    if (askSize !== state.prevAskSize[level]) state.askFlashAt[level] = nowMs;
    state.prevBidSize[level] = bidSize;
    state.prevAskSize[level] = askSize;
    state.bidPrice[level] = bid === null ? NaN : bid[0];
    state.askPrice[level] = ask === null ? NaN : ask[0];
    state.bidSize[level] = bidSize;
    state.askSize[level] = askSize;
  }

  state.mid = frame.mid;
  state.spreadTicks = frame.spread_ticks;
  state.tapeNs = frame.t_ns;
  state.seq = frame.seq;
  state.sessionId = frame.session_id;

  // Anchors the server dropped on our behalf. Sequence restarts at every
  // session, so a negative or huge step is a boundary rather than a loss and is
  // not counted as one.
  if (state.lastSeq >= 0 && frame.seq > state.lastSeq) {
    state.anchorsSkipped += frame.seq - state.lastSeq - 1;
  }
  state.lastSeq = frame.seq;
  state.framesReceived += 1;
  state.skippedPerFrame = state.anchorsSkipped / Math.max(1, state.framesReceived);

  if (frame.signal) {
    state.hasSignal = true;
    state.pDown = frame.signal.p_down;
    state.pFlat = frame.signal.p_flat;
    state.pUp = frame.signal.p_up;
    state.score = frame.signal.score;
    state.warmupRemaining = 0;
    pushScore(state, frame.signal.score);
    if (typeof frame.serving_infer_us === "number") {
      state.servingUs = frame.serving_infer_us;
      pushLatency(state, frame.serving_infer_us);
    }
  } else {
    // Not an error state, and never smoothed over: after a boundary the
    // normalisation history is destroyed on purpose and 599 anchors must pass.
    // Holding the last good signal here would be the precise dishonesty the
    // rest of this project exists to avoid.
    state.hasSignal = false;
    state.score = 0;
    state.servingUs = NaN;
    state.warmupRemaining = frame.warmup ? frame.warmup.anchors_remaining : 0;
    state.warmupRequired = frame.warmup ? frame.warmup.anchors_required : state.warmupRequired;
  }

  state.hasPendingFrame = true;
}

/**
 * Fold a session boundary in: reset what the boundary invalidates, and owe the
 * tape a gap of the right width.
 *
 * The width is computed from `gap_ns`, which the server only reports for a real
 * resync. A loop wrap or a seek reports zero because its discontinuity is an
 * artefact of the demo, so those get a one-column marker and no blank — the
 * client refuses to invent a duration for the same reason the server does.
 */
export function applyBoundary(state, message) {
  state.lastSeq = -1;
  state.hasSignal = false;
  state.score = 0;
  state.servingUs = NaN;
  // The score sparkline is emptied too. Scores either side of a boundary come
  // from models normalised against different history, across market nobody
  // observed, so a line joining them would be the same fabrication the tape
  // refuses to draw — just at 240 pixels instead of 1,000.
  state.scoreCount = 0;
  state.scoreHead = -1;

  if (message.gap_ns > 0 && state.nsPerColumn > 0) {
    // The blank is the unobserved time drawn at the tape's own measured time
    // scale, so the hole in the picture is the size of the hole in the data.
    //
    // CLAMPED, and the clamp says so. The two real gaps in the committed set
    // are 422 s and 544 s of unobserved market, against a panel that spans
    // roughly 130 s — drawn to scale they would blank the tape three times over
    // and leave the viewer with no context for what they were looking at. So
    // the blank is capped and the banner states the true duration, because a
    // gap drawn smaller than it is would understate exactly the thing the gap
    // exists to communicate.
    const wanted = Math.round(message.gap_ns / state.nsPerColumn);
    state.gapColumnsPending = Math.max(4, Math.min(GAP_COLUMN_CAP, wanted));
    const seconds = (message.gap_ns / 1e9).toFixed(1);
    const clamped = wanted > GAP_COLUMN_CAP ? " — blank clamped to fit the panel" : "";
    state.gapMessage =
      `resync gap — feed discontinuity, no data for ${seconds} s of market${clamped}`;
    // Dim the whole screen for as long as the banner is up. Only for a real
    // gap, and only briefly: at 10x the committed set crosses a boundary every
    // few seconds, and a dashboard that spent a third of its life greyed out
    // would train a viewer to ignore the one signal that means "these numbers
    // are not continuous with the ones you just read".
    state.gapDimUntil = performance.now() + GAP_DIM_MS;
  } else {
    state.gapColumnsPending = 0;
    state.gapMessage = null;
  }
  state.markPending = message.reason === "loop" ? "loop restart" : message.reason;
}

/**
 * Write the current book into the tape ring as one column.
 *
 * Called by the render loop, once per rendered frame, and only when a frame is
 * actually pending. Returns true if a column was written, which is what tells
 * the tape panel to blit.
 */
export function commitColumn(state) {
  if (state.gapColumnsPending > 0) {
    // Gap columns are written first and all at once — the hole is a duration,
    // not a fade, and drawing it over the next four seconds would make it look
    // like slow data rather than absent data.
    const owed = state.gapColumnsPending;
    for (let index = 0; index < owed; index += 1) {
      advanceColumn(state);
      state.colFlags[state.colHead] = FLAG_GAP;
      state.colMid[state.colHead] = NaN;
    }
    state.gapColumnsPending = 0;
    return owed;
  }

  if (!state.hasPendingFrame) return 0;
  state.hasPendingFrame = false;

  const previousNs = state.colCount > 0 ? state.colTapeNs[state.colHead] : 0;
  advanceColumn(state);
  const head = state.colHead;
  const base = head * DEPTH;
  const tick = state.tickSize;

  for (let level = 0; level < DEPTH; level += 1) {
    const hasBid = level < state.bidLevels;
    const hasAsk = level < state.askLevels;
    // Offset in ticks from mid, which is the axis. Sizes are stored as
    // log1p because that is how they are drawn — measured across the three
    // committed tapes the median resting size is ~0.0005 BTC and p99 is ~10,
    // four orders of magnitude, so a linear intensity ramp would render the
    // whole book as either black or white.
    state.colBidOffset[base + level] = hasBid ? (state.mid - state.bidPrice[level]) / tick : NaN;
    state.colAskOffset[base + level] = hasAsk ? (state.askPrice[level] - state.mid) / tick : NaN;
    state.colBidSize[base + level] = hasBid ? Math.log1p(state.bidSize[level]) : 0;
    state.colAskSize[base + level] = hasAsk ? Math.log1p(state.askSize[level]) : 0;
  }

  state.colMid[head] = state.mid;
  state.colTapeNs[head] = state.tapeNs;
  state.colScore[head] = state.hasSignal ? state.score : 0;
  // Confidence is the model's belief in whatever class it picked, which is what
  // the ribbon's opacity encodes. |score| already encodes direction and size,
  // so reusing it here would encode one quantity twice and none of the other.
  state.colConfidence[head] = state.hasSignal
    ? Math.max(state.pDown, state.pFlat, state.pUp)
    : 0;

  let flags = state.hasSignal ? 0 : FLAG_WARMUP;
  if (state.markPending !== null) flags |= FLAG_MARK;
  state.markPending = null;
  state.colFlags[head] = flags;

  // Exponential mean of tape time per column, used to size a gap. Only sane
  // deltas contribute: a boundary produces a negative or absurd step and must
  // not drag the estimate that decides how wide the next gap is drawn.
  if (previousNs > 0) {
    const delta = state.tapeNs - previousNs;
    if (delta > 0 && delta < 5e9) {
      state.nsPerColumn = state.nsPerColumn === 0 ? delta : state.nsPerColumn * 0.98 + delta * 0.02;
    }
  }
  return 1;
}

function advanceColumn(state) {
  state.colHead = (state.colHead + 1) & TAPE_MASK;
  state.colCount = Math.min(state.colCount + 1, TAPE_COLUMNS);
  state.columnsCommitted += 1;
}

export function columnIndex(state, stepsBack) {
  return (state.colHead - stepsBack + TAPE_COLUMNS * 2) & TAPE_MASK;
}

function pushScore(state, value) {
  state.scoreHead = (state.scoreHead + 1) & SIGNAL_MASK;
  state.scoreHistory[state.scoreHead] = value;
  state.scoreCount = Math.min(state.scoreCount + 1, SIGNAL_HISTORY);
}

function pushLatency(state, value) {
  state.latencyHead = (state.latencyHead + 1) & LATENCY_MASK;
  state.latency[state.latencyHead] = value;
  state.latencyCount = Math.min(state.latencyCount + 1, LATENCY_SAMPLES);
}

export function pushRenderMs(state, value) {
  state.renderHead = (state.renderHead + 1) & RENDER_MASK;
  state.renderMs[state.renderHead] = value;
  state.renderCount = Math.min(state.renderCount + 1, RENDER_SAMPLES);
}

/**
 * Percentiles from a ring, without allocating.
 *
 * Copies into the caller's scratch array, sorts the used prefix in place and
 * indexes it. `TypedArray.prototype.sort` is numeric by default — unlike
 * `Array.prototype.sort`, which is lexicographic and would report the p99 of
 * [9, 100] as 9.
 */
export function percentiles(source, count, scratch, out) {
  if (count === 0) return null;
  scratch.set(source.subarray(0, count));
  const used = scratch.subarray(0, count);
  used.sort();
  out.min = used[0];
  out.max = used[count - 1];
  out.p50 = used[Math.min(count - 1, Math.floor(count * 0.5))];
  out.p99 = used[Math.min(count - 1, Math.floor(count * 0.99))];
  out.p999 = used[Math.min(count - 1, Math.floor(count * 0.999))];
  out.count = count;
  return out;
}
