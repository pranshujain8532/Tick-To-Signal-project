/* stream.js — the data loop. It never draws, and it never waits for a frame.
 *
 * WHAT
 *     One websocket to `/ws/stream?flow=ack`, one exponential-backoff
 *     reconnect, and the four commands the server accepts (speed, seek, pause,
 *     ack). Messages mutate `state`; that is the entire contract with the rest
 *     of the dashboard.
 *
 * WHY `?flow=ack` IS NOT OPTIONAL DECORATION — measured, in Stage 8a.
 *     The default mode is fire-and-forget, and against the running container it
 *     was proven insufficient: a client reading one frame every 250 ms while
 *     the server produced ~244 frames/s received frames with strictly
 *     CONSECUTIVE sequence numbers, advancing 1 per read. The server had
 *     accepted 239 sends for 31 receives — roughly 208 stale frames sitting in
 *     uvicorn's write buffer and the socket buffer below it, being delivered in
 *     order, minutes behind the market.
 *
 *     `send_text()` returns when the frame reaches the transport, not when it
 *     reaches the client, so bounding the server's queue at one frame does not
 *     bound what the client sees. Credit mode does: the server hands over at
 *     most one unacknowledged frame, and this client acks once per RENDERED
 *     frame. Nothing can buffer, because nothing is sent until the pixels for
 *     the previous frame exist. Re-measured in the same container, the sequence
 *     advance per read went from 1 to a median of 34.
 *
 *     A consequence worth stating: when this tab is backgrounded the browser
 *     stops calling requestAnimationFrame, so the acks stop, so the server
 *     stops sending. The flow is render-driven end to end and a hidden tab
 *     costs the server nothing.
 *
 * DESIGN DECISION — the ack is sent AFTER the frame is drawn, not on receipt.
 *     Acking on receipt would make the credit window bound network delivery,
 *     which is not the thing that can fall behind here; the renderer is. Acking
 *     after the draw makes the window bound the whole client, which is what
 *     "the client is ready for another frame" is supposed to mean.
 *
 * DESIGN DECISION — one socket for data and control, rather than a socket plus
 * POSTs.
 *     Server-sent events would carry the frames perfectly well, but speed, seek
 *     and pause need a client-to-server path, and SSE downstream plus POST
 *     upstream is two transports, two failure modes and two reconnect stories
 *     where one suffices.
 */

import { applyFrame, applyBoundary } from "./state.js";

/** Backoff bounds. 250 ms is below human notice; 8 s keeps a dead server quiet. */
const BACKOFF_MIN_MS = 250;
const BACKOFF_MAX_MS = 8000;
const BACKOFF_FACTOR = 1.7;

export function createStream(state, options = {}) {
  const onStatusChange = options.onStatusChange || (() => {});
  const onServerError = options.onServerError || (() => {});

  let socket = null;
  let backoff = BACKOFF_MIN_MS;
  let pendingAcks = 0;
  let retryTimer = null;
  let closedByUs = false;

  function url() {
    const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${scheme}//${window.location.host}/ws/stream?flow=ack`;
  }

  function setConnection(value, detail) {
    if (state.connection === value) return;
    state.connection = value;
    onStatusChange(value, detail);
  }

  function connect() {
    closedByUs = false;
    setConnection(state.framesReceived > 0 ? "retrying" : "connecting");
    socket = new WebSocket(url());
    socket.onopen = handleOpen;
    socket.onmessage = handleMessage;
    socket.onclose = handleClose;
    // `onerror` carries no useful detail by design (the spec withholds it to
    // avoid leaking cross-origin information), so the close handler owns the
    // recovery and this one only stops the error reaching the console as an
    // unhandled event.
    socket.onerror = () => {};
  }

  function handleOpen() {
    backoff = BACKOFF_MIN_MS;
    pendingAcks = 0;
    setConnection("live");
    // The client's opinion of speed and pause is asserted on every connect,
    // including `pause: false`, and that is not redundant. The replay feed is a
    // single shared object — one producer serves every viewer, see
    // `serving/api.py:_produce` — so pause and speed are GLOBAL state that
    // outlives the connection that set them. A client that only sent `pause`
    // when it wanted to pause would load against a feed some previous viewer
    // had paused, show an un-pressed PAUSE button, and render a frozen tape
    // with no explanation. Asserting both makes the controls describe the feed
    // rather than only the last thing this tab did to it.
    //
    // The trade-off, stated because it is real: with two viewers the last one
    // to connect or to touch a control wins. A per-connection feed would fix
    // that and would run one model per viewer, which is the more expensive
    // wrong answer.
    send({ cmd: "speed", value: state.speed });
    send({ cmd: "pause", value: state.paused });
  }

  function handleMessage(event) {
    const message = JSON.parse(event.data);
    switch (message.type) {
      case "frame":
        applyFrame(state, message, performance.now());
        pendingAcks += 1;
        break;
      case "session_boundary":
        applyBoundary(state, message);
        // Boundaries travel on the same credit as frames, so they must be
        // acknowledged like frames. Acking only frames spends a credit that is
        // never returned, and after one boundary the stream stops dead.
        pendingAcks += 1;
        break;
      case "ack":
        // The server acknowledging a control command. Deliberately ignored:
        // the control is already showing the new state and re-applying it here
        // would make the UI stutter on every keypress.
        break;
      case "error":
        onServerError(message.detail);
        break;
      default:
        // An unknown message type is a version skew between this file and
        // serving/api.py, and silently dropping it is how that goes unnoticed.
        console.warn("unknown message type from /ws/stream:", message.type);
    }
  }

  function handleClose() {
    socket = null;
    pendingAcks = 0;
    if (closedByUs) return;
    setConnection("down");
    // Exponential backoff with jitter. The jitter matters even for one client:
    // a server that just restarted gets every open tab reconnecting on the same
    // schedule otherwise, and the thundering herd is self-inflicted.
    const delay = Math.min(BACKOFF_MAX_MS, backoff) * (0.75 + Math.random() * 0.5);
    backoff = Math.min(BACKOFF_MAX_MS, backoff * BACKOFF_FACTOR);
    retryTimer = window.setTimeout(connect, delay);
    setConnection("retrying", Math.round(delay));
  }

  function send(payload) {
    if (socket === null || socket.readyState !== WebSocket.OPEN) return false;
    socket.send(JSON.stringify(payload));
    return true;
  }

  return {
    connect,

    /**
     * Acknowledge the frames that have now been rendered.
     *
     * Exactly one ack per frame received, never more: the server's `grant()`
     * adds a credit per ack, so an extra ack would widen the window past one
     * and let the transport start buffering again — reintroducing, quietly, the
     * exact bug this mode exists to fix.
     */
    flushAcks() {
      while (pendingAcks > 0 && send({ cmd: "ack" })) {
        pendingAcks -= 1;
      }
    },

    setSpeed(value) {
      state.speed = value;
      send({ cmd: "speed", value });
    },

    setPaused(paused) {
      state.paused = paused;
      send({ cmd: "pause", value: paused });
    },

    seek(fraction) {
      send({ cmd: "seek", value: fraction });
    },

    close() {
      closedByUs = true;
      if (retryTimer !== null) window.clearTimeout(retryTimer);
      if (socket !== null) socket.close();
    },
  };
}
