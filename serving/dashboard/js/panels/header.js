/* header.js — the strip that says what you are looking at, and how fast.
 *
 * WHAT
 *     Symbol, feed mode, tape extent, the speed and scrub controls, the live
 *     serving latency, this client's own measurement of backpressure, and the
 *     render meter.
 *
 * WHY THE FEED-MODE BADGE IS THE LOUDEST THING UP HERE
 *     "DEMO REPLAY" and "LIVE" are different claims about the world. Everything
 *     else on this screen is identical in both modes, so the badge is the only
 *     thing standing between a viewer and the belief that they are watching an
 *     exchange. It is bordered in amber for replay and filled in coral for
 *     live, and the extent text beside it always states what is being replayed
 *     and at what multiple.
 *
 * WHY THE LATENCY READOUT IS LABELLED "SERVING (PYTHON / ONNX INT8)"
 *     Because a bare "latency" beside a project that also publishes an 11 µs
 *     C++ figure is an invitation to conflate them. This number is the ONNX
 *     int8 forward pass in the serving process, it is roughly 78x the C++
 *     figure, and the label carries the runtime so the two can never be read as
 *     the same measurement.
 *
 * WHY "ANCHORS SKIPPED / FRAME" IS HERE AT ALL
 *     It is this client's direct measurement of the server's backpressure
 *     policy, computed from gaps in the frame `seq` rather than reported by the
 *     server. `serving/api.py:ClientChannel` destroys a frame when a newer one
 *     arrives before the older has been written, because for a market view a
 *     stale frame is not merely late, it is wrong. The consequence is that this
 *     dashboard does not see every anchor, and the honest place to say so is on
 *     the dashboard, in a number, permanently.
 *
 * WHY THE DOM IS WRITTEN AT 4 Hz AND NOT 60 Hz
 *     Every write here changes text, which dirties layout. At 60 Hz that is
 *     eight layout invalidations per frame for numbers a human cannot read that
 *     fast anyway. Each field is also compared against its last value and
 *     skipped when unchanged, so a steady session writes nothing at all.
 */

import { microseconds, fixed, duration } from "../format.js";

const UPDATE_INTERVAL_MS = 250;
const RENDER_BUDGET_MS = 4;

export function createHeaderPanel(state, stream) {
  const elements = {
    symbol: document.getElementById("symbol"),
    mode: document.getElementById("feed-mode"),
    extent: document.getElementById("tape-extent"),
    servingP50: document.getElementById("serving-p50"),
    skipped: document.getElementById("skipped"),
    renderMs: document.getElementById("render-ms"),
    fps: document.getElementById("fps"),
    scrub: document.getElementById("scrub"),
    pause: document.getElementById("pause"),
  };
  const lastText = {};
  let lastUpdateAt = 0;
  // The scrub bar is an input the user drags; writing to it from the render
  // loop while they are dragging would fight them for control of it.
  let scrubbing = false;

  function setText(key, element, text) {
    if (lastText[key] === text) return;
    element.textContent = text;
    lastText[key] = text;
  }

  function setMeta(meta) {
    elements.symbol.textContent = meta.symbol;
    elements.mode.dataset.mode = meta.is_demo ? "demo" : "live";
    elements.mode.textContent = meta.is_demo ? "demo replay" : "live";
    state.tickSize = meta.tick_size;

    if (meta.is_demo) {
      const speed = meta.speed === null ? "" : ` · replayed at ${meta.speed}×`;
      elements.extent.textContent =
        `${meta.session_count} captured sessions · ${duration(meta.total_extent_s)}${speed}`;
    } else {
      // A live feed has no extent and no end. Reporting one would be a
      // fabrication, so the field says what it is instead.
      elements.extent.textContent = "live capture · no fixed extent";
      elements.scrub.disabled = true;
      elements.pause.disabled = true;
      for (const button of document.querySelectorAll("[data-speed]")) button.disabled = true;
    }
  }

  function draw(state, now) {
    if (now - lastUpdateAt < UPDATE_INTERVAL_MS) return;
    lastUpdateAt = now;

    setText("serving", elements.servingP50, microseconds(state.servingUs));
    setText("skipped", elements.skipped, fixed(state.skippedPerFrame, 1));
    setText(
      "render",
      elements.renderMs,
      `${fixed(state.renderP50, 2)} / ${fixed(state.renderMax, 2)} ms`
    );
    setText("fps", elements.fps, fixed(state.fps, 1));

    // No information by colour alone: the CSS turns the number amber-red AND
    // reveals an "over budget" tag beside it.
    elements.renderMs.classList.toggle("over-budget", state.renderP50 > RENDER_BUDGET_MS);

    if (!scrubbing && state.meta !== null && state.meta.is_demo) {
      // The scrub position is derived from the session the server says it is
      // in, which is the only position information the frame carries. It is an
      // approximation across sessions of unequal length and is not presented as
      // anything more precise than a position.
      const fraction = sessionFraction(state);
      if (fraction !== null) elements.scrub.value = String(Math.round(fraction * 1000));
    }
  }

  function sessionFraction(state) {
    if (state.sessionId === null || state.meta === null) return null;
    const sessions = state.meta.sessions;
    const name = state.sessionId.split("#")[0];
    const index = sessions.findIndex((session) => session.name === name);
    if (index === -1) return null;
    let elapsed = 0;
    for (let i = 0; i < index; i += 1) elapsed += sessions[i].extent_s;
    const within = sessions[index].anchors > 0 ? state.seq / sessions[index].anchors : 0;
    return Math.min(1, (elapsed + within * sessions[index].extent_s) / state.meta.total_extent_s);
  }

  // ------------------------------------------------------------- controls

  for (const button of document.querySelectorAll("[data-speed]")) {
    button.addEventListener("click", () => {
      const value = Number(button.dataset.speed);
      stream.setSpeed(value);
      for (const other of document.querySelectorAll("[data-speed]")) {
        other.setAttribute("aria-pressed", String(other === button));
      }
      // A speed change makes the tape's time axis discontinuous, so the tape
      // marks the column where it happened rather than silently changing scale.
      state.markPending = "speed change";
    });
  }

  elements.pause.addEventListener("click", () => {
    const paused = !state.paused;
    stream.setPaused(paused);
    elements.pause.setAttribute("aria-pressed", String(paused));
    elements.pause.textContent = paused ? "resume" : "pause";
  });

  elements.scrub.addEventListener("pointerdown", () => {
    scrubbing = true;
  });
  elements.scrub.addEventListener("pointerup", () => {
    scrubbing = false;
  });
  elements.scrub.addEventListener("change", () => {
    stream.seek(Number(elements.scrub.value) / 1000);
    scrubbing = false;
  });
  // Keyboard users get arrow-key seeking without a pointer ever being involved.
  elements.scrub.addEventListener("keyup", () => {
    stream.seek(Number(elements.scrub.value) / 1000);
  });

  return { draw, setMeta };
}
