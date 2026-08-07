/* render.js — one requestAnimationFrame loop, and the meter that polices it.
 *
 * WHAT
 *     A single rAF loop that commits one tape column, asks every panel to draw
 *     the current state, records how long that took, and only then acks the
 *     socket. Panels do not schedule their own frames; there is exactly one
 *     loop in this dashboard.
 *
 * WHY ONE LOOP, FULLY DECOUPLED FROM MESSAGE ARRIVAL
 *     This is the same separation Stage 7a drew between ingestion and compute,
 *     and it is here for the same reason. If drawing were triggered by
 *     `onmessage`, the render rate would be the arrival rate: 25 fps when the
 *     book is quiet, 300 fps when it is busy, and every burst would spend the
 *     frame budget redrawing states nobody sees. Worse, a slow draw would
 *     become backpressure on the socket read, which is how a UI stops reading
 *     its own socket and falls behind.
 *
 *     Instead: the socket mutates state, the loop draws the latest state. The
 *     data loop's rate and the render loop's rate are independent, and the only
 *     coupling that remains is deliberate — the ack, sent after the draw, which
 *     makes the SERVER'S send rate render-driven.
 *
 * WHY THE BUDGET IS 4 ms AND WHY IT IS ON SCREEN
 *     A 60 fps frame is 16.7 ms, of which the compositor and the browser want
 *     their share; 4 ms of scripting leaves room for the machine to be doing
 *     something else. It is on screen because this project's entire thesis is
 *     that tail latency is measured rather than asserted, and a dashboard that
 *     claimed 60 fps without an instrument would be making exactly the kind of
 *     unmeasured claim the rest of the repository refuses to make. The meter
 *     reports p50 AND max over the last ~2 s, because a p50 of 1.2 ms with a
 *     max of 40 ms is a dropped frame somebody can see, and the average would
 *     hide it.
 */

import { commitColumn, pushRenderMs, percentiles } from "./state.js";

export function createRenderLoop(state, panels, stream) {
  // Preallocated, because the percentile pass runs four times a second and the
  // one thing this loop may not do is allocate. Reused across every call.
  const renderStats = { min: 0, max: 0, p50: 0, p99: 0, p999: 0, count: 0 };
  let handle = 0;
  let lastTimestamp = 0;
  let smoothedFrameMs = 16.7;
  let lastMeterAt = 0;

  state.renderP50 = 0;
  state.renderMax = 0;
  state.fps = 0;

  function frame(timestamp) {
    // Scheduled first, so an exception in a panel cannot stop the loop dead and
    // leave a frozen screen that still looks live.
    handle = window.requestAnimationFrame(frame);

    if (lastTimestamp !== 0) {
      const delta = timestamp - lastTimestamp;
      // Exponential mean rather than 1/delta: instantaneous fps from a single
      // frame is mostly noise, and a number that flickers between 58 and 62 is
      // read as instability rather than as measurement error.
      if (delta > 0 && delta < 1000) smoothedFrameMs = smoothedFrameMs * 0.9 + delta * 0.1;
    }
    lastTimestamp = timestamp;

    const started = performance.now();

    // One column per rendered frame, and only when a frame is actually
    // pending — see state.js. This returns the number of columns written, which
    // is what the tape blits by.
    const columns = commitColumn(state);

    // Only the visible page's panels draw. A hidden page's canvases have zero
    // size and its data has not changed in any way a viewer can see, so drawing
    // it would spend the frame budget on pixels nobody is looking at — and the
    // expensive surface in this dashboard, the depth tape, is on exactly one of
    // the three pages. Chrome panels carry `page: null` and always draw.
    for (let index = 0; index < panels.length; index += 1) {
      const entry = panels[index];
      if (entry.page !== null && entry.page !== state.page) continue;
      entry.panel.draw(state, timestamp, columns);
    }

    pushRenderMs(state, performance.now() - started);

    // AFTER the draw: the credit window bounds the whole client, not just the
    // network. See stream.js.
    stream.flushAcks();

    if (timestamp - lastMeterAt > 250) {
      lastMeterAt = timestamp;
      const stats = percentiles(state.renderMs, state.renderCount, state.renderScratch, renderStats);
      if (stats !== null) {
        state.renderP50 = stats.p50;
        state.renderMax = stats.max;
      }
      state.fps = 1000 / smoothedFrameMs;
    }
  }

  return {
    start() {
      if (handle === 0) handle = window.requestAnimationFrame(frame);
    },
    stop() {
      if (handle !== 0) window.cancelAnimationFrame(handle);
      handle = 0;
    },
  };
}
