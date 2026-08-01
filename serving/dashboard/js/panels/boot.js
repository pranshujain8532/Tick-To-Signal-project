/* boot.js — the one animation in this dashboard that is not data.
 *
 * WHAT
 *     Five pipeline stages — CAPTURE, TAPE, FEATURES, MODEL, SIGNAL — that
 *     resolve one by one as the thing each of them names actually arrives, over
 *     about 2.5 seconds, and then get out of the way.
 *
 * WHY IT EXISTS, GIVEN THE RULE THAT MOTION MUST REPRESENT DATA
 *     Because the first ten seconds decide what a viewer thinks this is. Coming
 *     up stage by stage frames it as a pipeline with an order — capture feeds a
 *     tape, the tape feeds features, features feed a model, the model produces
 *     a signal — rather than as a page of charts that happen to share a screen.
 *     That is a claim about the architecture, and it is a true one.
 *
 * WHY EACH STAGE RESOLVES ON REAL DATA AND NOT ON A TIMER
 *     A scripted sequence that always looks the same is a loading screen
 *     pretending to be a system. These resolve when the socket opens, when the
 *     first frame lands, when the model's own description arrives from /meta,
 *     and when the first signal is produced. If a stage does not resolve, it
 *     stays dim and the dashboard behind it shows why — which is more useful
 *     than a progress bar that always reaches 100%.
 *
 *     SIGNAL frequently does not resolve inside the window, and that is correct
 *     rather than a bug: 599 anchors of history are required after every
 *     boundary, which is about 2.4 s of wall clock at 10x, so a page loaded
 *     just after a session boundary genuinely has no signal yet.
 *
 * WHY IT NEVER REPLAYS ON RECONNECT
 *     A reconnect is not a boot. Replaying this on every socket blip would turn
 *     a piece of framing into an interruption, and would hide the thing the
 *     viewer actually needs to see, which is that the connection dropped.
 *
 * prefers-reduced-motion renders the final state instantly: every stage
 * resolved, overlay gone on the next frame.
 */

const STAGE_ORDER = ["capture", "tape", "features", "model", "signal"];
const TOTAL_MS = 2500;

export function createBootSequence(state) {
  const root = document.getElementById("boot");
  const elements = {};
  for (const name of STAGE_ORDER) {
    elements[name] = root.querySelector(`[data-stage="${name}"]`);
  }
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const startedAt = performance.now();
  let finished = false;

  function resolve(name, value) {
    const element = elements[name];
    if (element === undefined || element.dataset.resolved === "true") return;
    element.dataset.resolved = "true";
    element.querySelector(".boot-stage-value").textContent = value;
  }

  function finish() {
    if (finished) return;
    finished = true;
    root.hidden = true;
  }

  if (reducedMotion) {
    for (const name of STAGE_ORDER) resolve(name, "ready");
    finish();
  }

  // Any key, any click. The hint says so, and an escape hatch nobody can find
  // is not an escape hatch.
  window.addEventListener("keydown", finish, { once: true });
  root.addEventListener("click", finish, { once: true });

  return {
    draw(state, now) {
      if (finished) return;

      if (state.connection === "live") resolve("capture", state.meta === null ? "connected" : feedLabel(state));
      if (state.colCount > 0) resolve("tape", `${state.colCount} columns`);
      if (state.framesReceived > 0) resolve("features", `${state.warmupRequired} rows`);
      if (state.meta !== null) resolve("model", modelLabel(state));
      if (state.hasSignal) resolve("signal", state.score.toFixed(3));

      if (now - startedAt > TOTAL_MS) finish();
    },
  };
}

function feedLabel(state) {
  return state.meta.is_demo ? "demo replay" : "live";
}

function modelLabel(state) {
  const engine = state.meta.engine;
  return `${engine.window_length}×40 int8`;
}
