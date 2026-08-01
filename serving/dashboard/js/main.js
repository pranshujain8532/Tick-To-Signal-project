/* main.js — wiring, and nothing else.
 *
 * WHAT
 *     Builds the state, the socket, the eight panels and the one render loop,
 *     fetches the five static records, and connects. No drawing happens here
 *     and no data is interpreted here.
 *
 * WHY A SEPARATE ENTRY MODULE
 *     `render.js` owns the loop and `stream.js` owns the socket; if either also
 *     owned construction, the loop would import the socket or the socket would
 *     import the panels, and the dependency graph would have a cycle in it
 *     within a week. Everything that knows about everything else lives in this
 *     one file, which is therefore the only file worth reading to learn how the
 *     dashboard is put together.
 *
 * DEGRADING HONESTLY
 *     A record that will not load produces an empty state NAMING the file that
 *     would have populated it, never a placeholder number and never a spinner
 *     that spins forever. The panels that depend on live data go dim when the
 *     socket is down, because a still picture of a market is indistinguishable
 *     from a live one that stopped moving.
 */

import { readTokens } from "./canvas.js";
import { createState } from "./state.js";
import { createStream } from "./stream.js";
import { createRenderLoop } from "./render.js";
import { createHeaderPanel } from "./panels/header.js";
import { createTapePanel } from "./panels/tape.js";
import { createLadderPanel } from "./panels/ladder.js";
import { createSignalPanel } from "./panels/signal.js";
import { createStabilityPanel } from "./panels/stability.js";
import { createLatencyPanel } from "./panels/latency.js";
import { createParetoPanel } from "./panels/pareto.js";
import { createEconomicsPanel } from "./panels/economics.js";
import { createBootSequence } from "./panels/boot.js";

/**
 * What actually populates each endpoint, for the empty state.
 *
 * The file patterns are the ones `serving/records.py` globs. Naming them means
 * a reader looking at a blank panel knows what to run, rather than knowing only
 * that something is broken.
 */
const RECORD_SOURCES = {
  "/meta": "the running feed — check that the API process is up",
  "/pareto": "benchmarks/python_variants_*.json and benchmarks/cpp_*_stream.json",
  "/stability": "benchmarks/evaluation_*.json (Stage 5)",
  "/economics": "benchmarks/evaluation_*.json (Stage 5)",
  "/latency": "the running feed's own rolling measurements",
};

async function loadRecord(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path} answered ${response.status}`);
  return response.json();
}

function emptyStateText(path, error) {
  return `No data from ${path} (${error.message}). Populated by: ${RECORD_SOURCES[path]}.`;
}

function main() {
  const tokens = readTokens();
  const state = createState();

  const app = document.getElementById("app");
  const stream = createStream(state, {
    onStatusChange(status, detail) {
      // The attribute has exactly one writer, below, so that a gap and a
      // dropped socket cannot fight over it. This handler only logs.
      if (status === "retrying" && typeof detail === "number") {
        console.info(`socket lost; retrying in ${detail} ms`);
      }
    },
    onServerError(detail) {
      // The server refuses unknown commands by name rather than ignoring them,
      // so this is a real disagreement between client and server and belongs in
      // the console where it can be read, not swallowed.
      console.warn("server rejected a command:", detail);
    },
  });

  const header = createHeaderPanel(state, stream);
  const stability = createStabilityPanel(tokens);
  const latency = createLatencyPanel(tokens);
  const pareto = createParetoPanel(tokens);
  const economics = createEconomicsPanel();

  // Draw order is z-order for panels that overlap nothing, so it is chosen for
  // cost instead: the tape blits first while the rest of the frame budget is
  // still untouched.
  const panels = [
    createTapePanel(tokens),
    createLadderPanel(tokens),
    createSignalPanel(tokens),
    stability,
    latency,
    pareto,
    economics,
    header,
    createBootSequence(state),
  ];

  // The connection indicator: the one piece of app-wide chrome that no single
  // panel owns. It reconciles two independent reasons the screen must dim —
  // the socket being down, and the feed being inside a session gap — into the
  // one attribute the stylesheet reads. Written as a panel rather than a
  // timer so it lives on the same clock as everything else.
  panels.push({
    draw(state, now) {
      const dimmed = state.gapDimUntil > now;
      const wanted = dimmed ? "gap" : state.connection;
      if (app.dataset.connection !== wanted) app.dataset.connection = wanted;
    },
  });

  const loop = createRenderLoop(state, panels, stream);
  loop.start();
  stream.connect();

  loadRecord("/meta")
    .then((meta) => {
      state.meta = meta;
      header.setMeta(meta);
    })
    .catch((error) => {
      state.failures["/meta"] = emptyStateText("/meta", error);
      console.error(state.failures["/meta"]);
    });

  loadRecord("/stability")
    .then((payload) => {
      state.stability = payload;
      stability.setData(payload);
    })
    .catch((error) => stability.setUnavailable(emptyStateText("/stability", error)));

  loadRecord("/economics")
    .then((payload) => {
      state.economics = payload;
      economics.setData(payload);
    })
    .catch((error) => economics.setUnavailable(emptyStateText("/economics", error)));

  loadRecord("/pareto")
    .then((payload) => {
      state.pareto = payload;
      pareto.setData(payload);
    })
    .catch((error) => pareto.setUnavailable(emptyStateText("/pareto", error)));

  loadRecord("/latency")
    .then((payload) => {
      state.latencyRecord = payload;
      latency.setRecord(payload);
    })
    .catch((error) => console.error(emptyStateText("/latency", error)));

  // Space pauses, which is the one shortcut a viewer will try without being
  // told. Everything else is a real focusable control.
  window.addEventListener("keydown", (event) => {
    if (event.code !== "Space" || event.target !== document.body) return;
    event.preventDefault();
    document.getElementById("pause").click();
  });
}

main();
