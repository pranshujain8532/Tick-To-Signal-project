/* main.js — wiring and routing, and nothing else.
 *
 * WHAT
 *     Builds the state, the socket, the panels and the one render loop; routes
 *     between the three pages; fetches the five static records. No drawing
 *     happens here and no data is interpreted here.
 *
 * WHY A SEPARATE ENTRY MODULE
 *     `render.js` owns the loop and `stream.js` owns the socket. If either also
 *     owned construction, the loop would import the socket or the socket would
 *     import the panels, and the dependency graph would have a cycle in it
 *     within a week. Everything that knows about everything else lives here,
 *     which makes this the only file worth reading to learn how the dashboard
 *     is put together.
 *
 * ROUTING, IN EIGHT LINES
 *     The URL hash is the router. `#evidence` shows the evidence page; anything
 *     unrecognised falls back to `live`. Rejected: a routing library, and a
 *     history-API router — both need a server that rewrites unknown paths back
 *     to index.html, and the server here is a StaticFiles mount whose whole
 *     appeal is that it does nothing clever.
 *
 *     A page that is not showing is `hidden`, so its canvases have zero size
 *     and the render loop skips its panels. The page switch is a performance
 *     boundary as well as a navigational one: the Live page's tape is the
 *     expensive surface, and it costs nothing while somebody is reading the
 *     Evidence page.
 *
 * DEGRADING HONESTLY
 *     A record that will not load produces an empty state NAMING the file that
 *     would have populated it — never a placeholder number, never a spinner
 *     that spins forever. Panels that depend on live data dim when the socket
 *     is down, because a still picture of a market is indistinguishable from a
 *     live one that stopped moving.
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
import { createSystemPanel } from "./panels/system.js";
import { createBootSequence } from "./panels/boot.js";

const PAGES = ["live", "evidence", "system"];

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
      if (status === "retrying" && typeof detail === "number") {
        console.info(`socket lost; retrying in ${detail} ms`);
      }
    },
    onServerError(detail) {
      // The server refuses unknown commands by name rather than ignoring them,
      // so this is a real disagreement between client and server and belongs in
      // the console rather than swallowed.
      console.warn("server rejected a command:", detail);
    },
  });

  const header = createHeaderPanel(state, stream);
  const stability = createStabilityPanel(tokens);
  const latency = createLatencyPanel(tokens);
  const pareto = createParetoPanel(tokens);
  const economics = createEconomicsPanel();
  const system = createSystemPanel();

  // `page: null` means chrome that is on every page. Everything else is drawn
  // only while its own page is showing.
  const panels = [
    { page: "live", panel: createTapePanel(tokens) },
    { page: "live", panel: createLadderPanel(tokens) },
    { page: "live", panel: createSignalPanel(tokens) },
    { page: "evidence", panel: stability },
    { page: "evidence", panel: latency },
    { page: "evidence", panel: pareto },
    { page: "evidence", panel: economics },
    { page: "system", panel: system },
    { page: null, panel: header },
    { page: null, panel: createBootSequence(state) },
    { page: null, panel: connectionIndicator(app) },
  ];

  // ------------------------------------------------------------- routing

  function showPage(name) {
    const page = PAGES.includes(name) ? name : "live";
    state.page = page;
    app.dataset.page = page;
    for (const candidate of PAGES) {
      document.getElementById(`page-${candidate}`).hidden = candidate !== page;
    }
    for (const tab of document.querySelectorAll(".tab")) {
      if (tab.dataset.page === page) {
        tab.setAttribute("aria-current", "page");
      } else {
        tab.removeAttribute("aria-current");
      }
    }
    if (window.location.hash.slice(1) !== page) {
      window.location.hash = page;
    }
  }

  for (const tab of document.querySelectorAll(".tab")) {
    tab.addEventListener("click", () => showPage(tab.dataset.page));
  }
  window.addEventListener("hashchange", () => showPage(window.location.hash.slice(1)));

  window.addEventListener("keydown", (event) => {
    if (event.target !== document.body) return;
    const index = ["Digit1", "Digit2", "Digit3"].indexOf(event.code);
    if (index !== -1) {
      showPage(PAGES[index]);
      return;
    }
    // Space pauses, which is the one shortcut a viewer will try untold.
    if (event.code === "Space") {
      event.preventDefault();
      document.getElementById("pause").click();
    }
  });

  showPage(window.location.hash.slice(1));

  const loop = createRenderLoop(state, panels, stream);
  loop.start();
  stream.connect();

  // ------------------------------------------------------- static records

  loadRecord("/meta")
    .then((meta) => {
      state.meta = meta;
      header.setMeta(meta);
      system.setMeta(meta);
    })
    .catch((error) => {
      state.failures["/meta"] = emptyStateText("/meta", error);
      console.error(state.failures["/meta"]);
    });

  loadRecord("/stability")
    .then((payload) => {
      state.stability = payload;
      stability.setData(payload);
      system.setStability(payload);
    })
    .catch((error) => stability.setUnavailable(emptyStateText("/stability", error)));

  loadRecord("/economics")
    .then((payload) => {
      state.economics = payload;
      economics.setData(payload);
      system.setEconomics(payload);
    })
    .catch((error) => economics.setUnavailable(emptyStateText("/economics", error)));

  loadRecord("/pareto")
    .then((payload) => {
      state.pareto = payload;
      pareto.setData(payload);
      system.setPareto(payload);
    })
    .catch((error) => pareto.setUnavailable(emptyStateText("/pareto", error)));

  loadRecord("/latency")
    .then((payload) => {
      state.latencyRecord = payload;
      latency.setRecord(payload);
    })
    .catch((error) => console.error(emptyStateText("/latency", error)));
}

/**
 * The one piece of chrome no panel owns: the connection attribute and its label.
 *
 * Written as a panel rather than as a socket callback so it lives on the same
 * clock as everything else, and so there is exactly one writer for the
 * attribute the stylesheet reads.
 */
function connectionIndicator(app) {
  const label = document.getElementById("connection-state");
  let shown = "";
  return {
    draw(state) {
      if (state.connection === shown) return;
      shown = state.connection;
      app.dataset.connection = shown;
      label.textContent = shown === "live" ? "live · flow=ack" : shown;
    },
  };
}

main();
