/* system.js — the pipeline, this session's own numbers, and what was not measured.
 *
 * WHAT
 *     Three DOM panels. The pipeline lists every hop from exchange socket to
 *     this browser with the number that was measured at it. "This session"
 *     counts what THIS page has observed since it loaded. "Known gaps" lists
 *     what the project has not measured.
 *
 * WHY THIS PAGE EXISTS
 *     Because the two questions that follow a live demo are "what is actually
 *     running?" and "what are you not telling me?", and both deserve a screen
 *     rather than a sentence. Putting the gaps on the same page as the pipeline
 *     also makes a point that a footnote cannot: the unmeasured items are part
 *     of the architecture diagram, not a disclaimer appended to it.
 *
 * WHY IT IS DOM AND NOT CANVAS
 *     Nothing here is redrawn at frame rate — the pipeline and the gaps are
 *     static, and the session counters change meaningfully about twice a
 *     second. Canvas would buy nothing and would cost the two things that
 *     matter most for this content: the text would not be selectable, and it
 *     would not be in the accessibility tree. This is the page a reader is most
 *     likely to want to quote.
 *
 * WHERE THE NUMBERS COME FROM
 *     Every figure the API serves is read from the API — the model's accuracy
 *     and latency from /pareto, the IC from /stability, the economics from
 *     /economics — so this page cannot drift from the records the way a
 *     hand-maintained diagram would.
 *
 *     Two hops quote figures no endpoint serves: capture throughput and the
 *     storage comparison. Those are stated with the record they came from named
 *     in the text, which is the honest version of a number this page cannot
 *     verify for itself.
 */

import { microseconds, fixed, signed, duration } from "../format.js";

const UPDATE_INTERVAL_MS = 500;

export function createSystemPanel() {
  const pipeline = document.getElementById("pipeline-body");
  const health = document.getElementById("health-body");
  const gaps = document.getElementById("gaps-body");

  // Filled from the API as each record lands; the pipeline redraws when any of
  // them changes, which happens at most four times in the life of the page.
  const facts = { meta: null, pareto: null, stability: null, economics: null };
  let lastHealth = "";
  let lastUpdateAt = 0;

  renderGaps();
  renderPipeline();

  function setMeta(meta) {
    facts.meta = meta;
    renderPipeline();
  }

  function setPareto(payload) {
    facts.pareto = payload;
    renderPipeline();
  }

  function setStability(payload) {
    facts.stability = payload;
    renderPipeline();
  }

  function setEconomics(payload) {
    facts.economics = payload;
    renderPipeline();
  }

  function rowFor(variant) {
    if (facts.pareto === null) return null;
    return facts.pareto.rows.find((row) => row.variant === variant) || null;
  }

  function renderPipeline() {
    const serving = rowFor("student_int8");
    const cpp = rowFor("cpp_incremental");
    const cppFull = rowFor("cpp_full");
    const teacher = rowFor("pytorch_eager_fp32");

    const hops = [
      {
        name: "capture",
        detail:
          "Binance depth diffs + REST snapshot, book rebuilt and cross-checked " +
          "against the exchange every 60 s. <b>70.7 msgs/s</b>, <b>0 drops</b>, " +
          "<b>0 resyncs</b> over 283 s — the rate Binance sent, not a ceiling " +
          "(benchmarks/stage1_capture_20260727.md).",
      },
      {
        name: "tape",
        detail:
          "Fixed-width mmap-able records. <b>261×</b> faster vectorised reads " +
          "than JSON replay, and <span class='warn'>6.56× LARGER than gzipped " +
          "JSON</span> — the format lost on size and the record says so " +
          "(benchmarks/binfmt_20260727T162516Z.json).",
      },
      {
        name: "features",
        detail: facts.meta === null
          ? "40 features per snapshot, causal rolling z-score."
          : `40 features per snapshot, causal rolling z-score over ` +
            `<b>${facts.meta.engine.normalisation_lookback}</b> rows; a model input ` +
            `is <b>${facts.meta.engine.window_length}×40</b>, so ` +
            `<b>${facts.meta.engine.history_rows_required}</b> anchors are needed ` +
            `after every boundary.`,
      },
      {
        name: "model",
        detail: teacher === null
          ? "CNN + inception + TCN, trained from scratch."
          : `CNN + inception + TCN, <b>${teacher.params.toLocaleString()}</b> params, ` +
            `macro-F1 <b>${fixed(teacher.macro_f1, 4)}</b> — against ` +
            `<b>0.5317</b> for logistic regression on a single snapshot.`,
      },
      {
        name: "evaluation",
        detail: facts.stability === null || facts.economics === null
          ? "Per-block IC, permutation null, decay fit, cost-aware simulation."
          : `Per-block IC <b>${signed(facts.stability.mean, 3)}</b>, IR ` +
            `<b>${fixed(facts.stability.information_ratio, 2)}</b>, positive in ` +
            `<b>${Math.round(facts.stability.fraction_positive * facts.stability.block_ics.length)}` +
            ` of ${facts.stability.block_ics.length}</b> blocks. Gross ` +
            `<b>${signed(facts.economics.gross_bps_per_trade, 3)} bps</b> against a ` +
            `<b>${fixed(facts.economics.breakeven_fee_bps, 3)} bps/side</b> breakeven — ` +
            `<span class='warn'>${fixed(facts.economics.tiers[0].shortfall_multiple, 0)}× short</span>.`,
      },
      {
        name: "compression",
        detail: serving === null
          ? "ONNX export, int8 quantisation, distillation into a 32k student."
          : `ONNX export → int8 → a <b>${serving.params.toLocaleString()}</b>-param ` +
            `student at <b>${fixed(serving.size_kib, 0)} KiB</b>, ` +
            `<b>${microseconds(serving.p50_us)}</b> p50. Distillation measured ` +
            `<span class='warn'>no benefit against its own control</span>.`,
      },
      {
        name: "c++ forward pass",
        detail: cpp === null || cppFull === null
          ? "Hand-written, no BLAS, no libtorch, parity-checked against PyTorch."
          : `No BLAS, no libtorch. Parity <b>1000/1000</b> argmax vs PyTorch. Full ` +
            `recompute <b>${microseconds(cppFull.p50_us)}</b> → incremental ` +
            `<b>${microseconds(cpp.p50_us)}</b> p50: the win is the algorithm. ` +
            `<span class='warn'>Feature construction is not in this number.</span>`,
      },
      {
        name: "serving",
        detail:
          "FastAPI push socket, credit-based flow control, ONNX int8 in-process. " +
          "This page is the client: one rAF loop, preallocated ring buffers, " +
          "rendering at the display's refresh rate against a <b>4 ms</b> " +
          "per-frame budget.",
      },
    ];

    pipeline.innerHTML = hops
      .map(
        (hop) => `
        <div class="hop">
          <span class="hop-name">${hop.name}</span>
          <span class="hop-detail">${hop.detail}</span>
        </div>`
      )
      .join("");
  }

  function renderGaps() {
    // Deliberately hard-coded prose: these are statements about what does not
    // exist, so there is no record to read them from. Kept in step with the
    // README's Known gaps section by hand, and short enough to check.
    const items = [
      [
        "C++ feature construction",
        "TODO(measure)",
        "The 11 µs covers the forward pass from a prepared feature column. " +
          "Feature construction is still Python and its C++ cost has never been measured.",
      ],
      [
        "capture throughput ceiling",
        "not measured",
        "70.7 msgs/s is what the exchange sent, not what the capture path can absorb.",
      ],
      [
        "end-to-end serving latency under load",
        "not measured",
        "This harness measures service time in a closed loop. An open-loop harness " +
          "with a fixed arrival schedule is a different program.",
      ],
      [
        "live mode",
        "unexercised",
        "TTS_MODE=live and the capture compose profile have never been run against " +
          "a live exchange.",
      ],
      [
        "generalisation",
        "untested",
        "One venue, one symbol, three sessions on 2026-07-27. Nothing here shows " +
          "that any of it generalises.",
      ],
    ];

    gaps.innerHTML = items
      .map(
        ([name, status, detail]) => `
        <div class="row">
          <span class="row-name">${name}</span>
          <span class="shortfall">${status}</span>
          <span></span>
        </div>
        <div class="hop-detail" style="padding: 0 0 6px 0">${detail}</div>`
      )
      .join("");
  }

  function draw(state, now) {
    if (now - lastUpdateAt < UPDATE_INTERVAL_MS) return;
    lastUpdateAt = now;

    // `now` is the rAF timestamp, which is milliseconds since navigation start.
    const uptime = now / 1000;
    const summary = [
      ["frames delivered to this browser", state.framesReceived.toLocaleString(), ""],
      ["tape columns committed", state.columnsCommitted.toLocaleString(), ""],
      [
        "anchors skipped by the server",
        state.anchorsSkipped.toLocaleString(),
        `${fixed(state.skippedPerFrame, 1)} per frame`,
      ],
      ["serving forward pass, latest", microseconds(state.servingUs), ""],
      [
        "render p50 / max",
        `${fixed(state.renderP50, 2)} / ${fixed(state.renderMax, 2)} ms`,
        state.renderP50 > 4 ? "over budget" : "budget 4 ms",
      ],
      ["frame rate", `${fixed(state.fps, 1)} fps`, ""],
      ["page open for", duration(uptime), ""],
      ["socket", state.connection, "credit 1, ack per rendered frame"],
    ];

    const markup = summary
      .map(
        ([name, value, note]) => `
        <div class="row">
          <span class="row-name">${name}</span>
          <span class="row-strong">${value}</span>
          <span class="row-name">${note}</span>
        </div>`
      )
      .join("");

    // Compared before writing: a steady session rewrites nothing, so this panel
    // costs one string comparison per half second when nothing has changed.
    if (markup !== lastHealth) {
      health.innerHTML = markup;
      lastHealth = markup;
    }
  }

  return { draw, setMeta, setPareto, setStability, setEconomics };
}
