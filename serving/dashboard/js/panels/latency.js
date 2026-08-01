/* latency.js — the live distribution of the forward pass this server runs.
 *
 * WHAT
 *     A histogram of `serving_infer_us` on a log x-axis, with vertical markers
 *     at p50, p99 and p99.9, built from the samples this client has received.
 *     The mean is deliberately absent.
 *
 * WHAT THIS NUMBER IS, EXACTLY
 *     The ONNX int8 forward pass inside the serving process, for one
 *     [1, 100, 40] input, measured with `perf_counter_ns` — and nothing else.
 *     It EXCLUDES feature construction, which is not small: measured on this
 *     machine at p50 ~1,600 µs against ~960 µs of inference, so the per-anchor
 *     cost is dominated by the part this panel does not show. That number is
 *     printed beneath the histogram rather than folded in, because folding it
 *     in would make this figure incomparable with the Stage 6 frontier that was
 *     measured the same way.
 *
 *     It is ALSO NOT the ~11 µs C++ figure on the Pareto panel. That is a
 *     different program, measured by a different harness, from an already
 *     prepared feature column. The header calls this readout "SERVING (Python /
 *     ONNX int8)" for that reason and never "latency".
 *
 * WHY LOG X
 *     p50 is around 1 ms and the max is around 12 ms. On a linear axis the
 *     entire body of the distribution occupies the leftmost tenth and the tail
 *     — the part that decides anything — is a flat line at zero height. Log x
 *     gives the tail the width it needs to be read.
 *
 * WHY NO MEAN
 *     A mean of 1.2 ms next to a p99.9 of 5 ms invites the reader to average
 *     away the thing that matters. A trade is not decided by the typical
 *     response; it is decided by the response you got. Tails, not averages.
 *
 * WHY IT REBINS AT 4 Hz AND NOT 60 Hz
 *     A distribution over 4,096 samples does not visibly change between two
 *     adjacent frames, and rebinning is the most expensive arithmetic in this
 *     dashboard: a sort of the sample ring plus a pass to fill the bins. Doing
 *     it four times a second costs a fortieth of doing it every frame and the
 *     picture is identical. Panels are allowed to have different natural
 *     cadences; what they are not allowed to do is have their own rAF loop.
 */

import { createSurface, hairline, labelText, numberText, wrappedText } from "../canvas.js";
import { percentiles } from "../state.js";
import { microseconds } from "../format.js";

const BINS = 48;
const REBIN_INTERVAL_MS = 250;

export function createLatencyPanel(tokens) {
  let dirty = true;
  const surface = createSurface(document.getElementById("latency-canvas"), {
    onResize: () => {
      dirty = true;
    },
  });
  const samplesLabel = document.getElementById("latency-samples");
  const relevanceNote = document.getElementById("relevance-note");

  // Everything the rebin needs, allocated once. The counts are integers and the
  // stats object is reused so the percentile pass produces no garbage.
  const counts = new Int32Array(BINS);
  const stats = { min: 0, max: 0, p50: 0, p99: 0, p999: 0, count: 0 };
  let hasStats = false;
  let lastRebinAt = 0;
  let logLow = 0;
  let logHigh = 1;
  let peak = 1;
  let featureNote = null;
  let lastSamplesText = "";

  /**
   * The measured note from /latency, rendered verbatim beside the histogram.
   *
   * The static sentence already in the markup says the same thing in fewer
   * words and survives a failed fetch; this replaces it with the server's own
   * prose when the server answers, so the two can never disagree about what
   * the current record says.
   */
  function setRecord(payload) {
    if (payload.relevance_note) relevanceNote.textContent = payload.relevance_note;
    if (payload.feature_construction) {
      featureNote = payload.feature_construction.p50_us;
    }
    dirty = true;
  }

  function rebin(state) {
    if (state.latencyCount === 0) {
      hasStats = false;
      return;
    }
    hasStats = percentiles(state.latency, state.latencyCount, state.latencyScratch, stats) !== null;
    if (!hasStats) return;

    // Bin edges span min..p99.9 in log space. Samples beyond p99.9 land in the
    // last bin rather than being dropped: an overflow that vanishes is how a
    // histogram quietly loses its own tail.
    logLow = Math.log10(Math.max(stats.min, 1e-3));
    logHigh = Math.log10(Math.max(stats.p999, stats.min * 1.5 + 1e-3));
    const span = Math.max(logHigh - logLow, 1e-6);

    counts.fill(0);
    for (let index = 0; index < state.latencyCount; index += 1) {
      const value = state.latency[index];
      if (value <= 0) continue;
      const position = (Math.log10(value) - logLow) / span;
      const bin = Math.min(BINS - 1, Math.max(0, Math.floor(position * BINS)));
      counts[bin] += 1;
    }
    peak = 1;
    for (let index = 0; index < BINS; index += 1) {
      if (counts[index] > peak) peak = counts[index];
    }
  }

  function draw(state, now) {
    if (now - lastRebinAt > REBIN_INTERVAL_MS) {
      lastRebinAt = now;
      rebin(state);
      dirty = true;
      const text = hasStats
        ? `${stats.count} live samples · window 4,096`
        : "no samples yet — the engine is warming";
      if (text !== lastSamplesText) {
        samplesLabel.textContent = text;
        lastSamplesText = text;
      }
    }
    if (!dirty) return;
    dirty = false;

    const ctx = surface.ctx;
    const width = surface.width;
    const height = surface.height;
    ctx.clearRect(0, 0, width, height);

    if (!hasStats) {
      wrappedText(ctx, tokens, "Waiting for the first inference of this session — the engine "
        + "discards its history at every boundary and needs 599 anchors.", 10, 20, width - 20);
      return;
    }

    const left = 10;
    const right = width - 10;
    const usable = right - left;
    // Two rows of headroom above the bars, because the percentile labels stack
    // into a second row when they crowd (see below) and must never sit on top
    // of the histogram they annotate.
    const top = 62;
    const bottom = height - 26;
    const span = Math.max(logHigh - logLow, 1e-6);

    const barWidth = usable / BINS;
    for (let index = 0; index < BINS; index += 1) {
      const barHeight = (counts[index] / peak) * (bottom - top);
      ctx.fillStyle = tokens.alpha.signal(0.55);
      ctx.fillRect(left + index * barWidth, bottom - barHeight, Math.max(1, barWidth - 1), barHeight);
    }
    hairline(ctx, left, bottom, usable, 0, tokens.hairline, surface.dpr);

    // p50, p99 and p99.9 are frequently within a few pixels of each other on a
    // log axis — on a well-behaved distribution p99 and p99.9 can differ by
    // less than 10% — so each label is stacked one row lower than the previous
    // when it would otherwise be drawn through it. Two percentile labels
    // overprinted is not a cosmetic problem: it produces a number on screen
    // that is neither of them.
    const markers = [
      { value: stats.p50, text: "p50" },
      { value: stats.p99, text: "p99" },
      { value: stats.p999, text: "p99.9" },
    ];
    let previousX = -Infinity;
    let row = 0;
    for (const marker of markers) {
      const position = (Math.log10(marker.value) - logLow) / span;
      const x = left + Math.max(0, Math.min(1, position)) * usable;
      // Alternating between exactly two rows, rather than stacking without
      // limit: three crowded markers would otherwise put the third row on top
      // of the bars.
      //
      // The threshold is 96 px because a label is a name plus a value — about
      // 50 px — and the rightmost one is flipped to the left of its line. Two
      // labels 89 px apart, one growing right and one growing left, still
      // collided at 64 px and printed a value that was neither percentile.
      row = x - previousX < 96 ? (row + 1) % 2 : 0;
      previousX = x;
      const labelY = top - 40 + row * 24;

      hairline(ctx, x, labelY + 4, 0, bottom - labelY - 4, tokens.alpha.text(0.45), surface.dpr);
      // The rightmost marker is at the right edge by definition, so its label
      // flips inward: a marker whose value is clipped off the canvas looks like
      // a line with no number.
      const flip = x > right - 62;
      const textX = flip ? x - 4 : x + 3;
      const align = flip ? "right" : "left";
      labelText(ctx, tokens, marker.text, textX, labelY, {
        colour: tokens.alpha.text(0.7),
        align,
        upper: false,
      });
      numberText(ctx, tokens, microseconds(marker.value), textX, labelY + 12, {
        size: 11,
        colour: tokens.text,
        align,
      });
    }

    labelText(ctx, tokens, `${microseconds(stats.min)} · log µs`, left, bottom + 12, {
      colour: tokens.dim,
      upper: false,
    });
    labelText(ctx, tokens, microseconds(stats.max), right, bottom + 12, {
      colour: tokens.dim,
      align: "right",
      upper: false,
    });

    // The boundary, on the panel itself and not only in the note below it.
    const boundary = featureNote === null
      ? "forward pass only — feature construction excluded"
      : `forward pass only — feature construction is a further ${microseconds(featureNote)} p50, excluded`;
    labelText(ctx, tokens, boundary, left, 10, { colour: tokens.dim, upper: false });
  }

  return { draw, setRecord };
}
