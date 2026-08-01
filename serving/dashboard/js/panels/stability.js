/* stability.js — the panel that decides whether this project is honest.
 *
 * WHAT
 *     Eighteen per-block information coefficients as bars around zero, the mean
 *     and ±1 sigma marked, the information ratio printed large, and how many
 *     blocks were positive. The pooled IC appears once, dimmed, labelled as not
 *     tradeable.
 *
 * WHY THIS PANEL IS NON-NEGOTIABLE, AND WHY IT IS SIZED LIKE THIS
 *     The pooled IC is +0.421. It is the largest, most flattering number this
 *     project can produce, and it answers a question nobody can trade: it is
 *     computed across the whole test block at once, which lets slow common
 *     drift inflate the correlation. The number that describes what a trader
 *     would actually experience is the per-block mean, +0.073, with an
 *     information ratio of 0.21 — an edge that is real (z = 4.25 against a
 *     200-trial null) and unstable (positive in 12 of 18 blocks).
 *
 *     If one number can be shown, it is 0.073. So 0.073 and the IR are drawn at
 *     the size of a headline, the eighteen blocks are drawn so the six negative
 *     ones are impossible to miss, and the pooled figure is dimmed contrast
 *     text with its correction attached in the markup.
 *
 * WHY THE BARS ARE NOT SORTED
 *     Sorting them would produce a tidy descending shape that reads as a
 *     distribution. In time order they read as what they are: a sequence, with
 *     runs of negative blocks that a trader would have lived through
 *     consecutively.
 *
 * WHY IT REDRAWS ONLY WHEN SOMETHING CHANGES
 *     Nothing here comes from the socket — it is one fetch of /stability, which
 *     reads a file. Redrawing static bars sixty times a second would spend the
 *     frame budget on a picture that cannot have changed. It repaints when the
 *     data arrives and when the panel is resized, and not otherwise.
 */

import { createSurface, hairline, labelText, numberText, wrappedText } from "../canvas.js";
import { signed, fixed } from "../format.js";

export function createStabilityPanel(tokens) {
  let dirty = true;
  const surface = createSurface(document.getElementById("stability-canvas"), {
    onResize: () => {
      dirty = true;
    },
  });
  const blocksLabel = document.getElementById("stability-blocks");
  const pooledLabel = document.getElementById("pooled-ic");

  let data = null;
  let failure = null;

  function setData(payload) {
    data = payload;
    dirty = true;
    const positive = Math.round(payload.fraction_positive * payload.block_ics.length);
    blocksLabel.textContent =
      `positive in ${positive} of ${payload.block_ics.length} blocks · ${payload.block_size} rows each`;
    pooledLabel.textContent = signed(payload.pooled_ic_not_tradeable, 3);
  }

  function setUnavailable(reason) {
    data = null;
    failure = reason;
    dirty = true;
  }

  function draw() {
    if (!dirty) return;
    dirty = false;

    const ctx = surface.ctx;
    const width = surface.width;
    const height = surface.height;
    ctx.clearRect(0, 0, width, height);

    if (data === null) {
      wrappedText(ctx, tokens, failure || "loading /stability…", 10, 20, width - 20);
      return;
    }

    // The headline. IR first because it is the number that says "this edge is
    // small relative to its own variability", which is the finding.
    numberText(ctx, tokens, fixed(data.information_ratio, 2), 10, 30, {
      size: 34,
      colour: tokens.text,
    });
    labelText(ctx, tokens, "information ratio", 10, 54, { colour: tokens.dim });

    numberText(ctx, tokens, signed(data.mean, 3), 130, 30, {
      size: 34,
      colour: data.mean > 0 ? tokens.bid : tokens.ask,
    });
    labelText(ctx, tokens, "per-block mean ic", 130, 54, { colour: tokens.dim });

    drawBlocks(ctx, width, height);
  }

  function drawBlocks(ctx, width, height) {
    const top = 72;
    const bottom = height - 14;
    const centre = (top + bottom) / 2;
    const half = (bottom - top) / 2;
    const left = 10;
    const usable = width - 20;
    const blocks = data.block_ics;
    const barWidth = Math.max(2, usable / blocks.length - 3);

    // Scaled by the largest absolute block IC so the worst block touches the
    // edge; the axis value is printed so the scale is not a mystery.
    let extreme = 0;
    for (let index = 0; index < blocks.length; index += 1) {
      extreme = Math.max(extreme, Math.abs(blocks[index]));
    }
    const scale = half / Math.max(extreme, 0.1);

    // ±1 sigma around the mean, drawn as a band. Its overlap with zero is the
    // point: a band that straddles zero is an edge you cannot rely on.
    const sigmaTop = centre - (data.mean + data.std) * scale;
    const sigmaBottom = centre - (data.mean - data.std) * scale;
    ctx.fillStyle = tokens.alpha.text(0.05);
    ctx.fillRect(left, sigmaTop, usable, sigmaBottom - sigmaTop);
    hairline(ctx, left, centre - data.mean * scale, usable, 0, tokens.alpha.text(0.55), surface.dpr);

    for (let index = 0; index < blocks.length; index += 1) {
      const value = blocks[index];
      const x = left + (usable / blocks.length) * index;
      const barHeight = Math.abs(value) * scale;
      ctx.fillStyle = value >= 0 ? tokens.alpha.bid(0.85) : tokens.alpha.ask(0.85);
      ctx.fillRect(x, value >= 0 ? centre - barHeight : centre, barWidth, barHeight);
    }

    hairline(ctx, left, centre, usable, 0, tokens.alpha.hairline(1), surface.dpr);
    labelText(ctx, tokens, `± ${fixed(extreme, 2)}`, left, top - 4, {
      colour: tokens.dim,
      upper: false,
    });
    labelText(ctx, tokens, "mean ± 1 sigma", left + 70, top - 4, {
      colour: tokens.alpha.text(0.55),
    });
    labelText(ctx, tokens, "blocks in time order →", width - 10, bottom + 8, {
      colour: tokens.dim,
      align: "right",
    });
  }

  return { draw, setData, setUnavailable };
}
