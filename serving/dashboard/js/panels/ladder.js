/* ladder.js — the price ladder: ten levels a side, and the spread between them.
 *
 * WHAT
 *     Twenty rows of price, resting size and cumulative depth, asks descending
 *     to the touch and bids descending away from it, with the spread in the
 *     gutter between. A level flashes when its size changes and decays over
 *     ~120 ms.
 *
 * WHY THE WHOLE PANEL IS ONE CANVAS AND NOT TWENTY DOM ROWS
 *     Every one of these numbers changes up to sixty times a second. Twenty
 *     rows of three text nodes plus a width-animated bar is sixty DOM writes
 *     per frame, each of which dirties layout for the whole panel; the flash
 *     decay would add twenty style mutations on top. Canvas draws the same
 *     information in about sixty primitives with no layout at all, and the
 *     decay is arithmetic on a timestamp rather than a CSS transition the
 *     browser has to track per element.
 *
 *     The cost of that choice is real and worth naming: canvas text is not
 *     selectable and not in the accessibility tree. It is paid for here by the
 *     panel carrying an aria-label, by the numbers also being available as JSON
 *     from the same server, and by nothing in this panel being the only place a
 *     fact appears.
 *
 * WHY THE FLASH IS ON SIZE AND NOT ON PRICE
 *     A price change moves the whole ladder, so flashing on price would flash
 *     all twenty rows at once and mean nothing. What a trader is watching for
 *     is size appearing or leaving a level that is still there, which is what
 *     `state.js:applyFrame` timestamps.
 *
 * THE SPREAD IN BPS IS NOT A TYPO
 *     One 0.01 USDT tick on a ~64,973 USDT mid is 0.0015 bps. Stage 5 caught
 *     this as a 100x error — it had been reported as 0.15 bps — and the
 *     corrected scale is what makes the economics panel's numbers add up, since
 *     the median spread is the cost of crossing it. Four decimals are shown for
 *     that reason: two would render the entire column as 0.00.
 */

import { createSurface, hairline, labelText, numberText } from "../canvas.js";
import { DEPTH } from "../state.js";
import { price as formatPrice, size as formatSize, bps as formatBps } from "../format.js";

/** How long a changed level stays lit. Long enough to see, short enough to read. */
const FLASH_MS = 120;
/** The gutter between the two sides, in CSS pixels. */
const GUTTER_HEIGHT = 34;

export function createLadderPanel(tokens) {
  const surface = createSurface(document.getElementById("ladder-canvas"));
  // Read once at construction. A user who changes the OS setting mid-session
  // gets it on reload, which is the same contract the boot sequence has.
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function draw(state, now) {
    const ctx = surface.ctx;
    const width = surface.width;
    const height = surface.height;
    ctx.clearRect(0, 0, width, height);

    const rowsPerSide = DEPTH;
    const available = height - GUTTER_HEIGHT;
    const rowHeight = available / (rowsPerSide * 2);
    const barLeft = width - 74;
    const barWidth = 62;

    // Cumulative depth is normalised against the deeper side, so the two halves
    // of the ladder are comparable to each other rather than each to itself —
    // an imbalanced book must LOOK imbalanced.
    let bidTotal = 0;
    let askTotal = 0;
    for (let level = 0; level < state.bidLevels; level += 1) bidTotal += state.bidSize[level];
    for (let level = 0; level < state.askLevels; level += 1) askTotal += state.askSize[level];
    const scale = Math.max(bidTotal, askTotal, 1e-9);

    // Asks are drawn worst-price-first so the touch sits against the gutter and
    // the two best prices are adjacent, which is how a ladder is read.
    let cumulative = askTotal;
    for (let level = state.askLevels - 1; level >= 0; level -= 1) {
      const y = (state.askLevels - 1 - level) * rowHeight;
      drawRow(ctx, state, {
        y,
        rowHeight,
        price: state.askPrice[level],
        size: state.askSize[level],
        cumulative,
        scale,
        colour: tokens.ask,
        alpha: tokens.alpha.ask,
        flashAt: state.askFlashAt[level],
        now,
        barLeft,
        barWidth,
        width,
      });
      cumulative -= state.askSize[level];
    }

    drawGutter(ctx, state, rowsPerSide * rowHeight, width);

    cumulative = 0;
    for (let level = 0; level < state.bidLevels; level += 1) {
      cumulative += state.bidSize[level];
      const y = rowsPerSide * rowHeight + GUTTER_HEIGHT + level * rowHeight;
      drawRow(ctx, state, {
        y,
        rowHeight,
        price: state.bidPrice[level],
        size: state.bidSize[level],
        cumulative,
        scale,
        colour: tokens.bid,
        alpha: tokens.alpha.bid,
        flashAt: state.bidFlashAt[level],
        now,
        barLeft,
        barWidth,
        width,
      });
    }
  }

  function drawRow(ctx, state, row) {
    const centre = row.y + row.rowHeight / 2;
    if (!Number.isFinite(row.price)) return;

    // The flash decays linearly to nothing. Under prefers-reduced-motion the
    // decay is skipped entirely and the row simply never lights: the flash is
    // the one piece of motion here that is decoration rather than data, since
    // the size number itself already changed.
    //
    // It lights the SIZE CELL and not the whole row. Full-width rows were the
    // first version and they were useless: on this book every level's size
    // changes almost every frame, so twenty full-width flashes at 60 Hz read as
    // a solid coloured wash rather than as twenty separate events.
    const age = row.now - row.flashAt;
    if (age >= 0 && age < FLASH_MS && !reducedMotion) {
      ctx.fillStyle = row.alpha(0.3 * (1 - age / FLASH_MS));
      ctx.fillRect(row.barLeft - 66, row.y + 1, 62, row.rowHeight - 2);
    }

    // Cumulative depth, drawn as a bar that grows leftward from the right edge
    // so the two sides mirror around the gutter.
    const fraction = Math.min(1, row.cumulative / row.scale);
    ctx.fillStyle = row.alpha(0.3);
    ctx.fillRect(row.barLeft, centre - 3, row.barWidth * fraction, 6);

    numberText(ctx, tokens, formatPrice(row.price), 10, centre, {
      size: 12,
      colour: row.colour,
    });
    numberText(ctx, tokens, formatSize(row.size), row.barLeft - 10, centre, {
      size: 12,
      align: "right",
      colour: tokens.text,
    });
  }

  /**
   * The centre gutter: spread in ticks and in bps, plus the imbalance.
   *
   * Spread is served by the API as `spread_ticks`, computed from the tick size
   * the feed MEASURED rather than one hardcoded here — so this panel cannot be
   * silently wrong if the demo is pointed at another symbol.
   */
  function drawGutter(ctx, state, y, width) {
    hairline(ctx, 0, y, width, 0, tokens.hairline, surface.dpr);
    hairline(ctx, 0, y + GUTTER_HEIGHT, width, 0, tokens.hairline, surface.dpr);

    const centre = y + GUTTER_HEIGHT / 2;
    labelText(ctx, tokens, "spread", 10, centre - 8, { colour: tokens.dim });

    const ticks = Number.isFinite(state.spreadTicks) ? state.spreadTicks : NaN;
    const spreadBps = Number.isFinite(ticks) && Number.isFinite(state.mid)
      ? ((ticks * state.tickSize) / state.mid) * 10000
      : NaN;

    numberText(
      ctx,
      tokens,
      Number.isFinite(ticks) ? `${ticks.toFixed(0)} tick${ticks === 1 ? "" : "s"}` : "—",
      10,
      centre + 7,
      { size: 12, colour: tokens.text }
    );
    numberText(ctx, tokens, `${formatBps(spreadBps)} bps`, width - 10, centre + 7, {
      size: 12,
      align: "right",
      colour: tokens.tape,
    });
    labelText(ctx, tokens, "0.01 usdt tick", width - 10, centre - 8, {
      colour: tokens.dim,
      align: "right",
    });
  }

  return { draw };
}
