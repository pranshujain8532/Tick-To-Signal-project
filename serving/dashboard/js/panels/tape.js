/* tape.js — the depth tape and the mid-drift strip. The signature panel.
 *
 * WHAT
 *     A continuously scrolling picture of where liquidity rests: x is time, y
 *     is distance from the mid, intensity is resting size. Beneath it, a thin
 *     strip showing where the mid itself actually went.
 *
 * WHY THE Y-AXIS IS DISTANCE FROM MID AND NOT PRICE — the measurement that
 * forces it.
 *     Stage 3 measured P(mid unchanged between adjacent snapshots) = 0.991 to
 *     0.994 on this book, with a 0.01 USDT tick against a ~64,973 USDT mid. On
 *     an absolute-price axis the entire ten-level book spans a few hundredths
 *     of a percent of the price, every level lands in the same pixel row, and
 *     the "tape" is a flat line. The subject of this panel is book SHAPE —
 *     where size rests, where it thickens, where it evaporates — so the mid is
 *     a straight centreline by construction and everything is drawn relative
 *     to it.
 *
 * WHY THE AXIS IS LOG-SPACED IN TICKS AND NOT LINEAR ±20 — measured, and this
 * corrected the plan.
 *     A linear ±20 tick axis was the intent, and the three committed tapes say
 *     it would show almost nothing. Level 10 rests at a MEDIAN 18-56 ticks from
 *     the mid depending on the session, p99 260-320 ticks, max 400; not one
 *     snapshot in any tape has its tenth level inside 20 ticks. A ±20 window
 *     would render levels 1-2 and clip the other eight off-screen entirely.
 *
 *     Auto-ranging linearly to ~±300 instead fails the other way: the touch,
 *     which is where every interesting change happens, would collapse into
 *     three pixels around the centreline. The book is sparse and roughly
 *     geometric in level spacing, so the axis is signed log1p(ticks) — the same
 *     log1p intuition Stage 3 used for sizes, applied to distance. Gridlines
 *     are labelled with real tick values (1, 10, 100) so the compression is
 *     visible and stated rather than silently flattering.
 *
 * WHY THE TAPE BLITS AND THE DRIFT STRIP REDRAWS — a cost comparison, not taste.
 *     The tape holds ~1,000 columns of 20 levels. Redrawing history every frame
 *     is ~20,000 fillRect calls at 60 fps, which is 1.2 M rectangles a second
 *     and will not fit in a 4 ms budget. Scrolling by blitting the canvas onto
 *     itself one column left and drawing only the newest column is one
 *     drawImage plus ~20 rectangles: three orders of magnitude less work, and
 *     it is exactly correct because a column, once drawn, never changes.
 *
 *     The drift strip cannot use the same trick, and the reason is instructive.
 *     Its y-range auto-ranges to the mid, and the mid moves — so a blitted
 *     history would be pixels drawn at a scale that no longer applies, which is
 *     a quietly wrong chart rather than a slow one. It is one polyline of ~1,000
 *     points per frame, which is affordable. Where history can go stale, redraw;
 *     where it cannot, blit.
 *
 *     The same argument is why a y-range change or a resize triggers a full
 *     repaint from the column ring in state.js rather than leaving the old
 *     pixels in place.
 *
 * REJECTED ALTERNATIVE — two ping-pong offscreen canvases for the scroll.
 *     Blitting a canvas onto itself is defined behaviour (the source is the
 *     bitmap as it was when drawImage was called), so the second buffer would
 *     double the memory of the largest surface on screen to solve a problem the
 *     specification already solves.
 */

import { createSurface, hairline, labelText, numberText } from "../canvas.js";
import { DEPTH, FLAG_GAP, FLAG_WARMUP, FLAG_MARK, columnIndex } from "../state.js";

/** Axis bounds in ticks. Floor covers the touch; ceiling covers the measured max. */
const MIN_RANGE_TICKS = 32;
const MAX_RANGE_TICKS = 512;
/** A level is at least this visible, so "there is size here" never disappears. */
const MIN_LEVEL_ALPHA = 0.16;
/**
 * Alpha is quantised to this many steps and the fill strings are built once.
 *
 * MEASURED, and this is the Stage 7b lesson one language up. The first version
 * computed `rgba(...)` per rectangle, which meant a full repaint allocated
 * ~21,000 strings and made the 2D context re-parse a colour for every one of
 * them: repaints were measured at 60-128 ms, or eight dropped frames. Hoisting
 * the colour lookup out of the loop is exactly the change that moved Stage 7b's
 * p99.99 by 2.4x while leaving p50 untouched — rare, enormous costs, removed by
 * not doing work in a loop that could be done once. Sixteen steps is below what
 * the eye resolves in an intensity ramp.
 */
const ALPHA_STEPS = 16;
/** Half-height of the ribbon at |score| = 1, in device pixels before dpr. */
const RIBBON_MAX_HALF = 9;
/** At most one full repaint per this many ms: it is the one expensive frame. */
const REPAINT_COOLDOWN_MS = 5000;

export function createTapePanel(tokens) {
  // Declared before the surfaces, and that ordering is load-bearing:
  // `createSurface` sizes the canvas immediately and therefore fires onResize
  // synchronously, inside this constructor. A `let` declared below would still
  // be in its temporal dead zone at that moment.
  let rangeTicks = 64;
  let sizeReference = Math.log1p(8); // ~p99 resting size in BTC, then auto-ranged
  let offsetEma = 0; // smoothed outermost level, in ticks; drives the y-range
  let needsRepaint = true;
  let needsOverlay = true;
  let lastRepaintAt = 0;
  let bannerUntil = 0;
  let lastAxisText = "";

  const scroll = createSurface(document.getElementById("tape-scroll"), {
    deviceSpace: true,
    onResize: () => {
      needsRepaint = true;
      needsOverlay = true;
    },
  });
  const overlay = createSurface(document.getElementById("tape-overlay"), {
    onResize: () => {
      needsOverlay = true;
    },
  });
  const drift = createSurface(document.getElementById("drift-canvas"));
  const banner = document.getElementById("tape-banner");
  const axisNote = document.getElementById("tape-extent-note");

  // Every fill string this panel can ever use, built once at construction.
  const bidFills = buildRamp(tokens.alpha.bid, MIN_LEVEL_ALPHA, 1);
  const askFills = buildRamp(tokens.alpha.ask, MIN_LEVEL_ALPHA, 1);
  // The ribbon stops short of opaque so the touch levels and the mid line stay
  // visible underneath it. The model's opinion sits ON the market here; it is
  // not allowed to replace it.
  const ribbonFills = buildRamp(tokens.alpha.signal, 0.12, 0.8);
  const markFill = tokens.alpha.dim(0.55);
  // The last colour handed to the context, so a run of levels at the same
  // intensity costs one assignment instead of one per rectangle.
  let lastFill = "";

  /** Device pixels one column advances per commit. Integer, always: see canvas.js. */
  function step() {
    return Math.max(1, Math.round(scroll.dpr));
  }

  /**
   * Screen y for an offset from mid, in the signed-log tick axis.
   * `side` is +1 for asks (above the centreline) and -1 for bids (below).
   */
  function yForOffset(offsetTicks, side, centre, half, logRange) {
    const magnitude = Math.log1p(Math.abs(offsetTicks)) / logRange;
    return centre - side * Math.min(1, magnitude) * half;
  }

  function drawColumn(state, ring, x, width) {
    const ctx = scroll.ctx;
    const height = scroll.height;
    const centre = height / 2;
    const half = centre - 2 * scroll.dpr;
    const logRange = Math.log1p(rangeTicks);
    const flags = state.colFlags[ring];
    const blockHeight = Math.max(2, Math.round(2.5 * scroll.dpr));

    // A gap is drawn as nothing at all. Not a dimmed column, not an
    // interpolation, not a line joining the two sessions — the market either
    // side of a resync is not comparable and the picture says so by being empty.
    setFill(ctx, tokens.panel);
    ctx.fillRect(x, 0, width, height);
    if (flags & FLAG_GAP) return;

    const base = ring * DEPTH;
    for (let level = 0; level < DEPTH; level += 1) {
      const bidOffset = state.colBidOffset[base + level];
      if (!Number.isNaN(bidOffset)) {
        const y = yForOffset(bidOffset, -1, centre, half, logRange);
        setFill(ctx, bidFills[intensityStep(state.colBidSize[base + level])]);
        ctx.fillRect(x, y - blockHeight / 2, width, blockHeight);
      }
      const askOffset = state.colAskOffset[base + level];
      if (!Number.isNaN(askOffset)) {
        const y = yForOffset(askOffset, 1, centre, half, logRange);
        setFill(ctx, askFills[intensityStep(state.colAskSize[base + level])]);
        ctx.fillRect(x, y - blockHeight / 2, width, blockHeight);
      }
    }

    // The signal ribbon rides the centreline: thickness is |score|, opacity is
    // the model's confidence in whichever class it picked. It is drawn only
    // when there IS a signal — during the 599-anchor warmup after a boundary
    // the centreline is bare, which is the honest picture of "no opinion yet".
    if (!(flags & FLAG_WARMUP)) {
      const magnitude = Math.min(1, Math.abs(state.colScore[ring]));
      const thickness = Math.max(1, magnitude * RIBBON_MAX_HALF * scroll.dpr);
      const confidence = state.colConfidence[ring];
      const step = Math.min(ALPHA_STEPS - 1, Math.round(confidence * (ALPHA_STEPS - 1)));
      setFill(ctx, ribbonFills[step]);
      ctx.fillRect(x, centre - thickness, width, thickness * 2);
    }

    // A boundary or a speed change gets a full-height hairline, because the
    // axis is discontinuous there and an unmarked discontinuity is a lie about
    // continuity.
    if (flags & FLAG_MARK) {
      setFill(ctx, markFill);
      ctx.fillRect(x, 0, Math.max(1, scroll.dpr), height);
    }
  }

  /** Assign a fill only when it actually changes; see ALPHA_STEPS. */
  function setFill(ctx, fill) {
    if (fill === lastFill) return;
    ctx.fillStyle = fill;
    lastFill = fill;
  }

  function buildRamp(colour, lowest, highest) {
    const ramp = new Array(ALPHA_STEPS);
    for (let step = 0; step < ALPHA_STEPS; step += 1) {
      const fraction = step / (ALPHA_STEPS - 1);
      ramp[step] = colour(lowest + (highest - lowest) * fraction);
    }
    return ramp;
  }

  function intensityStep(logSize) {
    if (logSize <= 0) return 0;
    const scaled = Math.min(1, logSize / sizeReference);
    return Math.round(scaled * (ALPHA_STEPS - 1));
  }

  /** Redraw every column the ring holds. The one expensive frame, rate-limited. */
  function repaint(state) {
    const ctx = scroll.ctx;
    setFill(ctx, tokens.panel);
    ctx.fillRect(0, 0, scroll.width, scroll.height);
    const columnWidth = step();
    const visible = Math.min(state.colCount, Math.floor(scroll.width / columnWidth));
    for (let back = 0; back < visible; back += 1) {
      const x = scroll.width - (back + 1) * columnWidth;
      drawColumn(state, columnIndex(state, back), x, columnWidth);
    }
    needsRepaint = false;
  }

  /** Track the axis range and the intensity reference against what is arriving. */
  function updateRanges(state, now) {
    if (state.colCount === 0) return;
    const base = state.colHead * DEPTH;
    let maxOffset = 0;
    let maxSize = 0;
    for (let level = 0; level < DEPTH; level += 1) {
      const bid = state.colBidOffset[base + level];
      const ask = state.colAskOffset[base + level];
      if (!Number.isNaN(bid) && bid > maxOffset) maxOffset = bid;
      if (!Number.isNaN(ask) && ask > maxOffset) maxOffset = ask;
      if (state.colBidSize[base + level] > maxSize) maxSize = state.colBidSize[base + level];
      if (state.colAskSize[base + level] > maxSize) maxSize = state.colAskSize[base + level];
    }
    // Decay slowly upward and even more slowly downward, so a single unusually
    // deep book does not rescale the whole panel and then rescale it back.
    sizeReference = Math.max(sizeReference * 0.999, maxSize * 0.9 + sizeReference * 0.1);

    // The axis follows a SMOOTHED outermost level, not the instantaneous one.
    // Measured on these tapes the tenth level wanders between roughly 18 and
    // 400 ticks from the mid within a single session, so a range driven by the
    // current column's maximum crosses a threshold every few seconds — and each
    // crossing costs a full repaint. Averaging first is what turns rescaling
    // from a recurring cost into a startup transient.
    offsetEma = offsetEma === 0 ? maxOffset : offsetEma * 0.99 + maxOffset * 0.01;

    const wanted = clampRange(offsetEma * 1.6);
    // Hysteresis plus a cooldown on top of the smoothing. A rescale is the one
    // frame this dashboard cannot draw inside its budget, so the bands are set
    // so that once the range has settled on this book it stops moving: only a
    // genuine regime change in book depth shifts it again.
    const tooSmall = wanted > rangeTicks;
    const tooLarge = wanted < rangeTicks * 0.4;
    if ((tooSmall || tooLarge) && now - lastRepaintAt > REPAINT_COOLDOWN_MS) {
      rangeTicks = wanted;
      lastRepaintAt = now;
      needsRepaint = true;
      needsOverlay = true;
    }
  }

  function clampRange(ticks) {
    if (!Number.isFinite(ticks)) return MIN_RANGE_TICKS;
    // Snap to a 1-2-5 ladder so the axis labels are round numbers rather than
    // whatever the last book happened to be.
    const nice = [32, 64, 128, 256, 512];
    for (let index = 0; index < nice.length; index += 1) {
      if (ticks <= nice[index]) return nice[index];
    }
    return MAX_RANGE_TICKS;
  }

  /** Centreline, gridlines and labels. Redrawn only when the axis changes. */
  function drawOverlay(state) {
    const ctx = overlay.ctx;
    const width = overlay.width;
    const height = overlay.height;
    const centre = height / 2;
    const half = centre - 2;
    const logRange = Math.log1p(rangeTicks);
    ctx.clearRect(0, 0, width, height);

    for (const ticks of [1, 10, 100]) {
      if (ticks > rangeTicks) continue;
      const offset = (Math.log1p(ticks) / logRange) * half;
      for (const side of [-1, 1]) {
        const y = centre - side * offset;
        hairline(ctx, 0, y, width, 0, tokens.alpha.hairline(0.9), overlay.dpr);
        labelText(ctx, tokens, `${ticks}`, 6, y - 7, { colour: tokens.alpha.dim(0.75) });
      }
    }

    // The mid is a straight line by construction, and it is amber because it is
    // the market. Everything blue on this panel is the model.
    hairline(ctx, 0, centre, width, 0, tokens.alpha.tape(0.5), overlay.dpr);
    labelText(ctx, tokens, "mid", 6, centre - 8, { colour: tokens.alpha.tape(0.8) });
    labelText(ctx, tokens, "asks", width - 34, 12, { colour: tokens.alpha.ask(0.9) });
    labelText(ctx, tokens, "bids", width - 34, height - 12, { colour: tokens.alpha.bid(0.9) });
    // Bottom-left, because the top-left corner is where the outermost gridline
    // label lands and the two were drawing over each other.
    labelText(
      ctx,
      tokens,
      `± ${rangeTicks} ticks · log`,
      6,
      height - 12,
      { colour: tokens.dim, upper: false }
    );
    needsOverlay = false;
  }

  /**
   * The mid-drift strip: absolute price, its own auto-range, same time axis.
   *
   * A gap column has a NaN mid and breaks the path rather than being skipped,
   * so two sessions are never joined by a line across market nobody observed.
   */
  function drawDrift(state) {
    const ctx = drift.ctx;
    const width = drift.width;
    const height = drift.height;
    ctx.clearRect(0, 0, width, height);
    if (state.colCount === 0) return;

    const columnWidth = Math.max(1, Math.round(drift.dpr)) / drift.dpr;
    const visible = Math.min(state.colCount, Math.floor(width / columnWidth));

    let low = Infinity;
    let high = -Infinity;
    for (let back = 0; back < visible; back += 1) {
      const value = state.colMid[columnIndex(state, back)];
      if (!Number.isFinite(value)) continue;
      if (value < low) low = value;
      if (value > high) high = value;
    }
    if (!Number.isFinite(low)) return;
    // A flat window would divide by zero and a book whose mid never moves is
    // the normal case here, so the range has a floor of one tick.
    const span = Math.max(high - low, state.tickSize);
    const top = 4;
    const usable = height - 12;

    ctx.beginPath();
    ctx.strokeStyle = tokens.tape;
    ctx.lineWidth = 1;
    let drawing = false;
    for (let back = visible - 1; back >= 0; back -= 1) {
      const value = state.colMid[columnIndex(state, back)];
      const x = width - (back + 1) * columnWidth;
      if (!Number.isFinite(value)) {
        drawing = false;
        continue;
      }
      const y = top + usable * (1 - (value - low) / span);
      if (drawing) {
        ctx.lineTo(x, y);
      } else {
        ctx.moveTo(x, y);
        drawing = true;
      }
    }
    ctx.stroke();

    numberText(ctx, tokens, high.toFixed(2), width - 6, 8, {
      align: "right",
      size: 10,
      colour: tokens.alpha.text(0.7),
    });
    numberText(ctx, tokens, low.toFixed(2), width - 6, height - 8, {
      align: "right",
      size: 10,
      colour: tokens.alpha.text(0.7),
    });
    // The label goes on whichever half of the strip the trace is NOT currently
    // in at its left edge. This strip is 62 px tall, so a fixed corner puts a
    // 1 px amber line straight through the text roughly half the time.
    const oldest = state.colMid[columnIndex(state, visible - 1)];
    const oldestY = Number.isFinite(oldest)
      ? top + usable * (1 - (oldest - low) / span)
      : height;
    // Clamped away from both edges: the adaptive placement put it flush against
    // the panel border, where it read as part of the divider rather than as a
    // label belonging to this strip.
    labelText(ctx, tokens, "mid drift · absolute", 8, oldestY < height / 2 ? height - 14 : 14, {
      colour: tokens.dim,
    });
  }

  /** How much tape time the panel currently spans, stated rather than implied. */
  function updateAxisNote(state) {
    const columnWidth = step();
    const visible = Math.min(state.colCount, Math.floor(scroll.width / columnWidth));
    if (visible < 2 || state.nsPerColumn <= 0) return;
    const seconds = (visible * state.nsPerColumn) / 1e9;
    const text = `${seconds.toFixed(0)} s of tape across the panel · ${visible} delivered frames`;
    if (text !== lastAxisText) {
      axisNote.textContent = text;
      lastAxisText = text;
    }
  }

  return {
    draw(state, now, columns) {
      if (state.gapMessage !== null) {
        banner.textContent = state.gapMessage;
        banner.dataset.shown = "true";
        bannerUntil = now + 2500;
        state.gapMessage = null;
      } else if (bannerUntil !== 0 && now > bannerUntil) {
        banner.dataset.shown = "false";
        bannerUntil = 0;
      }

      if (columns > 0) updateRanges(state, now);

      if (needsRepaint) {
        repaint(state);
      } else if (columns > 0) {
        const columnWidth = step();
        const shift = columns * columnWidth;
        if (shift >= scroll.width) {
          repaint(state);
        } else {
          // The scroll itself: one drawImage of the surface onto itself, then
          // the new columns. History is never touched again.
          scroll.ctx.drawImage(scroll.canvas, -shift, 0);
          for (let back = 0; back < columns; back += 1) {
            const x = scroll.width - (back + 1) * columnWidth;
            drawColumn(state, columnIndex(state, back), x, columnWidth);
          }
        }
      }

      if (needsOverlay) drawOverlay(state);
      drawDrift(state);
      if (columns > 0) updateAxisNote(state);
    },
  };
}
