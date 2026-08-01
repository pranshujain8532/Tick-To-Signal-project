/* signal.js — what the model currently thinks, and how steady it has been.
 *
 * WHAT
 *     Three class probabilities as one horizontal stacked bar, the score as a
 *     large interpolating number, and a short history sparkline beneath it.
 *
 * WHY A STACKED BAR AND NOT A GAUGE
 *     A gauge spends most of its pixels on an arc that carries no information
 *     and reads as a dashboard template rather than as an instrument. The
 *     stacked bar shows all three probabilities simultaneously, sums to one by
 *     construction so a viewer can see that it is a distribution, and fits a
 *     300 px column. The three classes are also labelled in text, because
 *     distinguishing them by colour alone would fail a colour-blind reader.
 *
 * WHY THE BIG NUMBER INTERPOLATES
 *     Motion here is data: the number is easing toward a value that genuinely
 *     changed, which makes the direction of change legible in peripheral
 *     vision. Nothing else on this panel moves unless the data does. The
 *     interpolation is a fixed fraction per frame rather than a duration-based
 *     tween, so it is frame-rate-independent in the only sense that matters
 *     here — it always converges, and it never overshoots.
 *
 * WHY THE PANEL GOES BLANK RATHER THAN HOLDING THE LAST VALUE
 *     After every session boundary the engine destroys its normalisation
 *     history on purpose and 599 anchors must pass before an input exists. The
 *     server sends `signal: null` with a countdown; this panel draws the
 *     countdown. Freezing on the last good score would be indistinguishable, to
 *     a viewer, from a live signal that happened to be steady — and that is the
 *     precise dishonesty the rest of the project is built to avoid.
 */

import { createSurface, hairline, labelText, numberText } from "../canvas.js";
import { SIGNAL_HISTORY } from "../state.js";
import { signed } from "../format.js";

export function createSignalPanel(tokens) {
  const surface = createSurface(document.getElementById("signal-canvas"));
  const stateLabel = document.getElementById("signal-state");

  // The displayed score, easing toward the true one. Kept here rather than in
  // state.js because it is a property of the drawing, not of the data.
  let shownScore = 0;
  let lastStateText = "";

  function draw(state) {
    const ctx = surface.ctx;
    const width = surface.width;
    const height = surface.height;
    ctx.clearRect(0, 0, width, height);

    const target = state.hasSignal ? state.score : 0;
    shownScore += (target - shownScore) * 0.18;

    setStateText(
      state.hasSignal
        ? "live"
        : `warming · ${state.warmupRemaining} of ${state.warmupRequired} anchors`
    );

    // Laid out from the bottom up, from the panel's actual height. The first
    // version positioned the three blocks independently from the top and drew
    // the score straight through the sparkline label, because the panel is
    // ~146 px tall once the permanent note beneath it has taken its space —
    // which is less than the sum of three blocks sized for a taller box.
    const sparkTop = height - 38;
    const scoreBaseline = sparkTop - 26;
    const barTop = 20;

    drawStackedBar(ctx, state, width, barTop);
    drawScore(ctx, state, width, scoreBaseline);
    drawSparkline(ctx, state, width, sparkTop, 30);
  }

  function setStateText(text) {
    if (text === lastStateText) return;
    stateLabel.textContent = text;
    lastStateText = text;
  }

  function drawStackedBar(ctx, state, width, y) {
    const left = 10;
    const barWidth = width - 20;
    const barHeight = 14;

    if (!state.hasSignal) {
      ctx.fillStyle = tokens.alpha.dim(0.12);
      ctx.fillRect(left, y, barWidth, barHeight);
      labelText(ctx, tokens, "no signal — history discarded at boundary", left, y + barHeight + 12, {
        colour: tokens.dim,
      });
      return;
    }

    // Down, flat, up — in the order the classes are numbered, so the bar reads
    // left-to-right as "sell ... hold ... buy" and matches the score's sign.
    const segments = [
      { value: state.pDown, colour: tokens.ask, text: "down" },
      { value: state.pFlat, colour: tokens.dim, text: "flat" },
      { value: state.pUp, colour: tokens.bid, text: "up" },
    ];
    let x = left;
    for (let index = 0; index < segments.length; index += 1) {
      const segment = segments[index];
      ctx.fillStyle = segment.colour;
      ctx.fillRect(x, y, barWidth * segment.value, barHeight);
      x += barWidth * segment.value;
    }

    // The legend row is at FIXED thirds, not under each segment. This model is
    // frequently near-certain — p(up) above 0.97 is common — which collapses two
    // of the three segments to zero width and stacked all three labels on the
    // same few pixels. The bar shows the proportions; the row below always shows
    // all three numbers, which is also what keeps the classes distinguishable
    // without relying on colour.
    const labelY = y + barHeight + 12;
    const aligns = ["left", "center", "right"];
    const positions = [left, left + barWidth / 2, left + barWidth];
    for (let index = 0; index < segments.length; index += 1) {
      const segment = segments[index];
      labelText(
        ctx,
        tokens,
        `${segment.text} ${(segment.value * 100).toFixed(0)}%`,
        positions[index],
        labelY,
        { colour: segment.colour, align: aligns[index], upper: false }
      );
    }
  }

  /**
   * The score, large.
   *
   * Its definition is NOT repeated here: the permanent note under this panel
   * already says "score = p(up) − p(down), computed server-side". Drawing it
   * twice cost a whole row in a panel that does not have one to spare.
   */
  function drawScore(ctx, state, width, y) {
    numberText(ctx, tokens, state.hasSignal ? signed(shownScore, 3) : "—", 10, y, {
      size: 32,
      colour: state.hasSignal ? tokens.signal : tokens.dim,
    });
  }

  /**
   * Recent score history.
   *
   * Fixed at ±0.5 rather than auto-ranged: an auto-ranged sparkline makes a
   * flat signal look dramatic, which for a model whose per-block IC mean is
   * +0.073 would be actively misleading. The scale is printed for that reason.
   */
  function drawSparkline(ctx, state, width, top, usable) {
    const left = 10;
    const drawWidth = width - 20;
    const centre = top + usable / 2;

    hairline(ctx, left, centre, drawWidth, 0, tokens.alpha.hairline(1), surface.dpr);
    // Right-aligned so it shares a row with the score rather than needing one.
    labelText(ctx, tokens, "score history · fixed ±0.5", width - 10, top - 8, {
      colour: tokens.dim,
      align: "right",
      upper: false,
    });

    if (state.scoreCount === 0) return;
    const visible = Math.min(state.scoreCount, SIGNAL_HISTORY);
    const stepX = drawWidth / visible;

    ctx.beginPath();
    ctx.strokeStyle = tokens.signal;
    ctx.lineWidth = 1;
    for (let back = visible - 1; back >= 0; back -= 1) {
      const index = (state.scoreHead - back + SIGNAL_HISTORY * 2) % SIGNAL_HISTORY;
      const value = Math.max(-0.5, Math.min(0.5, state.scoreHistory[index]));
      const x = left + (visible - 1 - back) * stepX;
      const y = centre - (value / 0.5) * (usable / 2);
      if (back === visible - 1) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  return { draw };
}
