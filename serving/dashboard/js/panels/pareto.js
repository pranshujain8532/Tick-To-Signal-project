/* pareto.js — the whole frontier, with the instrument that measured each point.
 *
 * WHAT
 *     Seven points: five Python-harness variants and both C++ paths, on a log
 *     latency axis against macro-F1. The variant this server is actually
 *     running is marked distinctly. Hovering a point shows its exact numbers.
 *
 * WHY THE TWO HARNESSES GET DIFFERENT MARKS
 *     This is the single most attackable chart in the project. A Python-harness
 *     number and a C++-harness number on one axis, drawn identically, is an
 *     invitation to read the 11 µs point as a continuation of the 729 µs one —
 *     and it is not. Different program, different instrument, different
 *     boundary: the C++ points time a forward pass from an already-prepared
 *     [40] feature column and exclude the feature construction that is inside
 *     every Python point. Filled circles are the Python harness, open squares
 *     are the C++ harness, the legend says so in words, and the note below the
 *     panel states the boundary permanently.
 *
 * WHY BOTH C++ POINTS ARE HERE
 *     Showing only the incremental path would present a ~193x algorithmic win
 *     as though it were what "rewriting it in C++" buys. The full pass is the
 *     honest baseline and it sits at 2,119 µs — SLOWER than ONNX Runtime on the
 *     same model, because ONNX Runtime ships vectorised kernels a hand-written
 *     nested loop does not match. The gap between the two C++ points is the
 *     Stage 7b finding; the gap between C++ full and ONNX int8 is the part that
 *     makes it honest.
 *
 * WHY EACH POINT HAS A WHISKER TO ITS p99
 *     A frontier plotted on p50 alone says the fast thing is fast. The whisker
 *     says how fast it is when it is not, which for this project is the whole
 *     argument: the C++ incremental point's p50 is 11 µs and its p99.9 is 87 µs,
 *     and a reader who only saw the dot would not know that.
 *
 * WHY ACCURACY ON THE C++ POINTS IS DIMMED
 *     No C++ program in this repository has ever scored a test block. Those two
 *     y-coordinates are inherited from the fp32 student the C++ path is a
 *     parity-checked re-implementation of, and the tooltip says "inherited, not
 *     measured" for both. Presenting an inherited number as a measured one is
 *     exactly the fabrication the constitution bans.
 */

import { createSurface, hairline, labelText, numberText, wrappedText } from "../canvas.js";
import { microseconds, fixed } from "../format.js";

export function createParetoPanel(tokens) {
  let dirty = true;
  const surface = createSurface(document.getElementById("pareto-canvas"), {
    onResize: () => {
      dirty = true;
    },
  });
  const tooltip = document.getElementById("pareto-tooltip");

  let rows = null;
  let failure = null;
  let hovered = -1;
  // Point positions in CSS pixels, recomputed on draw and reused for hit
  // testing so the pointer handler never repeats the projection arithmetic.
  let positions = [];

  function setData(payload) {
    rows = payload.rows;
    positions = rows.map(() => ({ x: 0, y: 0 }));
    dirty = true;
  }

  function setUnavailable(reason) {
    rows = null;
    failure = reason;
    dirty = true;
  }

  function project(row, box) {
    const x = box.left + ((Math.log10(row.p50_us) - box.logLow) / box.logSpan) * box.width;
    const y = box.bottom - ((row.macro_f1 - box.f1Low) / box.f1Span) * box.height;
    return { x, y };
  }

  function draw() {
    if (!dirty) return;
    dirty = false;

    const ctx = surface.ctx;
    const width = surface.width;
    const height = surface.height;
    ctx.clearRect(0, 0, width, height);

    if (rows === null) {
      wrappedText(ctx, tokens, failure || "loading /pareto…", 10, 20, width - 20);
      return;
    }

    const box = layout(width, height);
    drawAxes(ctx, box);

    for (let index = 0; index < rows.length; index += 1) {
      positions[index] = project(rows[index], box);
    }
    assignLabelSides(box);
    for (let index = 0; index < rows.length; index += 1) {
      drawPoint(ctx, rows[index], positions[index], box, index === hovered);
    }
  }

  /**
   * Decide which side of each point its label goes on, and nudge collisions.
   *
   * Three of the seven points sit within a factor of two of each other in
   * latency and within 0.003 of each other in macro-F1, so labels placed at a
   * fixed offset overlap and become unreadable. This is a deliberately simple
   * rule — label to the right unless that would run off the plot, then push
   * down while the label would collide with one already placed — because the
   * alternative is a force-directed layout, which is a lot of machinery for
   * seven fixed points that only move when a benchmark is re-run.
   */
  function assignLabelSides(box) {
    const placed = [];
    for (let index = 0; index < rows.length; index += 1) {
      const position = positions[index];
      const width = 8 + rows[index].label.length * 6.2;
      position.labelRight = position.x + width < box.right + 100;
      position.labelY = position.y - 10;
      let attempts = 0;
      while (attempts < 6 && collides(placed, position, width)) {
        position.labelY += 12;
        attempts += 1;
      }
      placed.push({ x: position.x, y: position.labelY, width, right: position.labelRight });
    }
  }

  function collides(placed, position, width) {
    for (let index = 0; index < placed.length; index += 1) {
      const other = placed[index];
      if (Math.abs(other.y - position.labelY) > 11) continue;
      const leftA = position.labelRight ? position.x : position.x - width;
      const leftB = other.right ? other.x : other.x - other.width;
      if (leftA < leftB + other.width && leftB < leftA + width) return true;
    }
    return false;
  }

  function layout(width, height) {
    const left = 34;
    // Room on the right for the widest point label. Without it the rightmost
    // point — PyTorch eager, which is the slowest and therefore always at the
    // edge — has its label clipped by the panel, and a clipped label on the
    // slowest variant is the one a reader most wants to identify.
    const right = width - 108;
    const top = 22;
    const bottom = height - 28;
    let logLow = Infinity;
    let logHigh = -Infinity;
    let f1Low = Infinity;
    let f1High = -Infinity;
    for (let index = 0; index < rows.length; index += 1) {
      const row = rows[index];
      logLow = Math.min(logLow, Math.log10(row.p50_us));
      logHigh = Math.max(logHigh, Math.log10(row.p99_us));
      f1Low = Math.min(f1Low, row.macro_f1);
      f1High = Math.max(f1High, row.macro_f1);
    }
    const f1Pad = Math.max((f1High - f1Low) * 0.18, 0.004);
    return {
      left,
      right,
      top,
      bottom,
      width: right - left,
      height: bottom - top,
      logLow: logLow - 0.15,
      logSpan: logHigh - logLow + 0.35,
      f1Low: f1Low - f1Pad,
      f1Span: f1High - f1Low + f1Pad * 2,
    };
  }

  function drawAxes(ctx, box) {
    hairline(ctx, box.left, box.bottom, box.width, 0, tokens.hairline, surface.dpr);
    hairline(ctx, box.left, box.top, 0, box.height, tokens.hairline, surface.dpr);

    // Decade gridlines, because a log axis without them is just a squashed one.
    const firstDecade = Math.ceil(box.logLow);
    for (let decade = firstDecade; decade <= box.logLow + box.logSpan; decade += 1) {
      const x = box.left + ((decade - box.logLow) / box.logSpan) * box.width;
      hairline(ctx, x, box.top, 0, box.height, tokens.alpha.hairline(0.8), surface.dpr);
      const value = Math.pow(10, decade);
      labelText(ctx, tokens, microseconds(value), x + 3, box.bottom + 10, {
        colour: tokens.dim,
        upper: false,
      });
    }
    // The axis title sits on its own row beneath the decade labels; sharing a
    // row with them put "p50 latency, log" underneath the 100 µs tick.
    labelText(ctx, tokens, "p50 latency, log · whisker to p99", box.left, box.bottom + 22, {
      colour: tokens.dim,
    });
    labelText(ctx, tokens, "macro f1", 6, box.top - 6, { colour: tokens.dim });
  }

  function drawPoint(ctx, row, position, box, isHovered) {
    const isCpp = row.measured_by === "cpp harness";
    const colour = row.is_serving ? tokens.signal : isCpp ? tokens.tape : tokens.text;

    // The whisker to p99: how slow this variant is when it is not fast.
    const p99x = box.left + ((Math.log10(row.p99_us) - box.logLow) / box.logSpan) * box.width;
    ctx.strokeStyle = tokens.alpha.dim(0.55);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(position.x, position.y);
    ctx.lineTo(p99x, position.y);
    ctx.stroke();

    ctx.lineWidth = 1.5;
    if (isCpp) {
      // Open square: a different instrument, and it must not be mistakable for
      // a filled circle at a glance or in a screenshot.
      ctx.strokeStyle = colour;
      ctx.strokeRect(position.x - 4, position.y - 4, 8, 8);
    } else {
      ctx.fillStyle = colour;
      ctx.beginPath();
      ctx.arc(position.x, position.y, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    if (row.is_serving) {
      // What the server is running right now, ringed and named. Without this
      // the reader has no way to know which of seven points is the live one.
      ctx.strokeStyle = tokens.signal;
      ctx.beginPath();
      ctx.arc(position.x, position.y, 9, 0, Math.PI * 2);
      ctx.stroke();
      labelText(ctx, tokens, "serving", position.x + 12, position.y + 13, {
        colour: tokens.signal,
      });
    }

    if (isHovered) {
      ctx.strokeStyle = tokens.text;
      ctx.strokeRect(position.x - 8, position.y - 8, 16, 16);
    }

    // Inherited accuracy is dimmed. The two C++ points have never scored a test
    // block; their height on this axis is borrowed from the fp32 student they
    // are a parity-checked re-implementation of, and dimming is the visual form
    // of the tooltip's "inherited, not measured".
    labelText(ctx, tokens, row.label, position.labelRight ? position.x + 8 : position.x - 8, position.labelY, {
      colour: row.macro_f1_is_inherited ? tokens.dim : tokens.alpha.text(0.85),
      align: position.labelRight ? "left" : "right",
    });
  }

  /** Nearest point within a forgiving radius, or -1. */
  function hitTest(x, y) {
    let best = -1;
    let bestDistance = 18 * 18;
    for (let index = 0; index < positions.length; index += 1) {
      const dx = positions[index].x - x;
      const dy = positions[index].y - y;
      const distance = dx * dx + dy * dy;
      if (distance < bestDistance) {
        bestDistance = distance;
        best = index;
      }
    }
    return best;
  }

  surface.canvas.addEventListener("mousemove", (event) => {
    if (rows === null) return;
    const bounds = surface.canvas.getBoundingClientRect();
    const index = hitTest(event.clientX - bounds.left, event.clientY - bounds.top);
    if (index === hovered) return;
    hovered = index;
    dirty = true;
    if (index === -1) {
      tooltip.style.display = "none";
      return;
    }
    const row = rows[index];
    tooltip.textContent = describe(row);
    tooltip.style.display = "block";
    tooltip.style.left = `${Math.min(positions[index].x + 14, surface.cssWidth - 210)}px`;
    tooltip.style.top = `${Math.max(4, positions[index].y - 60)}px`;
  });

  surface.canvas.addEventListener("mouseleave", () => {
    hovered = -1;
    tooltip.style.display = "none";
    dirty = true;
  });

  function describe(row) {
    const accuracy = row.macro_f1_is_inherited
      ? `macro F1 ${fixed(row.macro_f1, 4)}  (inherited from ${row.macro_f1_inherited_from}, not measured)`
      : `macro F1 ${fixed(row.macro_f1, 4)}`;
    const size = row.size_kib === null ? "" : `\n${fixed(row.size_kib, 1)} KiB`;
    return (
      `${row.label}\n` +
      `p50 ${microseconds(row.p50_us)}   p99 ${microseconds(row.p99_us)}   ` +
      `p99.9 ${microseconds(row.p99_9_us)}\n` +
      `${accuracy}\n` +
      `${row.params.toLocaleString()} params${size}\n` +
      `measured by ${row.measured_by}`
    );
  }

  return { draw, setData, setUnavailable };
}
