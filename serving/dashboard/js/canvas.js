/* canvas.js — device-pixel handling, shared by all eight canvas panels.
 *
 * WHAT
 *     `createSurface` owns one canvas: its backing-store size, its
 *     devicePixelRatio, and whether its drawing coordinates are CSS pixels or
 *     device pixels. `readTokens` lifts the nine locked colours out of
 *     tokens.css so no panel contains a colour literal.
 *
 * WHY A SURFACE OBJECT AND NOT `canvas.width = rect.width * dpr` INLINE
 *     Eight panels would each get that line slightly wrong. More importantly,
 *     reading `clientWidth` forces the browser to flush layout — do it inside
 *     the render loop and every frame pays a synchronous reflow to learn a
 *     number that changes maybe twice an hour. So size is read in a
 *     ResizeObserver callback, outside the loop, and the loop only ever reads
 *     cached fields.
 *
 * THE TWO COORDINATE MODES, AND WHY THE TAPE NEEDS THE SECOND ONE
 *     Most panels want CSS pixels: `setTransform(dpr, 0, 0, dpr, 0, 0)` and
 *     then draw in layout units. The depth tape cannot. It scrolls by blitting
 *     itself one column to the left every frame, and on a 1.25x display a
 *     1-CSS-pixel blit is a 1.25-device-pixel blit — a fractional copy, which
 *     resamples, and resampling the same image sixty times a second turns a
 *     sharp book into mush within a few seconds. It is a compounding error:
 *     invisible in a screenshot, and obvious ten seconds into a live view.
 *
 *     So the scrolling surfaces work in DEVICE pixels with the identity
 *     transform, and scroll by a whole number of them. Text and line widths in
 *     those surfaces are multiplied by `dpr` explicitly.
 */

/** Colour values, read once from the stylesheet that owns them. */
export function readTokens() {
  const style = getComputedStyle(document.documentElement);
  const token = (name) => style.getPropertyValue(name).trim();
  const rgb = (name, alpha) => `rgba(${token(name).replace(/\s+/g, ", ")}, ${alpha})`;
  return {
    ground: token("--ground"),
    panel: token("--panel"),
    hairline: token("--hairline"),
    text: token("--text"),
    dim: token("--dim"),
    tape: token("--tape"),
    signal: token("--signal"),
    bid: token("--bid"),
    ask: token("--ask"),
    /** Alpha variants, composed from the same nine colours. */
    alpha: {
      tape: (a) => rgb("--tape-rgb", a),
      signal: (a) => rgb("--signal-rgb", a),
      bid: (a) => rgb("--bid-rgb", a),
      ask: (a) => rgb("--ask-rgb", a),
      text: (a) => rgb("--text-rgb", a),
      dim: (a) => rgb("--dim-rgb", a),
      hairline: (a) => rgb("--hairline-rgb", a),
    },
    mono: style.getPropertyValue("--mono").trim(),
    sans: style.getPropertyValue("--sans").trim(),
  };
}

/**
 * Wrap a canvas element with cached size and a resize hook.
 *
 * `onResize` fires after the backing store has been resized and cleared, which
 * is the moment a panel holding scrolled history has to repaint it — see
 * panels/tape.js:repaint.
 */
export function createSurface(canvas, options = {}) {
  const deviceSpace = options.deviceSpace === true;
  const context = canvas.getContext("2d", { alpha: options.alpha !== false });

  const surface = {
    canvas,
    ctx: context,
    dpr: 1,
    /** Drawing-space dimensions: CSS pixels normally, device pixels if deviceSpace. */
    width: 0,
    height: 0,
    /** Always CSS pixels, for hit-testing against pointer events. */
    cssWidth: 0,
    cssHeight: 0,
    deviceSpace,
    onResize: options.onResize || null,
  };

  function apply() {
    const cssWidth = canvas.clientWidth;
    const cssHeight = canvas.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    if (cssWidth === 0 || cssHeight === 0) return;

    const backingWidth = Math.round(cssWidth * dpr);
    const backingHeight = Math.round(cssHeight * dpr);
    const unchanged =
      canvas.width === backingWidth && canvas.height === backingHeight && surface.dpr === dpr;
    if (unchanged) return;

    // Assigning width/height resets the backing store AND every context
    // property, which is why the transform is reapplied here rather than once
    // at construction.
    canvas.width = backingWidth;
    canvas.height = backingHeight;
    surface.dpr = dpr;
    surface.cssWidth = cssWidth;
    surface.cssHeight = cssHeight;
    if (deviceSpace) {
      context.setTransform(1, 0, 0, 1, 0, 0);
      surface.width = backingWidth;
      surface.height = backingHeight;
    } else {
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      surface.width = cssWidth;
      surface.height = cssHeight;
    }
    context.textBaseline = "middle";
    if (surface.onResize) surface.onResize(surface);
  }

  surface.sync = apply;

  // ResizeObserver rather than a window resize listener: a panel changes size
  // when the grid reflows, which a window event does not always imply, and the
  // observer fires outside the render loop so the layout read is not in the
  // frame budget.
  const observer = new ResizeObserver(apply);
  observer.observe(canvas);

  // devicePixelRatio changes when the window moves between monitors. There is
  // no event for it; the documented idiom is a matchMedia query that must be
  // re-registered after every change because the threshold moves with it.
  function watchPixelRatio() {
    const query = window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
    query.addEventListener("change", () => {
      apply();
      watchPixelRatio();
    }, { once: true });
  }
  watchPixelRatio();

  apply();
  return surface;
}

/**
 * A true one-device-pixel line, whatever the display density.
 *
 * A 1-unit line drawn on a half-pixel boundary is antialiased across two rows
 * and reads as a soft 2px grey smear — the exact thing that makes a dense
 * interface look cheap. Snapping to the device grid and using a rectangle
 * rather than a stroke sidesteps the whole lineWidth/half-pixel question.
 */
export function hairline(ctx, x, y, width, height, colour, dpr = 1) {
  const thin = 1 / dpr;
  ctx.fillStyle = colour;
  if (height === 0) {
    ctx.fillRect(x, Math.round(y * dpr) / dpr, width, thin);
  } else if (width === 0) {
    ctx.fillRect(Math.round(x * dpr) / dpr, y, thin, height);
  }
}

/**
 * Uppercase tracked label text, drawn the way the CSS `.label` class draws it.
 *
 * `upper: false` exists because uppercasing is not safe for text containing
 * units: `"1397 µs".toUpperCase()` is `"1397 ΜS"` — micro becomes a Greek
 * capital Mu, which renders as an M, and the label then reads "1397 ms". That
 * is a 1000x error in a latency figure, produced by a text transform. Any label
 * carrying a number or a unit passes `upper: false`.
 */
export function labelText(ctx, tokens, text, x, y, options = {}) {
  const scale = options.scale || 1;
  ctx.save();
  ctx.font = `${10 * scale}px ${tokens.sans}`;
  ctx.fillStyle = options.colour || tokens.dim;
  ctx.textAlign = options.align || "left";
  ctx.textBaseline = options.baseline || "middle";
  // Canvas has no letter-spacing in every engine this must run in, so the
  // tracking that CSS gets from `letter-spacing: 0.12em` is drawn per glyph.
  const tracking = 1.2 * scale;
  const upper = options.upper === false ? text : text.toUpperCase();
  if (ctx.textAlign === "left") {
    let cursor = x;
    for (let index = 0; index < upper.length; index += 1) {
      const glyph = upper[index];
      ctx.fillText(glyph, cursor, y);
      cursor += ctx.measureText(glyph).width + tracking;
    }
  } else {
    ctx.fillText(upper, x, y);
  }
  ctx.restore();
}

/**
 * Word-wrapped body text, for the one thing a canvas panel must say in prose:
 * that it has no data, and which file would have given it some.
 *
 * Not uppercased and not tracked, because this is a sentence rather than a
 * label — and because an empty state that runs off the edge of its own panel
 * fails at the one job it has. The first version used `labelText` and clipped
 * the filename it existed to name.
 */
export function wrappedText(ctx, tokens, text, x, y, maxWidth, options = {}) {
  ctx.save();
  ctx.font = `11px ${tokens.sans}`;
  ctx.fillStyle = options.colour || tokens.dim;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";

  const words = text.split(" ");
  let line = "";
  let cursor = y;
  for (let index = 0; index < words.length; index += 1) {
    const candidate = line === "" ? words[index] : `${line} ${words[index]}`;
    if (ctx.measureText(candidate).width > maxWidth && line !== "") {
      ctx.fillText(line, x, cursor);
      cursor += 15;
      line = words[index];
    } else {
      line = candidate;
    }
  }
  if (line !== "") ctx.fillText(line, x, cursor);
  ctx.restore();
}

/** Monospace numerals, the loudest thing on the screen. */
export function numberText(ctx, tokens, text, x, y, options = {}) {
  ctx.save();
  ctx.font = `${options.weight ? `${options.weight} ` : ""}${options.size || 12}px ${tokens.mono}`;
  ctx.fillStyle = options.colour || tokens.text;
  ctx.textAlign = options.align || "left";
  ctx.textBaseline = options.baseline || "middle";
  ctx.fillText(text, x, y);
  ctx.restore();
}
