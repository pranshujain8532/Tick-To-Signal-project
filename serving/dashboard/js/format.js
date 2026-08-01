/* format.js — how every number on this screen is written.
 *
 * WHAT
 *     Fixed-width numeric formatting for a screen where the digits must not
 *     move. Nothing here draws; it exists so that eight panels round, pad and
 *     sign their numbers identically.
 *
 * WHY THIS IS A MODULE AND NOT `toFixed` AT EACH CALL SITE
 *     Two panels showing the same quantity to different precision is the
 *     cheapest way to look untrustworthy. The rules live in one file so
 *     "microseconds are shown to one decimal below 100 and none above" is a
 *     decision made once.
 *
 * ON ALLOCATION — the honest version of "zero allocation in the render loop".
 *     Every function here allocates a small string, and that cannot be avoided
 *     in JavaScript without a glyph atlas and manual digit blitting, which
 *     would be a genuinely worse artefact. What the render loop avoids is
 *     ARRAY and OBJECT allocation: no `map`, no spread, no literals per frame.
 *     That is the allocation that matters. Stage 7b measured the mechanism one
 *     language down: hoisting a lookup out of the hot loop moved p50 by 0% and
 *     p99.99 by 2.4x, because allocation cost is rare and large rather than
 *     small and constant. Short-lived strings die in the nursery; a per-frame
 *     array of 20 level objects is what would eventually stop the world.
 */

/** Microseconds, at the precision the magnitude deserves. */
export function microseconds(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (value >= 10000) return `${(value / 1000).toFixed(1)} ms`;
  if (value >= 100) return `${value.toFixed(0)} µs`;
  if (value >= 10) return `${value.toFixed(1)} µs`;
  return `${value.toFixed(2)} µs`;
}

/** A signed value that keeps its sign column even when positive. */
export function signed(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const text = Math.abs(value).toFixed(digits);
  if (value > 0) return `+${text}`;
  if (value < 0) return `−${text}`; // U+2212, which is digit-width in a mono face
  return ` ${text}`;
}

export function fixed(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

/** A duration in the "3m30s" form the header uses for tape extent. */
export function duration(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "—";
  const whole = Math.floor(seconds);
  const minutes = Math.floor(whole / 60);
  const remainder = whole % 60;
  if (minutes === 0) return `${seconds.toFixed(1)}s`;
  return `${minutes}m${String(remainder).padStart(2, "0")}s`;
}

/**
 * Basis points at the scale this book actually trades on.
 *
 * Four decimals, because one 0.01 USDT tick on a ~65,000 USDT mid is 0.0015 bps
 * and two decimals would render the entire spread column as "0.00". Stage 5
 * caught a 100x error in exactly this conversion; showing enough digits to see
 * the value is part of not repeating it.
 */
export function bps(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (Math.abs(value) >= 1) return value.toFixed(3);
  return value.toFixed(4);
}

/** A price, always to the venue's two decimals. */
export function price(value) {
  if (!Number.isFinite(value)) return "—";
  return value.toFixed(2);
}

/**
 * A resting size in BTC.
 *
 * Three decimals is not decoration: measured across the three committed tapes
 * the median resting size is ~0.0005 BTC and p90 is ~0.7, so a two-decimal
 * column would show half the book as "0.00" and imply the level is empty when
 * it is not.
 */
export function size(value) {
  if (!Number.isFinite(value)) return "—";
  if (value >= 100) return value.toFixed(1);
  return value.toFixed(3);
}
