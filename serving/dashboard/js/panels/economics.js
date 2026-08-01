/* economics.js — the result that decides the project, rendered plainly.
 *
 * WHAT
 *     Gross edge per trade, the breakeven fee it implies, every published
 *     Binance taker tier with its shortfall multiple, and Stage 5's verdict
 *     sentence in full.
 *
 * WHY THIS PANEL IS DOM AND NOT CANVAS — the one deliberate exception.
 *     Every other data panel here is canvas, because it redraws. This one never
 *     redraws: it is one fetch of a file that changes when a benchmark is
 *     re-run. Canvas would buy nothing and would cost the two things that
 *     matter most for exactly this content — the text would not be selectable,
 *     and it would not be in the accessibility tree. This is the panel a reader
 *     is most likely to want to quote, and the one whose absence from a
 *     screen-reader would be least defensible.
 *
 * WHY IT GETS MORE THAN HALF THE BOTTOM ROW
 *     Because the finding is negative. A beautiful dashboard that made the
 *     negative result small would be a worse artefact than a plain one that
 *     made it large. The layout gives this panel more width than the Pareto
 *     frontier beside it, and that is a choice about honesty rather than about
 *     composition.
 *
 * THE ARITHMETIC, SO IT CAN BE CHECKED
 *     Gross edge is +0.285 bps per trade, one-way. Breakeven is half of that —
 *     0.142 bps per side — because a round trip pays the fee twice. The
 *     cheapest published taker tier is 4.0 bps, which is 28x the breakeven; the
 *     tier a new account actually gets is 10.0 bps, which is 70x. Every tier is
 *     shown, not the flattering subset, and every one of them is negative.
 */

import { signed, fixed } from "../format.js";

export function createEconomicsPanel() {
  const body = document.getElementById("economics-body");
  const verdict = document.getElementById("verdict");
  const source = document.getElementById("economics-source");

  function setData(payload) {
    // Built once, from a template string, rather than by forty appendChild
    // calls. This runs a single time in the life of the page.
    const tiers = payload.tiers
      .map(
        (tier) => `
        <div class="row">
          <span class="row-name">${escapeHtml(tier.name)}</span>
          <span>${fixed(tier.fee_bps, 1)} bps</span>
          <span class="shortfall">${fixed(tier.shortfall_multiple, 0)}× breakeven</span>
        </div>`
      )
      .join("");

    body.innerHTML = `
      <div class="row">
        <span class="row-name">gross edge per trade</span>
        <span>${signed(payload.gross_bps_per_trade, 3)} bps</span>
        <span class="row-name">${payload.trade_count.toLocaleString()} trades</span>
      </div>
      <div class="row">
        <span class="row-name">breakeven fee</span>
        <span>${fixed(payload.breakeven_fee_bps, 3)} bps/side</span>
        <span class="row-name">half the gross, paid twice</span>
      </div>
      <div class="row">
        <span class="row-name">median spread</span>
        <span>${fixed(payload.median_spread_bps, 4)} bps</span>
        <span class="row-name">one 0.01 tick</span>
      </div>
      <div style="height:6px"></div>
      <div class="row">
        <span class="row-name label">binance taker tiers, checked ${payload.tiers_checked_on}</span>
        <span></span><span></span>
      </div>
      ${tiers}`;

    verdict.textContent = `Stage 5's verdict: ${payload.verdict}`;
    source.textContent = payload.source;
  }

  function setUnavailable(reason) {
    body.innerHTML = `<div class="empty-state">${escapeHtml(reason)}</div>`;
  }

  /**
   * Values from the API are inserted as text, never as markup.
   *
   * They come from a file this repository wrote, so this is not defending
   * against an attacker — it is defending against a tier named "VIP 0 <BNB>"
   * silently disappearing because a browser read it as a tag.
   */
  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  // Nothing to do per frame: this panel has no live data. The render loop still
  // calls draw() on every panel uniformly, and this one returning immediately
  // is cheaper than special-casing it in the loop.
  return { draw() {}, setData, setUnavailable };
}
