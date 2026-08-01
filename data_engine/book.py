"""Limit order book reconstruction from a snapshot + diff stream.

WHAT
    An `OrderBook` that holds one venue's L2 depth for one symbol. You seed it
    with a REST snapshot, then fold in `depthUpdate` diff events in sequence
    order. It exposes the touch (best bid/ask), the mid, the top-N ladder as
    numpy arrays, and an invariant check that is meant to be run in anger.

WHY
    Every feature, label, and latency number downstream is a function of the
    book state. A subtly corrupted book does not look corrupted — it produces
    smooth, plausible features and a model that learns noise. So this module
    treats "prove the book is right" as the product, and reconstruction as the
    easy part.

DESIGN DECISION — integer fixed-point prices (price * 10^8) as dict keys,
never floats.
    Rejected alternative: `dict[float, float]` keyed on the parsed price.
    Binance sends prices as decimal strings ("94210.51000000"). Binary floats
    cannot represent most decimal ticks exactly, so two messages naming the
    same price level can produce two different float keys — the book silently
    grows a duplicate level that never gets deleted, because the "qty 0"
    removal arrives keyed to the *other* float. That failure is invisible for
    hours and then shows up as a book that is one level too deep. Integers on
    the tick grid make level identity exact by construction, which also makes
    equality, ordering, and the binary format in Stage 2 exact for free.
    Scale 10^8 covers every spot tick size Binance quotes; BTC at 10^5 scales
    to 10^13, comfortably inside int64.

DESIGN DECISION — two dicts (price -> qty), not sorted arrays or a tree.
    (This supersedes the Stage-0 stub docstring, which proposed sorted
    ladders.) A diff event is a scatter of unrelated price levels, and the
    dominant operation is "set or delete this one level": O(1) in a dict,
    O(k) memmove in a sorted array. Deletion — the `qty == 0` case — is a
    single `del` with no shifting and no tombstones. The cost we accept is
    that ordered reads (`best_bid`, `top_n`) are O(k) / O(k log k) rather
    than O(1), because ordering is not maintained incrementally.
    That trade is right for Stage 1, where reads happen a few times per
    second (heartbeat, cross-check) and writes happen on every event. It is
    the wrong trade for the Stage 7 hot path, which reads the top 10 levels
    on every tick — that path gets a contiguous sorted ladder, and the
    difference between the two is itself a thing worth being able to explain.

DESIGN DECISION — a sequence gap raises, it never self-heals.
    See `apply_diff`. The reasoning is long enough to live at its call site.

INFORMATION HORIZON
    Book state after applying an event with final update id `u` is a function
    of events with update id <= u only. Nothing here reads forward.

Reference for the diff semantics and the sync algorithm:
    https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams
    (section "How to manage a local order book correctly")
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Iterable, Sequence

import numpy as np

# Fixed-point scale. Prices and quantities are stored as
# round(real_value * 10^8) and only converted back to float at the edges
# (display, features), never for comparison or as a dict key.
PRICE_SCALE = 10 ** 8
QTY_SCALE = 10 ** 8

# `last_update_id` sentinel meaning "no snapshot has been applied yet".
# Binance update ids are positive, so -1 can never collide with a real one.
_UNINITIALISED = -1


class SequenceGapError(Exception):
    """Raised when a depth diff does not continue from the local book state.

    Carries the ids so the caller can log exactly how many updates were lost,
    which is the difference between "one dropped frame" and "the socket was
    dead for a minute" when reading incident logs later.
    """

    def __init__(self, expected_first_id: int, event_first_id: int, event_final_id: int) -> None:
        self.expected_first_id = expected_first_id
        self.event_first_id = event_first_id
        self.event_final_id = event_final_id
        self.missed_updates = event_first_id - expected_first_id
        super().__init__(
            f"sequence gap: expected an event starting at U={expected_first_id}, "
            f"got U={event_first_id} u={event_final_id} "
            f"({self.missed_updates} update ids missed)"
        )


class BookInvariantError(Exception):
    """Raised by `OrderBook.assert_valid` when the book is not a valid book."""


def to_fixed(value: str, scale: int) -> int:
    """Convert a decimal string from the exchange into scaled integer units.

    Uses `Decimal` rather than `float(value) * scale` because the float path
    has to be rescued by a `round()` whose correctness is an argument about
    representation error, while the Decimal path is exact by construction.
    This sits off the latency-critical path (Stage 1 is I/O bound on the
    socket); if profiling in a later stage says otherwise, the replacement is
    a hand-rolled integer string parser, not a float.
    """
    scaled = Decimal(value) * scale
    as_integer = int(scaled)
    # A non-zero remainder means the venue quoted finer precision than our
    # scale can hold. Truncating would corrupt the tick grid silently, so we
    # refuse — a scale that is too small is a bug in this file, not in the data.
    if scaled != as_integer:
        raise ValueError(f"{value!r} does not fit the 10^{len(str(scale)) - 1} fixed-point grid")
    return as_integer


class OrderBook:
    """Local L2 book for one symbol, driven by snapshot + diff events."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol.upper()
        self.bids: dict[int, int] = {}
        self.asks: dict[int, int] = {}
        self.last_update_id: int = _UNINITIALISED

    # ---------------------------------------------------------------- state

    @property
    def is_initialised(self) -> bool:
        return self.last_update_id != _UNINITIALISED

    def apply_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Replace the whole book with a REST depth snapshot.

        Expects the raw `GET /api/v3/depth` response: `lastUpdateId`, `bids`,
        `asks`. Replaces rather than merges — a snapshot is the complete truth
        for the levels it contains, and merging would keep stale levels that
        the snapshot has already removed.
        """
        self.bids = self._parse_side(snapshot["bids"])
        self.asks = self._parse_side(snapshot["asks"])
        self.last_update_id = int(snapshot["lastUpdateId"])

    def apply_diff(self, event: dict[str, Any]) -> bool:
        """Apply one `depthUpdate` event. Returns False if it was stale.

        Implements Binance's documented update procedure:
          * `u` <= local update id  -> already reflected, ignore.
          * `U` >  local id + 1     -> updates were missed, raise.
          * otherwise               -> apply, then adopt `u` as the local id.

        WHY RAISE INSTEAD OF PATCHING THE HOLE:
        the tempting alternative is to keep going — apply the event anyway, or
        quietly refetch a snapshot and merge. Both are wrong here. A gap means
        some set of levels changed and we did not see how; every level in the
        book is now *possibly* stale, and we cannot tell which ones, because a
        diff only mentions levels that moved. A book that is wrong at one
        unknown price level still produces perfectly plausible features, so the
        damage is undetectable downstream and poisons training data silently.
        Raising converts an invisible data-quality problem into a loud,
        countable event: the caller tears the book down and resyncs from a
        fresh snapshot, and the resync counter tells us how often it happened.
        Losing a second of data is cheap. Not knowing which second is not.

        Note on `<=` vs the docs' `<`: the update procedure says ignore when
        `u` is *less than* the local id, while step 5 of the sync algorithm
        discards buffered events where `u <= lastUpdateId`. We use `<=`
        throughout. An event whose final id equals our current id is entirely
        contained in what we have already applied, so skipping it cannot lose
        information, and using one rule in both places means the buffered
        replay and the live path cannot disagree.
        """
        if not self.is_initialised:
            raise RuntimeError("apply_snapshot must be called before apply_diff")

        first_id = int(event["U"])
        final_id = int(event["u"])

        if final_id <= self.last_update_id:
            return False

        expected_first_id = self.last_update_id + 1
        if first_id > expected_first_id:
            raise SequenceGapError(expected_first_id, first_id, final_id)

        self._apply_levels(self.bids, event["b"])
        self._apply_levels(self.asks, event["a"])
        self.last_update_id = final_id
        return True

    def apply_level_update(self, is_bid: bool, price: int, quantity: int) -> None:
        """Set or delete a single price level, in fixed-point units.

        Exists for replaying a **binary tape**, where a depth diff has already
        been decomposed into one record per changed level and the sequence
        numbers are gone — they did their job at capture time, where the gap
        check ran and a gap forced a resync. Re-checking here would be
        checking a property of a file we wrote ourselves.

        This is therefore the one entry point that mutates the book without
        sequence validation, and it is deliberately named so that its use is
        obvious in a diff. Live feed data must go through `apply_diff`;
        routing it here instead would silently disable gap detection, which is
        the single most valuable safety property in `data_engine`.
        """
        side = self.bids if is_bid else self.asks
        if quantity == 0:
            side.pop(price, None)
        else:
            side[price] = quantity

    # ------------------------------------------------------------ mutation

    @staticmethod
    def _parse_side(raw_levels: Iterable[Sequence[str]]) -> dict[int, int]:
        """Build a fresh side from snapshot levels, dropping zero quantities."""
        side: dict[int, int] = {}
        for price_str, qty_str in raw_levels:
            quantity = to_fixed(qty_str, QTY_SCALE)
            if quantity > 0:
                side[to_fixed(price_str, PRICE_SCALE)] = quantity
        return side

    @staticmethod
    def _apply_levels(side: dict[int, int], raw_levels: Iterable[Sequence[str]]) -> None:
        """Set or delete individual price levels in place.

        Binance encodes removal as a quantity of zero rather than a separate
        message type, so "delete" and "update" arrive through the same field.
        A zero for a level we do not hold is normal, not an error: it can
        refer to a level outside the snapshot's depth limit, so the discard
        is deliberate.
        """
        for price_str, qty_str in raw_levels:
            price = to_fixed(price_str, PRICE_SCALE)
            quantity = to_fixed(qty_str, QTY_SCALE)
            if quantity == 0:
                side.pop(price, None)
            else:
                side[price] = quantity

    # ------------------------------------------------------------- reading

    def best_bid(self) -> int | None:
        """Highest bid price in fixed-point units, or None if the side is empty."""
        if not self.bids:
            return None
        return max(self.bids)

    def best_ask(self) -> int | None:
        """Lowest ask price in fixed-point units, or None if the side is empty."""
        if not self.asks:
            return None
        return min(self.asks)

    def mid_price(self) -> float | None:
        """Mid in real price units, or None if either side is empty.

        This is the one place we deliberately leave integer land: the mid of
        an odd-tick spread is a half tick, so it is not representable on the
        grid. That is safe precisely because the mid is never used as a key or
        compared for equality — it is a derived quantity for features and
        display only.
        """
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / (2 * PRICE_SCALE)

    def spread(self) -> float | None:
        """Best ask minus best bid in real price units, or None if one-sided."""
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is None or ask is None:
            return None
        return (ask - bid) / PRICE_SCALE

    def depth(self) -> tuple[int, int]:
        """Number of live price levels on each side, as (bids, asks)."""
        return len(self.bids), len(self.asks)

    def top_n(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the best `n` levels per side as int64 arrays of [price, qty].

        Bids come back descending (best first), asks ascending (best first),
        both in fixed-point units. Sides shorter than `n` return what exists
        rather than padding: a padded row is indistinguishable from a real
        level with zero size, and the caller deciding how to pad is better
        than this function guessing.
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        return (
            self._sorted_levels(self.bids, n, descending=True),
            self._sorted_levels(self.asks, n, descending=False),
        )

    @staticmethod
    def _sorted_levels(side: dict[int, int], n: int, descending: bool) -> np.ndarray:
        """Sort one side by price and return the best `n` rows as [price, qty]."""
        prices = sorted(side, reverse=descending)[:n]
        if not prices:
            return np.empty((0, 2), dtype=np.int64)
        rows = [(price, side[price]) for price in prices]
        return np.array(rows, dtype=np.int64)

    # ---------------------------------------------------------- invariants

    def assert_valid(self) -> None:
        """Raise `BookInvariantError` unless this is a well-formed book.

        Checks, in order: quantities and prices are strictly positive; the
        bid ladder is strictly descending and the ask ladder strictly
        ascending; the book is not crossed or locked (best_bid < best_ask).

        Monotonicity is structurally guaranteed by sorting unique dict keys,
        so checking it is not paranoia about the dicts — it is a test of
        `_sorted_levels` itself, which is the function every consumer of this
        book actually reads through. Cheap enough to run on every event
        during development and in the fuzz test.
        """
        self._assert_side_positive(self.bids, "bid")
        self._assert_side_positive(self.asks, "ask")

        all_bids, all_asks = self.top_n(max(len(self.bids), len(self.asks), 1))
        self._assert_strictly_ordered(all_bids[:, 0], descending=True, side_name="bid")
        self._assert_strictly_ordered(all_asks[:, 0], descending=False, side_name="ask")

        bid = self.best_bid()
        ask = self.best_ask()
        if bid is not None and ask is not None and bid >= ask:
            raise BookInvariantError(
                f"book is crossed or locked: best_bid={bid} >= best_ask={ask} "
                f"(last_update_id={self.last_update_id})"
            )

    @staticmethod
    def _assert_side_positive(side: dict[int, int], side_name: str) -> None:
        for price, quantity in side.items():
            if price <= 0:
                raise BookInvariantError(f"non-positive {side_name} price: {price}")
            if quantity <= 0:
                raise BookInvariantError(
                    f"non-positive {side_name} quantity {quantity} at price {price} "
                    "(zero quantities must delete the level, not store it)"
                )

    @staticmethod
    def _assert_strictly_ordered(prices: np.ndarray, descending: bool, side_name: str) -> None:
        if prices.size < 2:
            return
        differences = np.diff(prices)
        is_ordered = bool(np.all(differences < 0)) if descending else bool(np.all(differences > 0))
        if not is_ordered:
            direction = "descending" if descending else "ascending"
            raise BookInvariantError(f"{side_name} ladder is not strictly {direction}")

    def __repr__(self) -> str:
        bid = self.best_bid()
        ask = self.best_ask()
        return (
            f"OrderBook({self.symbol}, last_update_id={self.last_update_id}, "
            f"depth={self.depth()}, best_bid={bid}, best_ask={ask})"
        )
