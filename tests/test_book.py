"""Tests for `data_engine.book`.

Three groups, in increasing order of how much they would worry an interviewer:

  1. Hand-written sequences with known answers — proves the diff semantics
     match Binance's documented behaviour on cases we can check by eye.
  2. Failure cases — proves a sequence gap is loud and that a book cannot be
     used before it is seeded.
  3. A randomised fuzz over 10,000 synthetic diffs — proves the invariants
     hold on sequences nobody wrote by hand, which is the only way to gain
     confidence about a state machine that will run for days.

The fuzz test carries a shadow model of the book built with completely
different code (plain dicts of floats-as-ints, no class involved). Testing an
implementation against a re-implementation is the point: if both had the same
bug, they would have to have it for different reasons.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from data_engine.book import (
    PRICE_SCALE,
    QTY_SCALE,
    BookInvariantError,
    OrderBook,
    SequenceGapError,
    to_fixed,
)

# A small, readable snapshot: three levels a side around a mid of 100.00.
SNAPSHOT = {
    "lastUpdateId": 1000,
    "bids": [["99.98", "1.0"], ["99.97", "2.0"], ["99.96", "3.0"]],
    "asks": [["100.02", "1.5"], ["100.03", "2.5"], ["100.04", "3.5"]],
}


def make_diff(first_id: int, final_id: int, bids=None, asks=None) -> dict:
    """Build a `depthUpdate` payload with the venue's field names."""
    return {
        "e": "depthUpdate",
        "s": "BTCUSDT",
        "U": first_id,
        "u": final_id,
        "b": bids or [],
        "a": asks or [],
    }


@pytest.fixture()
def book() -> OrderBook:
    seeded = OrderBook("BTCUSDT")
    seeded.apply_snapshot(SNAPSHOT)
    return seeded


# ------------------------------------------------- 1. known-answer sequences


def test_fixed_point_conversion_is_exact_where_float_is_not():
    """The concrete reason prices are integers and not floats.

    `0.29` and `0.57` are not representable in binary floating point, and the
    product with 10^8 lands just below the integer. Truncating gives a price
    one tick off, so the same level would occupy two different keys and the
    zero-quantity delete would miss one of them.
    """
    assert to_fixed("0.29", PRICE_SCALE) == 29_000_000
    assert int(float("0.29") * PRICE_SCALE) == 28_999_999

    assert to_fixed("0.57", PRICE_SCALE) == 57_000_000
    assert int(float("0.57") * PRICE_SCALE) == 56_999_999

    assert to_fixed("94210.51000000", PRICE_SCALE) == 9_421_051_000_000
    assert to_fixed("0.00000001", QTY_SCALE) == 1


def test_fixed_point_rejects_precision_it_cannot_hold():
    """Silently truncating a finer tick grid would corrupt level identity."""
    with pytest.raises(ValueError, match="fixed-point grid"):
        to_fixed("0.000000001", PRICE_SCALE)


def test_snapshot_populates_both_sides(book: OrderBook):
    assert book.last_update_id == 1000
    assert book.depth() == (3, 3)
    assert book.best_bid() == to_fixed("99.98", PRICE_SCALE)
    assert book.best_ask() == to_fixed("100.02", PRICE_SCALE)
    assert book.mid_price() == pytest.approx(100.00)
    assert book.spread() == pytest.approx(0.04)
    book.assert_valid()


def test_diff_updates_inserts_and_deletes_levels(book: OrderBook):
    """One event exercising all three level operations at once."""
    applied = book.apply_diff(
        make_diff(
            first_id=1001,
            final_id=1005,
            bids=[["99.98", "0"], ["99.99", "4.0"]],   # delete the touch, insert a better bid
            asks=[["100.03", "9.0"]],                  # resize an existing level
        )
    )

    assert applied is True
    assert book.last_update_id == 1005
    assert book.best_bid() == to_fixed("99.99", PRICE_SCALE)
    assert to_fixed("99.98", PRICE_SCALE) not in book.bids
    assert book.asks[to_fixed("100.03", PRICE_SCALE)] == to_fixed("9.0", QTY_SCALE)
    book.assert_valid()


def test_deleting_an_unknown_level_is_not_an_error(book: OrderBook):
    """Zero quantities for levels outside the snapshot's depth are normal."""
    book.apply_diff(make_diff(1001, 1001, bids=[["50.00", "0"]]))
    assert book.depth() == (3, 3)


def test_stale_event_is_ignored_not_applied(book: OrderBook):
    """An event fully covered by the snapshot must change nothing."""
    before = dict(book.bids)

    assert book.apply_diff(make_diff(900, 1000, bids=[["99.98", "999.0"]])) is False
    assert book.bids == before
    assert book.last_update_id == 1000


def test_event_overlapping_the_snapshot_is_applied(book: OrderBook):
    """The bracketing case from the docs: U <= lastUpdateId <= u."""
    assert book.apply_diff(make_diff(998, 1002, bids=[["99.98", "7.0"]])) is True
    assert book.bids[to_fixed("99.98", PRICE_SCALE)] == to_fixed("7.0", QTY_SCALE)
    assert book.last_update_id == 1002


def test_event_abutting_the_snapshot_is_applied(book: OrderBook):
    """The contiguous case: U == lastUpdateId + 1, no overlap and no hole."""
    assert book.apply_diff(make_diff(1001, 1002, asks=[["100.02", "8.0"]])) is True
    assert book.last_update_id == 1002


def test_multi_event_sequence_produces_the_expected_book(book: OrderBook):
    """A short realistic run, checked against a book written out by hand."""
    book.apply_diff(make_diff(1001, 1010, bids=[["99.97", "0"]], asks=[["100.05", "1.0"]]))
    book.apply_diff(make_diff(1011, 1020, bids=[["99.95", "5.0"]]))
    book.apply_diff(make_diff(1021, 1030, asks=[["100.02", "0"], ["100.03", "0"]]))

    expected_bids = {"99.98": "1.0", "99.96": "3.0", "99.95": "5.0"}
    expected_asks = {"100.04": "3.5", "100.05": "1.0"}
    assert book.bids == {
        to_fixed(price, PRICE_SCALE): to_fixed(qty, QTY_SCALE)
        for price, qty in expected_bids.items()
    }
    assert book.asks == {
        to_fixed(price, PRICE_SCALE): to_fixed(qty, QTY_SCALE)
        for price, qty in expected_asks.items()
    }
    assert book.last_update_id == 1030
    book.assert_valid()


def test_top_n_is_sorted_best_first_and_truncates(book: OrderBook):
    bids, asks = book.top_n(2)

    assert bids.tolist() == [
        [to_fixed("99.98", PRICE_SCALE), to_fixed("1.0", QTY_SCALE)],
        [to_fixed("99.97", PRICE_SCALE), to_fixed("2.0", QTY_SCALE)],
    ]
    assert asks.tolist() == [
        [to_fixed("100.02", PRICE_SCALE), to_fixed("1.5", QTY_SCALE)],
        [to_fixed("100.03", PRICE_SCALE), to_fixed("2.5", QTY_SCALE)],
    ]
    assert bids.dtype == np.int64


def test_top_n_does_not_pad_a_short_side(book: OrderBook):
    """Fewer levels than asked for returns fewer rows, never zero-filled rows."""
    bids, _asks = book.top_n(10)
    assert bids.shape == (3, 2)


def test_empty_book_reports_none_rather_than_guessing():
    empty = OrderBook("BTCUSDT")
    empty.apply_snapshot({"lastUpdateId": 1, "bids": [], "asks": []})

    assert empty.best_bid() is None
    assert empty.best_ask() is None
    assert empty.mid_price() is None
    assert empty.spread() is None
    assert empty.top_n(5)[0].shape == (0, 2)
    empty.assert_valid()


# ------------------------------------------------------------ 2. failure cases


def test_sequence_gap_raises_and_reports_how_much_was_missed(book: OrderBook):
    """The central correctness rule: a hole is never silently continued."""
    with pytest.raises(SequenceGapError) as caught:
        book.apply_diff(make_diff(1005, 1010, bids=[["99.98", "1.0"]]))

    error = caught.value
    assert error.expected_first_id == 1001
    assert error.event_first_id == 1005
    assert error.missed_updates == 4


def test_book_is_unchanged_after_a_gap(book: OrderBook):
    """Raising must not leave the book half-updated for a caller that retries."""
    before = dict(book.bids)
    with pytest.raises(SequenceGapError):
        book.apply_diff(make_diff(2000, 2001, bids=[["99.98", "42.0"]]))
    assert book.bids == before
    assert book.last_update_id == 1000


def test_diff_before_snapshot_is_refused():
    unseeded = OrderBook("BTCUSDT")
    with pytest.raises(RuntimeError, match="apply_snapshot"):
        unseeded.apply_diff(make_diff(1, 2))


def test_assert_valid_detects_a_crossed_book(book: OrderBook):
    """Reach past the public API to build the corruption we must catch."""
    book.bids[to_fixed("100.10", PRICE_SCALE)] = to_fixed("1.0", QTY_SCALE)

    with pytest.raises(BookInvariantError, match="crossed or locked"):
        book.assert_valid()


def test_assert_valid_detects_a_stored_zero_quantity(book: OrderBook):
    book.bids[to_fixed("99.90", PRICE_SCALE)] = 0

    with pytest.raises(BookInvariantError, match="non-positive"):
        book.assert_valid()


# ---------------------------------------------------------------- 3. the fuzz


def _format_fixed(value: int, scale: int) -> str:
    """Render a scaled integer back into the decimal string the venue sends."""
    return f"{value / scale:.8f}"


def _random_resizes(rng: random.Random, prices: list[int]) -> list[tuple[int, int]]:
    """Randomly delete or resize up to two of the given levels."""
    updates = []
    for price in rng.sample(prices, min(2, len(prices))):
        quantity = 0 if rng.random() < 0.5 else rng.randint(1, 500) * 1_000
        updates.append((price, quantity))
    return updates


def _generate_diff(
    rng: random.Random,
    shadow_bids: dict[int, int],
    shadow_asks: dict[int, int],
    first_id: int,
) -> tuple[dict, int]:
    """Produce one synthetic depthUpdate and fold it into the shadow model.

    Generates the two things that actually stress the book: levels appearing
    and disappearing near the touch, and the touch itself jumping to a new
    price. A jump is emitted as a *single* event that both deletes the levels
    being crossed and installs the new ones, which is how a real sweep
    arrives — the book never observes the intermediate crossed state, so a
    correct implementation never has to tolerate one.

    The generator keeps itself honest with a boundary price: after every
    event, every bid is strictly below the boundary and every ask strictly
    above it. Old levels on the wrong side are swept, new levels are placed
    on the right side, and surviving levels are only resized within the side
    they already occupy. Without a rule this explicit it is easy to write a
    generator that emits crossed books and then blame the book for them.
    """
    tick = 10_000
    boundary = rng.randint(9_000, 11_000) * tick

    bid_updates = [(price, 0) for price in shadow_bids if price >= boundary]
    ask_updates = [(price, 0) for price in shadow_asks if price <= boundary]

    for level in range(1, rng.randint(2, 7)):
        bid_updates.append((boundary - level * tick, rng.randint(1, 500) * 1_000))
        ask_updates.append((boundary + level * tick, rng.randint(1, 500) * 1_000))

    surviving_bids = [price for price in sorted(shadow_bids) if price < boundary]
    surviving_asks = [price for price in sorted(shadow_asks) if price > boundary]
    bid_updates.extend(_random_resizes(rng, surviving_bids))
    ask_updates.extend(_random_resizes(rng, surviving_asks))

    _apply_to_shadow(shadow_bids, bid_updates)
    _apply_to_shadow(shadow_asks, ask_updates)

    final_id = first_id + rng.randint(0, 5)
    event = make_diff(
        first_id,
        final_id,
        bids=[[_format_fixed(p, PRICE_SCALE), _format_fixed(q, QTY_SCALE)] for p, q in bid_updates],
        asks=[[_format_fixed(p, PRICE_SCALE), _format_fixed(q, QTY_SCALE)] for p, q in ask_updates],
    )
    return event, final_id


def _apply_to_shadow(side: dict[int, int], updates: list[tuple[int, int]]) -> None:
    """Independent re-implementation of the level update rule, on purpose."""
    for price, quantity in updates:
        if quantity == 0:
            side.pop(price, None)
        else:
            side[price] = quantity


def test_fuzz_ten_thousand_diffs_preserves_every_invariant():
    """Property test: no synthetic sequence of 10k diffs can break the book.

    Asserts after every single event rather than at the end, so a failure
    reports the event that broke it instead of the wreckage afterwards.
    """
    rng = random.Random(20260727)
    book = OrderBook("BTCUSDT")
    book.apply_snapshot({"lastUpdateId": 0, "bids": [], "asks": []})

    shadow_bids: dict[int, int] = {}
    shadow_asks: dict[int, int] = {}
    next_first_id = 1

    for step in range(10_000):
        event, final_id = _generate_diff(rng, shadow_bids, shadow_asks, next_first_id)
        previous_id = book.last_update_id

        assert book.apply_diff(event) is True, f"event {step} was unexpectedly treated as stale"
        book.assert_valid()

        assert book.last_update_id == final_id
        assert book.last_update_id >= previous_id, "update ids must never go backwards"
        next_first_id = final_id + 1

    assert book.bids == shadow_bids
    assert book.asks == shadow_asks


def test_fuzz_gaps_are_always_detected():
    """Inject a hole at a random point in each of many short sequences."""
    rng = random.Random(11)

    for _trial in range(200):
        book = OrderBook("BTCUSDT")
        book.apply_snapshot(SNAPSHOT)
        current_id = 1000

        for _step in range(rng.randint(1, 10)):
            final_id = current_id + rng.randint(1, 4)
            book.apply_diff(make_diff(current_id + 1, final_id, bids=[["99.90", "1.0"]]))
            current_id = final_id

        skipped = rng.randint(1, 50)
        with pytest.raises(SequenceGapError) as caught:
            book.apply_diff(make_diff(current_id + 1 + skipped, current_id + 60))
        assert caught.value.missed_updates == skipped
