"""Deterministic replay of a binary tape into book states.

WHAT
    Memory-maps a `binfmt` tape and offers two ways to read it: a vectorised
    path (`load_snapshots`, `load_events`) that hands back numpy views over
    the mapped pages with no copy and no parse, and a sequential path
    (`iter_books`) that reconstructs `OrderBook` states event by event. Both
    read the same bytes; neither redefines a dtype.

WHY
    One replay implementation means training and backtesting cannot disagree
    about what the market looked like at time t. The classic and very
    expensive research bug is a train/backtest skew where two code paths
    reconstruct state slightly differently; the cheapest defence is to have
    exactly one path.

DESIGN DECISION — mmap plus a numpy view, not `file.read()` into a buffer.
    Rejected alternative: read the file into `bytes` and parse it. That costs
    one full copy of the tape into the heap before any work happens, and it
    caps working-set size at RAM. Memory mapping lets the OS page in only what
    is touched, shares those pages between processes reading the same tape,
    and — because the layout is fixed-width and aligned — lets numpy address
    the pages directly. The cost, stated plainly: every array handed out by
    this module is a window onto a mapping, and holding one keeps that mapping
    alive after the reader is closed. That is the real price of zero-copy —
    see `TapeReader.close` for why it is paid this way round.

DESIGN DECISION — the book is re-anchored at every snapshot, not carried
across blocks.
    Each block begins with a fresh top-of-book snapshot, and `iter_books`
    rebuilds the book from it rather than continuing to accumulate. Rejected
    alternative: seed once and apply every event to the end of the file. That
    sounds more faithful, but it makes the state at event *n* depend on all
    *n* preceding events, so seeking to the middle of a file would produce a
    different book than reading it from the start — a seek that silently
    returns something else is worse than no seek. Re-anchoring makes
    `seek + replay` and `full replay` produce *identical* state by
    construction, which is a property a test can assert, and it bounds
    reconstruction drift to one block.

DESIGN DECISION — a generator over events, not a materialised DataFrame.
    Rejected alternative: load the whole session into memory and index it.
    More convenient in a notebook, but it makes look-ahead *possible* — with
    the full array in scope nothing stops a feature function from reaching
    forward, and nothing catches it if one does. A forward-only generator
    makes look-ahead structurally awkward, which is a stronger guarantee than
    a code review.

DESIGN DECISION — replay is wall-clock-free.
    We advance on record order, not by sleeping until the next timestamp.
    Pacing is only needed for a serving demo, so it does not belong in the
    path that runs on every training epoch.

INFORMATION HORIZON
    Everything this module yields at step t is derived from records at or
    before t. Labels are not produced here — they live in `ml/labels.py`,
    which is the only place allowed to look forward, and which must say by
    how much.
"""

from __future__ import annotations

import logging
import mmap
from pathlib import Path
from typing import Iterator

import numpy as np

from data_engine.binfmt import (
    EVENT_DIFF,
    EVENT_PADDING,
    HEADER_SIZE,
    KIND_SNAPSHOT,
    SIDE_BID,
    FormatError,
    TapeHeader,
)
from data_engine.book import OrderBook

LOGGER = logging.getLogger("replay")


class TapeReader:
    """Read-only view over one binary tape.

    Use as a context manager. Arrays returned by `load_snapshots` and
    `load_events` are views over the memory mapping; they stay valid after the
    reader closes, because holding one keeps the mapping alive.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._handle = open(self.path, "rb")
        try:
            self._mapping = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)
        except ValueError as error:  # empty file: mmap refuses a zero-length map
            self._handle.close()
            raise FormatError(f"{self.path} is empty, so it cannot contain a tape header") from error

        try:
            self.header = TapeHeader.from_bytes(self._mapping[:HEADER_SIZE])
            self.block_dtype = self.header.block_dtype()
            self.block_count = self._count_whole_blocks()
            self.blocks = np.frombuffer(
                self._mapping, dtype=self.block_dtype, count=self.block_count, offset=HEADER_SIZE
            )
        except Exception:
            self.close()
            raise

    def _count_whole_blocks(self) -> int:
        """How many complete blocks the file holds, tolerating a torn tail.

        The writer pads the final block so a cleanly closed tape divides
        exactly. A file that does not divide exactly was therefore cut short —
        a killed process, a full disk — and the sane response is to read the
        blocks that are whole and say so, rather than throw away hours of good
        data because the last four kilobytes are missing.
        """
        payload_bytes = len(self._mapping) - HEADER_SIZE
        block_bytes = self.block_dtype.itemsize
        whole = payload_bytes // block_bytes
        leftover = payload_bytes - whole * block_bytes
        if leftover:
            LOGGER.warning(
                "%s has %d trailing bytes that do not form a whole %d-byte block; "
                "the tape was probably truncated. Reading %d whole blocks.",
                self.path.name,
                leftover,
                block_bytes,
                whole,
            )
        return whole

    # ------------------------------------------------------ vectorised path

    def load_snapshots(self) -> np.ndarray:
        """Every snapshot record in the file, as a zero-copy structured view.

        This is the whole reason the block layout is fixed. Field access on a
        structured array returns a view, so `blocks["snapshot"]` is a strided
        window onto the mapped pages — no allocation, no parse, no Python
        loop, whatever the size of the tape.
        """
        return self.blocks["snapshot"]

    def load_events(self) -> np.ndarray:
        """Every event record, shaped `(block_count, snapshot_interval)`.

        Also a view. Padding records are included — filtering them would mean
        copying, and callers that care can mask on
        `events["event_type"] != EVENT_PADDING` themselves.
        """
        return self.blocks["events"]

    def snapshot_timestamps(self) -> np.ndarray:
        """Local receive timestamp of each block's anchor, for seeking."""
        return self.load_snapshots()["local_ts_ns"]

    def block_index_for_timestamp(self, local_ts_ns: int) -> int:
        """Index of the last block anchored at or before `local_ts_ns`.

        Binary search over a contiguous int64 array — O(log n) with no file
        access beyond the pages it touches. Timestamps are non-decreasing
        because the writer appends in receive order, which is what makes the
        search valid; `test_binfmt` asserts it.
        """
        timestamps = self.snapshot_timestamps()
        position = int(np.searchsorted(timestamps, local_ts_ns, side="right")) - 1
        return max(position, 0)

    # ------------------------------------------------------ sequential path

    def iter_books(self, start_ts_ns: int | None = None) -> Iterator[tuple[int, OrderBook]]:
        """Yield `(local_ts_ns, book)` for the anchor and every real event.

        Seeks to the block covering `start_ts_ns` and replays forward from
        there. Because every block re-anchors the book from its own snapshot,
        the state produced here is identical to the state a full replay would
        reach at the same record — seeking is exact, not approximate.

        The same `OrderBook` instance is yielded every time and mutated in
        place. That is deliberate — allocating a book per event would dominate
        the cost of reading the tape — but it means a consumer that wants to
        keep a state must copy what it needs out of it, typically via
        `top_n`. Holding the reference and reading it later gives the state at
        the *end* of the loop, which is a bug this docstring exists to prevent.
        """
        book = OrderBook(self.header.symbol)
        first_block = 0 if start_ts_ns is None else self.block_index_for_timestamp(start_ts_ns)

        for block_index in range(first_block, self.block_count):
            block = self.blocks[block_index]
            _seed_book_from_snapshot(book, block["snapshot"])
            anchor_ts = int(block["snapshot"]["local_ts_ns"])
            if start_ts_ns is None or anchor_ts >= start_ts_ns:
                yield anchor_ts, book

            for event_type, side, price, quantity, event_ts in _decode_block_events(block["events"]):
                if event_type == EVENT_PADDING:
                    break  # padding only ever occupies the tail of the last block
                if event_type == EVENT_DIFF:
                    book.apply_level_update(is_bid=side == SIDE_BID, price=price, quantity=quantity)
                if start_ts_ns is None or event_ts >= start_ts_ns:
                    yield event_ts, book

    # ------------------------------------------------------------ lifecycle

    def close(self) -> None:
        """Release our handle on the tape.

        Deliberately does *not* call `mmap.close()`. Every array handed out by
        `load_snapshots` and `load_events` exports a buffer onto the mapping,
        and `mmap.close()` raises `BufferError` while any of them is alive —
        so an explicit close would force callers to `del` their arrays before
        leaving the `with` block, which is a trap rather than an API. Dropping
        our reference instead lets ordinary refcounting release the mapping
        once the last view dies.

        The consequence, stated plainly: arrays outlive the reader, and the
        mapping outlives it with them. That is usually what you want. It does
        mean a process that opens many tapes and keeps one array from each
        holds all those mappings open, so long-lived callers should keep only
        what they need — usually a copy of a few columns rather than the view.
        """
        self.blocks = None
        self._mapping = None
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "TapeReader":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"TapeReader({self.path.name}, symbol={self.header.symbol}, "
            f"blocks={self.block_count}, interval={self.header.snapshot_interval})"
        )


def _decode_block_events(events: np.ndarray) -> Iterator[tuple[int, int, int, int, int]]:
    """Decode one block's event records into plain Python ints, column at a time.

    `.tolist()` converts a whole column in a single C-level pass. The obvious
    alternative — indexing the structured array record by record and calling
    `int()` on each field — boxes a numpy scalar for every field of every
    record, and that boxing, not the file access, dominates the cost of a
    sequential replay. Doing it per column instead is what keeps the
    row-at-a-time path worth having next to the vectorised one.
    """
    return zip(
        events["event_type"].tolist(),
        events["side"].tolist(),
        events["price"].tolist(),
        events["qty"].tolist(),
        events["local_ts_ns"].tolist(),
    )


def _seed_book_from_snapshot(book: OrderBook, snapshot: np.void) -> None:
    """Reset `book` to the ladder held in one snapshot record.

    Leaves `last_update_id` at its uninitialised sentinel on purpose. The tape
    does not carry exchange update ids — they did their job at capture time,
    where a gap forced a resync — so a book replayed from a tape has no
    meaningful sequence position. Leaving the sentinel means that calling
    `apply_diff` on such a book raises immediately instead of inventing a
    sequence check against a number that means nothing.
    """
    book.bids.clear()
    book.asks.clear()
    _fill_side(book.bids, snapshot["bids"])
    _fill_side(book.asks, snapshot["asks"])


def _fill_side(side: dict[int, int], levels: np.ndarray) -> None:
    """Load `[price, qty]` rows into a book side, skipping the zero padding."""
    for price, quantity in levels:
        # A real level always has a strictly positive price, so price == 0 is
        # unambiguously the writer's zero-fill for a side shorter than depth.
        if price > 0:
            side[int(price)] = int(quantity)


# ------------------------------------------------- vectorised conveniences


def snapshot_best_prices(snapshots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best bid and best ask across every snapshot, in fixed-point units.

    Column 0 of row 0 of each ladder, which the writer guarantees is the best
    level because `OrderBook.top_n` returns best-first. Vectorised over the
    whole tape: this is the shape of every analysis in Stage 3, and the reason
    `load_snapshots` returning a real array rather than an iterator matters.
    """
    return snapshots["bids"][:, 0, 0], snapshots["asks"][:, 0, 0]


def snapshot_spreads(snapshots: np.ndarray, price_scale: int) -> np.ndarray:
    """Spread in real price units for every snapshot, as float64.

    Returns NaN where either side was empty, rather than a misleading zero or
    a negative number — an absent quote is not a zero spread, and a plot
    should show a hole where there was no market.
    """
    best_bid, best_ask = snapshot_best_prices(snapshots)
    spread = (best_ask - best_bid).astype(np.float64) / price_scale
    one_sided = (best_bid <= 0) | (best_ask <= 0)
    return np.where(one_sided, np.nan, spread)


def snapshot_mid_prices(snapshots: np.ndarray, price_scale: int) -> np.ndarray:
    """Mid price in real units for every snapshot, NaN where one-sided."""
    best_bid, best_ask = snapshot_best_prices(snapshots)
    mid = (best_bid + best_ask).astype(np.float64) / (2 * price_scale)
    one_sided = (best_bid <= 0) | (best_ask <= 0)
    return np.where(one_sided, np.nan, mid)
