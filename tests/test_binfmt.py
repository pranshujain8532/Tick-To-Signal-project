"""Tests for `data_engine.binfmt` and `data_engine.replay`.

A storage format has one job — give back exactly what it was given — and
exactly one way to fail catastrophically, which is to give back something
plausible but wrong. So the tests here are heavier on *exactness* than on
behaviour: byte offsets, alignment, round-trips at scale, and the property
that seeking cannot change what you read.

The four the constitution names as mandatory for this stage:
  * header round-trip,
  * 100k-record write/read exact equality,
  * seek correctness (seek + replay == full replay),
  * a corrupted file raising cleanly rather than returning garbage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data_engine.binfmt import (
    EVENT_DIFF,
    EVENT_DTYPE,
    EVENT_PADDING,
    EVENT_TRADE,
    HEADER_DTYPE,
    HEADER_SIZE,
    KIND_EVENT,
    KIND_SNAPSHOT,
    MAGIC,
    SIDE_ASK,
    SIDE_BID,
    BinaryWriter,
    FormatError,
    TapeHeader,
    block_dtype,
    bytes_per_event,
    snapshot_dtype,
)
from data_engine.replay import TapeReader, snapshot_mid_prices, snapshot_spreads

PRICE_SCALE = 10 ** 8
QTY_SCALE = 10 ** 8


def make_header(**overrides) -> TapeHeader:
    defaults = dict(
        symbol="BTCUSDT",
        price_scale=PRICE_SCALE,
        qty_scale=QTY_SCALE,
        capture_start_ns=1_785_164_539_514_000_000,
        snapshot_interval=100,
        depth_levels=10,
    )
    defaults.update(overrides)
    return TapeHeader(**defaults)


def make_ladder(best_price: int, step: int, depth: int = 10) -> np.ndarray:
    """A synthetic `[price, qty]` ladder walking away from `best_price`."""
    prices = best_price + step * np.arange(depth, dtype=np.int64)
    quantities = np.arange(1, depth + 1, dtype=np.int64) * 1_000_000
    return np.stack([prices, quantities], axis=1)


def write_tape(path: Path, event_count: int, header: TapeHeader | None = None) -> TapeHeader:
    """Write a tape of `event_count` synthetic events, snapshotting on cadence.

    Every field is a distinct function of the event index so that a reader
    that mixes two records up, or reads one field at the wrong offset, cannot
    accidentally still pass.
    """
    header = header or make_header()
    with BinaryWriter(path, header) as writer:
        for index in range(event_count):
            if writer.needs_snapshot:
                writer.write_snapshot(
                    local_ts_ns=2_000_000_000 + index,
                    exchange_ts_ns=1_000_000_000 + index,
                    bids=make_ladder(6_468_000_000_000 - index, -1_000_000, header.depth_levels),
                    asks=make_ladder(6_468_100_000_000 + index, 1_000_000, header.depth_levels),
                )
            writer.write_event(
                local_ts_ns=2_000_000_000 + index,
                exchange_ts_ns=1_000_000_000 + index,
                event_type=EVENT_DIFF if index % 3 else EVENT_TRADE,
                side=SIDE_BID if index % 2 else SIDE_ASK,
                price=6_468_000_000_000 + index,
                qty=index * 7 + 1,
            )
    return header


# ------------------------------------------------------------ layout itself


def test_record_sizes_are_exactly_what_the_format_promises():
    assert HEADER_DTYPE.itemsize == HEADER_SIZE == 64
    assert EVENT_DTYPE.itemsize == 40
    assert snapshot_dtype(10).itemsize == 344          # 24 header bytes + 40 levels * 8
    assert block_dtype(100, 10).itemsize == 344 + 100 * 40


def test_every_int64_field_is_eight_byte_aligned():
    """The reason the padding bytes exist at all.

    A misaligned int64 cannot be described by a numpy view over the mapping,
    which would silently turn the whole zero-copy design back into a parse
    loop. This asserts the property directly rather than trusting that the
    offsets were typed correctly.
    """
    for dtype in (HEADER_DTYPE, EVENT_DTYPE, snapshot_dtype(10)):
        for name in dtype.names:
            field_dtype, offset = dtype.fields[name][0], dtype.fields[name][1]
            base = field_dtype.base
            if base.itemsize == 8 and base.kind in "iu":
                assert offset % 8 == 0, f"{name} at offset {offset} is not 8-byte aligned"


def test_block_size_keeps_every_block_aligned():
    """Blocks must not knock the following block off an 8-byte boundary."""
    for interval in (1, 2, 7, 100, 1000):
        assert block_dtype(interval, 10).itemsize % 8 == 0


def test_all_multibyte_fields_are_little_endian():
    """Endianness is a decision, so it gets an assertion and not a comment."""
    for dtype in (HEADER_DTYPE, EVENT_DTYPE, snapshot_dtype(10)):
        for name in dtype.names:
            base = dtype.fields[name][0].base
            if base.itemsize > 1 and base.kind in "iu":
                assert base.byteorder in "<|=", f"{name} is not little-endian"


def test_amortised_bytes_per_event_accounts_for_the_snapshot():
    header = make_header(snapshot_interval=100, depth_levels=10)
    assert bytes_per_event(header) == pytest.approx(40 + 344 / 100)


# ------------------------------------------------------------ header round-trip


def test_header_round_trips_exactly():
    original = make_header()

    restored = TapeHeader.from_bytes(original.to_bytes())

    assert restored == original


def test_header_is_exactly_64_bytes_and_starts_with_the_magic():
    encoded = make_header().to_bytes()

    assert len(encoded) == HEADER_SIZE
    assert encoded[:4] == MAGIC


def test_header_refuses_a_symbol_that_does_not_fit():
    with pytest.raises(FormatError, match="8-byte header field"):
        make_header(symbol="TOOLONGSYMBOL").to_bytes()


# ------------------------------------------------------------- corrupt files


def test_bad_magic_raises_cleanly(tmp_path: Path):
    """The failure an interviewer will ask about: someone hands you a .zip."""
    path = tmp_path / "not_a_tape.tape"
    path.write_bytes(b"PK\x03\x04" + bytes(HEADER_SIZE * 2))

    with pytest.raises(FormatError, match="bad magic"):
        TapeReader(path)


def test_future_version_raises_rather_than_guessing(tmp_path: Path):
    path = tmp_path / "from_the_future.tape"
    path.write_bytes(make_header().to_bytes().replace(b"\x01\x00", b"\x63\x00", 1) + bytes(4344))

    with pytest.raises(FormatError, match="format version"):
        TapeReader(path)


def test_truncated_header_raises(tmp_path: Path):
    path = tmp_path / "stub.tape"
    path.write_bytes(MAGIC + b"\x01\x00")

    with pytest.raises(FormatError, match="too short"):
        TapeReader(path)


def test_empty_file_raises(tmp_path: Path):
    path = tmp_path / "empty.tape"
    path.write_bytes(b"")

    with pytest.raises(FormatError, match="empty"):
        TapeReader(path)


def test_zero_snapshot_interval_is_refused(tmp_path: Path):
    """A zero interval would divide by zero deep inside the block arithmetic."""
    broken = bytearray(make_header().to_bytes())
    broken[32:36] = (0).to_bytes(4, "little")
    path = tmp_path / "zero_interval.tape"
    path.write_bytes(bytes(broken))

    with pytest.raises(FormatError, match="must be positive"):
        TapeReader(path)


def test_truncated_tail_is_tolerated_not_fatal(tmp_path: Path):
    """A killed writer must not cost us the blocks that are whole."""
    path = tmp_path / "torn.tape"
    header = write_tape(path, event_count=500)
    block_size = block_dtype(header.snapshot_interval, header.depth_levels).itemsize

    truncated = path.read_bytes()[: HEADER_SIZE + 3 * block_size + 17]
    path.write_bytes(truncated)

    with TapeReader(path) as reader:
        assert reader.block_count == 3
        assert len(reader.load_snapshots()) == 3


# ------------------------------------------------------------ writer cadence


def test_writer_refuses_an_event_when_a_snapshot_is_due(tmp_path: Path):
    header = make_header(snapshot_interval=4)
    with BinaryWriter(tmp_path / "cadence.tape", header) as writer:
        assert writer.needs_snapshot is True
        with pytest.raises(FormatError, match="snapshot is due"):
            writer.write_event(1, 1, EVENT_DIFF, SIDE_BID, 100, 5)


def test_writer_refuses_a_snapshot_off_cadence(tmp_path: Path):
    header = make_header(snapshot_interval=4)
    with BinaryWriter(tmp_path / "cadence.tape", header) as writer:
        writer.write_snapshot(1, 1, make_ladder(100, -1), make_ladder(200, 1))
        writer.write_event(2, 2, EVENT_DIFF, SIDE_BID, 100, 5)
        with pytest.raises(FormatError, match="off cadence"):
            writer.write_snapshot(3, 3, make_ladder(100, -1), make_ladder(200, 1))


def test_final_block_is_padded_so_the_file_divides_exactly(tmp_path: Path):
    """The invariant every other guarantee in the reader is built on."""
    path = tmp_path / "padded.tape"
    header = write_tape(path, event_count=250)   # 2 full blocks + 50 events
    block_size = block_dtype(header.snapshot_interval, header.depth_levels).itemsize

    assert (path.stat().st_size - HEADER_SIZE) % block_size == 0

    with TapeReader(path) as reader:
        events = reader.load_events()
        assert reader.block_count == 3
        assert int((events["event_type"] == EVENT_PADDING).sum()) == 50


def test_a_tape_with_no_records_is_just_a_header(tmp_path: Path):
    path = tmp_path / "headeronly.tape"
    with BinaryWriter(path, make_header()):
        pass

    assert path.stat().st_size == HEADER_SIZE
    with TapeReader(path) as reader:
        assert reader.block_count == 0
        assert len(reader.load_snapshots()) == 0


# --------------------------------------------------- the 100k round-trip


def test_one_hundred_thousand_records_round_trip_exactly(tmp_path: Path):
    """Write 100k events, read them back, demand every field byte-identical."""
    path = tmp_path / "big.tape"
    event_count = 100_000
    header = write_tape(path, event_count)

    with TapeReader(path) as reader:
        assert reader.header == header
        assert reader.block_count == event_count // header.snapshot_interval

        events = reader.load_events().reshape(-1)
        assert len(events) == event_count
        assert int((events["event_type"] == EVENT_PADDING).sum()) == 0

        index = np.arange(event_count, dtype=np.int64)
        assert np.array_equal(events["kind"], np.full(event_count, KIND_EVENT, dtype=np.uint8))
        assert np.array_equal(events["price"], 6_468_000_000_000 + index)
        assert np.array_equal(events["qty"], index * 7 + 1)
        assert np.array_equal(events["local_ts_ns"], 2_000_000_000 + index)
        assert np.array_equal(events["exchange_ts_ns"], 1_000_000_000 + index)
        expected_side = np.where(index % 2, SIDE_BID, SIDE_ASK).astype(np.uint8)
        assert np.array_equal(events["side"], expected_side)


def test_snapshots_round_trip_exactly(tmp_path: Path):
    path = tmp_path / "snaps.tape"
    header = write_tape(path, event_count=1_000)

    with TapeReader(path) as reader:
        snapshots = reader.load_snapshots()
        assert len(snapshots) == 10
        assert np.array_equal(snapshots["kind"], np.full(10, KIND_SNAPSHOT, dtype=np.uint8))

        # Block k anchors at event index k * interval, by cadence.
        anchors = np.arange(10, dtype=np.int64) * header.snapshot_interval
        assert np.array_equal(snapshots["local_ts_ns"], 2_000_000_000 + anchors)
        for block, anchor in enumerate(anchors):
            assert np.array_equal(
                snapshots["bids"][block], make_ladder(6_468_000_000_000 - anchor, -1_000_000)
            )


def test_load_snapshots_is_a_view_and_not_a_copy(tmp_path: Path):
    """Zero-copy is the point of the format, so it gets asserted, not assumed."""
    path = tmp_path / "view.tape"
    write_tape(path, event_count=500)

    with TapeReader(path) as reader:
        snapshots = reader.load_snapshots()
        assert snapshots.flags.owndata is False
        assert snapshots.base is not None


# ------------------------------------------------------------ seek behaviour


def test_seek_lands_on_the_last_block_at_or_before_the_timestamp(tmp_path: Path):
    path = tmp_path / "seek.tape"
    write_tape(path, event_count=1_000)

    with TapeReader(path) as reader:
        timestamps = reader.snapshot_timestamps()
        assert reader.block_index_for_timestamp(int(timestamps[4])) == 4
        assert reader.block_index_for_timestamp(int(timestamps[4]) + 5) == 4
        assert reader.block_index_for_timestamp(int(timestamps[0]) - 10_000) == 0
        assert reader.block_index_for_timestamp(int(timestamps[-1]) + 10 ** 9) == 9


def test_snapshot_timestamps_are_non_decreasing(tmp_path: Path):
    """Binary search over them is only valid if they are sorted."""
    path = tmp_path / "sorted.tape"
    write_tape(path, event_count=5_000)

    with TapeReader(path) as reader:
        timestamps = reader.snapshot_timestamps()
        assert np.all(np.diff(timestamps.astype(np.int64)) >= 0)


def test_seek_then_replay_equals_full_replay(tmp_path: Path):
    """The mandatory seek test: seeking must not change what you read.

    Collect the book state at a given record via a full replay from the start
    of the file, then again by seeking straight to it, and demand the two
    books are identical — same levels, same quantities, both sides. This holds
    because every block re-anchors from its own snapshot, which is the
    property that design decision buys.
    """
    path = tmp_path / "parity.tape"
    write_tape(path, event_count=2_000)

    with TapeReader(path) as reader:
        target_ts = int(reader.snapshot_timestamps()[13]) + 40

        full_bids = full_asks = None
        for timestamp, book in reader.iter_books():
            if timestamp == target_ts:
                full_bids, full_asks = dict(book.bids), dict(book.asks)
                break

        seek_bids = seek_asks = None
        for timestamp, book in reader.iter_books(start_ts_ns=target_ts):
            seek_bids, seek_asks = dict(book.bids), dict(book.asks)
            break

    assert full_bids is not None and seek_bids is not None
    assert full_bids == seek_bids
    assert full_asks == seek_asks


def test_iter_books_never_yields_a_padding_record(tmp_path: Path):
    path = tmp_path / "nopad.tape"
    write_tape(path, event_count=150)   # leaves 50 padding records in block 1

    with TapeReader(path) as reader:
        yielded = sum(1 for _ in reader.iter_books())

    # 2 anchors + 150 real events; the 50 filler records must not appear.
    assert yielded == 152


def test_replayed_book_refuses_a_live_diff(tmp_path: Path):
    """A tape-seeded book has no sequence position, and must say so."""
    path = tmp_path / "nodiff.tape"
    write_tape(path, event_count=200)

    with TapeReader(path) as reader:
        _timestamp, book = next(iter(reader.iter_books()))
        assert book.is_initialised is False
        with pytest.raises(RuntimeError, match="apply_snapshot"):
            book.apply_diff({"U": 1, "u": 2, "b": [], "a": []})


# --------------------------------------------------- vectorised conveniences


def test_spreads_and_mids_are_computed_across_the_whole_tape(tmp_path: Path):
    path = tmp_path / "spread.tape"
    header = write_tape(path, event_count=300)

    with TapeReader(path) as reader:
        snapshots = reader.load_snapshots()
        spreads = snapshot_spreads(snapshots, header.price_scale)
        mids = snapshot_mid_prices(snapshots, header.price_scale)

    assert len(spreads) == 3
    assert np.all(np.isfinite(spreads))
    assert np.all(spreads > 0)
    assert np.all(mids > 0)


def test_one_sided_snapshot_reports_nan_not_a_fake_spread(tmp_path: Path):
    """An absent quote is not a zero spread, and a plot should show a hole."""
    path = tmp_path / "onesided.tape"
    header = make_header(snapshot_interval=2)
    empty = np.empty((0, 2), dtype=np.int64)
    with BinaryWriter(path, header) as writer:
        writer.write_snapshot(1, 1, make_ladder(100, -1), empty)
        writer.write_event(2, 2, EVENT_DIFF, SIDE_ASK, 200, 5)

    with TapeReader(path) as reader:
        spreads = snapshot_spreads(reader.load_snapshots(), header.price_scale)

    assert np.isnan(spreads[0])
