"""Custom fixed-width binary storage format for order-book tapes.

WHAT
    Defines the on-disk layout — a 64-byte file header followed by a sequence
    of identical blocks, each block being one book snapshot plus a fixed
    number of event records — and the writer that produces it. The numpy
    structured dtypes that describe every byte are defined here **once** and
    exported; the reader in `replay.py` and everything in `ml/` import them
    and never redefine them.

WHY
    The tape is read far more often than it is written: every training epoch,
    every backtest, every latency benchmark walks it end to end. A layout that
    a memory-mapped numpy view can address directly removes parsing from the
    hot path entirely — not "makes parsing faster", removes it. Shrinking the
    record also shrinks the working set, which is what actually determines
    throughput on a laptop.

DESIGN DECISION — fixed-width binary records over CSV, JSON, or Parquet.
    Rejected: CSV and JSON — roughly an order of magnitude more bytes and a
    full text parse per row, with decimal-to-float round-tripping that is
    lossy in maddening ways. Rejected: Parquet — genuinely good, and the
    honest answer in a job. Rejected *here* because the entire point is to be
    able to say in an interview exactly what every byte on disk means and why
    the record is the size it is; Parquet outsources that understanding to a
    library. The cost we accept is owning versioning, endianness, alignment
    and corruption handling ourselves. Hence the mandatory round-trip test.

DESIGN DECISION — the file is a sequence of identical blocks, not a free
mixture of records.
    A block is `[1 snapshot][N events]`, always, with the final block padded
    out with explicitly tagged filler events so that every file is exactly
    `header + M * block_size` bytes. Rejected alternative: a free stream of
    tagged variable-width records, which is more flexible and is what the
    first draft of this format was. It cannot be memory-mapped into a single
    numpy view, because a numpy view needs a constant stride — so
    `load_snapshots()` would have to gather records one at a time, which is a
    copy and a Python loop, i.e. exactly what this format exists to avoid.
    Fixing the block shape buys: O(1) seek by arithmetic, a genuine zero-copy
    view of every snapshot in the file, and a trivially checkable invariant
    (file size must be an exact multiple of the block size). It costs at most
    `N-1` filler events per file, about 4 KB.

DESIGN DECISION — the record-type tag is byte 0 of each record, not a loose
byte between records.
    The tag is still one byte and still precedes every record's payload, but
    it lives *inside* the record's 8-byte-aligned prefix. Rejected
    alternative: writing a bare tag byte between records, which is the
    obvious reading of "a 1-byte tag before each record". That shifts every
    following field by one byte, so a 40-byte record starts at offset 1, then
    42, then 83 — every `int64` lands on an odd address, no numpy view can
    describe it, and on some architectures the loads fault outright. Alignment
    is not a micro-optimisation here; it is the difference between the format
    working and not working.

DESIGN DECISION — little-endian, explicitly, everywhere.
    Rejected alternative: network byte order (big-endian), the traditional
    choice for anything written to a wire. Every machine this will ever run
    on — x86-64 and ARM64 — is little-endian, so big-endian would mean a
    byte swap on every field on every read, buying portability to hardware
    that does not exist in our deployment. The dtypes below say `<` on every
    multi-byte field so the file is identical whatever the host does.

DESIGN DECISION — scaled `int64`, never `float64`.
    Prices live on a tick grid, so an `int64` of ticks is exact where a float
    is not, and exactness is what makes level identity, equality comparison
    and sorting correct near the touch. Inherited from `book.py`, and the
    reason the scales are recorded in the header rather than assumed.

WHAT THIS FORMAT DELIBERATELY LOSES
    A snapshot stores the top `depth_levels` of each side, not the whole book,
    so a book reconstructed by replaying events onto a snapshot is exact near
    the touch and progressively less complete further out. That is a
    deliberate trade for a pipeline whose features only ever read the top 10
    levels, and the snapshot interval bounds how far it can drift before the
    book is re-anchored exactly. It is recorded here rather than discovered
    later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = b"TTS1"
FORMAT_VERSION = 1
HEADER_SIZE = 64

# Record kinds, stored in byte 0 of every record.
KIND_SNAPSHOT = 1
KIND_EVENT = 2

# Event kinds, stored in byte 1 of an event record.
EVENT_DIFF = 0
EVENT_TRADE = 1
# Filler written only to round the final block out to its fixed width. Readers
# must skip it. It is tagged rather than zeroed so that "unused" is a state the
# format states explicitly, instead of one a reader has to infer from a
# suspiciously zero price.
EVENT_PADDING = 2

# Sides, stored in byte 2 of an event record.
SIDE_BID = 0
SIDE_ASK = 1
SIDE_NONE = 2  # trades carry an aggressor side; use this when it is unknown

DEFAULT_SNAPSHOT_INTERVAL = 100
DEFAULT_DEPTH_LEVELS = 10

# ---------------------------------------------------------------- the layout

# 64 bytes. Fixed size, not a length-prefixed or self-describing header,
# because a fixed size means the first record starts at a compile-time-known
# offset: `mmap[64:]` is the data, with no parse step before you can address
# it. Every field is placed on its natural alignment by hand, and the trailing
# reserved span exists so that a future version can add fields without moving
# anything that already exists.
HEADER_DTYPE = np.dtype(
    {
        "names": [
            "magic",
            "version",
            "reserved_after_version",
            "symbol",
            "price_scale",
            "qty_scale",
            "capture_start_ns",
            "snapshot_interval",
            "depth_levels",
            "reserved_tail",
        ],
        "formats": ["S4", "<u2", "V2", "S8", "<u4", "<u4", "<u8", "<u4", "<u4", "V24"],
        "offsets": [0, 4, 6, 8, 16, 20, 24, 32, 36, 40],
        "itemsize": HEADER_SIZE,
    }
)

# 40 bytes: one level change or one trade.
#   0      kind        u8    always KIND_EVENT
#   1      event_type  u8    EVENT_DIFF / EVENT_TRADE / EVENT_PADDING
#   2      side        u8    SIDE_BID / SIDE_ASK / SIDE_NONE
#   3-7    padding     5 B   pushes the first int64 to offset 8
#   8-15   local_ts_ns   u64
#   16-23  exchange_ts_ns u64
#   24-31  price       i64   fixed point, price_scale from the header
#   32-39  qty         i64   fixed point, qty_scale from the header
# The five padding bytes are the whole alignment story in miniature: the
# payload needs 36 bytes, but spending 4 more to keep every int64 on an
# 8-byte boundary is what lets numpy address this file without copying it.
EVENT_DTYPE = np.dtype(
    {
        "names": ["kind", "event_type", "side", "padding", "local_ts_ns", "exchange_ts_ns", "price", "qty"],
        "formats": ["u1", "u1", "u1", "V5", "<u8", "<u8", "<i8", "<i8"],
        "offsets": [0, 1, 2, 3, 8, 16, 24, 32],
        "itemsize": 40,
    }
)

EVENT_SIZE = EVENT_DTYPE.itemsize


def snapshot_dtype(depth_levels: int = DEFAULT_DEPTH_LEVELS) -> np.dtype:
    """Build the snapshot record dtype for a given ladder depth.

    At the default depth of 10 this is 344 bytes:
      0      kind         u8   always KIND_SNAPSHOT
      1-7    padding      7 B  pushes the first int64 to offset 8
      8-15   local_ts_ns    u64
      16-23  exchange_ts_ns u64
      24-183 bids   (10, 2) i64   [price, qty], best first, descending
      184-343 asks  (10, 2) i64   [price, qty], best first, ascending

    A level of `[0, 0]` means "this side had fewer than `depth_levels` levels".
    That is unambiguous rather than merely conventional: a real level always
    has a strictly positive price, which `OrderBook.assert_valid` enforces, so
    a zero price cannot be a real level.
    """
    level_bytes = depth_levels * 2 * 8
    return np.dtype(
        {
            "names": ["kind", "padding", "local_ts_ns", "exchange_ts_ns", "bids", "asks"],
            "formats": ["u1", "V7", "<u8", "<u8", ("<i8", (depth_levels, 2)), ("<i8", (depth_levels, 2))],
            "offsets": [0, 1, 8, 16, 24, 24 + level_bytes],
            "itemsize": 24 + 2 * level_bytes,
        }
    )


def block_dtype(snapshot_interval: int, depth_levels: int = DEFAULT_DEPTH_LEVELS) -> np.dtype:
    """Build the dtype for one `[snapshot][N events]` block.

    This is the dtype that makes the format work. Mapping a file as an array
    of these gives `blocks["snapshot"]` — every snapshot in the file, as a
    strided numpy view over the mapped pages, with no copy and no parse. Field
    access on a structured array returns a view, which is the entire reason
    the block shape is fixed.

    Every block size is a multiple of 8 (`344 + 40N`), so blocks never knock
    the following block out of alignment.
    """
    snapshot = snapshot_dtype(depth_levels)
    return np.dtype(
        {
            "names": ["snapshot", "events"],
            "formats": [snapshot, (EVENT_DTYPE, (snapshot_interval,))],
            "offsets": [0, snapshot.itemsize],
            "itemsize": snapshot.itemsize + snapshot_interval * EVENT_SIZE,
        }
    )


class FormatError(Exception):
    """Raised when a file is not a tape this module can read."""


@dataclass(frozen=True)
class TapeHeader:
    """The 64 bytes at the front of every tape, parsed.

    Frozen because a header describes a file that has already been written;
    mutating one would only ever mean the in-memory copy disagreed with disk.
    """

    symbol: str
    price_scale: int
    qty_scale: int
    capture_start_ns: int
    snapshot_interval: int = DEFAULT_SNAPSHOT_INTERVAL
    depth_levels: int = DEFAULT_DEPTH_LEVELS
    version: int = FORMAT_VERSION

    def to_bytes(self) -> bytes:
        encoded = self.symbol.upper().encode("ascii")
        if len(encoded) > 8:
            raise FormatError(f"symbol {self.symbol!r} does not fit the 8-byte header field")
        record = np.zeros(1, dtype=HEADER_DTYPE)
        record["magic"] = MAGIC
        record["version"] = self.version
        record["symbol"] = encoded
        record["price_scale"] = self.price_scale
        record["qty_scale"] = self.qty_scale
        record["capture_start_ns"] = self.capture_start_ns
        record["snapshot_interval"] = self.snapshot_interval
        record["depth_levels"] = self.depth_levels
        return record.tobytes()

    @classmethod
    def from_bytes(cls, raw: bytes) -> "TapeHeader":
        """Parse and validate a header, refusing anything we cannot read.

        Checked in order of how badly wrong each failure is: a wrong magic
        means this is not our file at all, a newer version means it is our
        file but written by code that knew things this code does not, and a
        zero interval or depth would make the block arithmetic divide by zero
        further downstream where the error would be unrecognisable.
        """
        if len(raw) < HEADER_SIZE:
            raise FormatError(f"file is {len(raw)} bytes, too short to hold a {HEADER_SIZE}-byte header")
        record = np.frombuffer(raw[:HEADER_SIZE], dtype=HEADER_DTYPE)[0]

        magic = bytes(record["magic"])
        if magic != MAGIC:
            raise FormatError(f"bad magic {magic!r}: expected {MAGIC!r}, this is not a tick-to-signal tape")

        version = int(record["version"])
        if version > FORMAT_VERSION:
            raise FormatError(f"tape is format version {version}; this build understands up to {FORMAT_VERSION}")

        snapshot_interval = int(record["snapshot_interval"])
        depth_levels = int(record["depth_levels"])
        if snapshot_interval <= 0 or depth_levels <= 0:
            raise FormatError(
                f"header declares snapshot_interval={snapshot_interval} depth_levels={depth_levels}; "
                "both must be positive for the block layout to make sense"
            )

        return cls(
            symbol=record["symbol"].decode("ascii"),
            price_scale=int(record["price_scale"]),
            qty_scale=int(record["qty_scale"]),
            capture_start_ns=int(record["capture_start_ns"]),
            snapshot_interval=snapshot_interval,
            depth_levels=depth_levels,
            version=version,
        )

    def block_dtype(self) -> np.dtype:
        return block_dtype(self.snapshot_interval, self.depth_levels)

    def snapshot_dtype(self) -> np.dtype:
        return snapshot_dtype(self.depth_levels)


# ---------------------------------------------------------------- the writer


class BinaryWriter:
    """Writes a tape. Buffers records; never one syscall per record.

    The cadence is enforced rather than assumed: the writer refuses an event
    when a snapshot is due and refuses a snapshot when it is not. The reader's
    O(1) seek and its zero-copy view both depend on every block having exactly
    `snapshot_interval` events, so letting a caller drift off cadence would
    silently produce a file that reads back as garbage. Better to fail at the
    write.
    """

    def __init__(
        self,
        path: Path,
        header: TapeHeader,
        buffer_bytes: int = 1 << 20,
    ) -> None:
        self.path = Path(path)
        self.header = header
        self.snapshots_written = 0
        self.events_written = 0

        self._snapshot_dtype = header.snapshot_dtype()
        self._buffer = bytearray()
        self._buffer_limit = buffer_bytes
        # Start "due" so the very first record must be a snapshot: a block
        # cannot begin with events, because there would be nothing to apply
        # them to.
        self._events_in_block = header.snapshot_interval
        self._snapshot_scratch = np.zeros(1, dtype=self._snapshot_dtype)
        self._event_scratch = np.zeros(1, dtype=EVENT_DTYPE)

        self._handle = open(self.path, "wb")
        self._handle.write(header.to_bytes())

    @property
    def needs_snapshot(self) -> bool:
        """True when the next record written must be a snapshot."""
        return self._events_in_block >= self.header.snapshot_interval

    def write_snapshot(self, local_ts_ns: int, exchange_ts_ns: int, bids: np.ndarray, asks: np.ndarray) -> None:
        """Start a new block with a top-of-book snapshot.

        `bids` and `asks` are `[price, qty]` int64 arrays straight out of
        `OrderBook.top_n`, best level first. Shorter sides are zero-filled to
        `depth_levels`.
        """
        if not self.needs_snapshot:
            raise FormatError(
                f"snapshot written off cadence: {self._events_in_block} of "
                f"{self.header.snapshot_interval} events into the current block"
            )
        record = self._snapshot_scratch
        record["kind"] = KIND_SNAPSHOT
        record["local_ts_ns"] = local_ts_ns
        record["exchange_ts_ns"] = exchange_ts_ns
        record["bids"] = _pad_levels(bids, self.header.depth_levels)
        record["asks"] = _pad_levels(asks, self.header.depth_levels)

        self._append(record.tobytes())
        self._events_in_block = 0
        self.snapshots_written += 1

    def write_event(self, local_ts_ns: int, exchange_ts_ns: int, event_type: int, side: int, price: int, qty: int) -> None:
        """Append one level change or trade to the current block."""
        if self.needs_snapshot:
            raise FormatError(
                f"event written when a snapshot is due (block already holds "
                f"{self._events_in_block} of {self.header.snapshot_interval} events)"
            )
        record = self._event_scratch
        record["kind"] = KIND_EVENT
        record["event_type"] = event_type
        record["side"] = side
        record["local_ts_ns"] = local_ts_ns
        record["exchange_ts_ns"] = exchange_ts_ns
        record["price"] = price
        record["qty"] = qty

        self._append(record.tobytes())
        self._events_in_block += 1
        self.events_written += 1

    def _append(self, encoded: bytes) -> None:
        self._buffer.extend(encoded)
        if len(self._buffer) >= self._buffer_limit:
            self._flush()

    def _flush(self) -> None:
        if self._buffer:
            self._handle.write(self._buffer)
            self._buffer.clear()

    def _pad_final_block(self) -> None:
        """Fill the open block with tagged filler so the file ends on a block.

        Costs at most `snapshot_interval - 1` records — about 4 KB at the
        default — and in exchange the reader can assert that the file size is
        an exact multiple of the block size, and map the whole thing in one
        view with no special case for the tail.
        """
        if self.snapshots_written == 0:
            return
        missing = self.header.snapshot_interval - self._events_in_block
        if missing <= 0:
            return
        filler = np.zeros(missing, dtype=EVENT_DTYPE)
        filler["kind"] = KIND_EVENT
        filler["event_type"] = EVENT_PADDING
        filler["side"] = SIDE_NONE
        self._append(filler.tobytes())
        self._events_in_block = self.header.snapshot_interval

    def close(self) -> None:
        if self._handle.closed:
            return
        self._pad_final_block()
        self._flush()
        self._handle.close()

    def __enter__(self) -> "BinaryWriter":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def _pad_levels(levels: np.ndarray, depth_levels: int) -> np.ndarray:
    """Zero-fill a `[price, qty]` ladder up to exactly `depth_levels` rows."""
    if levels.ndim != 2 or levels.shape[1] != 2:
        raise FormatError(f"expected a (n, 2) [price, qty] array, got shape {levels.shape}")
    if len(levels) > depth_levels:
        raise FormatError(f"got {len(levels)} levels, more than the header's depth_levels={depth_levels}")
    padded = np.zeros((depth_levels, 2), dtype=np.int64)
    padded[: len(levels)] = levels
    return padded


def block_size_bytes(header: TapeHeader) -> int:
    return header.block_dtype().itemsize


def bytes_per_event(header: TapeHeader) -> float:
    """Amortised on-disk cost of one event, including its share of snapshots.

    An event record is 40 bytes, but each block also carries one snapshot, so
    the honest per-event figure spreads that snapshot across the events it
    serves. This is the number the README quotes; quoting the bare 40 would
    be understating the format's real cost.
    """
    snapshot_bytes = header.snapshot_dtype().itemsize
    return EVENT_SIZE + snapshot_bytes / header.snapshot_interval
