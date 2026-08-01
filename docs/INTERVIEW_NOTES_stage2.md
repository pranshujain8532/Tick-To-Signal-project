# Interview notes — Stage 2: custom binary storage format

What shipped: a fixed-width binary tape format with a 64-byte header and a
block structure, a memory-mapped zero-copy reader, the writer wired into the
capture daemon alongside the raw archive, 29 new tests (66 total), and a
benchmark that measured the format honestly enough to contradict the plan.

Files: [data_engine/binfmt.py](../data_engine/binfmt.py),
[data_engine/replay.py](../data_engine/replay.py),
[tests/test_binfmt.py](../tests/test_binfmt.py),
[benchmarks/bench_binfmt.py](../benchmarks/bench_binfmt.py),
[notebooks/02_binary_format_design.ipynb](../notebooks/02_binary_format_design.ipynb).

Measured results: [benchmarks/binfmt_20260727T162516Z.json](../benchmarks/binfmt_20260727T162516Z.json).

---

## 1. Design decisions and the alternatives rejected

### 1.1 Fixed-width binary over CSV, JSON, or Parquet — `binfmt.py:1-40`

Rejected CSV and JSON: ~10x the bytes uncompressed and a full text parse per
row. Rejected Parquet: genuinely good and the honest answer in a job, but the
entire point here is being able to say what every byte means; Parquet
outsources that understanding to a library. The cost accepted is owning
versioning, endianness, alignment and corruption handling — hence the
mandatory round-trip test.

### 1.2 A sequence of identical blocks, not a free stream of tagged records

This is the decision the whole format turns on, and it was **not** the first
draft. The obvious design — a free stream of variable-width tagged records —
cannot be memory-mapped into a single numpy view, because a numpy view needs
a *constant stride*. `load_snapshots()` would degrade into a per-record
gather in a Python loop, which is exactly the work the format exists to
delete.

Fixing the block shape to `[1 snapshot][N events]`, always, buys:
- **O(1) seek**: block `k` is at `64 + k * block_size`, no index, no scan.
- **A genuine zero-copy view**: `blocks["snapshot"]` is a strided window.
- **A checkable invariant**: file size must be an exact multiple of the block.

It costs at most `N-1` filler events per file (~4 KB), and the rigidity that
the writer must never drift off cadence — enforced, not assumed, by
`BinaryWriter` refusing an event when a snapshot is due.

### 1.3 The record-type tag is byte 0 of the record, not a byte between records

The stage spec asked for "a 1-byte record-type tag before each record". Taken
literally — a bare byte *between* records — every following field shifts by
one: 40-byte records start at offsets 1, 42, 83, every `int64` lands on an odd
address, no numpy view can describe it, and on some architectures the loads
fault outright.

So the tag is still one byte and still precedes every payload, but it lives
inside the record's 8-byte-aligned prefix. Same property, zero cost. This is
the deviation from spec I would raise first in a review, because it is a
deviation in letter that preserves the intent.

### 1.4 Explicit padding to keep every `int64` 8-byte aligned — `binfmt.py:EVENT_DTYPE`

The event payload needs 36 bytes; the record spends 40. Three bytes at offset
3 push `local_ts_ns` to offset 8, and the rest follows. `test_every_int64_field_is_eight_byte_aligned`
asserts the property directly rather than trusting the offsets were typed
correctly — layout constants are right when written and wrong after the third
edit. Block size `344 + 40N` is a multiple of 8 for any `N`, so blocks cannot
knock each other off a boundary.

### 1.5 Little-endian, declared rather than native

Rejected big-endian ("network byte order"): every machine this will run on is
little-endian, so it would cost a byte swap per field per read for portability
to hardware not in the deployment. Declaring `<` rather than leaving it native
is the important half — "native" means a tape written on one machine could be
misread on another, and the symptom would be absurd values rather than an
error. Asserted by `test_all_multibyte_fields_are_little_endian`.

### 1.6 The final block is padded with tagged filler

Rejected: leave a ragged tail and special-case it in the reader. Padding costs
~4 KB per file and in exchange every file is exactly `header + M * block_size`,
so the reader maps the whole thing in one view with no tail logic. The filler
is tagged `EVENT_PADDING` rather than zeroed, so "unused" is a state the
format states explicitly instead of one a reader infers from a suspiciously
zero price.

### 1.7 The book is re-anchored at every snapshot — `replay.py:iter_books`

Rejected: seed once and apply every event to the end of the file. That sounds
more faithful, but it makes state at event *n* depend on all *n* preceding
events, so seeking into the middle would produce a *different* book than
reading from the start — a seek that silently returns something else is worse
than no seek. Re-anchoring makes `seek + replay` and `full replay` produce
identical state **by construction**, which is a property a test can assert
(`test_seek_then_replay_equals_full_replay`), and it bounds reconstruction
drift to one block.

### 1.8 `close()` does not close the mmap — `replay.py:TapeReader.close`

Found by tests, not by reasoning: six tests failed with
`BufferError: cannot close exported pointers exist`. Every array from
`load_snapshots` exports a buffer onto the mapping, so `mmap.close()` raises
while any of them is alive — which would force callers to `del` their arrays
before leaving a `with` block. That is a trap, not an API. Dropping our
reference instead lets refcounting release the mapping when the last view
dies. Stated cost: arrays outlive the reader and keep the mapping alive with
them, so long-lived callers should keep a copy of the columns they need rather
than the view.

### 1.9 Bulk column decode in the sequential path — `replay.py:_decode_block_events`

Indexing the structured array record by record boxes a numpy scalar per field
per record, and that boxing — not file access — dominates a sequential replay.
`.tolist()` converts a whole column in one C-level pass. This is why the
row-at-a-time path is worth having next to the vectorised one at all.

### 1.10 A new mutation entry point on `OrderBook` — `book.py:apply_level_update`

Tape replay needs to apply one level at a time, and the tape does not carry
exchange update ids — they did their job at capture time, where a gap forced a
resync. So `apply_level_update` performs no sequence checking, and is named so
that its use is obvious in a diff. A book seeded from a tape keeps
`last_update_id` at its uninitialised sentinel, so feeding it a live diff
raises immediately instead of inventing a check against a meaningless number
(`test_replayed_book_refuses_a_live_diff`). Notebook 01 was updated in the
same session, as the constitution requires.

---

## 2. The ten hardest questions

**Q1. Why a fixed 64-byte header rather than a length-prefixed one?**
Because a fixed size means the first record starts at a compile-time-known
offset, so a reader can address any record by arithmetic before parsing
anything — a variable header forces a parse step before the data can even be
located, which is the class of work this format exists to eliminate. The 24
reserved bytes let a future version add fields without moving existing ones.
The cost is a hard 8-byte symbol field, which raises rather than truncating.
→ `binfmt.py:HEADER_DTYPE`, `TapeHeader.to_bytes`.

**Q2. Why is the type tag inside the record instead of before it?**
A loose byte between records shifts every subsequent field by one, putting
every `int64` on an odd address. numpy cannot describe that with a strided
view, so the zero-copy design collapses into a per-record gather, and
misaligned 8-byte loads fault on some architectures. Byte 0 of an aligned
record keeps "tag precedes payload" at zero cost.
→ `binfmt.py:1-40` (design decision), `tests/test_binfmt.py:test_every_int64_field_is_eight_byte_aligned`.

**Q3. Walk me through what makes `load_snapshots` actually zero-copy.**
The file is `header + M` identical blocks, so a block dtype describes the
whole payload as a contiguous array; `np.frombuffer` over the mapping builds
that array without allocating, and structured *field* access returns a view,
so `blocks["snapshot"]` strides one 344-byte record out of every 4,344-byte
block. The notebook demonstrates it rather than asserting it: `owndata=False`,
a non-`None` `base`, 34,400 bytes described in ~32 µs, and the time does not
grow with file size because nothing proportional to the file happened.
→ `binfmt.py:block_dtype`, `replay.py:load_snapshots`, notebook 02 §7.

**Q4. Your format is 6.6x LARGER than gzipped JSON. Did it fail?**
On size, yes — and the plan predicted 10–20x smaller, so this is a
falsified prediction, not a caveat. Two causes, both explainable: we store 16
bytes of timestamp on *every* level record where one JSON frame carries ~39
levels sharing one pair, and JSON of decimal strings gzips extraordinarily
well. But size was the wrong objective for an artefact written once and read
every training epoch: measured read throughput is 4.9x faster sequentially and
261x faster for the vectorised access pattern Stage 3 actually uses. The
specific fix if disk ever binds is known — hoist shared timestamps into a
per-frame header record and drop the event record to 24 bytes.
→ `benchmarks/binfmt_20260727T162516Z.json`, notebook 02 §9.

**Q5. Why is the sequential speedup only 4.9x when the vectorised one is 261x?**
Because the sequential comparison is Python loop against Python loop — both
sides are dominated by interpreter overhead, not by the format, so it is the
*pessimistic* bound on what the format buys. The vectorised gap is the real
thesis: getting a spread series out of JSON requires parsing every frame and
rebuilding the book because JSON has no addressable structure, while getting
it out of the tape is a strided view and a column subtraction. One cost is
proportional to the data, the other to the answer.

**Q6. How do you choose the snapshot interval?**
It is a direct seek-versus-size dial. `N=10` costs 34.4 B/event of snapshot
overhead (86%) and never replays more than 10 events after a seek; `N=1000`
costs 0.34 B/event (0.9%) and may replay 1,000. We chose 100: 3.44 B/event,
8.6% overhead, and replaying 100 events costs microseconds. It lives in the
header rather than in a reader constant so that tapes written with different
intervals stay readable.
→ notebook 02 §6, `binfmt.py:bytes_per_event`.

**Q7. What happens if the capture process is killed mid-write?**
The tape ends with a partial block, because padding happens at close. The
reader computes whole blocks, logs a warning naming the leftover byte count,
and reads what is intact — throwing away hours of good data because the last
four kilobytes are missing would be the wrong call. This is not hypothetical:
it happened during Stage 2 when a shell pipeline closed early and killed the
daemon.
→ `replay.py:_count_whole_blocks`, `tests/test_binfmt.py:test_truncated_tail_is_tolerated_not_fatal`.

**Q8. Your snapshots only hold 10 levels. Isn't the reconstructed book wrong?**
It is deliberately lossy, and the docstring says so rather than leaving it to
be discovered. A book replayed from a tape is exact near the touch and
progressively less complete further out, because a snapshot re-anchors only
the top 10. That is the right trade for a pipeline whose features read the top
10 and nothing else, and the snapshot interval bounds how far it can drift
before being re-anchored exactly. If Stage 3 ever needs depth 20, it is a
header field, not a format change.
→ `binfmt.py` "WHAT THIS FORMAT DELIBERATELY LOSES".

**Q9. Why keep the raw JSON archive if you have a binary tape?**
Because the binary format needs something independent to be validated
*against*, and the raw frames are the only ground truth that does not depend
on our own parsing being correct. The tape is derived; the archive is
evidence. Dropping it would leave the format checkable only against itself.
The capture daemon writes both, and the benchmark compares them on data from
the same session for exactly this reason.
→ `capture.py:CaptureConfig.write_binary_tape`.

**Q10. What would you change?**
Three things, in order. Hoist per-frame timestamps out of level records —
that is a 40% cut in event size and the single highest-value change. Add a
CRC or checksum per block, since right now a bit flip in the middle of a tape
is undetectable and would surface as a strange price rather than an error.
And record the exchange update id on the tape after all: it costs 8 bytes but
would let a replayed book be re-validated against the venue offline, which is
currently only possible via the raw archive.

---

## 3. What Stage 2 measured

Produced by `benchmarks/bench_binfmt.py`, run on one capture session that
wrote both artefacts from the same market data. Machine: 13th Gen Intel Core
i5-13420H, 15.7 GB RAM, Windows 11 (10.0.26200), Python 3.13.12, numpy 2.4.3.
No core pinning, no thermal control — these are throughput numbers spanning
seconds, where scheduling noise averages out. 7 timed passes after one warmup;
median reported, never mean.

Input: 12,000 websocket messages / 258 s → 107,223 tape events, 1,073 blocks.

### Size

| Artefact | Bytes | Bytes/event |
|---|---|---|
| Raw JSONL, uncompressed | 7,321,739 | 68.3 |
| Raw JSONL, gzipped | 710,446 | 6.6 |
| Binary tape | 4,661,176 | 43.5 |
| Binary tape, gzipped | 798,022 | 7.4 |

- Binary is **1.57x smaller** than uncompressed JSON.
- Binary is **6.56x larger** than gzipped JSON.
- Gzipped binary is **1.12x larger** than gzipped JSON.

### Read throughput

| Path | Median | Rate |
|---|---|---|
| Binary sequential replay (`iter_books`) | 0.054 s | 2,009,049 records/s |
| JSON sequential replay | 0.262 s | 410,441 records/s |
| Binary vectorised spread series | 0.001 s | 783,956 snapshots/s |
| JSON replay to the same series | 0.358 s | 2,741 snapshots/s |

- Sequential replay: **4.86x faster**.
- Vectorised extraction: **261x faster**.

### Caveats recorded with the numbers

- The binary record count includes one block anchor per 100 events (~1% of
  records); the JSON count has no equivalent. Wall-clock ratios are the honest
  comparison, not the per-record rates.
- `json_plain_bytes` includes the REST depth snapshots the daemon archives for
  offline cross-checking; the tape does not store those, so JSON carries a
  little data the binary format does not.
- Laptop, desktop OS, no pinning, no thermal control.
