# Bundled sample data

Small slices of real captured BTCUSDT data, committed so the notebooks execute
from a clean checkout with no network and no prior capture run. Everything here
is genuine venue data recorded by `data_engine.capture` — nothing is synthetic
or hand-edited.

| File | Size | What it is | Used by |
|---|---|---|---|
| `btcusdt_sample.jsonl.gz` | 380 KB | 8,002 raw websocket frames + the REST snapshots the daemon fetched | notebook 01 |
| `btcusdt_sample.tape` | 424 KB | First 100 blocks of a `snapshot_interval=100` tape | notebook 02 |
| `btcusdt_dense_sample.tape` | 1.1 MB | First 1,500 blocks of a `snapshot_interval=10` tape | notebook 03 |
| `btcusdt_session_mids.npz` | 74 KB | Mid-price series and timestamps for three complete capture sessions | notebook 03 |
| `btcusdt_replay_s0/s1/s2.tape` | 4.1 MB | ~70 s from each of the three capture sessions, for the serving demo | Stage 8 |

## Provenance

All captured 2026-07-27, BTCUSDT, `btcusdt@depth@100ms` + `btcusdt@trade`.

```bash
# notebook 01 and 02 samples
python -m data_engine.capture --symbol btcusdt --out data/ --max-messages 8000
python -m data_engine.capture --symbol btcusdt --out data/ --max-messages 12000

# notebook 03 sample: denser snapshot grid for the ML pipeline
python -m data_engine.capture --symbol btcusdt --out data/stage3 \
    --snapshot-interval 10 --max-messages 45000
```

The `.tape` files are byte-prefixes of the captured tapes. Truncating on a
block boundary leaves a completely valid file — that is a property of the
format, demonstrated in notebook 02 — so these are not re-encoded, just cut.

## Why the `.npz` exists

Notebook 03 chooses the label parameters `(k, alpha)` from a class-balance
grid, and that grid is only meaningful over a **whole capture session**: a
short slice is directionally biased, so it produces a misleading balance. The
first 1,500 blocks of session 0 give an imbalance ratio of 5.9 where the full
14,109-snapshot session gives 1.08 — the slice is simply a stretch of uptrend.

Bundling all three full sessions as tapes would cost 29 MB. The grid only needs
the mid series, which compresses to 74 KB because the mid is unchanged 99% of
the time. So the `.npz` carries the mids for all three sessions, and the tape
sample carries the full book depth needed to demonstrate feature construction.

This is derived data, and the notebook treats it as such: it re-derives the
mids from `btcusdt_dense_sample.tape` and asserts they are bit-identical to the
first 1,500 entries of `session_0_mids`, so the derived file cannot silently
drift from the tape it came from.

### Contents of the `.npz`

| Array | Length | Notes |
|---|---|---|
| `session_0_mids`, `session_0_ts_ns` | 14,109 | 483 s |
| `session_1_mids`, `session_1_ts_ns` | 19,614 | 604 s |
| `session_2_mids`, `session_2_ts_ns` | 5,536 | 181 s |

Three sessions rather than one because the capture daemon resynced twice, and
each session is a separate contiguous tape. Keeping them separate is deliberate
— concatenating them would invent mid moves across the gaps that never happened.

## Why the serving demo needs three tapes, not one

`btcusdt_dense_sample.tape` is a prefix of session 0 alone, so it contains **no
session boundary** — and at 1,500 blocks it is 65.8 s of market, which loops
every 6.6 s at the demo's 10x. Both are wrong for Stage 8: the dashboard's whole
point is that a resync is visible, and a 6.6 s loop is too short to watch.

`btcusdt_replay_s0/s1/s2.tape` are ~70 s prefixes of the three real sessions,
cut on block boundaries the same way as the other samples:

| tape | blocks | extent | anchor rate | source |
|---|---|---|---|---|
| `btcusdt_replay_s0.tape` | 1,571 | 70.0 s | 22.4 Hz | `btcusdt_20260727T191845_898907Z.tape` |
| `btcusdt_replay_s1.tape` | 1,704 | 69.8 s | 24.4 Hz | `btcusdt_20260727T192658_360768Z.tape` |
| `btcusdt_replay_s2.tape` | 2,270 | 69.9 s | 32.5 Hz | `btcusdt_20260727T193712_001784Z.tape` |

Total 209.7 s across two real resync boundaries, which is a **21.0 s loop at
10x** — long enough to watch, short enough that a recruiter sees the whole thing
twice in a minute, and it crosses a boundary every 7 s.

**A note on the gaps the feed reports.** `SessionBoundary.gap_ns` between these
prefixes is 422.5 s and 543.8 s, not the 9.1 s and 9.4 s that separated the full
sessions. That is correct rather than a bug: the gap is measured between the last
anchor actually present in one tape and the first in the next, so it includes the
~413 s of session 0 that truncation removed as well as the real resync. It is the
number a consumer needs — the amount of market this replay did not observe — and
computing a return across it would be wrong for both reasons.

Cut with, for each session, `HEADER_SIZE + block_bytes * n` bytes where `n` is
the first anchor beyond 70 s:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
from data_engine.binfmt import HEADER_SIZE
from data_engine.replay import TapeReader
for i, src in enumerate(sorted(Path("data/stage3").glob("*.tape"))):
    with TapeReader(src) as r:
        ts = r.snapshot_timestamps().astype(np.int64)
        n = int(np.searchsorted(ts - ts[0], 70.0 * 1e9, side="right"))
        size = HEADER_SIZE + r.block_dtype.itemsize * n
    Path(f"notebooks/sample_data/btcusdt_replay_s{i}.tape").write_bytes(src.read_bytes()[:size])
PY
```
