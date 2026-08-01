# Stage 1 capture run — 2026-07-27

**What this is, and what it is not.** These are recorded logs from live runs of
`data_engine.capture`, not the output of a benchmark harness. They establish
that the daemon runs, stays synchronised, and proves itself against the
exchange. They do **not** establish a throughput ceiling: every rate below is
the rate Binance chose to send, not the rate this machine can absorb. The
capacity number needs a harness that feeds the book as fast as it will go.
That harness was never written: the ceiling is listed among the known gaps in
the [README](../README.md) rather than carried as a pending table row.

## Machine

| | |
|---|---|
| CPU | 13th Gen Intel Core i5-13420H, 8 physical / 12 logical cores |
| RAM | 15.7 GB |
| OS | Windows 11 Home Single Language, 10.0.26200 |
| Python | 3.13.12 (Anaconda build) |
| Network | residential broadband, no colocation, no tuning |
| Thermal state | untracked — laptop, unpinned, other processes running |

The thermal and pinning caveats do not matter for these numbers because none
of them is a latency measurement. They will matter in Stage 7, where
`docs/benchmark_methodology.md` will record them properly.

## Configuration

`python -m data_engine.capture --symbol btcusdt --out data/ --max-messages N`

Streams: `btcusdt@depth@100ms` + `btcusdt@trade`, combined stream endpoint.
REST snapshots: `GET /api/v3/depth?symbol=BTCUSDT&limit=1000`.
Cross-check interval 60 s, heartbeat interval 10 s.

## Run A — 20,000 messages, 283 seconds

Selected lines, verbatim:

```
20:25:39 INFO  connected: wss://stream.binance.com:9443/stream?streams=btcusdt@depth@100ms/btcusdt@trade
20:25:40 INFO  synchronised: snapshot_id=97897822513 buffered=1 applied=0 depth=(1000, 1000)
20:25:50 INFO  heartbeat msgs/s=58.5  total=657   depth=1095/1031 mid=64674.00 spread=0.01 resyncs=0 xcheck=0/0/0 uptime=11s
20:26:10 INFO  heartbeat msgs/s=142.8 total=2354  depth=1117/1096 mid=64670.00 spread=0.01 resyncs=0 xcheck=0/0/0 uptime=31s
20:26:20 INFO  heartbeat msgs/s=16.4  total=2518  depth=1135/1118 mid=64670.00 spread=0.01 resyncs=0 xcheck=0/0/0 uptime=41s
20:26:30 INFO  heartbeat msgs/s=186.4 total=4383  depth=1089/1267 mid=64629.54 spread=0.01 resyncs=0 xcheck=0/0/0 uptime=51s
20:26:40 INFO  cross-check PASS local_id=97897899093 snapshot_id=97897899093
20:27:41 WARN  cross-check SKEW levels_differing=1  local_id=97897955623 snapshot_id=97897955598 (1 in a row)
20:28:41 INFO  cross-check PASS local_id=97898014227 snapshot_id=97898014227
20:29:42 WARN  cross-check SKEW levels_differing=20 local_id=97898066098 snapshot_id=97898065203 (1 in a row)
20:30:21 INFO  shutdown: messages=20000 depth_applied=2749 resyncs=0 uptime=283s
```

| Quantity | Value |
|---|---|
| Messages received | 20,000 |
| Depth diffs applied | 2,749 |
| Wall time | 283 s |
| Mean message rate | 70.7 msgs/s |
| Observed 10 s rate range | 16.4 – 186.4 msgs/s |
| Resyncs | 0 |
| Sequence gaps | 0 |
| Cross-checks | 2 PASS, 2 SKEW, 0 FAIL |

**The result that matters.** Both PASS lines have `local_id == snapshot_id`
exactly, and both reported zero differing levels across the top 10 of each
side. Both SKEW lines have a non-zero id delta (25 and 895 updates). That is
the SKEW/FAIL classifier behaving exactly as designed: when the two states
describe the same instant they match perfectly, and every mismatch is
accompanied by evidence that they describe different instants. It is also the
strongest correctness evidence in Stage 1 — an independent confirmation from
the exchange that 2,749 diffs were folded in without drift.

## Run B — 8,000 messages, 97 seconds (the notebook sample)

```
20:32:18 INFO  connected: ...
20:32:19 INFO  snapshot is newer than every buffered event; nothing to replay
20:32:19 INFO  synchronised: snapshot_id=97898281344 buffered=1 applied=0 depth=(1000, 1000)
20:33:19 WARN  cross-check SKEW levels_differing=11 local_id=97898364162 snapshot_id=97898364069 (1 in a row)
20:33:54 INFO  shutdown: messages=8000 depth_applied=868 resyncs=0 uptime=97s
```

| Quantity | Value |
|---|---|
| Messages received | 8,000 (7,128 trades, 868 applied depth diffs, 3 discarded as stale) |
| Wall time | 97 s |
| Mean message rate | 82.5 msgs/s |
| Resyncs | 0 |
| Archive size | 388,954 bytes for 8,002 records — ~48.6 B/record gzipped |
| Book depth after 90 s | 1,119 bids / 1,183 asks |

This run's archive is committed as
`notebooks/sample_data/btcusdt_sample.jsonl.gz` and drives notebook 01.

**Offline reproduction.** Replaying that archive in notebook 01 reconstructs
the book to update id 97,898,364,162 at the cross-check point — the same
`local_id` the daemon logged — and finds 4 + 7 = 11 differing rows, the same
`levels_differing=11`. The offline path reproduces the online daemon exactly,
which is what makes the archived cross-check snapshots usable as a permanent,
re-runnable correctness proof.

## Not measured

- **24-hour unattended uptime.** The implementation plan's Stage-1 Definition
  of Done asks for it; the longest run here is 283 s. Still open — see the
  addendum below.
- **Throughput ceiling.** Needs a harness that feeds the book as fast as it will
  go. Still open — see the addendum below.
- **Bytes/event vs the binary format.** The 48.6 B/record above is the gzipped
  JSONL baseline the Stage 2 format was compared against — **now measured**, see
  the addendum below.

## Addendum, added in Stage 8c

This report is dated and its body is left as it was written on 2026-07-27. Two
of the three items above have since been answered, and one has not:

- **Bytes/event vs the binary format: MEASURED.** Stage 2 settled it, and not
  in this format's favour: 43.5 B/event against 68.3 B for raw JSON (1.57x
  smaller) but **6.56x LARGER than gzipped JSON**. See
  [binfmt_20260727T162516Z.json](binfmt_20260727T162516Z.json) and
  [../notebooks/02_binary_format_design.ipynb](../notebooks/02_binary_format_design.ipynb).
- **Throughput ceiling: still not measured, and no longer pending.** The Stage 2
  replay harness measures *read* throughput, not the capture path's ingest
  capacity, so it does not answer this question. It is listed as a known gap in
  the [README](../README.md) rather than carried as a pending marker here.
- **24-hour unattended uptime: still not measured.** The longest run remains
  283 s. Also a known gap in the README.

Neither open item carries a `TODO(measure)` marker any more, because that marker
means "a number belongs in a results table and has not been measured yet", and
neither of these has a row waiting for it. They are stated limitations.
