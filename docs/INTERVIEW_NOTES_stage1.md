# Interview notes — Stage 1: capture engine and order-book reconstruction

What shipped: a Binance spot L2 capture daemon with documented snapshot-sync,
gap detection, auto-resync, a self-sufficient raw archive, and a 60-second
online correctness proof; plus an `OrderBook` with integer fixed-point prices
and 37 passing tests, including a 10,000-event fuzz.

Files: [data_engine/book.py](../data_engine/book.py),
[data_engine/capture.py](../data_engine/capture.py),
[tests/test_book.py](../tests/test_book.py),
[tests/test_capture.py](../tests/test_capture.py),
[notebooks/01_orderbook_capture_walkthrough.ipynb](../notebooks/01_orderbook_capture_walkthrough.ipynb).

Protocol reference, checked 2026-07-27:
<https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams>
(section "How to manage a local order book correctly") and
<https://developers.binance.com/docs/binance-spot-api-docs/rest-api>
(`GET /api/v3/depth`).

---

## 1. Design decisions and the alternatives rejected

### 1.1 Integer fixed-point prices, not floats — `book.py:44-56`, `book.py:88-104`

Prices and quantities are stored as `round(value * 10^8)` in `int`, parsed
from the venue's decimal strings through `Decimal`.

**Rejected: `dict[float, float]`.** Binary floating point cannot represent
most decimal ticks exactly, so the same price level arriving in two messages
can land on two different float keys. The book grows a phantom duplicate, and
the `quantity = 0` message that should delete it removes only one of the two.
The book is then one level too deep at a price that no longer exists —
invisible for hours, and fatal to any feature computed from the touch.

Demonstrated concretely: `int(float("0.29") * 10**8) == 28999999`, not
`29000000`. Asserted in
[tests/test_book.py](../tests/test_book.py) —
`test_fixed_point_conversion_is_exact_where_float_is_not`.

**Rejected: `float(value) * scale` then `round()`.** Rounding does rescue the
common cases, but its correctness is an argument about representation error
rather than a property of the code. `Decimal` is exact by construction. The
cost is speed, which is acceptable because Stage 1 is I/O bound on the socket;
`to_fixed`'s docstring records that the replacement, if profiling ever demands
one, is a hand-rolled integer string parser and not a float.

`to_fixed` **raises** when a value carries more precision than the 10^8 grid
holds, rather than truncating — truncation would be the same bug in a
different coat.

### 1.2 Two dicts price→qty, not sorted arrays — `book.py:29-42`

A diff event is a scatter of unrelated price levels and the dominant operation
is "set or delete this one level": O(1) in a dict, O(k) memmove in a sorted
array. Deletion is a single `del`, no shifting and no tombstones.

**Cost accepted:** ordered reads (`best_bid`, `top_n`) are O(k) / O(k log k)
because ordering is not maintained incrementally. Right for Stage 1, where
reads happen a few times a second and writes happen on every event. **Wrong
for Stage 7**, whose hot path reads the top 10 levels on every tick — that
path gets a contiguous sorted ladder, and being able to explain why the same
program wants two different data structures at two different stages is the
actual interview answer.

Note: this supersedes the Stage-0 stub docstring, which had proposed sorted
ladders before the update pattern was clear.

### 1.3 A sequence gap raises; it never self-heals — `book.py:130-166`

`apply_diff` implements the venue's update procedure and raises
`SequenceGapError` when `U > last_update_id + 1`.

**Rejected: patch the hole in place** (apply anyway, or quietly refetch a
snapshot and merge while the socket stays up). See Q2 below for the full
reasoning. Binance's own procedure says to discard the book and restart, and
the simpler control flow is worth the second of data it costs: there is
exactly one code path that produces a valid book, so exactly one path to get
right.

### 1.4 `<=` rather than the docs' `<` for the stale test — `book.py:157-166`

The update procedure says ignore when `u` is *less than* the local id; step 5
of the sync algorithm discards buffered events where `u <= lastUpdateId`. We
use `<=` in both places. An event whose final id equals the local id is
entirely contained in what was already applied, so skipping it cannot lose
information, and one rule in both places means the buffered replay and the
live path cannot disagree.

### 1.5 The bracket check is the weaker, sufficient condition — `capture.py:_verify_snapshot_is_bracketed`

The docs describe the expected post-discard state as "the first buffered event
should now have `lastUpdateId` within its `[U;u]` range", i.e. `U <= L <= u`.
That is not the only correct case: an event with `U == L + 1` starts exactly
where the snapshot ends, which is perfectly contiguous and loses nothing, yet
falls outside the doc's bracket. So the **fatal** check is `U <= L + 1` (no
hole) and the doc-shaped bracket is only logged.

This matters in practice, not just pedantically — the sample capture hit the
abutting case on the very first sync (`snapshot is newer than every buffered
event; nothing to replay`). Both cases are tested:
`test_bracketed_first_event_is_accepted` and
`test_abutting_first_event_is_accepted`.

### 1.6 Archive raw frames verbatim, and put the REST snapshots on the tape too — `capture.py:RawArchive`

**Rejected: archive reconstructed book snapshots.** Far less disk, but it
bakes today's reconstruction logic into the archive forever: a bug in the
sequencing rules would corrupt data we could never recover, and Stage 2's
binary format would have nothing independent to validate against.

**Rejected: archive the parsed message re-serialised.** That puts *this
module's* json handling into the permanent record. The websocket text frame is
stored as an opaque string; `fetch_depth_snapshot` returns the undecoded body
alongside the parse for the same reason.

**Added during the stage:** the first live run produced a tape containing only
websocket frames — which cannot be replayed into an absolute book, because a
diff stream describes only changes. Sync snapshots and 60-second cross-check
snapshots now go on the tape as well, tagged by `source`. The tape is
self-sufficient (notebook 01 needs no network) and carries its own correctness
proof (the cross-check can be re-run offline). Asserted by
`test_bundled_sample_is_self_sufficient`.

### 1.7 The cross-check distinguishes SKEW from FAIL — `capture.py:_record_cross_check`

**Rejected: compare and call any mismatch a failure.** A REST snapshot and the
live book describe different instants — the book keeps moving during the HTTP
round trip — so naive comparison false-alarms in fast markets, and an alarm
that cries wolf is worse than no alarm.

A mismatch is reported as SKEW (advisory) and escalates to FAIL only after
three consecutive mismatches, which is the signature of real corruption rather
than latency: transient skew resolves, corruption does not. Empirically
validated by the capture run below — every exact-id comparison passed with
zero differing levels, and every mismatch coincided with an id delta.

### 1.8 stdlib `urllib` on a thread, not `aiohttp`/`requests` — `capture.py:_http_get_text`

Neither is in CLAUDE.md's dependency list, and adding an HTTP stack for two
GET requests a minute is not a trade worth making. `urllib.request` inside
`asyncio.to_thread` is blocking-but-off-the-event-loop, which is sufficient at
this rate. `limit=1000` costs 50 request weight against a 6000/minute budget.

### 1.9 Archive filenames carry microseconds and a counter — `capture.py:RawArchive._rotate`

Found by a test, not by reasoning: `test_archive_rotates_once_the_size_budget_is_spent`
failed because back-to-back rotations landed in the same second, the
second-resolution filename collided, and `gzip.open(..., "at")` silently
appended instead of rotating. The file stayed readable — gzip concatenates
members — so the only symptom was that the size bound quietly stopped being
enforced. Fixed with microseconds plus a per-process counter, and opened with
`"xt"` so any residual collision fails loudly rather than silently merging.

---

## 2. The ten hardest questions

**Q1. Why integer fixed-point prices instead of floats? Give me the failure.**
Binary floats cannot represent most decimal ticks exactly, so the same level
arriving twice can produce two different keys; the book grows a phantom
duplicate and the `qty = 0` delete removes only one of them, leaving a level
at a price that no longer exists. `int(float("0.29") * 10**8)` is `28999999`,
one tick low. Integers on the tick grid make level identity exact by
construction and carry straight into Stage 2's binary record.
→ `book.py:88-104` (`to_fixed`), `tests/test_book.py` (`test_fixed_point_conversion_is_exact_where_float_is_not`).

**Q2. Why raise on a sequence gap instead of refetching and patching?**
A diff only names the levels that changed, so a missed event leaves us unable
to say *which* levels are stale — the whole book is suspect. A book that is
wrong at one unknown price still produces smooth, plausible features, so the
corruption is undetectable downstream and silently poisons training data.
Raising converts an invisible data-quality problem into a loud, countable one:
tear down, resync from a fresh snapshot, increment the counter. Losing a
second of data is cheap; not knowing which second is not.
→ `book.py:130-166`, `capture.py:run_capture` (resync counting).

**Q3. Walk me through the sync algorithm. How does it avoid both missing and
double-applying a diff?**
The snapshot is a complete state at id `L`; the diff stream is a contiguous
run of id ranges. The two failure modes are a hole *before* `L` and a replay
of history already *inside* `L`, and one step is aimed at each. Step 4 —
refetch if `lastUpdateId < U` of the first buffered event — rules out the
snapshot being older than the buffer, killing the hole. Step 5 — discard
buffered events with `u <= L` — rules out replaying anything already baked
into the snapshot, killing the double-apply. What survives both filters is
exactly the events overlapping or immediately following `L`, and `apply_diff`
enforces contiguity from there.
→ `capture.py:_synchronise_book` (docstring carries the numbered steps).

**Q4. Your cross-check reported 11 differing levels. Is your book wrong?**
No. The replay was 93 update ids ahead of the snapshot, so the two describe
different instants; 9/10 bid *prices* still agreed and it was mostly the
resting quantities that had moved — the signature of elapsed time, not
corruption. That is why an isolated mismatch is SKEW and only three
consecutive mismatches are a FAIL. The authoritative case is an exact id
match, and both times that occurred during capture the daemon logged PASS with
zero differing levels.
→ `capture.py:_record_cross_check`, notebook 01 §9.

**Q5. If a mismatch is usually skew, is the cross-check worth anything?**
Yes, in two ways. When the ids coincide the comparison is exact and
authoritative — that is a genuine end-to-end proof of the whole reconstruction
chain, and it fired twice in a five-minute run. When they do not coincide, the
*persistence* of a mismatch is still informative: skew is uncorrelated between
checks whereas a real reconstruction bug produces a mismatch every single
time, which is what the three-in-a-row rule detects. A systematic drift cannot
hide in the skew.

**Q6. Why is the daemon's book deeper than the 1,000 levels the snapshot gave
you? Is that a leak?**
No — the snapshot's `limit=1000` is a window on the top of book, and diffs
subsequently arrive for levels outside it, which we accumulate. The sample run
ends at 1,119 bids / 1,183 asks. The real consequence is the other direction:
levels beyond the snapshot window that we were never told about are unknown
until they next change, so deep levels are the least trustworthy part of the
book. This is intrinsic to any depth-diff reconstruction and is why features
only ever use the top N near the touch.
→ `capture.py:fetch_depth_snapshot` docstring, notebook 01 §5.

**Q7. You keep two timestamps per message. Why not one?**
`recv_ns` is the wall clock, needed to align this tape with anything else in
the world. `recv_mono_ns` is monotonic and is the only one safe for measuring
inter-arrival gaps, because the wall clock can step backwards under NTP and a
negative inter-arrival time would quietly become a negative feature in Stage
3. Neither is the exchange's `E` field, which we also keep — trusting the
venue's clock alone would make any measured "latency" a property of their
clock discipline rather than of our system.
→ `capture.py:RawArchive` docstring.

**Q8. How do you know your book is correct and not just self-consistent?**
Three independent layers. `assert_valid` checks structural invariants (strict
ladder ordering, positive sizes, uncrossed touch) and runs after every event
in the fuzz and in the notebook. The 10,000-event fuzz test compares the
`OrderBook` against a shadow model written with different code, so a shared
bug would have to arise for two different reasons. And the 60-second
cross-check compares against the exchange's own answer — invariants prove
self-consistency, only the exchange proves correctness.
→ `book.py:assert_valid`, `tests/test_book.py:test_fuzz_ten_thousand_diffs_preserves_every_invariant`.

**Q9. What breaks first if you run this for a month?**
Ranked by likelihood: (a) disk — raw JSONL is the dominant cost and is exactly
what Stage 2's binary format exists to fix; (b) the 24-hour forced disconnect,
which is handled as a normal reconnect-and-resync and logged at INFO rather
than ERROR; (c) unbounded book growth from stale deep levels that were removed
while outside our snapshot window and never re-mentioned — the periodic resync
bounds this in practice, since each resync rebuilds from a fresh snapshot.
What does *not* break is correctness after a gap, because a gap always forces
a full rebuild rather than a patch.

**Q10. What would you do differently with more time?**
Three things, in order. Replace the reconnect backoff with something jittered
and capped, since a fixed 2 s reconnect from many processes would synchronise
against the venue. Record the websocket receive timestamp inside the library's
frame callback rather than after `recv()` returns, which currently includes
event-loop scheduling delay in `recv_ns` — this matters for Stage 7's
tick-to-signal measurement and is noted there, not silently ignored. And add a
second venue behind the same book, since the current design is exchange-shaped
in exactly one place (`_unwrap_combined` and the field names), which is where
an adapter would go — though not before a second implementation actually
exists, per CLAUDE.md.

---

## 3. What Stage 1 measured

Produced by running the daemon, not estimated. Machine: Windows 11, Python
3.13.12. Two runs on 2026-07-27, BTCUSDT, `btcusdt@depth@100ms` +
`btcusdt@trade`.

| Quantity | Value | Source |
|---|---|---|
| Sustained throughput | 16–186 msgs/s, varying with market activity | heartbeat lines |
| Mean throughput, 283 s run | ~71 msgs/s (20,000 messages) | daemon shutdown line |
| Resyncs | 0 in 283 s and 0 in 97 s | resync counter |
| Cross-checks | 2 PASS, 2 SKEW, 0 FAIL over 283 s | cross-check log |
| Depth-diff : trade ratio | ~1 : 8.2 | notebook 01 §5 |
| Raw archive size | 388,954 bytes for 8,002 records (~49 B/record gzipped) | sample file |
| Book depth after 90 s | 1,119 bids / 1,183 asks | notebook 01 §5 |
| Spread at seed | 0.01 USDT on a 64,680 mid (0.0015 bps) | notebook 01 §4 |

**Correction (Stage 5).** This row originally read "~0.15 bps", which was wrong
by a factor of 100 — one basis point at a 64,680 mid is 6.47 USDT, so a
one-cent spread is 0.0015 bps. Caught while computing Stage 5's cost analysis,
where the relationship between spread and fee decides the conclusion. The
"117x the half-spread" figure quoted for the label dead zone in Stage 3 was
computed in price units and is unaffected.

Not yet measured, and deliberately not guessed: 24-hour unattended uptime
(the Stage-1 Definition of Done in the implementation plan asks for this, and
the longest run so far is 283 seconds), and the capture path's throughput
ceiling — every rate reported here is the rate Binance chose to send.

Bytes-per-event versus the binary format *was* since measured, and the format
lost: 6.56x larger than gzipped JSON
([binfmt](../benchmarks/binfmt_20260727T162516Z.json)). The two that remain open
are listed under Known gaps in the [README](../README.md); they no longer carry
`TODO(measure)` markers, because that marker means a results row is waiting for
a number and neither of these has one.
