"""Assemble training samples from captured tapes.

WHAT
    Turns a list of `.tape` files into a chronologically ordered set of
    `(window, label)` samples, and exposes them as a `torch.utils.data.Dataset`
    that slices windows on demand.

WHY
    Stages 3's pieces — causal features, forward-looking labels, embargoed
    splits — each do one job. Something has to hold them together, decide what
    happens at a session boundary, and hand the result to a training loop.
    That gluing is where a leakage bug would most plausibly reappear, so it
    lives in one small file with its own tests rather than being retyped in
    every notebook.

DESIGN DECISION — windows are sliced on demand, never materialised.
    Rejected alternative: build one `[N, 100, 40]` float32 array and index it.
    Adjacent windows overlap by 99 of 100 rows, so materialising our ~37,000
    samples costs 592 MB to represent 6 MB of underlying data. Keeping the
    normalised `[T, 40]` matrix per session and slicing in `__getitem__` costs
    a copy of one window per sample fetched, which the dataloader overlaps
    with compute anyway.

DESIGN DECISION — each capture session is a separate, self-contained block.
    Rejected alternative: concatenate every tape into one long series and
    treat it as continuous. The tapes are separated by resyncs — gaps of
    seconds to minutes where we saw nothing. Concatenating would invent
    windows that straddle a gap and mid moves that never happened, and the
    labels computed across the join would be fiction. So no window and no
    label ever crosses a session boundary, and the rolling normalisation
    restarts per session rather than carrying stale statistics across a hole.

DESIGN DECISION — samples are ordered by capture time across sessions.
    The global sample index is chronological, which is what makes
    `ml.splits.walk_forward_splits` meaningful when applied to it. Sessions
    are sorted by their first timestamp rather than by filename, because
    filenames are only incidentally ordered.

INFORMATION HORIZON
    A sample at global index i is built from the feature rows
    `[end - 99, end]` of its session and a label reading forward to
    `end + k`. Nothing here reaches outside that, and `ml/splits.py` is what
    keeps two samples' spans from overlapping across a train/test boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data_engine.replay import TapeReader
from ml.features import (
    DEFAULT_NORMALISATION_LOOKBACK,
    WINDOW_LENGTH,
    build_feature_matrix,
    rolling_zscore,
)
from ml.labels import DEFAULT_ALPHA, DEFAULT_SMOOTHING_K, build_labels


@dataclass
class Session:
    """One capture session: normalised features, labels, and what is usable.

    Carries the *executable* prices and the wall-clock timestamps alongside the
    model inputs. Stage 5 needs both and neither can be recovered from the
    normalised features: the z-score deliberately destroys the price level, and
    a mid is not a price anyone can trade at. Keeping them here means the
    evaluation reads the same rows the model saw, rather than re-deriving them
    from the tape and risking a one-row misalignment between signal and fill.
    """

    name: str
    features: np.ndarray  # [T, 40] float32, causally normalised
    labels: np.ndarray  # [T] int64
    usable: np.ndarray  # [T] bool — row has a full window behind it and a valid label
    mids: np.ndarray  # [T] float64, real price units
    best_bids: np.ndarray  # [T] float64 — where a sell actually fills
    best_asks: np.ndarray  # [T] float64 — where a buy actually fills
    timestamps_ns: np.ndarray  # [T] int64, local receive clock
    first_timestamp_ns: int

    def usable_end_rows(self) -> np.ndarray:
        """Row indices that can serve as the final row of a training window."""
        return np.flatnonzero(self.usable)

    def elapsed_seconds(self) -> float:
        if len(self.timestamps_ns) < 2:
            return 0.0
        return float(self.timestamps_ns[-1] - self.timestamps_ns[0]) / 1e9


def load_session(
    tape_path: Path | str,
    window_length: int = WINDOW_LENGTH,
    lookback: int = DEFAULT_NORMALISATION_LOOKBACK,
    smoothing_k: int = DEFAULT_SMOOTHING_K,
    alpha: float = DEFAULT_ALPHA,
) -> Session:
    """Read one tape and build its features, labels and usability mask."""
    with TapeReader(tape_path) as reader:
        snapshots = reader.load_snapshots()
        price_scale = reader.header.price_scale
        matrix, row_valid = build_feature_matrix(snapshots, price_scale, reader.header.qty_scale)
        best_bid = np.array(snapshots["bids"][:, 0, 0].astype(np.float64) / price_scale)
        best_ask = np.array(snapshots["asks"][:, 0, 0].astype(np.float64) / price_scale)
        mids = (best_bid + best_ask) / 2.0
        timestamps = np.array(snapshots["local_ts_ns"].astype(np.int64))
        first_timestamp = int(timestamps[0]) if len(timestamps) else 0

    normalised, normalisation_valid = rolling_zscore(matrix, lookback)
    label_set = build_labels(mids, smoothing_k, alpha)

    usable = _usable_end_rows(row_valid, normalisation_valid, label_set.valid, window_length)
    return Session(
        name=Path(tape_path).stem,
        features=normalised,
        labels=label_set.labels.astype(np.int64),
        usable=usable,
        mids=mids,
        best_bids=best_bid,
        best_asks=best_ask,
        timestamps_ns=timestamps,
        first_timestamp_ns=first_timestamp,
    )


def _usable_end_rows(
    row_valid: np.ndarray,
    normalisation_valid: np.ndarray,
    label_valid: np.ndarray,
    window_length: int,
) -> np.ndarray:
    """A row can end a sample only if the whole window behind it is clean.

    Three conditions, all necessary: every row of the window had two sides to
    compute a mid from, every row had a full normalisation lookback, and the
    end row carries a real label. Rows failing any of them are dropped, never
    patched — the Stage 3 rule that a fabricated sample is worse than a
    missing one applies here too.
    """
    clean_row = row_valid & normalisation_valid
    if len(clean_row) < window_length:
        return np.zeros(len(clean_row), dtype=bool)

    window_clean = np.lib.stride_tricks.sliding_window_view(clean_row, window_length).all(axis=1)
    usable = np.zeros(len(clean_row), dtype=bool)
    usable[window_length - 1 :] = window_clean
    return usable & label_valid


def load_sessions(tape_paths: list[Path | str], **kwargs) -> list[Session]:
    """Load several tapes and return them in capture order."""
    sessions = [load_session(path, **kwargs) for path in tape_paths]
    sessions.sort(key=lambda session: session.first_timestamp_ns)
    return sessions


@dataclass(frozen=True)
class SampleIndex:
    """A flat, chronological index over every usable sample in every session.

    Two parallel arrays rather than a list of tuples: the split functions in
    `ml/splits.py` work on integer ranges, and keeping this as arrays means a
    fold is a slice rather than a Python loop.
    """

    session_of_sample: np.ndarray  # [N] int64
    end_row_of_sample: np.ndarray  # [N] int64
    labels: np.ndarray  # [N] int64

    def __len__(self) -> int:
        return len(self.session_of_sample)

    def class_balance(self) -> dict[str, float]:
        counts = np.bincount(self.labels, minlength=3)
        return {name: float(count / max(len(self), 1)) for name, count in zip(("down", "flat", "up"), counts)}


def build_sample_index(sessions: list[Session]) -> SampleIndex:
    """Flatten every session's usable rows into one chronological index."""
    session_ids = []
    end_rows = []
    labels = []
    for session_id, session in enumerate(sessions):
        rows = session.usable_end_rows()
        session_ids.append(np.full(len(rows), session_id, dtype=np.int64))
        end_rows.append(rows.astype(np.int64))
        labels.append(session.labels[rows])
    if not session_ids:
        empty = np.empty(0, dtype=np.int64)
        return SampleIndex(empty, empty, empty)
    return SampleIndex(
        session_of_sample=np.concatenate(session_ids),
        end_row_of_sample=np.concatenate(end_rows),
        labels=np.concatenate(labels),
    )


def gather_windows(
    sessions: list[Session],
    sample_index: SampleIndex,
    positions: np.ndarray,
    window_length: int = WINDOW_LENGTH,
) -> np.ndarray:
    """Build `[N, window_length, 40]` for the given samples in one gather per session.

    WHY THIS EXISTS. Fetching windows one at a time through a
    `Dataset.__getitem__` costs a Python call and a slice per sample, which
    measured at ~600 samples/s — 46 s per epoch on our 27,674 training samples
    and roughly seven minutes per epoch on FI-2010's 254,750. Because every
    window is a contiguous run of rows, a whole batch is one fancy-index
    gather instead, and the Python overhead disappears.

    Samples from different sessions are gathered separately, since their rows
    index different matrices. There are only a handful of sessions, so the
    loop is over sessions and not over samples — that distinction is the whole
    speedup.
    """
    positions = np.asarray(positions, dtype=np.int64)
    feature_count = sessions[0].features.shape[1]
    windows = np.empty((len(positions), window_length, feature_count), dtype=np.float32)
    if len(positions) == 0:
        return windows

    session_ids = sample_index.session_of_sample[positions]
    end_rows = sample_index.end_row_of_sample[positions]
    offsets = np.arange(-window_length + 1, 1, dtype=np.int64)
    for session_id in np.unique(session_ids):
        selected = session_ids == session_id
        rows = end_rows[selected][:, None] + offsets[None, :]
        windows[selected] = sessions[int(session_id)].features[rows]
    return windows


def gather_last_rows(
    sessions: list[Session],
    sample_index: SampleIndex,
    positions: np.ndarray,
) -> np.ndarray:
    """The final feature row of each window, as `[N, 40]` — what the baseline sees."""
    positions = np.asarray(positions, dtype=np.int64)
    feature_count = sessions[0].features.shape[1]
    rows = np.empty((len(positions), feature_count), dtype=np.float32)
    session_ids = sample_index.session_of_sample[positions]
    end_rows = sample_index.end_row_of_sample[positions]
    for session_id in np.unique(session_ids):
        selected = session_ids == session_id
        rows[selected] = sessions[int(session_id)].features[end_rows[selected]]
    return rows


class BatchedWindowLoader:
    """Iterates `(windows, labels)` batches, gathering each batch in one shot.

    Replaces `DataLoader` on the hot path. Deliberately not a `DataLoader`
    subclass and deliberately single-process: the gather is fast enough that
    worker processes would add spawn cost and Windows pickling headaches for
    no gain, and a single process keeps the run exactly reproducible from the
    seed.
    """

    def __init__(
        self,
        sessions: list[Session],
        sample_index: SampleIndex,
        positions: np.ndarray,
        batch_size: int,
        shuffle: bool = False,
        window_length: int = WINDOW_LENGTH,
        seed: int = 0,
    ) -> None:
        self.sessions = sessions
        self.sample_index = sample_index
        self.positions = np.asarray(positions, dtype=np.int64)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.window_length = window_length
        self._generator = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Number of batches, so a scheduler can be sized before the first epoch."""
        return max(1, int(np.ceil(len(self.positions) / self.batch_size)))

    def __iter__(self):
        order = self._generator.permutation(len(self.positions)) if self.shuffle else np.arange(len(self.positions))
        for start in range(0, len(order), self.batch_size):
            chosen = self.positions[order[start : start + self.batch_size]]
            windows = gather_windows(self.sessions, self.sample_index, chosen, self.window_length)
            labels = self.sample_index.labels[chosen]
            yield windows, labels

    def labels_for_positions(self) -> np.ndarray:
        return self.sample_index.labels[self.positions]


class WindowDataset:
    """A `torch.utils.data.Dataset` over a subset of the sample index.

    Kept for per-sample access — the notebook inspects individual windows, and
    a `Dataset` is the clearest way to show what one sample is. The training
    loop uses `BatchedWindowLoader` instead, for the reason given there.

    Deliberately not declared as a `torch.utils.data.Dataset` subclass: `torch`
    is a heavy import and `data_engine` must stay usable without it. PyTorch's
    `DataLoader` only requires `__len__` and `__getitem__`.
    """

    def __init__(
        self,
        sessions: list[Session],
        sample_index: SampleIndex,
        positions: np.ndarray,
        window_length: int = WINDOW_LENGTH,
    ) -> None:
        self.sessions = sessions
        self.sample_index = sample_index
        self.positions = np.asarray(positions, dtype=np.int64)
        self.window_length = window_length

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, item: int) -> tuple[np.ndarray, int]:
        position = int(self.positions[item])
        session = self.sessions[int(self.sample_index.session_of_sample[position])]
        end_row = int(self.sample_index.end_row_of_sample[position])
        window = session.features[end_row - self.window_length + 1 : end_row + 1]
        return window, int(self.sample_index.labels[position])

    def labels_for_positions(self) -> np.ndarray:
        """Labels of exactly the samples this subset covers, in order."""
        return self.sample_index.labels[self.positions]

    def last_rows(self) -> np.ndarray:
        """The final feature row of every window, as `[N, 40]`.

        This is what the logistic-regression baseline sees: the current book
        state and nothing else. Materialising it is cheap because it is one
        row per sample rather than a hundred.
        """
        return gather_last_rows(self.sessions, self.sample_index, self.positions)


def describe_sessions(sessions: list[Session], sample_index: SampleIndex) -> str:
    """One line per session plus a total, for logs and the notebook."""
    lines = [f"{'session':<40} {'rows':>8} {'usable':>8} {'down':>7} {'flat':>7} {'up':>7}"]
    for session_id, session in enumerate(sessions):
        rows = session.usable_end_rows()
        counts = np.bincount(session.labels[rows], minlength=3) / max(len(rows), 1)
        lines.append(
            f"{session.name[:40]:<40} {len(session.features):>8,} {len(rows):>8,} "
            f"{counts[0]:>7.3f} {counts[1]:>7.3f} {counts[2]:>7.3f}"
        )
    balance = sample_index.class_balance()
    lines.append(
        f"{'TOTAL':<40} {'':>8} {len(sample_index):>8,} "
        f"{balance['down']:>7.3f} {balance['flat']:>7.3f} {balance['up']:>7.3f}"
    )
    return "\n".join(lines)
