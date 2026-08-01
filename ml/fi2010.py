"""Loader for the public FI-2010 limit-order-book benchmark.

WHAT
    Reads the FI-2010 benchmark files into the same `[T, 40]` feature matrix
    and `[T]` label vector shape that `ml/dataset.py` produces from our own
    tapes, so `ml/model.py` and `ml/train.py` can train on either without
    knowing which they were handed.

WHY
    Everything else in this project is measured on data we captured
    ourselves, which makes the numbers honest but incomparable — nobody else
    has our tape, so "macro-F1 0.6" means nothing to a reader. FI-2010 is the
    standard public benchmark for exactly this task, so training the same
    architecture on it produces one number that can be placed next to
    published results. It is the only external yardstick in the project.

SOURCE
    Canonical dataset record (Ntakaris, Magris, Kanniainen, Gabbouj,
    Iosifidis, 2018), hosted by Finland's Fairdata service:
        https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649
    Paper: "Benchmark dataset for mid-price forecasting of limit order book
    data with machine learning methods", Journal of Forecasting 37(8),
    pp. 852-866.

    We read the widely-used preprocessed copy mirrored in the DeepLOB
    reference implementation, because it is the exact split that published
    results are quoted on:
        https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books
    Downloading a different preprocessing of the same underlying data would
    produce a number that looks comparable and is not, which is worse than
    having no number.

FILE FORMAT (verified by inspection, 2026-07-28)
    Each file is a whitespace-separated text matrix stored **transposed**:
    149 lines, one per feature, each holding one value per event. For the
    `NoAuction_DecPre_CF` files used here:

        lines 0-39     the 40 raw LOB values, ordered per level as
                       (ask price, ask volume, bid price, bid volume)
        lines 40-143   104 engineered features from the original paper
        lines 144-148  labels for horizons k = 1, 2, 3, 5, 10,
                       encoded 1 = up, 2 = stationary, 3 = down

DESIGN DECISION — use only the 40 raw LOB values, not the 104 engineered ones.
    Rejected alternative: feed all 144 features, which would almost certainly
    score better. The point of this comparison is to test *our architecture*
    against published ones on the same input the published ones used — DeepLOB
    and its successors take the 40 raw values — so adding hand-engineered
    features would make our number better and meaningless.

DESIGN DECISION — take the data's normalisation as given.
    The `DecPre` files are already normalised by decimal scaling. Rejected
    alternative: re-normalise with our own causal rolling z-score for
    consistency with our tapes. That would change the preprocessing the
    published numbers assume and break the only comparison this module exists
    to make. The asymmetry is deliberate and is stated in the results table.

INFORMATION HORIZON
    The labels ship with the dataset and were computed by its authors over a
    forward horizon of k events. This module does not compute labels; it reads
    them. The horizon is whichever `k` the caller selects, and the same
    `k`-row embargo logic applies when splitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Row layout inside a file, as documented above.
RAW_FEATURE_ROWS = 40
FIRST_LABEL_ROW = 144
# The five horizons the dataset ships labels for, in file order.
LABEL_HORIZONS = (1, 2, 3, 5, 10)

# The dataset encodes 1 = up, 2 = stationary, 3 = down. We use the project's
# ordering (0 = down, 1 = flat, 2 = up) so that `label - 1` is the sign of the
# move in every module, and so the confusion matrices read the same way
# whichever dataset produced them.
_FI2010_TO_PROJECT = {1: 2, 2: 1, 3: 0}

DEFAULT_FILENAMES = {
    "train": ("Train_Dst_NoAuction_DecPre_CF_7.txt",),
    "test": (
        "Test_Dst_NoAuction_DecPre_CF_7.txt",
        "Test_Dst_NoAuction_DecPre_CF_8.txt",
        "Test_Dst_NoAuction_DecPre_CF_9.txt",
    ),
}


class FI2010NotAvailable(FileNotFoundError):
    """Raised when the benchmark files are not on disk, with how to get them."""


@dataclass
class FI2010Split:
    """One split of the benchmark, shaped like a `ml.dataset.Session`."""

    name: str
    features: np.ndarray  # [T, 40] float32
    labels: np.ndarray  # [T] int64, in project encoding
    horizon: int

    def class_balance(self) -> dict[str, float]:
        counts = np.bincount(self.labels, minlength=3)
        return {name: float(count / len(self.labels)) for name, count in zip(("down", "flat", "up"), counts)}


def _read_matrix(path: Path, wanted_rows: set[int]) -> dict[int, np.ndarray]:
    """Parse only the rows we need out of a transposed text matrix.

    The training file is 607 MB of text holding ~38 million numbers, and we
    want 45 of its 149 rows. Parsing line by line and skipping the rest turns
    a multi-minute `np.loadtxt` into a few seconds: the bytes still have to be
    read, but two thirds of them never become floats.
    """
    rows: dict[int, np.ndarray] = {}
    with open(path, "r") as handle:
        for index, line in enumerate(handle):
            if index in wanted_rows:
                rows[index] = np.fromstring(line, sep=" ", dtype=np.float32)
    missing = wanted_rows - rows.keys()
    if missing:
        raise ValueError(f"{path.name} has no row {sorted(missing)}; is this an FI-2010 file?")
    return rows


def load_file(path: Path | str, horizon: int = 10) -> FI2010Split:
    """Load one benchmark file at the given prediction horizon."""
    if horizon not in LABEL_HORIZONS:
        raise ValueError(f"horizon must be one of {LABEL_HORIZONS}, got {horizon}")
    path = Path(path)
    if not path.exists():
        raise FI2010NotAvailable(_download_hint(path.parent))

    label_row = FIRST_LABEL_ROW + LABEL_HORIZONS.index(horizon)
    wanted = set(range(RAW_FEATURE_ROWS)) | {label_row}
    rows = _read_matrix(path, wanted)

    features = np.stack([rows[index] for index in range(RAW_FEATURE_ROWS)], axis=1)
    raw_labels = rows[label_row].astype(np.int64)
    labels = _to_project_encoding(raw_labels)
    return FI2010Split(name=path.stem, features=_reorder_to_bid_first(features), labels=labels, horizon=horizon)


def _to_project_encoding(raw_labels: np.ndarray) -> np.ndarray:
    """Map the dataset's {1 up, 2 stationary, 3 down} onto our {0,1,2}."""
    unexpected = set(np.unique(raw_labels).tolist()) - set(_FI2010_TO_PROJECT)
    if unexpected:
        raise ValueError(f"unexpected label values {sorted(unexpected)}; expected {sorted(_FI2010_TO_PROJECT)}")
    mapped = np.empty_like(raw_labels)
    for source, target in _FI2010_TO_PROJECT.items():
        mapped[raw_labels == source] = target
    return mapped


def _reorder_to_bid_first(features: np.ndarray) -> np.ndarray:
    """Swap each level from (ask, bid) to (bid, ask) ordering.

    FI-2010 stores each level as (ask price, ask volume, bid price, bid
    volume); our tapes store (bid price, bid size, ask price, ask size). The
    convolutional front end fuses columns in fixed pairs, so either ordering
    would train — but "fuse the two sides" would mean the mirror image of what
    the comments in `ml/model.py` claim. Reordering costs one array copy and
    keeps every explanation in the project true of both datasets.
    """
    reordered = features.copy()
    reordered[:, 0::4] = features[:, 2::4]  # bid price into slot 0
    reordered[:, 1::4] = features[:, 3::4]  # bid volume into slot 1
    reordered[:, 2::4] = features[:, 0::4]  # ask price into slot 2
    reordered[:, 3::4] = features[:, 1::4]  # ask volume into slot 3
    return reordered


def load_benchmark(data_dir: Path | str, horizon: int = 10) -> tuple[FI2010Split, FI2010Split]:
    """Load the standard train and test splits, concatenating the test days.

    Returns `(train, test)`. The three test files are consecutive days and are
    concatenated in order, which is the setup published results use.
    """
    data_dir = Path(data_dir)
    train_files = [data_dir / name for name in DEFAULT_FILENAMES["train"]]
    test_files = [data_dir / name for name in DEFAULT_FILENAMES["test"]]
    for path in train_files + test_files:
        if not path.exists():
            raise FI2010NotAvailable(_download_hint(data_dir))

    train = load_file(train_files[0], horizon)
    test_parts = [load_file(path, horizon) for path in test_files]
    test = FI2010Split(
        name="Test_Dst_NoAuction_DecPre_CF_7-9",
        features=np.concatenate([part.features for part in test_parts]),
        labels=np.concatenate([part.labels for part in test_parts]),
        horizon=horizon,
    )
    return train, test


def to_session(split: FI2010Split, window_length: int = 100):
    """Adapt a benchmark split into a `ml.dataset.Session`.

    Lets the benchmark reuse the same loaders, training loop and metrics as
    our own tapes — which is the point of running it at all. If FI-2010 needed
    its own training path, a difference in the two numbers could always be
    blamed on the plumbing rather than on the data.

    `usable` starts once a full window exists behind a row. Unlike our tapes
    there is no normalisation warmup to skip, because the dataset arrives
    pre-normalised, and no label tail to drop, because the authors ship a
    label for every row.
    """
    from ml.dataset import Session  # imported here to keep the import graph acyclic

    usable = np.zeros(len(split.features), dtype=bool)
    usable[window_length - 1 :] = True
    return Session(
        name=split.name,
        features=np.ascontiguousarray(split.features, dtype=np.float32),
        labels=split.labels.astype(np.int64),
        usable=usable,
        # The benchmark is normalised and carries no raw prices or timestamps,
        # so nothing here can be marked to a fill or placed on a clock. NaN
        # rather than a plausible-looking zero, so that any attempt to run
        # Stage 5's cost analysis on FI-2010 produces NaN immediately instead
        # of a number that looks like money. That analysis runs on our own
        # tapes only, and this makes it impossible to forget.
        mids=np.full(len(split.features), np.nan),
        best_bids=np.full(len(split.features), np.nan),
        best_asks=np.full(len(split.features), np.nan),
        timestamps_ns=np.zeros(len(split.features), dtype=np.int64),
        first_timestamp_ns=0,
    )


def is_available(data_dir: Path | str) -> bool:
    """True when every file `load_benchmark` needs is present."""
    data_dir = Path(data_dir)
    every_file = DEFAULT_FILENAMES["train"] + DEFAULT_FILENAMES["test"]
    return all((data_dir / name).exists() for name in every_file)


def _download_hint(data_dir: Path) -> str:
    return (
        f"FI-2010 benchmark files not found in {data_dir}.\n"
        "The canonical record is https://etsin.fairdata.fi/dataset/"
        "73eb48d7-4dbc-4a10-a52a-da745b47a649 ; the preprocessed copy that published\n"
        "results are quoted on can be fetched with:\n"
        f"    mkdir -p {data_dir}\n"
        "    curl -L -o data.zip https://raw.githubusercontent.com/zcakhaa/"
        "DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/master/data/data.zip\n"
        "    unzip data.zip"
    )
