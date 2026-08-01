"""Tests for `ml.dataset`, `ml.metrics`, `ml.baseline` and `ml.fi2010`.

The dataset module is where Stage 3's separate guarantees get glued together,
which makes it the most likely place for a leakage bug to reappear. So the
tests here are mostly about boundaries: a window must never straddle two
capture sessions, a sample must never exist without a real label, and the fast
batch gather must return exactly what the slow per-sample path returns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.dataset import (
    BatchedWindowLoader,
    SampleIndex,
    Session,
    WindowDataset,
    build_sample_index,
    gather_last_rows,
    gather_windows,
    load_session,
)
from ml.features import WINDOW_LENGTH
from ml.metrics import accuracy, confusion_matrix, macro_f1, majority_class_baseline, per_class_scores

SAMPLE_TAPE = Path(__file__).resolve().parents[1] / "notebooks" / "sample_data" / "btcusdt_dense_sample.tape"
FI2010_DIR = Path(__file__).resolve().parents[1] / "data" / "fi2010"


def make_session(name: str, rows: int, seed: int, usable_from: int = WINDOW_LENGTH - 1) -> Session:
    rng = np.random.default_rng(seed)
    usable = np.zeros(rows, dtype=bool)
    usable[usable_from:] = True
    mids = np.full(rows, 65_000.0)
    return Session(
        name=name,
        features=rng.normal(size=(rows, 40)).astype(np.float32),
        labels=rng.integers(0, 3, size=rows).astype(np.int64),
        usable=usable,
        mids=mids,
        best_bids=mids - 0.005,
        best_asks=mids + 0.005,
        # ~29 snapshots a second, matching the captured tapes.
        timestamps_ns=seed * 1_000_000_000 + np.arange(rows, dtype=np.int64) * 34_000_000,
        first_timestamp_ns=seed * 1_000_000_000,
    )


# --------------------------------------------------------------- the metrics


def test_confusion_matrix_rows_are_truth_and_columns_are_prediction():
    truth = np.array([0, 0, 1, 2, 2, 2])
    prediction = np.array([0, 1, 1, 2, 2, 0])

    matrix = confusion_matrix(truth, prediction)

    assert matrix[0, 0] == 1 and matrix[0, 1] == 1
    assert matrix[1, 1] == 1
    assert matrix[2, 2] == 2 and matrix[2, 0] == 1
    assert matrix.sum() == len(truth)


def test_a_class_the_model_never_predicts_scores_zero_not_nan():
    """Macro-F1 must penalise ignoring a class, not quietly drop it."""
    truth = np.array([0, 1, 2, 0, 1, 2])
    always_flat = np.full(6, 1)

    scores = per_class_scores(truth, always_flat)

    assert scores["f1"][0] == 0.0 and scores["f1"][2] == 0.0
    assert np.all(np.isfinite(scores["f1"]))
    assert macro_f1(truth, always_flat) == pytest.approx(scores["f1"].mean())


def test_perfect_prediction_scores_one():
    truth = np.array([0, 1, 2, 0, 1, 2])
    assert macro_f1(truth, truth) == pytest.approx(1.0)
    assert accuracy(truth, truth) == pytest.approx(1.0)


def test_majority_baseline_uses_the_train_split_to_choose_its_class():
    """Picking the majority class from test labels would be a peek at the answer."""
    train = np.array([0, 0, 0, 0, 1, 2])
    test = np.array([1, 1, 1, 2, 2, 0])

    result = majority_class_baseline(train, test)

    assert result["predicted_class"] == 0.0
    assert result["accuracy"] == pytest.approx(1 / 6)


def test_macro_f1_on_a_balanced_three_class_majority_guess_is_about_one_sixth():
    """The number every result has to clear before it means anything."""
    truth = np.repeat([0, 1, 2], 100)
    always_zero = np.zeros(300, dtype=np.int64)

    assert macro_f1(truth, always_zero) == pytest.approx(0.1667, abs=0.01)


# ------------------------------------------------------- session boundaries


def test_sample_index_is_chronological_across_sessions():
    sessions = [make_session("a", 400, seed=1), make_session("b", 300, seed=2)]

    index = build_sample_index(sessions)

    session_ids = index.session_of_sample
    assert np.all(np.diff(session_ids) >= 0), "samples must not interleave between sessions"
    assert session_ids[0] == 0 and session_ids[-1] == 1


def test_no_window_straddles_a_session_boundary():
    """The property that stops us inventing market data across a resync gap.

    Every window is drawn from one session's matrix, so a window can only span
    a boundary if the index is wrong. Checked by confirming every sample's row
    range lies inside its own session.
    """
    sessions = [make_session("a", 400, seed=1), make_session("b", 300, seed=2)]
    index = build_sample_index(sessions)

    for position in range(len(index)):
        session = sessions[int(index.session_of_sample[position])]
        end_row = int(index.end_row_of_sample[position])
        assert end_row - WINDOW_LENGTH + 1 >= 0
        assert end_row < len(session.features)


def test_rows_without_a_full_window_are_excluded():
    sessions = [make_session("a", 400, seed=3)]

    index = build_sample_index(sessions)

    assert int(index.end_row_of_sample.min()) >= WINDOW_LENGTH - 1
    assert len(index) == 400 - (WINDOW_LENGTH - 1)


def test_an_empty_session_list_produces_an_empty_index():
    index = build_sample_index([])

    assert len(index) == 0
    assert index.class_balance() == {"down": 0.0, "flat": 0.0, "up": 0.0}


# --------------------------------------------------- the fast path is honest


def test_batched_gather_matches_the_per_sample_path_exactly():
    """The optimisation must be invisible in the output, or it is a bug.

    `gather_windows` exists purely for speed; if it disagreed with
    `WindowDataset.__getitem__` by even one row the model would train on
    something other than what the notebook shows.
    """
    sessions = [make_session("a", 400, seed=4), make_session("b", 350, seed=5)]
    index = build_sample_index(sessions)
    positions = np.array([0, 17, 200, 301, len(index) - 1])

    fast = gather_windows(sessions, index, positions)
    dataset = WindowDataset(sessions, index, positions)

    for slot in range(len(positions)):
        slow_window, slow_label = dataset[slot]
        assert np.array_equal(fast[slot], slow_window)
        assert slow_label == int(index.labels[positions[slot]])


def test_gathered_window_ends_on_the_sample_row():
    """Row -1 of the window must be the row the label is attached to."""
    sessions = [make_session("a", 400, seed=6)]
    index = build_sample_index(sessions)
    positions = np.array([5, 50, 120])

    windows = gather_windows(sessions, index, positions)
    last_rows = gather_last_rows(sessions, index, positions)

    for slot, position in enumerate(positions):
        end_row = int(index.end_row_of_sample[position])
        assert np.array_equal(windows[slot][-1], sessions[0].features[end_row])
        assert np.array_equal(last_rows[slot], sessions[0].features[end_row])


def test_loader_covers_every_sample_exactly_once():
    sessions = [make_session("a", 500, seed=7)]
    index = build_sample_index(sessions)
    positions = np.arange(len(index))
    loader = BatchedWindowLoader(sessions, index, positions, batch_size=64, shuffle=True, seed=1)

    seen = 0
    labels_seen = []
    for windows, labels in loader:
        seen += len(windows)
        labels_seen.append(labels)

    assert seen == len(positions)
    assert sorted(np.concatenate(labels_seen).tolist()) == sorted(index.labels.tolist())


def test_shuffling_changes_order_but_not_content():
    sessions = [make_session("a", 400, seed=8)]
    index = build_sample_index(sessions)
    positions = np.arange(len(index))

    ordered = np.concatenate([labels for _windows, labels in BatchedWindowLoader(sessions, index, positions, 32)])
    shuffled = np.concatenate(
        [labels for _windows, labels in BatchedWindowLoader(sessions, index, positions, 32, shuffle=True, seed=3)]
    )

    assert not np.array_equal(ordered, shuffled)
    assert np.array_equal(np.sort(ordered), np.sort(shuffled))


# ------------------------------------------------------- loading a real tape


@pytest.mark.skipif(not SAMPLE_TAPE.exists(), reason="bundled dense sample tape is missing")
def test_a_real_tape_loads_into_a_usable_session():
    session = load_session(SAMPLE_TAPE)

    assert session.features.shape[1] == 40
    assert session.features.dtype == np.float32
    assert len(session.labels) == len(session.features) == len(session.mids)
    assert session.usable.any(), "the sample tape should yield at least some usable rows"
    assert np.all(session.mids[session.usable] > 0)


@pytest.mark.skipif(not SAMPLE_TAPE.exists(), reason="bundled dense sample tape is missing")
def test_usable_rows_exclude_the_normalisation_warmup_and_the_label_tail():
    session = load_session(SAMPLE_TAPE)
    rows = session.usable_end_rows()

    # Nothing usable before the lookback and the window have both elapsed.
    assert int(rows.min()) >= WINDOW_LENGTH - 1
    # The last k rows have no forward window, so they cannot be labelled.
    assert int(rows.max()) < len(session.features) - 1


# ------------------------------------------------------------ the baselines


def test_queue_imbalance_is_positive_when_the_bid_holds_more_size():
    from ml.baseline import queue_imbalance

    features = np.zeros((2, 40), dtype=np.float32)
    features[0, 1::4] = 2.0  # bid sizes
    features[0, 3::4] = 1.0  # ask sizes
    features[1, 1::4] = 1.0
    features[1, 3::4] = 2.0

    imbalance = queue_imbalance(features)

    assert imbalance[0] > 0
    assert imbalance[1] < 0
    assert imbalance[0] == pytest.approx(-imbalance[1])


def test_imbalance_rule_dead_zone_widens_into_flat():
    from ml.baseline import imbalance_rule_predictions

    rng = np.random.default_rng(0)
    features = np.abs(rng.normal(size=(500, 40))).astype(np.float32)

    narrow = imbalance_rule_predictions(features, dead_zone=0.0)
    wide = imbalance_rule_predictions(features, dead_zone=10.0)

    assert np.count_nonzero(wide == 1) > np.count_nonzero(narrow == 1)
    assert np.all(wide == 1), "an enormous dead zone must call everything flat"


def test_imbalance_threshold_is_chosen_on_train_not_test():
    """Tuning the baseline's threshold on test would be look-ahead in miniature."""
    from ml.baseline import evaluate_imbalance_rule

    rng = np.random.default_rng(1)
    train_features = np.abs(rng.normal(size=(400, 40))).astype(np.float32)
    train_labels = rng.integers(0, 3, 400)
    test_features = np.abs(rng.normal(size=(200, 40))).astype(np.float32)
    test_labels = rng.integers(0, 3, 200)

    result = evaluate_imbalance_rule(train_features, train_labels, test_features, test_labels)

    assert 0.0 <= result["macro_f1"] <= 1.0
    assert result["dead_zone"] >= 0.0
    assert "train_macro_f1" in result


# ------------------------------------------------------------- FI-2010


def test_fi2010_label_encoding_maps_onto_the_project_convention():
    """The dataset ships 1=up, 2=stationary, 3=down; we use 0=down, 1=flat, 2=up."""
    from ml.fi2010 import _to_project_encoding

    mapped = _to_project_encoding(np.array([1, 2, 3, 1]))

    assert mapped.tolist() == [2, 1, 0, 2]


def test_fi2010_rejects_unexpected_label_values():
    from ml.fi2010 import _to_project_encoding

    with pytest.raises(ValueError, match="unexpected label values"):
        _to_project_encoding(np.array([1, 2, 7]))


def test_fi2010_reordering_puts_the_bid_first():
    """FI-2010 stores (ask, bid) per level; our model's comments assume (bid, ask)."""
    from ml.fi2010 import _reorder_to_bid_first

    original = np.arange(8, dtype=np.float32).reshape(1, 8)  # two levels

    reordered = _reorder_to_bid_first(original)

    assert reordered[0].tolist() == [2.0, 3.0, 0.0, 1.0, 6.0, 7.0, 4.0, 5.0]


def test_fi2010_missing_files_explain_how_to_get_them():
    from ml.fi2010 import FI2010NotAvailable, load_file

    with pytest.raises(FI2010NotAvailable, match="etsin.fairdata.fi"):
        load_file(Path("nowhere") / "Train_Dst_NoAuction_DecPre_CF_7.txt")


def test_fi2010_rejects_an_unsupported_horizon():
    from ml.fi2010 import load_file

    with pytest.raises(ValueError, match="horizon must be one of"):
        load_file(FI2010_DIR / "Train_Dst_NoAuction_DecPre_CF_7.txt", horizon=4)


@pytest.mark.skipif(
    not (FI2010_DIR / "Test_Dst_NoAuction_DecPre_CF_9.txt").exists(),
    reason="FI-2010 benchmark files not downloaded",
)
def test_fi2010_smallest_test_file_loads_with_the_documented_shape():
    from ml.fi2010 import load_file

    split = load_file(FI2010_DIR / "Test_Dst_NoAuction_DecPre_CF_9.txt", horizon=10)

    assert split.features.shape[1] == 40
    assert split.features.dtype == np.float32
    assert len(split.labels) == len(split.features)
    assert set(np.unique(split.labels).tolist()) <= {0, 1, 2}
    assert sum(split.class_balance().values()) == pytest.approx(1.0)
