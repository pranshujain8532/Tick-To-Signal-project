"""Tests for `ml.export_weights`.

The C++ parity test is the real proof that the exported artefacts are right,
but it lives in another language and another build system. These tests cover
the Python half so that a broken export is caught by `pytest` rather than by a
confusing C++ failure twenty minutes later:

  * the folding maths is exact (verified in float64, where rounding cannot hide
    an algebra error),
  * the file format round-trips,
  * and the fixtures actually correspond to the weights shipped beside them.
"""

from __future__ import annotations

import copy
import struct
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ml.distill import build_student  # noqa: E402
from ml.export_weights import (  # noqa: E402
    DTYPE_FLOAT32,
    FIXTURE_MAGIC,
    FORMAT_VERSION,
    WEIGHT_MAGIC,
    NamedTensor,
    build_folded_model,
    build_op_fixtures,
    fold_all,
    fold_conv_batchnorm,
    generate_fixtures,
    verify_folding,
    write_fixture_file,
    write_weight_file,
)
from ml.model import ModelConfig, build_model  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parents[1] / "inference_cpp" / "artifacts"


@pytest.fixture(scope="module")
def student():
    """An untrained student is enough: folding is exact regardless of values."""
    return build_student(seed=0).eval()


# ------------------------------------------------------------ the folding math


def test_folding_is_exact_in_float64():
    """THE claim that everything else rests on.

    Folding is pure algebra, so in float64 the folded model must reproduce the
    original to machine precision. Measured at ~4e-14 on the real student. If
    this test fails, the derivation or the Inception channel slicing is wrong —
    and no amount of tolerance-widening downstream would make that acceptable.
    """
    model = build_student(seed=1).double().eval()
    folded = build_folded_model(model).double().eval()

    generator = np.random.default_rng(0)
    probe = torch.from_numpy(generator.standard_normal((16, 100, 40))).double()
    with torch.no_grad():
        difference = float((model(probe) - folded(probe)).abs().max())

    assert difference < 1e-11, f"folding is not exact: {difference:.3e} in float64"


def test_folding_in_float32_is_within_rounding(student):
    """The float32 gap must be rounding-sized, and is judged relatively.

    An absolute threshold here was the first thing that failed — the same
    scale-invariance mistake Stage 6 made with the ONNX parity gate. What
    matters is the divergence relative to the size of the logits.
    """
    absolute, relative = verify_folding(student, trials=32)

    assert relative < 1e-5
    assert absolute > 0.0, "exactly zero would mean the fold did nothing at all"


def test_folded_conv_reproduces_conv_then_batchnorm():
    """A single layer, checked directly against `conv -> BN` in eval mode."""
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(3, 4, kernel_size=(2, 2), bias=True)
    norm = torch.nn.BatchNorm2d(4)
    # Give the norm non-trivial statistics; a freshly built one is the identity.
    norm.running_mean = torch.randn(4)
    norm.running_var = torch.rand(4) + 0.5
    norm.weight = torch.nn.Parameter(torch.randn(4))
    norm.bias = torch.nn.Parameter(torch.randn(4))
    norm.eval()

    weight, bias = fold_conv_batchnorm(conv.weight, conv.bias, norm)
    folded = torch.nn.Conv2d(3, 4, kernel_size=(2, 2), bias=True)
    folded.weight = torch.nn.Parameter(weight)
    folded.bias = torch.nn.Parameter(bias)

    probe = torch.randn(5, 3, 8, 8)
    with torch.no_grad():
        reference = norm(conv(probe))
        candidate = folded(probe)

    assert torch.allclose(reference, candidate, atol=1e-5)


def test_a_conv_without_bias_folds_correctly():
    """PyTorch omits the conv bias when a BatchNorm follows; zero is the answer."""
    torch.manual_seed(1)
    conv = torch.nn.Conv2d(2, 3, kernel_size=(1, 1), bias=False)
    norm = torch.nn.BatchNorm2d(3)
    norm.running_mean = torch.randn(3)
    norm.running_var = torch.rand(3) + 0.5
    norm.eval()

    weight, bias = fold_conv_batchnorm(conv.weight, conv.bias, norm)
    probe = torch.randn(4, 2, 5, 5)
    folded = torch.nn.Conv2d(2, 3, kernel_size=(1, 1), bias=True)
    folded.weight = torch.nn.Parameter(weight)
    folded.bias = torch.nn.Parameter(bias)

    with torch.no_grad():
        assert torch.allclose(norm(conv(probe)), folded(probe), atol=1e-6)


def test_inception_norm_splits_across_its_three_branches(student):
    """The one fold that is not a straight conv->BN pair.

    The Inception norm follows a concatenation, so its parameters must be
    sliced by branch. Getting the slice order wrong would mix `conv_five`'s
    statistics into `conv_three`, which is exactly the kind of error that still
    produces plausible logits.
    """
    names = {tensor.name for tensor in fold_all(student)}

    for branch in ("conv_three", "conv_five", "pool_project"):
        assert f"inception.{branch}.weight" in names
        assert f"inception.{branch}.bias" in names
    # The reductions feed an activation, not a norm, so they are NOT folded.
    assert "inception.reduce_three.weight" in names
    assert "inception.reduce_five.weight" in names
    # And the norm itself must not survive as a tensor.
    assert not any(name.startswith("inception.norm") for name in names)


def test_no_batchnorm_tensors_survive_the_fold(student):
    """Every norm must be absorbed; a leftover means a layer was missed."""
    names = [tensor.name for tensor in fold_all(student)]

    assert not [name for name in names if "norm" in name or "running_" in name]
    assert len(names) == len(set(names)), "duplicate tensor names would silently overwrite"


# -------------------------------------------------------------- the file format


def test_weight_file_round_trips(tmp_path):
    """Read the format back with plain struct unpacking, as the C++ does."""
    tensors = [
        NamedTensor("alpha", np.arange(6, dtype=np.float32).reshape(2, 3)),
        NamedTensor("beta", np.array([1.5, -2.5], dtype=np.float32)),
    ]
    path = write_weight_file(tensors, tmp_path / "w.ttsw")
    raw = path.read_bytes()

    assert raw[:4] == WEIGHT_MAGIC
    version, count = struct.unpack_from("<II", raw, 4)
    assert version == FORMAT_VERSION and count == 2

    offset = 12
    for expected in tensors:
        (name_length,) = struct.unpack_from("<I", raw, offset)
        offset += 4
        name = raw[offset : offset + name_length].decode("ascii")
        offset += name_length
        dtype, ndim = struct.unpack_from("<II", raw, offset)
        offset += 8
        dims = struct.unpack_from(f"<{ndim}I", raw, offset)
        offset += 4 * ndim
        values = int(np.prod(dims))
        data = np.frombuffer(raw, dtype=np.float32, count=values, offset=offset)
        offset += 4 * values

        assert name == expected.name
        assert dtype == DTYPE_FLOAT32
        assert dims == expected.array.shape
        assert np.array_equal(data.reshape(dims), expected.array)
    assert offset == len(raw), "no trailing bytes"


def test_weight_file_rejects_non_float32():
    with pytest.raises(ValueError, match="expected float32"):
        NamedTensor("bad", np.zeros((2, 2), dtype=np.float64))


def test_fixture_file_round_trips(tmp_path):
    inputs = np.random.default_rng(0).standard_normal((3, 100, 40)).astype(np.float32)
    outputs = np.random.default_rng(1).standard_normal((3, 3)).astype(np.float32)

    path = write_fixture_file(inputs, outputs, tmp_path / "f.ttsf")
    raw = path.read_bytes()

    assert raw[:4] == FIXTURE_MAGIC
    version, count, window, features, classes = struct.unpack_from("<IIIII", raw, 4)
    assert (version, count, window, features, classes) == (FORMAT_VERSION, 3, 100, 40, 3)

    offset = 24
    stride = window * features
    for index in range(count):
        recovered_input = np.frombuffer(raw, np.float32, stride, offset).reshape(window, features)
        offset += 4 * stride
        recovered_output = np.frombuffer(raw, np.float32, classes, offset)
        offset += 4 * classes
        assert np.array_equal(recovered_input, inputs[index])
        assert np.array_equal(recovered_output, outputs[index])
    assert offset == len(raw)


def test_fixtures_are_produced_by_the_model_they_ship_with(student):
    """Weights and expectations must come from one model, or parity is a lie."""
    inputs, outputs = generate_fixtures(student, count=8)

    with torch.no_grad():
        recomputed = student(torch.from_numpy(inputs)).numpy()

    assert np.allclose(outputs, recomputed, atol=0.0), "fixtures must be exactly reproducible"


def test_op_fixtures_cover_every_operator_the_model_uses():
    """A missing case means an operator with no independent test."""
    cases = {tensor.name.rsplit(".", 1)[0] for tensor in build_op_fixtures()}

    assert cases == {
        "conv2d_spatial", "conv2d_temporal", "conv2d_five", "conv1d_dilated",
        "maxpool_negative", "leaky_relu", "linear", "softmax",
    }


def test_the_maxpool_fixture_is_actually_negative():
    """The case only catches the zero-padding bug if its inputs are negative.

    PyTorch zero-pads before pooling, so a window overhanging the start must
    return 0.0 rather than the largest negative value. On positive data both
    implementations agree and the test proves nothing.
    """
    fixtures = {tensor.name: tensor.array for tensor in build_op_fixtures()}

    assert fixtures["maxpool_negative.input"].max() < 0.0
    # The expectation must therefore contain zeros from the padding.
    assert (fixtures["maxpool_negative.expected"] == 0.0).any()


# --------------------------------------------------- the shipped artefacts


@pytest.mark.skipif(not (ARTIFACTS / "student_weights.ttsw").exists(),
                    reason="run `python -m ml.export_weights` first")
def test_shipped_weight_file_matches_the_current_architecture():
    """Guards against the exported file drifting from ml/model.py."""
    raw = (ARTIFACTS / "student_weights.ttsw").read_bytes()

    assert raw[:4] == WEIGHT_MAGIC
    version, count = struct.unpack_from("<II", raw, 4)
    assert version == FORMAT_VERSION
    # Three fusion blocks x 3 convs x 2 tensors = 18; inception 5 convs x 2 = 10;
    # project 2; four TCN blocks x 2 convs x 2 = 16; classifier 2. Total 48.
    assert count == 48
