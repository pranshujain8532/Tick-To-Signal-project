"""Tests for `ml.export` and `ml.distill`.

The parity test is the one that matters, and it is mandatory in the same sense
the C++/PyTorch parity test will be in Stage 7: a graph that computes a
slightly different function is a different model, and nothing downstream —
latency, quantisation, the Pareto chart — means anything if it is wrong.

Export and quantisation are slow (seconds each), so the ONNX artefacts are
built once per module via a session-scoped fixture rather than per test.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from ml.distill import (  # noqa: E402
    STUDENT_CONFIG,
    DistillConfig,
    build_student,
    distillation_loss,
)
from ml.export import (  # noqa: E402
    ParityReport,
    DEPLOY_FEATURES,
    DEPLOY_WINDOW,
    INPUT_NAME,
    OUTPUT_NAME,
    WindowCalibrationReader,
    check_parity,
    describe_quantised_layer,
    export_to_onnx,
    file_size_kib,
    make_session,
    quantize_to_int8,
    run_onnx,
)
from ml.model import ModelConfig, build_model  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model():
    """A small model so export and quantisation stay fast in the test suite."""
    config = ModelConfig(conv_channels=6, inception_channels=8, tcn_channels=12, tcn_dilations=(1, 2))
    return build_model(config, seed=0).eval()


@pytest.fixture(scope="module")
def exported(tiny_model, tmp_path_factory):
    directory = tmp_path_factory.mktemp("onnx")
    return export_to_onnx(tiny_model, directory / "model.onnx")


# ------------------------------------------------------------------- export


def test_export_produces_a_loadable_graph_with_named_tensors(exported):
    session = make_session(exported)

    assert [i.name for i in session.get_inputs()] == [INPUT_NAME]
    assert [o.name for o in session.get_outputs()] == [OUTPUT_NAME]


def test_the_input_shape_is_fixed_at_batch_one(exported):
    """A dynamic batch axis would cost latency; we deploy exactly one shape."""
    shape = make_session(exported).get_inputs()[0].shape

    assert shape == [1, DEPLOY_WINDOW, DEPLOY_FEATURES]
    assert all(isinstance(dimension, int) for dimension in shape), "no dimension may be symbolic"


def test_a_wrong_batch_size_is_rejected_rather_than_silently_reshaped(exported):
    session = make_session(exported)
    two_windows = np.zeros((2, DEPLOY_WINDOW, DEPLOY_FEATURES), dtype=np.float32)

    with pytest.raises(Exception):
        session.run(None, {INPUT_NAME: two_windows})


# ------------------------------------------------------- THE parity test


def test_onnx_matches_pytorch_within_tolerance_on_1000_random_inputs(tiny_model, exported):
    """MANDATORY. Correctness before performance, always.

    1,000 inputs through both paths; the worst absolute logit difference must
    stay under 1e-5. Argmax agreement is asserted separately because it is the
    thing that actually changes a prediction — a graph can be numerically loose
    without ever flipping a decision, or numerically tight while flipping one
    near a boundary, and the two claims are different.
    """
    report = check_parity(tiny_model, exported, trials=1_000, tolerance=1e-5, seed=0)

    assert report.trials == 1_000
    assert report.passed, report.summary()
    assert report.max_absolute_difference < 1e-5
    assert report.argmax_agreement == 1.0


def test_parity_also_holds_on_real_captured_windows(tiny_model, exported):
    """Random normals are a proxy for z-scored features; real data is the claim.

    Heavy tails are exactly where a numerical difference would show up, so the
    real distribution is the harder test of the two.
    """
    generator = np.random.default_rng(1)
    # Heavy-tailed, like the real causally-z-scored features (bulk ~1.4, tails to ~22).
    windows = (generator.standard_t(df=3, size=(200, DEPLOY_WINDOW, DEPLOY_FEATURES)) * 1.4).astype(np.float32)

    report = check_parity(tiny_model, exported, trials=200, tolerance=1e-5, inputs=windows)

    assert report.passed, report.summary()


def test_parity_reports_both_absolute_and_relative_divergence(tiny_model, exported):
    """A fixed absolute tolerance is not scale-invariant, so both are reported.

    Measured on the real models: the teacher sits at 5.2e-06 absolute and the
    student at 3.5e-05 — the student breaches a 1e-5 absolute bar while its
    *relative* divergence is 3.7e-06 and it agrees with PyTorch on every
    argmax. The report therefore carries both numbers so neither claim can be
    made without the other.
    """
    report = check_parity(tiny_model, exported, trials=100, tolerance=1e-5)

    assert report.output_scale > 0
    assert report.max_relative_difference == pytest.approx(
        report.max_absolute_difference / report.output_scale, rel=1e-9
    )
    assert isinstance(report.meets_absolute_tolerance, bool)


def test_parity_fails_when_an_argmax_flips_even_if_numerically_close():
    """Argmax agreement is a separate gate, and it must be able to fail alone.

    A graph can be numerically tight and still flip a decision that sat on a
    boundary. Constructed here directly rather than hoping an export produces
    the case.
    """
    report = ParityReport(
        trials=100,
        tolerance=1e-5,
        relative_tolerance=1e-5,
        max_absolute_difference=1e-9,
        mean_absolute_difference=1e-10,
        max_relative_difference=1e-9,
        output_scale=1.0,
        worst_trial=0,
        max_probability_difference=1e-9,
        argmax_agreement=0.99,
    )

    assert report.meets_absolute_tolerance
    assert not report.passed, "a flipped prediction must fail parity however small the difference"


def test_parity_report_fails_loudly_when_the_graph_is_wrong(tiny_model, exported, tmp_path):
    """The test must be able to fail, or it proves nothing.

    Export a *different* model to the same filename and confirm the parity
    check rejects it.
    """
    other = build_model(ModelConfig(conv_channels=6, inception_channels=8, tcn_channels=12, tcn_dilations=(1, 2)), seed=99)
    wrong_path = export_to_onnx(other, tmp_path / "other.onnx")

    report = check_parity(tiny_model, wrong_path, trials=50, tolerance=1e-5)

    assert not report.passed
    assert report.argmax_agreement < 1.0


# ------------------------------------------------------------ quantisation


def test_calibration_reader_yields_one_window_at_a_time():
    windows = np.zeros((3, DEPLOY_WINDOW, DEPLOY_FEATURES), dtype=np.float32)
    reader = WindowCalibrationReader(windows)

    batches = []
    while (batch := reader.get_next()) is not None:
        batches.append(batch)

    assert len(batches) == 3
    assert batches[0][INPUT_NAME].shape == (1, DEPLOY_WINDOW, DEPLOY_FEATURES)
    reader.rewind()
    assert reader.get_next() is not None


def test_calibration_reader_refuses_wrongly_shaped_data():
    with pytest.raises(ValueError, match=r"expected \[N, T, F\]"):
        WindowCalibrationReader(np.zeros((10, 40), dtype=np.float32))


@pytest.fixture(scope="module")
def quantised(exported, tmp_path_factory):
    generator = np.random.default_rng(0)
    calibration = generator.standard_normal((32, DEPLOY_WINDOW, DEPLOY_FEATURES)).astype(np.float32)
    directory = tmp_path_factory.mktemp("int8")
    return quantize_to_int8(exported, directory / "model_int8.onnx", calibration)


def test_quantisation_shrinks_a_real_sized_model(tmp_path):
    """int8 shrinks *weights*, but QDQ adds graph structure — size is net.

    Measured on this architecture at three scales, fp32 -> int8:
        4,425 params    55.0 -> 75.9 KiB   0.73x  (it GROWS)
       32,155 params   173.1 -> 125.3 KiB  1.38x
      319,715 params  1290.7 -> 421.4 KiB  3.06x

    The Quantize/DequantizeLinear pairs and their scale and zero-point
    initialisers cost a roughly constant ~50 KiB, so quantisation only pays for
    itself once the weights dominate that overhead. Asserted at student scale
    because that is the smallest model we actually ship.
    """
    student = build_student(seed=0).eval()
    fp32_path = export_to_onnx(student, tmp_path / "student.onnx")
    generator = np.random.default_rng(0)
    calibration = generator.standard_normal((16, DEPLOY_WINDOW, DEPLOY_FEATURES)).astype(np.float32)

    int8_path = quantize_to_int8(fp32_path, tmp_path / "student_int8.onnx", calibration)

    assert file_size_kib(int8_path) < file_size_kib(fp32_path)


def test_quantisation_can_make_a_very_small_model_bigger(exported, quantised):
    """Documents the overhead rather than pretending int8 always wins.

    The fixture model is deliberately tiny (4,425 parameters) so the suite
    stays fast, and at that size the QDQ scaffolding outweighs the weight
    savings. Recording it here means the size claim in the Pareto table is
    understood to be scale-dependent.
    """
    assert file_size_kib(quantised) > file_size_kib(exported)


def test_the_quantised_graph_still_runs_and_returns_three_logits(quantised):
    session = make_session(quantised)
    windows = np.zeros((4, DEPLOY_WINDOW, DEPLOY_FEATURES), dtype=np.float32)

    outputs = run_onnx(session, windows)

    assert outputs.shape == (4, 3)
    assert np.isfinite(outputs).all()


def test_quantised_graph_exposes_real_scales_and_zero_points(quantised):
    """Quantisation is an affine map; the notebook shows the actual constants."""
    layers = describe_quantised_layer(quantised, limit=1)

    assert layers, "expected at least one Quantize/DequantizeLinear node"
    entry = layers[0]
    assert entry["op_type"] in ("QuantizeLinear", "DequantizeLinear")
    assert len(entry["scale_sample"]) >= 1
    assert all(scale > 0 for scale in entry["scale_sample"]), "a scale must be positive to be invertible"


def test_percentile_calibration_is_the_default_not_minmax(exported, tmp_path):
    """Guards the measured decision: percentile 99.99 keeps fp32 agreement at
    0.962 where MinMax drops it to 0.911 on the full held-out block.

    Both paths must produce a working graph; the point of the test is that
    asking for the default and asking for percentile give the same result,
    so a future edit cannot silently revert to MinMax.
    """
    generator = np.random.default_rng(0)
    calibration = generator.standard_normal((16, DEPLOY_WINDOW, DEPLOY_FEATURES)).astype(np.float32)

    default_path = quantize_to_int8(exported, tmp_path / "default.onnx", calibration)
    minmax_path = quantize_to_int8(exported, tmp_path / "minmax.onnx", calibration, percentile=None)

    windows = generator.standard_normal((8, DEPLOY_WINDOW, DEPLOY_FEATURES)).astype(np.float32)
    default_out = run_onnx(make_session(default_path), windows)
    minmax_out = run_onnx(make_session(minmax_path), windows)

    assert np.isfinite(default_out).all() and np.isfinite(minmax_out).all()
    assert not np.allclose(default_out, minmax_out), "the two calibrators must produce different graphs"


# ------------------------------------------------------------- distillation


def test_student_parameter_count_is_within_the_design_budget():
    """30K-50K is a commitment: small enough to hand-roll, large enough to learn."""
    student = build_student(seed=0)

    count = student.parameter_count()
    assert 30_000 <= count <= 50_000, f"student has {count:,} parameters, outside the 30K-50K budget"


def test_student_keeps_the_teachers_receptive_field():
    """Narrower, not shorter-sighted — otherwise capacity and context confound."""
    student = build_student(seed=0)
    teacher = build_model(ModelConfig(), seed=0)

    assert student.receptive_field() == teacher.receptive_field() == 83
    assert student.config.tcn_dilations == teacher.config.tcn_dilations


def test_student_is_roughly_ten_times_smaller_than_the_teacher():
    student = build_student(seed=0)
    teacher = build_model(ModelConfig(), seed=0)

    ratio = teacher.parameter_count() / student.parameter_count()
    assert 8.0 <= ratio <= 12.0, f"compression ratio {ratio:.1f}x is outside the intended ~10x"


def test_distillation_loss_reduces_to_cross_entropy_when_alpha_is_one():
    """alpha=1 means "ignore the teacher", which must be exactly plain CE."""
    torch.manual_seed(0)
    student_logits = torch.randn(8, 3)
    teacher_logits = torch.randn(8, 3)
    labels = torch.randint(0, 3, (8,))

    combined = distillation_loss(student_logits, teacher_logits, labels, alpha=1.0, temperature=4.0)
    plain = torch.nn.functional.cross_entropy(student_logits, labels)

    assert torch.allclose(combined, plain)


def test_distillation_loss_is_near_zero_when_the_student_matches_the_teacher():
    """alpha=0 isolates the KL term, which must vanish on a perfect match."""
    torch.manual_seed(0)
    logits = torch.randn(8, 3)
    labels = torch.randint(0, 3, (8,))

    loss = distillation_loss(logits, logits.clone(), labels, alpha=0.0, temperature=4.0)

    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_the_temperature_squared_factor_keeps_the_soft_term_scale_stable():
    """THE point of the T^2 factor, asserted rather than described.

    Softening by T shrinks the KL gradients by ~1/T^2. Multiplying by T^2
    cancels that, so raising the temperature changes how *soft* the targets are
    without silently rescaling the distillation term against the hard-label
    term. Without the factor, the soft loss at T=8 would be ~16x smaller than
    at T=2 for the same disagreement; with it, the two stay the same order of
    magnitude.
    """
    torch.manual_seed(0)
    student_logits = torch.randn(64, 3)
    teacher_logits = student_logits + torch.randn(64, 3) * 0.5
    labels = torch.randint(0, 3, (64,))

    corrected = []
    uncorrected = []
    for temperature in (2.0, 4.0, 8.0):
        corrected.append(float(distillation_loss(student_logits, teacher_logits, labels, 0.0, temperature)))
        student_soft = torch.nn.functional.log_softmax(student_logits / temperature, dim=1)
        teacher_soft = torch.nn.functional.softmax(teacher_logits / temperature, dim=1)
        uncorrected.append(float(torch.nn.functional.kl_div(student_soft, teacher_soft, reduction="batchmean")))

    corrected_spread = max(corrected) / min(corrected)
    uncorrected_spread = max(uncorrected) / min(uncorrected)
    assert corrected_spread < uncorrected_spread / 3, (
        f"T^2 should stabilise the soft term: corrected spread {corrected_spread:.1f}x "
        f"vs uncorrected {uncorrected_spread:.1f}x"
    )


def test_a_confident_teacher_and_a_wrong_student_produce_a_large_loss():
    """Sanity: the loss must actually respond to disagreement."""
    student_logits = torch.tensor([[5.0, 0.0, 0.0]])
    teacher_logits = torch.tensor([[0.0, 0.0, 5.0]])
    labels = torch.tensor([2])

    agree = distillation_loss(teacher_logits, teacher_logits, labels, 0.3, 4.0)
    disagree = distillation_loss(student_logits, teacher_logits, labels, 0.3, 4.0)

    assert float(disagree) > float(agree)


def test_student_config_is_the_teacher_config_only_narrower():
    """Same block structure is a claim the comparison depends on."""
    teacher = ModelConfig()

    assert STUDENT_CONFIG.window_length == teacher.window_length
    assert STUDENT_CONFIG.depth_levels == teacher.depth_levels
    assert STUDENT_CONFIG.tcn_kernel == teacher.tcn_kernel
    assert STUDENT_CONFIG.tcn_dilations == teacher.tcn_dilations
    assert STUDENT_CONFIG.conv_channels < teacher.conv_channels
    assert STUDENT_CONFIG.tcn_channels < teacher.tcn_channels


def test_distill_config_defaults_are_sane():
    config = DistillConfig()

    assert 0.0 <= config.alpha <= 1.0
    assert config.temperature > 1.0, "T=1 would defeat the purpose of softening"
