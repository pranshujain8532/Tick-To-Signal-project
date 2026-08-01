"""Tests for `ml.model`.

Four things are worth asserting about an architecture before it is ever
trained, and all four have caught real mistakes in this file:

  * **Shapes.** The front end collapses 40 feature columns to 1 in three
    deliberate steps; if any stride is wrong the model still runs and still
    trains, it just fuses the wrong things.
  * **Parameter count.** "About 300K" is a design commitment — small enough to
    distil and hand-roll in C++ later. A silent widening breaks Stages 6 and 7
    long before anyone notices.
  * **Causality.** The whole "why not an LSTM" argument rests on being able to
    advance the model one tick at a time, which requires that position t never
    depends on position t+1. Asserted by perturbation, not by reading the
    padding arguments.
  * **Overfitting one batch.** The classic sanity check, explained at its test.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ml.model import ModelConfig, TickToSignalNet, build_model  # noqa: E402


BATCH = 8
WINDOW = 100
FEATURES = 40


@pytest.fixture()
def model() -> TickToSignalNet:
    return build_model(seed=0)


# ----------------------------------------------------------------- shapes


def test_forward_maps_a_window_to_three_logits(model: TickToSignalNet):
    output = model(torch.randn(BATCH, WINDOW, FEATURES))

    assert output.shape == (BATCH, 3)
    assert torch.isfinite(output).all()


def test_front_end_collapses_the_feature_axis_in_three_steps(model: TickToSignalNet):
    """40 -> 20 -> 10 -> 1: pairs, then sides, then levels.

    Each of these is a microstructure claim, not just a shape. 40->20 fuses
    (price, size) into "what rests here"; 20->10 fuses bid against ask into a
    per-level imbalance; 10->1 fuses the levels into whole-book shape.
    """
    model.eval()
    hidden = torch.randn(BATCH, WINDOW, FEATURES).unsqueeze(1)

    with torch.no_grad():
        after_pairs = model.fuse_price_and_size(hidden)
        after_sides = model.fuse_sides(after_pairs)
        after_levels = model.fuse_levels(after_sides)

    assert after_pairs.shape == (BATCH, model.config.conv_channels, WINDOW, 20)
    assert after_sides.shape == (BATCH, model.config.conv_channels, WINDOW, 10)
    assert after_levels.shape == (BATCH, model.config.conv_channels, WINDOW, 1)


def test_time_axis_is_never_shortened(model: TickToSignalNet):
    """Causal padding keeps T=100 throughout; a shrinking T means a bug."""
    model.eval()
    hidden = torch.randn(BATCH, WINDOW, FEATURES).unsqueeze(1)

    with torch.no_grad():
        for stage in (model.fuse_price_and_size, model.fuse_sides, model.fuse_levels):
            hidden = stage(hidden)
            assert hidden.shape[2] == WINDOW
        assert model.inception(hidden).shape[2] == WINDOW


def test_a_wrongly_shaped_input_is_refused(model: TickToSignalNet):
    with pytest.raises(ValueError, match=r"expected \[batch, time, features\]"):
        model(torch.randn(BATCH, WINDOW, FEATURES, 1))


# -------------------------------------------------------- parameter budget


def test_parameter_count_is_within_the_design_budget(model: TickToSignalNet):
    """250K-400K is a commitment, not an observation.

    Small enough that Stage 6 can distil it and Stage 7 can hand-roll the
    forward pass; large enough to have some capacity. If this test fails
    because the model grew, the fix is to decide deliberately whether the
    later stages can still carry it, not to widen the bound.
    """
    count = model.parameter_count()

    assert 250_000 <= count <= 400_000, f"model has {count:,} parameters, outside the 250K-400K budget"


def test_receptive_field_fits_inside_the_window(model: TickToSignalNet):
    """A field longer than the window means paying for context we never feed it."""
    field = model.receptive_field()

    assert field == 83
    assert field <= model.config.window_length


# --------------------------------------------------------------- causality


def test_the_model_cannot_see_forward_in_time(model: TickToSignalNet):
    """Position t must not depend on any input after t. Tested by gradient.

    WHY GRADIENTS AND NOT PERTURBATION. The obvious test — nudge one timestep
    and diff the activations — does not work on an untrained network. Measured
    on this architecture, a +5.0 nudge at the last row reaches the logits as a
    change of about 1e-6: real, but far below `allclose`'s default tolerance,
    because at initialisation a deep stack of small random weights is nearly a
    constant function. A test built on that would have to pick a tolerance out
    of the air and would silently become vacuous.

    `d output[t] / d input[t']` is exact instead: it is *identically zero* when
    the dependency does not exist, whatever the weights happen to be. That is
    the property being claimed, so that is the thing to measure.

    Run in `eval` so BatchNorm uses frozen running statistics. In `train` mode
    BatchNorm normalises across batch *and* time, which mixes positions — not
    look-ahead (the window already ends at t and the label looks forward from
    t), but it does mean the strictly causal object is the inference-mode
    model, which is the one Stage 7 reimplements.
    """
    model.eval()
    inputs = torch.randn(1, WINDOW, FEATURES, requires_grad=True)
    read_at = 60

    activations = _temporal_stack_output(model, inputs)
    activations[0, :, read_at].sum().backward()
    gradient = inputs.grad[0].abs().sum(dim=1)

    assert torch.all(gradient[read_at + 1 :] == 0), "position 60 depends on the future"
    assert gradient[read_at] > 0, "position 60 does not depend on its own input"
    assert gradient[read_at - 1] > 0, "position 60 does not depend on the recent past"


def test_the_receptive_field_is_exactly_as_advertised(model: TickToSignalNet):
    """`receptive_field()` claims 83 steps; the gradient must agree.

    A receptive field that is only computed on paper drifts the moment a
    kernel width changes. This reads it off the actual computation graph.
    """
    model.eval()
    inputs = torch.randn(1, WINDOW, FEATURES, requires_grad=True)

    model(inputs).sum().backward()
    influential = torch.nonzero(inputs.grad[0].abs().sum(dim=1)).flatten()

    reached_back = WINDOW - int(influential.min())
    assert reached_back == model.receptive_field() == 83
    assert int(influential.max()) == WINDOW - 1, "the final timestep must be inside the field"


def _temporal_stack_output(model: TickToSignalNet, inputs: torch.Tensor) -> torch.Tensor:
    """Run the network up to the end of the TCN, keeping the full time axis."""
    hidden = inputs.unsqueeze(1)
    hidden = model.fuse_price_and_size(hidden)
    hidden = model.fuse_sides(hidden)
    hidden = model.fuse_levels(hidden)
    hidden = model.inception(hidden).squeeze(-1)
    hidden = model.project(hidden)
    for block in model.temporal_blocks:
        hidden = block(hidden)
    return hidden


def test_the_final_timestep_reaches_the_classifier(model: TickToSignalNet):
    """The read-out is `hidden[:, :, -1]`, so the newest row must matter most.

    Checked by gradient magnitude for the reason given above: at
    initialisation the *value* change is ~1e-6, but the dependency is real and
    the gradient shows it plainly.
    """
    model.eval()
    inputs = torch.randn(1, WINDOW, FEATURES, requires_grad=True)

    model(inputs).sum().backward()
    per_step = inputs.grad[0].abs().sum(dim=1)

    assert per_step[-1] > 0, "the most recent row does not reach the prediction"
    assert per_step[:-83].sum() == 0, "rows outside the receptive field must contribute nothing"


# ------------------------------------------------------- the sanity check


def test_model_can_overfit_a_single_batch():
    """The classic sanity check: can the model memorise 32 samples?

    WHY THIS TEST EXISTS. Before asking whether an architecture *generalises*,
    ask whether it can learn at all. A model that cannot drive the loss to zero
    on a batch it sees 200 times has a bug — a detached gradient, a wrong axis,
    a label misalignment, a dead activation — and no amount of tuning on the
    real dataset will reveal which. Overfitting on purpose separates "the
    plumbing is broken" from "the problem is hard", and those two failures look
    identical on a learning curve.

    It is also the fastest possible check that labels line up with inputs: if
    the pairing were shuffled, the model could still memorise, so we
    deliberately use *few* samples and *many* steps, where memorisation is easy
    and any wiring fault is not.
    """
    torch.manual_seed(0)
    model = build_model(seed=0)
    inputs = torch.randn(32, WINDOW, FEATURES)
    targets = torch.randint(0, 3, (32,))

    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_function = torch.nn.CrossEntropyLoss()

    model.train()
    accuracy = 0.0
    for step in range(200):
        optimiser.zero_grad()
        logits = model(inputs)
        loss = loss_function(logits, targets)
        loss.backward()
        optimiser.step()
        accuracy = float((logits.argmax(dim=1) == targets).float().mean())
        if accuracy == 1.0:
            break

    assert accuracy == 1.0, f"only reached {accuracy:.0%} on 32 samples after {step + 1} steps"


def test_a_narrower_config_still_builds_and_runs():
    """The config is a dataclass, so a smaller student model is one line.

    Stage 6 distils into something much smaller; this proves the architecture
    is parameterised rather than hardcoded, before that stage depends on it.
    """
    small = ModelConfig(conv_channels=8, inception_channels=16, tcn_channels=24, tcn_dilations=(1, 2))
    student = build_model(small)

    output = student(torch.randn(2, WINDOW, FEATURES))

    assert output.shape == (2, 3)
    assert student.parameter_count() < 60_000
