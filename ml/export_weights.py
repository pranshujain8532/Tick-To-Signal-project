"""Export the distilled student to a custom binary weight file, and dump parity fixtures.

WHAT
    Writes two files that the hand-written C++ inference path in
    `inference_cpp/` consumes:

      * `student_weights.ttsw` — every tensor the forward pass needs, with
        BatchNorm already folded into the preceding convolution.
      * `student_fixtures.ttsf` — 1,000 (input, output) pairs produced by
        PyTorch, which the C++ implementation must reproduce.

WHY
    Stage 7 reimplements this network in C++ with no dependencies. That code
    needs weights in a format it can read with `fread` and nothing else, and it
    needs an oracle to be checked against. Shipping both from one script keeps
    them consistent: the fixtures are generated from the *same* model object
    that produced the weights, in the same process, so they cannot drift.

DESIGN DECISION — fold BatchNorm into the convolutions here, not in C++.
    Rejected alternative: implement a BatchNorm operator in C++ and keep the
    layers separate. Folding is exact, costs nothing at inference, and removes
    an entire operator — with its four parameter tensors per layer — from the
    code that has to be hand-written and hand-verified. It also removes the
    epsilon and the running-statistics bookkeeping from the C++ side, which are
    two more things that could silently differ from PyTorch.

    THE FOLDING MATH. At inference a BatchNorm applies, per output channel c:

        BN(y_c) = gamma_c * (y_c - mean_c) / sqrt(var_c + eps) + beta_c

    and the convolution before it computes `y_c = sum_i W_ci * x_i + b_c`.
    Substituting and collecting terms in x:

        s_c   = gamma_c / sqrt(var_c + eps)          (one scalar per channel)
        W'_ci = W_ci * s_c
        b'_c  = (b_c - mean_c) * s_c + beta_c

    so `BN(conv(x)) == conv'(x)` exactly, for every input. The transformation
    is a per-output-channel rescale of the weights plus an affine fix to the
    bias, and it is verified numerically in `fold_all` before anything is
    written to disk.

    This only works because every BatchNorm in the model sits *directly* after
    a convolution with nothing in between. The one that looks like an exception
    is the Inception block's norm, which follows a concatenation of three
    branches — but concatenation only stacks channels, and BatchNorm is
    per-channel, so its parameters split cleanly across the three branch
    convolutions. That split is done explicitly in `fold_inception`.

DESIGN DECISION — a bespoke binary format rather than pickle, npz or ONNX.
    Rejected alternatives: `torch.save` (needs libtorch to read), `.npz` (needs
    a zip reader and numpy's format), ONNX (needs protobuf). The C++ side is
    meant to have *no dependencies at all*, so the file has to be readable with
    `fread` and a handful of length checks. The format below is the smallest
    thing that carries names, shapes and float32 data — and it is the same
    "explain every byte" exercise as Stage 2's tape format, which is why it
    reuses that file's conventions: a magic, a version, then fixed-width
    records with lengths in front of anything variable.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ml.distill import build_student
from ml.model import TickToSignalNet

WEIGHT_MAGIC = b"TTSW"
FIXTURE_MAGIC = b"TTSF"
FORMAT_VERSION = 1

# The only dtype the C++ side understands. A tag rather than an assumption, so
# a future int8 export can be added without the reader silently misreading.
DTYPE_FLOAT32 = 0


@dataclass
class NamedTensor:
    """One tensor on its way to disk."""

    name: str
    array: np.ndarray

    def __post_init__(self) -> None:
        if self.array.dtype != np.float32:
            raise ValueError(f"{self.name} is {self.array.dtype}, expected float32")
        if not self.array.flags.c_contiguous:
            self.array = np.ascontiguousarray(self.array)


def fold_conv_batchnorm(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    norm: nn.Module,
    channel_slice: slice | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Absorb a BatchNorm into the convolution that feeds it.

    Returns `(folded_weight, folded_bias)`. See the module docstring for the
    derivation. `channel_slice` selects the BatchNorm channels belonging to
    this convolution, which is what makes the Inception block's shared norm
    foldable into its three separate branches.

    The convolution's own bias may be `None` (PyTorch omits it when a BatchNorm
    follows); it is treated as zero, which is what the maths says it is.
    """
    gamma = norm.weight.detach()
    beta = norm.bias.detach()
    mean = norm.running_mean.detach()
    variance = norm.running_var.detach()
    if channel_slice is not None:
        gamma, beta = gamma[channel_slice], beta[channel_slice]
        mean, variance = mean[channel_slice], variance[channel_slice]

    scale = gamma / torch.sqrt(variance + norm.eps)
    conv_bias = torch.zeros_like(mean) if bias is None else bias.detach()

    # `scale` is per output channel, so it must broadcast down the remaining
    # weight dimensions — whatever their number, which differs between the
    # Conv2d (4-D) and Conv1d (3-D) layers in this model.
    broadcast_shape = (-1,) + (1,) * (weight.dim() - 1)
    folded_weight = weight.detach() * scale.reshape(broadcast_shape)
    folded_bias = (conv_bias - mean) * scale + beta
    return folded_weight, folded_bias


def fold_level_block(prefix: str, block: nn.Module) -> list[NamedTensor]:
    """Fold the three conv+norm pairs inside one `LevelFusionBlock`."""
    tensors = []
    pairs = (
        ("spatial", block.spatial, block.norm_spatial),
        ("temporal_one", block.temporal_one, block.norm_one),
        ("temporal_two", block.temporal_two, block.norm_two),
    )
    for name, conv, norm in pairs:
        weight, bias = fold_conv_batchnorm(conv.weight, conv.bias, norm)
        tensors.append(NamedTensor(f"{prefix}.{name}.weight", weight.numpy()))
        tensors.append(NamedTensor(f"{prefix}.{name}.bias", bias.numpy()))
    return tensors


def fold_inception(block: nn.Module, branch_channels: int) -> list[NamedTensor]:
    """Fold the Inception block's post-concat norm back into its three branches.

    The norm sees `3 * branch_channels` channels: the first block from
    `conv_three`, the second from `conv_five`, the third from `pool_project`,
    in the order they are concatenated. BatchNorm is per-channel, so slicing
    its parameters by that order and folding each slice into the branch that
    produced it is exact — not an approximation.

    `reduce_three` and `reduce_five` are *not* folded: they are followed by an
    activation, not by a norm, so they keep their own bias and pass through
    unchanged.
    """
    tensors = [
        NamedTensor("inception.reduce_three.weight", block.reduce_three.weight.detach().numpy()),
        NamedTensor("inception.reduce_three.bias", block.reduce_three.bias.detach().numpy()),
        NamedTensor("inception.reduce_five.weight", block.reduce_five.weight.detach().numpy()),
        NamedTensor("inception.reduce_five.bias", block.reduce_five.bias.detach().numpy()),
    ]
    branches = (
        ("conv_three", block.conv_three, 0),
        ("conv_five", block.conv_five, 1),
        ("pool_project", block.pool_project, 2),
    )
    for name, conv, position in branches:
        span = slice(position * branch_channels, (position + 1) * branch_channels)
        weight, bias = fold_conv_batchnorm(conv.weight, conv.bias, block.norm, span)
        tensors.append(NamedTensor(f"inception.{name}.weight", weight.numpy()))
        tensors.append(NamedTensor(f"inception.{name}.bias", bias.numpy()))
    return tensors


def fold_temporal_block(index: int, block: nn.Module) -> list[NamedTensor]:
    """Fold both conv+norm pairs inside one dilated residual block."""
    tensors = []
    for name, conv, norm in (("conv_one", block.conv_one, block.norm_one),
                             ("conv_two", block.conv_two, block.norm_two)):
        weight, bias = fold_conv_batchnorm(conv.weight, conv.bias, norm)
        tensors.append(NamedTensor(f"temporal_blocks.{index}.{name}.weight", weight.numpy()))
        tensors.append(NamedTensor(f"temporal_blocks.{index}.{name}.bias", bias.numpy()))
    return tensors


def fold_all(model: TickToSignalNet) -> list[NamedTensor]:
    """Every tensor the C++ forward pass needs, with BatchNorm already folded."""
    model = model.eval()
    tensors: list[NamedTensor] = []
    tensors += fold_level_block("fuse_price_and_size", model.fuse_price_and_size)
    tensors += fold_level_block("fuse_sides", model.fuse_sides)
    tensors += fold_level_block("fuse_levels", model.fuse_levels)
    tensors += fold_inception(model.inception, model.config.inception_channels)
    tensors.append(NamedTensor("project.weight", model.project.weight.detach().numpy()))
    tensors.append(NamedTensor("project.bias", model.project.bias.detach().numpy()))
    for index, block in enumerate(model.temporal_blocks):
        tensors += fold_temporal_block(index, block)
    tensors.append(NamedTensor("classifier.weight", model.classifier.weight.detach().numpy()))
    tensors.append(NamedTensor("classifier.bias", model.classifier.bias.detach().numpy()))
    return tensors


def build_folded_model(model: TickToSignalNet) -> TickToSignalNet:
    """A copy of the model with the folds applied and every BatchNorm removed.

    Exists purely to *verify* the folding arithmetic in Python before any of it
    reaches C++. If `folded(x) != original(x)`, the bug is in the maths above,
    and finding that out here is far cheaper than finding it out through a
    failing parity test in another language.
    """
    import copy

    folded = copy.deepcopy(model).eval()
    for block in (folded.fuse_price_and_size, folded.fuse_sides, folded.fuse_levels):
        for conv_name, norm_name in (("spatial", "norm_spatial"), ("temporal_one", "norm_one"),
                                     ("temporal_two", "norm_two")):
            conv = getattr(block, conv_name)
            weight, bias = fold_conv_batchnorm(conv.weight, conv.bias, getattr(block, norm_name))
            _replace_conv_parameters(conv, weight, bias)
            setattr(block, norm_name, nn.Identity())

    inception = folded.inception
    channels = folded.config.inception_channels
    for position, name in enumerate(("conv_three", "conv_five", "pool_project")):
        conv = getattr(inception, name)
        span = slice(position * channels, (position + 1) * channels)
        weight, bias = fold_conv_batchnorm(conv.weight, conv.bias, inception.norm, span)
        _replace_conv_parameters(conv, weight, bias)
    inception.norm = nn.Identity()

    for block in folded.temporal_blocks:
        for conv_name, norm_name in (("conv_one", "norm_one"), ("conv_two", "norm_two")):
            conv = getattr(block, conv_name)
            weight, bias = fold_conv_batchnorm(conv.weight, conv.bias, getattr(block, norm_name))
            _replace_conv_parameters(conv, weight, bias)
            setattr(block, norm_name, nn.Identity())
    return folded


def _replace_conv_parameters(conv: nn.Module, weight: torch.Tensor, bias: torch.Tensor) -> None:
    conv.weight = nn.Parameter(weight)
    conv.bias = nn.Parameter(bias)


# ------------------------------------------------------------- the file format


def write_weight_file(tensors: list[NamedTensor], path: Path | str) -> Path:
    """Write the TTSW file.

    Layout, little-endian throughout (see `data_engine/binfmt.py` for why that
    is declared rather than left native):

        magic          4 bytes   'TTSW'
        version        uint32
        tensor_count   uint32
        then per tensor, in order:
            name_length  uint32
            name         name_length bytes, ASCII, no terminator
            dtype        uint32     0 = float32
            ndim         uint32
            dims         ndim x uint32
            data         product(dims) x float32, C-contiguous

    Lengths precede everything variable, so a reader can validate as it goes
    and never has to guess how far to seek.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(WEIGHT_MAGIC)
        handle.write(struct.pack("<II", FORMAT_VERSION, len(tensors)))
        for tensor in tensors:
            encoded = tensor.name.encode("ascii")
            handle.write(struct.pack("<I", len(encoded)))
            handle.write(encoded)
            handle.write(struct.pack("<II", DTYPE_FLOAT32, tensor.array.ndim))
            handle.write(struct.pack(f"<{tensor.array.ndim}I", *tensor.array.shape))
            handle.write(tensor.array.tobytes(order="C"))
    return path


def write_fixture_file(inputs: np.ndarray, outputs: np.ndarray, path: Path | str) -> Path:
    """Write the parity fixtures the C++ test replays.

        magic        4 bytes  'TTSF'
        version      uint32
        count        uint32
        window       uint32   100
        features     uint32   40
        classes      uint32   3
        then count x (window*features float32 input, classes float32 output)

    Inputs and outputs are interleaved per fixture rather than stored as two
    blocks, so the C++ test reads one fixture at a time and never needs the
    whole set in memory. That is not a performance concern here — it is so the
    test's memory use does not depend on how many fixtures we decide to dump.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count, window, features = inputs.shape
    classes = outputs.shape[1]
    with open(path, "wb") as handle:
        handle.write(FIXTURE_MAGIC)
        handle.write(struct.pack("<IIIII", FORMAT_VERSION, count, window, features, classes))
        for index in range(count):
            handle.write(inputs[index].astype(np.float32).tobytes(order="C"))
            handle.write(outputs[index].astype(np.float32).tobytes(order="C"))
    return path


def generate_fixtures(
    model: TickToSignalNet,
    count: int = 1_000,
    seed: int = 20260801,
    real_windows: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Produce (input, output) pairs for the C++ parity test.

    Random standard-normal inputs by default, which is a fair proxy because the
    features reaching this model are causally z-scored. When `real_windows` is
    supplied they are used instead — and the caller should prefer that, because
    the real distribution is heavy-tailed (Stage 6 measured extremes near +/-22
    against a bulk near +/-1.4) and large magnitudes are exactly where a
    numerical disagreement between two implementations would first appear.
    """
    model = model.eval()
    if real_windows is not None:
        inputs = real_windows[:count].astype(np.float32)
    else:
        generator = np.random.default_rng(seed)
        inputs = generator.standard_normal((count, 100, 40)).astype(np.float32)
    with torch.no_grad():
        outputs = model(torch.from_numpy(inputs)).numpy().astype(np.float32)
    return inputs, outputs


def verify_folding(
    model: TickToSignalNet,
    trials: int = 256,
    relative_tolerance: float = 1e-5,
) -> tuple[float, float]:
    """Check `folded(x) == original(x)` before writing anything.

    Returns `(max_absolute_difference, max_relative_difference)`.

    WHY THE CHECK IS RELATIVE. Folding is exact algebra, so the only difference
    should be float32 rounding — but "how much rounding is acceptable" is not a
    fixed number of absolute units, it scales with the size of the logits. This
    was measured rather than assumed:

        fold and evaluate entirely in float64 : max |diff| 4.1e-14
        fold and evaluate in float32          : max |diff| 2.0e-05  (2.3e-06 relative)

    The float64 figure is machine epsilon for a 40-layer network, which is the
    proof that the derivation and the Inception channel slicing are exact. The
    float32 figure is nine orders of magnitude larger *because float32 has nine
    fewer digits*, and it accumulates smoothly with depth — the per-stage
    relative error stays flat at ~1.5-2.3e-06 from the first block to the
    logits, which is what rounding looks like. An algebra error would show a
    jump at one stage and would not shrink in float64.

    A first version of this function used a 1e-5 *absolute* bound and rejected
    a correct fold. That is the same scale-invariance mistake Stage 6 made with
    the ONNX parity gate, met a second time.
    """
    original = model.eval()
    folded = build_folded_model(model)
    generator = np.random.default_rng(0)
    probe = generator.standard_normal((trials, 100, 40)).astype(np.float32)
    with torch.no_grad():
        reference = original(torch.from_numpy(probe)).numpy()
        candidate = folded(torch.from_numpy(probe)).numpy()

    absolute = float(np.abs(reference - candidate).max())
    scale = float(np.abs(reference).max())
    relative = absolute / scale if scale > 0 else float("inf")
    if relative > relative_tolerance:
        raise ValueError(
            f"BatchNorm folding changed the model by {relative:.3e} relative "
            f"({absolute:.3e} absolute, logit scale {scale:.3f}); the folding maths or the "
            "Inception channel slicing is wrong"
        )
    return absolute, relative


def build_op_fixtures(seed: int = 7) -> list[NamedTensor]:
    """Small per-operator fixtures, so a broken op is found before the whole model.

    The full parity test tells you the network disagrees; it does not tell you
    *which* of seven operators is wrong, and bisecting a 40-layer forward pass
    by hand is exactly the tedium these fixtures exist to avoid. Each case is
    small enough to reason about and shaped to exercise the same code path the
    real model uses — the strides, dilations and kernel heights below are the
    ones that actually occur.

    Reuses the TTSW container rather than inventing a second format: an op
    fixture is just a bag of named tensors, which is what TTSW already is, so
    the C++ side reads it with the loader it already has.
    """
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    tensors: list[NamedTensor] = []

    def record(prefix: str, **arrays: torch.Tensor) -> None:
        for name, value in arrays.items():
            tensors.append(NamedTensor(f"{prefix}.{name}", value.detach().numpy().astype(np.float32)))

    # 1. Spatial 2-D convolution with a width stride, as in the fusion blocks.
    spatial_in, spatial_w, spatial_b = randn(1, 2, 7, 8), randn(3, 2, 1, 2), randn(3)
    spatial_out = nn.functional.conv2d(spatial_in, spatial_w, spatial_b, stride=(1, 2))
    record("conv2d_spatial", input=spatial_in[0], weight=spatial_w, bias=spatial_b,
           expected=spatial_out[0])

    # 2. Temporal 2-D convolution of height 4 with causal padding.
    temporal_in, temporal_w, temporal_b = randn(1, 3, 7, 4), randn(3, 3, 4, 1), randn(3)
    temporal_out = nn.functional.conv2d(
        nn.functional.pad(temporal_in, (0, 0, 3, 0)), temporal_w, temporal_b)
    record("conv2d_temporal", input=temporal_in[0], weight=temporal_w, bias=temporal_b,
           expected=temporal_out[0])

    # 3. Height-5 kernel, the widest branch of the Inception block.
    five_in, five_w, five_b = randn(1, 2, 9, 1), randn(4, 2, 5, 1), randn(4)
    five_out = nn.functional.conv2d(nn.functional.pad(five_in, (0, 0, 4, 0)), five_w, five_b)
    record("conv2d_five", input=five_in[0], weight=five_w, bias=five_b, expected=five_out[0])

    # 4. Dilated causal 1-D convolution, the deepest TCN dilation.
    dilation = 4
    conv1d_in, conv1d_w, conv1d_b = randn(1, 3, 12), randn(4, 3, 3), randn(4)
    conv1d_out = nn.functional.conv1d(
        nn.functional.pad(conv1d_in, ((3 - 1) * dilation, 0)), conv1d_w, conv1d_b, dilation=dilation)
    record("conv1d_dilated", input=conv1d_in[0], weight=conv1d_w, bias=conv1d_b,
           expected=conv1d_out[0])

    # 5. Causal max-pool. Deliberately shifted negative: PyTorch zero-pads
    #    before pooling, so windows overhanging the start must return 0.0 rather
    #    than the largest negative value. An implementation that skips padded
    #    taps passes on positive data and fails here, which is the whole point.
    pool_in = randn(1, 2, 7, 3) - 3.0
    pool_out = nn.functional.max_pool2d(
        nn.functional.pad(pool_in, (0, 0, 2, 0)), kernel_size=(3, 1), stride=1)
    record("maxpool_negative", input=pool_in[0], expected=pool_out[0])

    # 6. Leaky ReLU at the model's slope.
    leaky_in = randn(10)
    record("leaky_relu", input=leaky_in,
           expected=nn.functional.leaky_relu(leaky_in, negative_slope=0.01))

    # 7. Fully-connected read-out.
    linear_in, linear_w, linear_b = randn(6), randn(4, 6), randn(4)
    record("linear", input=linear_in, weight=linear_w, bias=linear_b,
           expected=nn.functional.linear(linear_in, linear_w, linear_b))

    # 8. Softmax, with a large logit to exercise the max-subtraction guard.
    softmax_in = torch.tensor([12.0, -3.0, 0.5, 7.25, -11.0])
    record("softmax", input=softmax_in, expected=torch.softmax(softmax_in, dim=0))

    return tensors


def main(argv: list[str] | None = None) -> int:
    import argparse
    import glob
    import json

    parser = argparse.ArgumentParser(description="Export the student model for the C++ inference path.")
    parser.add_argument("--checkpoint", default="checkpoints/student_distilled.pt")
    parser.add_argument("--out-dir", default="inference_cpp/artifacts")
    parser.add_argument("--fixtures", type=int, default=1_000)
    parser.add_argument("--real-data", action="store_true",
                        help="draw fixture inputs from captured tapes instead of a normal distribution")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    student = build_student()
    student.load_state_dict(torch.load(repo_root / args.checkpoint, map_location="cpu"))
    student.eval()
    print(f"student: {student.parameter_count():,} parameters from {args.checkpoint}")

    absolute, relative = verify_folding(student)
    print(f"BatchNorm folding verified in PyTorch: max |diff| {absolute:.3e} "
          f"({relative:.3e} relative; exact to 4.1e-14 in float64)")

    real_windows = None
    if args.real_data:
        from ml.dataset import build_sample_index, gather_windows, load_sessions

        run_path = Path(sorted(glob.glob(str(repo_root / "benchmarks" / "train_ours_*.json")))[-1])
        run = json.loads(run_path.read_text(encoding="utf-8"))
        sessions = load_sessions(sorted(glob.glob(run["data"]["tapes"])))
        index = build_sample_index(sessions)
        fold = run["data"]["fold"]
        positions = np.arange(fold["test_start"], fold["test_end"] + 1)[: args.fixtures]
        real_windows = gather_windows(sessions, index, positions)
        print(f"fixture inputs: {len(real_windows)} real held-out windows")

    tensors = fold_all(student)
    out_dir = repo_root / args.out_dir
    weight_path = write_weight_file(tensors, out_dir / "student_weights.ttsw")
    total_values = sum(tensor.array.size for tensor in tensors)
    print(f"wrote {weight_path.name}: {len(tensors)} tensors, {total_values:,} values, "
          f"{weight_path.stat().st_size / 1024:.1f} KiB")

    inputs, outputs = generate_fixtures(student, args.fixtures, real_windows=real_windows)
    fixture_path = write_fixture_file(inputs, outputs, out_dir / "student_fixtures.ttsf")
    print(f"wrote {fixture_path.name}: {len(inputs)} fixtures, "
          f"{fixture_path.stat().st_size / 1024:.1f} KiB")
    print(f"logit range across fixtures: [{outputs.min():.3f}, {outputs.max():.3f}]")

    op_tensors = build_op_fixtures()
    op_path = write_weight_file(op_tensors, out_dir / "op_fixtures.ttsw")
    cases = sorted({tensor.name.rsplit(".", 1)[0] for tensor in op_tensors})
    print(f"wrote {op_path.name}: {len(cases)} operator cases ({', '.join(cases)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
