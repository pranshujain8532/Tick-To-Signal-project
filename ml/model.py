"""The deep LOB model: a CNN feature extractor with a dilated-convolution head.

WHAT
    Maps one `[T=100, F=40]` window of normalised book features to three
    logits (down / flat / up). Four stages, in order:

      1. A DeepLOB-style convolutional front end that collapses the 40-column
         feature axis in three deliberate steps — fuse each (price, size)
         pair, fuse the two sides of each level, then fuse the ten levels.
      2. An Inception-lite block mixing three temporal kernel sizes, so the
         network can attend to fast and slow structure without us having to
         pick one kernel width up front.
      3. A TCN head: four dilated *causal* convolutions (dilations 1, 2, 4, 8)
         with residual connections.
      4. A linear read-out from the final timestep.

    319,715 parameters, deliberately small — small enough that Stage 6 can
    distil it and Stage 7 can hand-roll its forward pass in C++.
    `tests/test_model.py` fails the build if the count leaves 250K-400K.

WHY
    Written from first principles rather than assembled from a library of
    blocks, because the goal is to be able to explain every tensor shape,
    every receptive field, and every parameter under questioning. A model you
    cannot draw on a whiteboard is not a portfolio piece.

DESIGN DECISION — a TCN head instead of an LSTM.
    Rejected alternative: an LSTM or GRU over the window, which is the
    canonical choice in the DeepLOB literature and would work fine on
    accuracy. Rejected for three reasons, in the order they matter here:

      1. **Inference latency is the deliverable.** A recurrent step is
         sequentially dependent: T timesteps means T dependent matrix
         multiplies that cannot be overlapped. A dilated convolution stack is
         a fixed sequence of GEMMs with a statically known access pattern,
         which is what makes the Stage 7 C++ path a tractable weekend rather
         than a research project.
      2. **Incremental inference.** Because every layer here is causal, a new
         tick only needs the *new* column: state can be kept in a ring buffer
         and advanced by one, instead of recomputing a 100x40 forward pass per
         tick. That is not an optimisation detail, it is the difference
         between microseconds and milliseconds, and a recurrent net with a
         non-causal CNN in front of it cannot do it.
      3. **Determinism and parity.** Reproducing convolution arithmetic
         bit-for-bit between PyTorch and hand-written C++ is tractable;
         reproducing cuDNN's fused recurrent kernels is not, and the
         C++/PyTorch parity test in Stage 7 is mandatory.

    The cost we accept is a bounded receptive field — see `receptive_field`,
    which is 83 timesteps against a 100-step window. If Stage 5's measured
    signal half-life turns out to exceed that, this decision gets revisited in
    writing rather than quietly.

DESIGN DECISION — no transformer.
    Rejected alternative: self-attention over the 100 timesteps. At this
    sequence length and parameter budget attention buys nothing and costs
    exactly the thing the project is about. Attention is O(T^2) in both time
    and memory where the dilated stack is O(T); it has no incremental form
    (every new tick re-attends over the whole window, so the ring-buffer trick
    is gone); and a 100-step window with strong local structure is precisely
    the regime where convolutions already capture the dependencies attention
    would learn. We care about the p99, and attention's p99 is worse for no
    accuracy we could point to.

DESIGN DECISION — causal padding everywhere, not just in the TCN.
    Rejected alternative: DeepLOB's `same` padding in the convolutional front
    end. To be precise about *why*, because it is easy to get this argument
    wrong in an interview: `same` padding would **not** leak future
    information — the window ends at time t and there is no future inside it
    to see. Causality is required for a different reason, incremental
    inference: an output that depends only on past inputs can be advanced one
    tick at a time from a ring buffer, which is the whole Stage 7 story.
    Leakage is prevented by how the window and label are built (Stage 3), not
    by the padding.

    One honest caveat on that claim. `BatchNorm` over `[B, C, T]` computes its
    statistics across batch *and time*, so during **training** the activation
    at position t is influenced by every position in the window, future ones
    included — the convolutions are causal but the normalisation is not. This
    costs nothing that matters: it is not look-ahead, because the window
    already stops at t and the label looks forward from t; and at inference
    the layer uses frozen running statistics, so the deployed model is
    strictly causal and is the thing Stage 7 reimplements. The distinction is
    exactly the sort of detail worth knowing before someone asks.

DESIGN DECISION — read out the last timestep, not a pooled average.
    Rejected alternative: mean-pooling across time before the classifier.
    Pooling mixes the whole window into the prediction, which sounds
    harmless but destroys the incremental property: the pooled value changes
    with every expiring timestep, so it cannot be updated in O(1). Reading the
    final position keeps the model's output a function of "state right now",
    which is also the semantically correct thing to predict from.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    """Shape and width of the network. A dataclass, not a config system."""

    window_length: int = 100
    depth_levels: int = 10
    features_per_level: int = 4
    conv_channels: int = 32
    inception_channels: int = 64
    tcn_channels: int = 96
    tcn_kernel: int = 3
    tcn_dilations: tuple[int, ...] = (1, 2, 4, 8)
    num_classes: int = 3
    dropout: float = 0.1

    @property
    def feature_count(self) -> int:
        return self.depth_levels * self.features_per_level


def _causal_pad_time(inputs: Tensor, padding: int) -> Tensor:
    """Left-pad the time axis so a convolution cannot reach forward.

    Works for both `[B, C, T, W]` and `[B, C, T]`, padding the dimension that
    is `T` in each. Padding on the left only is what makes the layer causal:
    output position t is computed from inputs at positions <= t.
    """
    if padding == 0:
        return inputs
    if inputs.dim() == 4:
        return nn.functional.pad(inputs, (0, 0, padding, 0))
    return nn.functional.pad(inputs, (padding, 0))


class LevelFusionBlock(nn.Module):
    """One stage of the front end: squeeze the feature axis, then look in time.

    Each block applies a *spatial* convolution that reduces the 40-column
    feature axis, then two causal *temporal* convolutions that build structure
    along time at the resulting width. Splitting the two directions is the
    point of the DeepLOB design: mixing levels and mixing time are different
    operations and are easier to reason about when they are separate layers.
    """

    def __init__(self, in_channels: int, out_channels: int, spatial_kernel: int, spatial_stride: int) -> None:
        super().__init__()
        self.spatial = nn.Conv2d(in_channels, out_channels, kernel_size=(1, spatial_kernel), stride=(1, spatial_stride))
        self.temporal_one = nn.Conv2d(out_channels, out_channels, kernel_size=(4, 1))
        self.temporal_two = nn.Conv2d(out_channels, out_channels, kernel_size=(4, 1))
        self.norm_spatial = nn.BatchNorm2d(out_channels)
        self.norm_one = nn.BatchNorm2d(out_channels)
        self.norm_two = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.activation(self.norm_spatial(self.spatial(inputs)))
        hidden = self.activation(self.norm_one(self.temporal_one(_causal_pad_time(hidden, 3))))
        hidden = self.activation(self.norm_two(self.temporal_two(_causal_pad_time(hidden, 3))))
        return hidden


class InceptionLite(nn.Module):
    """Three temporal views of the same tensor, concatenated.

    Branches of kernel 1, 3 and 5 over time, plus a max-pooled branch. The
    motivation is that we do not know the right temporal scale a priori:
    queue depletion at the touch happens over a handful of events, while a
    directional drift plays out over tens. Rather than tuning one kernel
    width, the block learns how much of each to use.

    Rejected alternative: a single wide kernel. Same receptive field, more
    parameters, and no ability to weight scales differently.
    """

    def __init__(self, in_channels: int, branch_channels: int) -> None:
        super().__init__()
        self.reduce_three = nn.Conv2d(in_channels, branch_channels, kernel_size=(1, 1))
        self.conv_three = nn.Conv2d(branch_channels, branch_channels, kernel_size=(3, 1))
        self.reduce_five = nn.Conv2d(in_channels, branch_channels, kernel_size=(1, 1))
        self.conv_five = nn.Conv2d(branch_channels, branch_channels, kernel_size=(5, 1))
        self.pool_project = nn.Conv2d(in_channels, branch_channels, kernel_size=(1, 1))
        self.norm = nn.BatchNorm2d(branch_channels * 3)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, inputs: Tensor) -> Tensor:
        three = self.conv_three(_causal_pad_time(self.activation(self.reduce_three(inputs)), 2))
        five = self.conv_five(_causal_pad_time(self.activation(self.reduce_five(inputs)), 4))
        pooled = nn.functional.max_pool2d(_causal_pad_time(inputs, 2), kernel_size=(3, 1), stride=1)
        pooled = self.pool_project(pooled)
        merged = torch.cat([three, five, pooled], dim=1)
        return self.activation(self.norm(merged))


class TemporalBlock(nn.Module):
    """One dilated causal residual block: two convolutions plus a skip.

    The residual connection is what makes stacking dilations safe. Without it,
    four layers of dilated convolution is a deep enough chain that gradients
    at the early layers are unreliable; with it, each block only has to learn
    a correction to what it was handed.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv_one = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv_two = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm_one = nn.BatchNorm1d(channels)
        self.norm_two = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.conv_one(_causal_pad_time(inputs, self.padding))
        hidden = self.dropout(self.activation(self.norm_one(hidden)))
        hidden = self.conv_two(_causal_pad_time(hidden, self.padding))
        hidden = self.dropout(self.activation(self.norm_two(hidden)))
        return self.activation(hidden + inputs)


class TickToSignalNet(nn.Module):
    """The full model: `[B, T, F]` in, `[B, 3]` logits out."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        channels = self.config.conv_channels

        # Three fusion stages, each halving or collapsing the feature axis.
        # With the layout [bid_price, bid_qty, ask_price, ask_qty] per level:
        #   40 -> 20  a stride-2 kernel of width 2 fuses each (price, size)
        #             pair, so one number now summarises "what is resting at
        #             this level on this side".
        #   20 -> 10  the same again fuses the bid summary with the ask
        #             summary, so one number summarises the *imbalance* at a
        #             level, which is the single most predictive cheap feature
        #             in microstructure.
        #   10 -> 1   a full-width kernel fuses the ten levels into a picture
        #             of the whole visible book shape.
        self.fuse_price_and_size = LevelFusionBlock(1, channels, spatial_kernel=2, spatial_stride=2)
        self.fuse_sides = LevelFusionBlock(channels, channels, spatial_kernel=2, spatial_stride=2)
        self.fuse_levels = LevelFusionBlock(channels, channels, spatial_kernel=self.config.depth_levels, spatial_stride=1)

        self.inception = InceptionLite(channels, self.config.inception_channels)
        inception_out = self.config.inception_channels * 3

        self.project = nn.Conv1d(inception_out, self.config.tcn_channels, kernel_size=1)
        self.temporal_blocks = nn.ModuleList(
            [
                TemporalBlock(self.config.tcn_channels, self.config.tcn_kernel, dilation, self.config.dropout)
                for dilation in self.config.tcn_dilations
            ]
        )
        self.classifier = nn.Linear(self.config.tcn_channels, self.config.num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.dim() != 3:
            raise ValueError(f"expected [batch, time, features], got shape {tuple(inputs.shape)}")
        hidden = inputs.unsqueeze(1)  # [B, 1, T, F] — one channel, time as height

        hidden = self.fuse_price_and_size(hidden)
        hidden = self.fuse_sides(hidden)
        hidden = self.fuse_levels(hidden)
        hidden = self.inception(hidden)

        hidden = hidden.squeeze(-1)  # the feature axis is fully collapsed: [B, C, T]
        hidden = self.project(hidden)
        for block in self.temporal_blocks:
            hidden = block(hidden)

        # Read out the final timestep only. See the module docstring: pooling
        # would destroy the O(1) incremental update Stage 7 depends on.
        return self.classifier(hidden[:, :, -1])

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def receptive_field(self) -> int:
        """How many timesteps back the output at position t actually depends on.

        Reported rather than assumed, because the whole "why not an LSTM"
        argument rests on this number being big enough. Every layer is causal,
        so the field is the sum of each layer's `(kernel - 1) * dilation` plus
        one for the current step.
        """
        span = 0
        # Three fusion blocks, each with two temporal convolutions of width 4.
        span += 3 * 2 * (4 - 1)
        # Inception: the widest branch is kernel 5.
        span += 5 - 1
        # The dilated stack: two convolutions per block.
        for dilation in self.config.tcn_dilations:
            span += 2 * (self.config.tcn_kernel - 1) * dilation
        return span + 1


def build_model(config: ModelConfig | None = None, seed: int | None = None) -> TickToSignalNet:
    """Construct the model, optionally with a fixed initialisation seed."""
    if seed is not None:
        torch.manual_seed(seed)
    return TickToSignalNet(config)
