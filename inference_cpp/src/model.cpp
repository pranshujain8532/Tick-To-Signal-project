#include "model.hpp"

#include <cstring>

#include "ops.hpp"

namespace tts {

using namespace arch;

StudentModel::LayerWeights StudentModel::resolve(const std::string& name,
                                                 const std::vector<int>& shape) {
    LayerWeights resolved;
    resolved.weight = weights_.get(name + ".weight", shape).data();
    resolved.bias = weights_.get(name + ".bias", {shape[0]}).data();
    return resolved;
}

StudentModel::StudentModel(const std::string& weight_path) : weights_(weight_path) {
    // Every buffer the forward pass will ever need, sized once. Nothing below
    // this constructor allocates. Total is roughly 240 KB, dominated by the two
    // [8, 100, 20] buffers at 64 KB each.
    input_.reshape({1, kWindow, kFeatures});                      //  16,000 B
    pairs_a_.reshape({kConvChannels, kWindow, kWidthAfterPairs});  //  64,000 B
    pairs_b_.reshape({kConvChannels, kWindow, kWidthAfterPairs});  //  64,000 B
    sides_a_.reshape({kConvChannels, kWindow, kWidthAfterSides});  //  32,000 B
    sides_b_.reshape({kConvChannels, kWindow, kWidthAfterSides});  //  32,000 B
    levels_a_.reshape({kConvChannels, kWindow, kWidthAfterLevels});//   3,200 B
    levels_b_.reshape({kConvChannels, kWindow, kWidthAfterLevels});//   3,200 B
    reduce_.reshape({kBranchChannels, kWindow, kWidthAfterLevels});//   6,400 B
    pooled_.reshape({kConvChannels, kWindow, kWidthAfterLevels});  //   3,200 B
    concat_.reshape({kConcatChannels, kWindow, kWidthAfterLevels});//  19,200 B
    projected_.reshape({kTcnChannels, kWindow});                   //  12,800 B
    tcn_a_.reshape({kTcnChannels, kWindow});                       //  12,800 B
    tcn_b_.reshape({kTcnChannels, kWindow});                       //  12,800 B

    // Resolve every weight the forward pass will use, with its expected shape.
    // Two jobs at once: a mismatch between this file's architecture constants
    // and the exported model fails loudly here rather than producing
    // plausible-looking logits from the wrong tensor, and the hot path is left
    // with nothing to look up.
    //
    // IncrementalModel resolves the same names independently rather than
    // sharing a table with this class. The duplication is deliberate: the shape
    // arguments are the assertion, so two classes asserting the same shapes is
    // two chances to catch a divergence, and a shared table would be a third
    // place for the architecture to be described.
    const std::vector<int> fusion_temporal{kConvChannels, kConvChannels, 4, 1};

    pairs_w_.spatial = resolve("fuse_price_and_size.spatial", {kConvChannels, 1, 1, 2});
    pairs_w_.temporal_one = resolve("fuse_price_and_size.temporal_one", fusion_temporal);
    pairs_w_.temporal_two = resolve("fuse_price_and_size.temporal_two", fusion_temporal);

    sides_w_.spatial = resolve("fuse_sides.spatial", {kConvChannels, kConvChannels, 1, 2});
    sides_w_.temporal_one = resolve("fuse_sides.temporal_one", fusion_temporal);
    sides_w_.temporal_two = resolve("fuse_sides.temporal_two", fusion_temporal);

    levels_w_.spatial =
        resolve("fuse_levels.spatial", {kConvChannels, kConvChannels, 1, kDepthLevels});
    levels_w_.temporal_one = resolve("fuse_levels.temporal_one", fusion_temporal);
    levels_w_.temporal_two = resolve("fuse_levels.temporal_two", fusion_temporal);

    reduce_three_w_ = resolve("inception.reduce_three", {kBranchChannels, kConvChannels, 1, 1});
    conv_three_w_ = resolve("inception.conv_three", {kBranchChannels, kBranchChannels, 3, 1});
    reduce_five_w_ = resolve("inception.reduce_five", {kBranchChannels, kConvChannels, 1, 1});
    conv_five_w_ = resolve("inception.conv_five", {kBranchChannels, kBranchChannels, 5, 1});
    pool_project_w_ = resolve("inception.pool_project", {kBranchChannels, kConvChannels, 1, 1});

    project_w_ = resolve("project", {kTcnChannels, kConcatChannels, 1});
    for (int index = 0; index < kTcnBlocks; ++index) {
        const std::string prefix = "temporal_blocks." + std::to_string(index);
        const std::vector<int> shape{kTcnChannels, kTcnChannels, kTcnKernel};
        block_one_w_[index] = resolve(prefix + ".conv_one", shape);
        block_two_w_[index] = resolve(prefix + ".conv_two", shape);
    }
    classifier_w_ = resolve("classifier", {kClasses, kTcnChannels});
}

std::size_t StudentModel::activation_bytes() const {
    const Tensor* buffers[] = {&input_, &pairs_a_, &pairs_b_, &sides_a_, &sides_b_,
                               &levels_a_, &levels_b_, &reduce_, &pooled_, &concat_,
                               &projected_, &tcn_a_, &tcn_b_};
    std::size_t total = 0;
    for (const Tensor* buffer : buffers) total += buffer->size() * sizeof(float);
    return total;
}

// One LevelFusionBlock: a spatial convolution that narrows the feature axis,
// then two causal temporal convolutions, each followed by leaky ReLU. The
// BatchNorms that sit between conv and activation in PyTorch are gone - folded
// into these very weights by ml/export_weights.py.
void StudentModel::run_level_block(const FusionWeights& layer, const float* input, int in_channels,
                                   int in_width, int spatial_kernel, int spatial_stride,
                                   int out_width, Tensor& scratch_a, Tensor& scratch_b) {
    conv2d_causal_time(input, in_channels, kWindow, in_width, layer.spatial.weight,
                       layer.spatial.bias, kConvChannels, 1, spatial_kernel, spatial_stride,
                       scratch_a.data());
    leaky_relu(scratch_a.data(), static_cast<std::size_t>(kConvChannels) * kWindow * out_width);

    // Two temporal convolutions of height 4, ping-ponging between the two
    // scratch buffers so no third allocation is needed.
    const LayerWeights* stages[2] = {&layer.temporal_one, &layer.temporal_two};
    Tensor* source = &scratch_a;
    Tensor* destination = &scratch_b;
    for (int stage = 0; stage < 2; ++stage) {
        conv2d_causal_time(source->data(), kConvChannels, kWindow, out_width,
                           stages[stage]->weight, stages[stage]->bias, kConvChannels, 4, 1, 1,
                           destination->data());
        leaky_relu(destination->data(), static_cast<std::size_t>(kConvChannels) * kWindow * out_width);
        Tensor* swap = source;
        source = destination;
        destination = swap;
    }
    // After two swaps the result is back in `scratch_a`, which is what the
    // caller reads. Stated rather than left to be traced.
}

// The Inception block. Three branches are written directly into their slice of
// the concatenation buffer, which is why no separate concat step appears: the
// channel-major layout means branch outputs are contiguous runs of `concat_`.
void StudentModel::run_inception(const float* input) {
    const int plane = kWindow * kWidthAfterLevels;

    // Branch 1: 1x1 reduce -> leaky -> causal conv of height 3.
    conv2d_causal_time(input, kConvChannels, kWindow, kWidthAfterLevels, reduce_three_w_.weight,
                       reduce_three_w_.bias, kBranchChannels, 1, 1, 1, reduce_.data());
    leaky_relu(reduce_.data(), static_cast<std::size_t>(kBranchChannels) * plane);
    conv2d_causal_time(reduce_.data(), kBranchChannels, kWindow, kWidthAfterLevels,
                       conv_three_w_.weight, conv_three_w_.bias, kBranchChannels, 3, 1, 1,
                       concat_.data() + 0 * kBranchChannels * plane);

    // Branch 2: the same with a height-5 kernel.
    conv2d_causal_time(input, kConvChannels, kWindow, kWidthAfterLevels, reduce_five_w_.weight,
                       reduce_five_w_.bias, kBranchChannels, 1, 1, 1, reduce_.data());
    leaky_relu(reduce_.data(), static_cast<std::size_t>(kBranchChannels) * plane);
    conv2d_causal_time(reduce_.data(), kBranchChannels, kWindow, kWidthAfterLevels,
                       conv_five_w_.weight, conv_five_w_.bias, kBranchChannels, 5, 1, 1,
                       concat_.data() + 1 * kBranchChannels * plane);

    // Branch 3: causal max-pool over the raw input, then a 1x1 projection.
    // Note the pool runs on `input`, not on a reduced copy - matching
    // InceptionLite.forward in ml/model.py.
    maxpool2d_causal_time(input, kConvChannels, kWindow, kWidthAfterLevels, 3, pooled_.data());
    conv2d_causal_time(pooled_.data(), kConvChannels, kWindow, kWidthAfterLevels,
                       pool_project_w_.weight, pool_project_w_.bias, kBranchChannels, 1, 1, 1,
                       concat_.data() + 2 * kBranchChannels * plane);

    // The post-concat BatchNorm folded into the three branch convolutions
    // above, so only the activation is left to apply.
    leaky_relu(concat_.data(), static_cast<std::size_t>(kConcatChannels) * plane);
}

// One dilated residual block: two causal convolutions with leaky ReLU, then the
// input added back and a final activation.
void StudentModel::run_temporal_block(int index) {
    const int dilation = kDilations[index];
    const std::size_t count = static_cast<std::size_t>(kTcnChannels) * kWindow;

    conv1d_causal_dilated(projected_.data(), kTcnChannels, kWindow, block_one_w_[index].weight,
                          block_one_w_[index].bias, kTcnChannels, kTcnKernel, dilation,
                          tcn_a_.data());
    leaky_relu(tcn_a_.data(), count);

    conv1d_causal_dilated(tcn_a_.data(), kTcnChannels, kWindow, block_two_w_[index].weight,
                          block_two_w_[index].bias, kTcnChannels, kTcnKernel, dilation,
                          tcn_b_.data());
    leaky_relu(tcn_b_.data(), count);

    // Residual: out = leaky(conv_two_out + block_input). The block input is
    // still in `projected_`, so the sum is written there and becomes the next
    // block's input - which is exactly the chain PyTorch builds.
    add_into(tcn_b_.data(), projected_.data(), count);
    leaky_relu(tcn_b_.data(), count);
    std::memcpy(projected_.data(), tcn_b_.data(), count * sizeof(float));
}

void StudentModel::forward(const float* window, float* logits) {
    // Copy the caller's window into an owned, aligned buffer. The copy is 16 KB
    // and buys two things: the caller's pointer need not be aligned, and the
    // model never reads memory whose lifetime it does not control.
    std::memcpy(input_.data(), window, static_cast<std::size_t>(kWindow) * kFeatures * sizeof(float));

    // 40 -> 20: fuse each (price, size) pair.
    run_level_block(pairs_w_, input_.data(), 1, kFeatures, 2, 2, kWidthAfterPairs, pairs_a_,
                    pairs_b_);
    // 20 -> 10: fuse the bid side against the ask side, giving a per-level imbalance.
    run_level_block(sides_w_, pairs_a_.data(), kConvChannels, kWidthAfterPairs, 2, 2,
                    kWidthAfterSides, sides_a_, sides_b_);
    // 10 -> 1: fuse the ten levels into whole-book shape.
    run_level_block(levels_w_, sides_a_.data(), kConvChannels, kWidthAfterSides, kDepthLevels, 1,
                    kWidthAfterLevels, levels_a_, levels_b_);

    run_inception(levels_a_.data());

    // The feature axis is now width 1, so [48, 100, 1] is already the [48, 100]
    // layout the 1-D convolutions want - the squeeze is free.
    conv1d_causal_dilated(concat_.data(), kConcatChannels, kWindow, project_w_.weight,
                          project_w_.bias, kTcnChannels, 1, 1, projected_.data());

    for (int index = 0; index < kTcnBlocks; ++index) run_temporal_block(index);

    // Read out the final timestep only, which is what makes the model
    // incrementally updatable - see the design note in ml/model.py.
    float final_step[kTcnChannels];
    for (int c = 0; c < kTcnChannels; ++c) {
        final_step[c] = projected_.data()[c * kWindow + (kWindow - 1)];
    }
    linear(final_step, classifier_w_.weight, classifier_w_.bias, kTcnChannels, kClasses, logits);
}

void StudentModel::forward_probabilities(const float* window, float* probabilities) {
    float logits[kClasses];
    forward(window, logits);
    softmax(logits, kClasses, probabilities);
}

}  // namespace tts
