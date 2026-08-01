// model.hpp - the distilled student's forward pass, hand-written.
//
// WHAT
//     `StudentModel::forward(const float* window, float* logits)` maps one
//     [100, 40] book window to three logits, using the weights loaded from a
//     TTSW file. It is the same function as ml/model.py's `TickToSignalNet`
//     with the student configuration, with BatchNorm already folded in.
//
// WHY
//     This is the artefact Stage 7 exists to produce: an inference path with no
//     framework, no allocator traffic, and a shape that is fixed at compile
//     time. Stage 7a proves it computes the right answer; Stage 7b makes it
//     fast. In that order, deliberately.
//
// DESIGN DECISION - every buffer AND every weight pointer is resolved in the
// constructor, none on the hot path.
//     `forward()` performs zero heap allocation. Rejected alternative: allocate
//     activations per call, or return a vector. Three reasons, in order of how
//     much they matter here:
//
//       1. Determinism. An allocator can take a lock, hit a slow path, or ask
//          the OS for pages. Those events are rare and large, which puts them
//          precisely in the p99.9 that Stage 6 measured and Stage 7b is meant
//          to improve. A forward pass that cannot allocate cannot be surprised
//          by the allocator.
//       2. Cache behaviour. Reusing the same buffers every call keeps the whole
//          working set - about 240 KB - resident and warm, instead of touching
//          fresh cold pages each time.
//       3. It is a real constraint that shapes the design. Because the buffers
//          are permanent, the shapes must be known at construction, which is
//          only true because the input shape is fixed at [1, 100, 40]. That is
//          the payoff for the Stage 6 decision not to export a dynamic batch
//          axis.
//
//     The cost is that a `StudentModel` is not thread-safe: two threads calling
//     `forward` on one instance would write over each other's activations. That
//     is the right trade for a batch-1 tick-to-signal path - construct one per
//     thread - and it is stated here rather than discovered.
//
//     The weight pointers are resolved in the constructor for the same reason,
//     and that was NOT true when this file was written. Stage 7a looked every
//     weight up by name inside `forward`, building a std::string per layer to do
//     it - which allocates, since the names are longer than the small-string
//     buffer - and then walking a std::map. The docstring said "allocates
//     nothing" and the code allocated about fifty times per call. Amortised over
//     100 timesteps it was invisible; on the incremental path it would have been
//     a large fraction of the whole forward pass. See
//     docs/INTERVIEW_NOTES_stage7b.md for what it measured.
//
// DESIGN DECISION - architecture constants, not a config file.
//     The dimensions below are the student configuration from ml/distill.py,
//     written as compile-time constants. Rejected alternative: read the shapes
//     from the weight file and size everything dynamically. That would be a
//     small model-loading framework, and this code runs exactly one model. The
//     constants are cross-checked against the weight file at construction, so a
//     mismatch is an immediate, specific error rather than a silent misread.

#ifndef TICK_TO_SIGNAL_MODEL_HPP
#define TICK_TO_SIGNAL_MODEL_HPP

#include <string>
#include <vector>

#include "tensor.hpp"
#include "weights.hpp"

namespace tts {

// The student configuration from ml/distill.py: STUDENT_CONFIG.
namespace arch {
constexpr int kWindow = 100;         // timesteps per sample
constexpr int kFeatures = 40;        // 10 levels x (bid price, bid size, ask price, ask size)
constexpr int kConvChannels = 8;     // conv_channels
constexpr int kBranchChannels = 16;  // inception_channels, per branch
constexpr int kConcatChannels = kBranchChannels * 3;
constexpr int kTcnChannels = 32;     // tcn_channels
constexpr int kTcnKernel = 3;
constexpr int kTcnBlocks = 4;
constexpr int kClasses = 3;
constexpr int kDepthLevels = 10;

// Feature-axis width after each fusion stage: 40 -> 20 -> 10 -> 1.
constexpr int kWidthAfterPairs = kFeatures / 2;
constexpr int kWidthAfterSides = kWidthAfterPairs / 2;
constexpr int kWidthAfterLevels = 1;

constexpr int kDilations[kTcnBlocks] = {1, 2, 4, 8};
}  // namespace arch

class StudentModel {
public:
    explicit StudentModel(const std::string& weight_path);

    // The hot call. Reads `arch::kWindow * arch::kFeatures` floats in row-major
    // [time][feature] order - the same layout ml/features.py produces - and
    // writes `arch::kClasses` logits. Allocates nothing.
    void forward(const float* window, float* logits);

    // Convenience for callers that want probabilities; applies softmax to the
    // logits. Kept separate because the parity fixtures compare logits, and a
    // softmax would hide a constant offset between the two implementations.
    void forward_probabilities(const float* window, float* probabilities);

    std::size_t weight_values() const { return weights_.total_values(); }
    std::size_t activation_bytes() const;

private:
    // One convolution's resolved weight and bias. Held instead of a name so the
    // hot path never touches the weight map or builds a string.
    struct LayerWeights {
        const float* weight = nullptr;
        const float* bias = nullptr;
    };
    // The three convolutions of one LevelFusionBlock, kept together because
    // run_level_block wants exactly this group.
    struct FusionWeights {
        LayerWeights spatial, temporal_one, temporal_two;
    };

    LayerWeights resolve(const std::string& name, const std::vector<int>& shape);

    void run_level_block(const FusionWeights& layer, const float* input, int in_channels,
                         int in_width, int spatial_kernel, int spatial_stride,
                         int out_width, Tensor& scratch_a, Tensor& scratch_b);
    void run_inception(const float* input);
    void run_temporal_block(int index);

    WeightStore weights_;

    // ---- weight pointers, all resolved once in the constructor ----
    FusionWeights pairs_w_, sides_w_, levels_w_;
    LayerWeights reduce_three_w_, conv_three_w_, reduce_five_w_, conv_five_w_, pool_project_w_;
    LayerWeights project_w_;
    LayerWeights block_one_w_[arch::kTcnBlocks], block_two_w_[arch::kTcnBlocks];
    LayerWeights classifier_w_;

    // ---- activation buffers, all allocated once in the constructor ----
    // Named for the stage that fills them; sizes are in the .cpp next to the
    // allocation so each one's role and cost sit together.
    Tensor input_;        // [1, 100, 40]  the caller's window, copied in
    Tensor pairs_a_;      // [8, 100, 20]  fuse_price_and_size, ping
    Tensor pairs_b_;      // [8, 100, 20]  fuse_price_and_size, pong
    Tensor sides_a_;      // [8, 100, 10]  fuse_sides, ping
    Tensor sides_b_;      // [8, 100, 10]  fuse_sides, pong
    Tensor levels_a_;     // [8, 100, 1]   fuse_levels, ping
    Tensor levels_b_;     // [8, 100, 1]   fuse_levels, pong
    Tensor reduce_;       // [16, 100, 1]  inception 1x1 reduction output
    Tensor pooled_;       // [8, 100, 1]   inception max-pool branch
    Tensor concat_;       // [48, 100, 1]  the three branches written in place
    Tensor projected_;    // [32, 100]     after the 1x1 projection
    Tensor tcn_a_;        // [32, 100]     residual block, first convolution
    Tensor tcn_b_;        // [32, 100]     residual block, second convolution
};

}  // namespace tts

#endif  // TICK_TO_SIGNAL_MODEL_HPP
