#include "incremental.hpp"

#include <limits>

#include "ops.hpp"

#if defined(TTS_USE_AVX2_KERNEL) && defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace tts {

using namespace arch;

namespace {

// Kernel heights the architecture uses over the time axis. They are literals in
// model.cpp too; duplicating them here is safe because every weight this file
// resolves is shape-checked against them at construction, so a divergence is an
// immediate, named error rather than a silent misread.
constexpr int kFusionTemporalKernel = 4;   // LevelFusionBlock.temporal_one / _two
constexpr int kInceptionThree = 3;         // InceptionLite.conv_three
constexpr int kInceptionFive = 5;          // InceptionLite.conv_five
constexpr int kInceptionPool = 3;          // InceptionLite max-pool branch
constexpr int kLongestTimeKernel = 5;      // the largest of the above, for tap arrays

// ---------------------------------------------------------------------------
// Column operators: the same convolutions as ops.cpp, evaluated at exactly ONE
// output timestep.
//
// `history[k]` is the input column belonging to kernel tap k, oldest first, so
// `history[kernel - 1]` is always the current tick. The caller assembles those
// pointers from a ring, which is what turns "convolve over the window" into
// "convolve over four pointers".
//
// These are file-local rather than added to ops.hpp because the incremental path
// is their only caller, and ops.hpp is documented as the operator set the full
// forward pass needs. They are not covered by ops_test's per-operator fixtures;
// they are covered by incremental_test, which checks their composition against
// the fixture-verified full path on 300 consecutive ticks. That is the stronger
// check of the two - a per-op fixture would not catch a ring indexed one slot
// off, and the equivalence test does.
// ---------------------------------------------------------------------------

// Accumulation order (in_channel, then kernel height, then kernel width) is
// deliberately identical to conv2d_causal_time. Float addition is not
// associative, so matching the order is what makes the two paths agree to the
// last bit rather than merely to a tolerance.
//
// The loop nest is the reordered one - output width innermost, weight hoisted to
// a scalar - for the same reasons ops.cpp gives. Per output element the sequence
// of additions is unchanged, which is why both paths could be reordered
// independently and stay bit-identical to each other.
//
// Taps before the start of the stream are the ring's zeros. The full-window
// version skips them; multiplying zero by a weight and adding is exact in
// IEEE-754, so both reach the same sum, and skipping here would mean a branch
// on the hottest loop in the file for no numerical gain.
void conv2d_column(const float* const* __restrict history, int in_channels, int in_width,
                   const float* __restrict weight, const float* __restrict bias, int out_channels,
                   int kernel_height, int kernel_width, int stride_width,
                   float* __restrict output) {
    const int out_width = (in_width - kernel_width) / stride_width + 1;

    for (int oc = 0; oc < out_channels; ++oc) {
        float* __restrict out_row = output + oc * out_width;
        for (int ow = 0; ow < out_width; ++ow) out_row[ow] = bias[oc];

        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_height; ++kh) {
                const float* __restrict column = history[kh] + ic * in_width;
                for (int kw = 0; kw < kernel_width; ++kw) {
                    const float coefficient =
                        weight[((oc * in_channels + ic) * kernel_height + kh) * kernel_width + kw];
                    for (int ow = 0; ow < out_width; ++ow) {
                        out_row[ow] += column[ow * stride_width + kw] * coefficient;
                    }
                }
            }
        }
    }
}

// out[i] += scalar * coefficients[i]. THE hottest loop in the streaming path:
// the eight TCN convolutions call it 96 times each with count = 32, and the TCN
// is 53% of this model's multiply-accumulates.
//
// TWO IMPLEMENTATIONS, SELECTED AT COMPILE TIME, and the scalar one is never
// deleted. -DTTS_AVX2=ON picks the intrinsics; anything else, including any
// non-x86 target, gets the plain loop. Both are exercised by ctest, which is the
// only way the intrinsics can be trusted.
//
// WHY _mm256_fmadd_ps AND NOT A SEPARATE MULTIPLY AND ADD.
//     FMA computes a*b+c with ONE rounding where a separate multiply and add
//     round twice, so the two are not interchangeable and the choice has to
//     match what the rest of the codebase does. It does: GCC's default for C++
//     is -ffp-contract=fast, and inspecting the assembly for the scalar loop
//     under -march=native shows 16 vfmadd instructions and zero vmulps/vaddps -
//     the compiler is already contracting everywhere else. Writing mul+add here
//     would make this kernel the ODD ONE OUT and would break the bit-identity
//     that incremental_test checks.
//
//     That default is now pinned explicitly in CMakeLists.txt rather than
//     inherited, because it is load-bearing. Note FMA contraction is not the
//     thing -ffast-math would enable: contraction removes a rounding step and
//     never reorders a sum, whereas reassociation changes which numbers are
//     added together. The first is fine here; the second is refused.
//
// The remainder loop below handles a count that is not a multiple of 8. For
// this model it never runs - every 1-D convolution has 32 output channels, and
// 32 is 4 vectors exactly - but a kernel that is silently wrong for count 33 is
// a trap for whoever changes tcn_channels next.
#if defined(TTS_USE_AVX2_KERNEL) && defined(__AVX2__) && defined(__FMA__)
inline void axpy_into(float* __restrict out, const float* __restrict coefficients, float scalar,
                      int count) {
    const __m256 broadcast = _mm256_set1_ps(scalar);
    int i = 0;
    for (; i + 8 <= count; i += 8) {
        // Unaligned loads: `out` is a 64-byte-aligned Tensor, but `coefficients`
        // points partway into a transposed weight buffer at an offset that is
        // only a multiple of out_channels floats. _mm256_load_ps would fault.
        // On every part that has AVX2, an unaligned load that does not straddle
        // a cache line costs the same as an aligned one.
        const __m256 accumulated = _mm256_loadu_ps(out + i);
        const __m256 coefficient = _mm256_loadu_ps(coefficients + i);
        _mm256_storeu_ps(out + i, _mm256_fmadd_ps(broadcast, coefficient, accumulated));
    }
    for (; i < count; ++i) out[i] += scalar * coefficients[i];
}
#else
inline void axpy_into(float* __restrict out, const float* __restrict coefficients, float scalar,
                      int count) {
    for (int i = 0; i < count; ++i) out[i] += scalar * coefficients[i];
}
#endif

// The dilated 1-D case. Dilation does not appear here at all: it lives entirely
// in which ring slots the caller picked for `history`, which is the whole trick
// - a dilated convolution over a stream is a plain convolution over strided
// pointers.
//
// LOOP ORDER - output channel innermost, which is the only axis available.
//     conv2d_column can put the output WIDTH innermost. Here the width is 1, so
//     the original nesting left one float accumulator absorbing 96 dependent
//     additions (32 in_channels x 3 taps) per output channel: a ~4-cycle latency
//     chain 96 links long, 32 times over, in each of eight TCN convolutions.
//
//     Running `oc` innermost gives 32 independent accumulations instead. It also
//     requires `weight_by_output` rather than the file's layout - see
//     transpose_conv1d_weight for why, and why that is the change that actually
//     unlocked the vector units.
//
//     Per output channel the additions are still bias, then (ic, k) in order, so
//     this remains bit-identical to the nesting it replaced.
void conv1d_column(const float* const* __restrict history, int in_channels,
                   const float* __restrict weight_by_output, const float* __restrict bias,
                   int out_channels, int kernel, float* __restrict output) {
    for (int oc = 0; oc < out_channels; ++oc) output[oc] = bias[oc];

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int k = 0; k < kernel; ++k) {
            const float* __restrict row = weight_by_output + (ic * kernel + k) * out_channels;
            axpy_into(output, row, history[k][ic], out_channels);
        }
    }
}

// Causal max-pool at one timestep. The zeros a freshly reset ring holds are what
// stand in for PyTorch's zero padding, so an overhanging window competes against
// 0.0 exactly as maxpool2d_causal_time arranges with its `(ih < 0) ? 0.0f`.
void maxpool_column(const float* const* __restrict history, int channels, int width,
                    int kernel_height, float* __restrict output) {
    for (int c = 0; c < channels; ++c) {
        for (int w = 0; w < width; ++w) {
            float best = -std::numeric_limits<float>::infinity();
            for (int kh = 0; kh < kernel_height; ++kh) {
                const float value = history[kh][c * width + w];
                if (value > best) best = value;
            }
            output[c * width + w] = best;
        }
    }
}

}  // namespace

// ----------------------------------------------------------------- ColumnRing

void ColumnRing::configure(int capacity, int values_per_column) {
    capacity_ = capacity;
    values_ = values_per_column;
    storage_.reshape({capacity, values_per_column});
    reset();
}

void ColumnRing::reset() {
    storage_.zero();
    // Start one slot before the beginning so the first push lands at index 0.
    newest_ = capacity_ - 1;
}

float* ColumnRing::push() {
    newest_ = (newest_ + 1) % capacity_;
    return storage_.data() + static_cast<std::size_t>(newest_) * values_;
}

const float* ColumnRing::peek(int steps_back) const {
    const int index = ((newest_ - steps_back) % capacity_ + capacity_) % capacity_;
    return storage_.data() + static_cast<std::size_t>(index) * values_;
}

// ------------------------------------------------------------ IncrementalModel

IncrementalModel::ConvWeights IncrementalModel::resolve(const std::string& name,
                                                        const std::vector<int>& shape) {
    ConvWeights resolved;
    resolved.weight = weights_.get(name + ".weight", shape).data();
    resolved.bias = weights_.get(name + ".bias", {shape[0]}).data();
    return resolved;
}

// Rebuild a conv1d weight from PyTorch's [out][in][k] into [in][k][out].
//
// WHY THE LAYOUT CHANGES HERE, AND ONLY HERE
//     conv1d_column runs its inner loop over OUTPUT channels. It has to: the
//     feature axis is width 1 at that point in the network, so there is no other
//     axis to put innermost. In the file's layout the weights for consecutive
//     output channels sit in_channels * kernel floats apart - 384 bytes for the
//     TCN. That is a correct read and a strided one, and a strided read cannot
//     be vectorised at all, because there is no instruction that loads eight
//     floats 384 bytes apart.
//
//     Transposing to [in][k][out] puts those eight floats next to each other.
//     This is the change that made the vector units reachable; the intrinsics in
//     axpy_into only decide whether the compiler or this file writes the vector
//     code, and the commit message reports which won.
//
//     conv2d_column is deliberately NOT transposed: it already runs the output
//     WIDTH innermost, and in that loop PyTorch's layout is already contiguous.
//
//     The transpose runs once, at construction, into memory this class owns. The
//     on-disk format is untouched - ml/export_weights.py still writes PyTorch's
//     order - so there is one layout in the file and exactly one place where a
//     second one is derived from it.
void transpose_conv1d_weight(const float* source, int out_channels, int in_channels, int kernel,
                             float* destination) {
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int k = 0; k < kernel; ++k) {
                destination[(ic * kernel + k) * out_channels + oc] =
                    source[(oc * in_channels + ic) * kernel + k];
            }
        }
    }
}

IncrementalModel::ConvWeights IncrementalModel::resolve_conv1d(const std::string& name,
                                                               int out_channels, int in_channels,
                                                               int kernel) {
    ConvWeights resolved = resolve(name, {out_channels, in_channels, kernel});

    transposed_.emplace_back();
    Tensor& copy = transposed_.back();
    copy.reshape({in_channels, kernel, out_channels});
    transpose_conv1d_weight(resolved.weight, out_channels, in_channels, kernel, copy.data());
    resolved.weight_by_output = copy.data();
    return resolved;
}

IncrementalModel::IncrementalModel(const std::string& weight_path) : weights_(weight_path) {
    // Projection plus two per temporal block. Reserved so the vector never
    // reallocates while ConvWeights members already point into its Tensors.
    transposed_.reserve(1 + 2 * kTcnBlocks);

    const std::vector<int> fusion_temporal{kConvChannels, kConvChannels, kFusionTemporalKernel, 1};

    pairs_spatial_w_ = resolve("fuse_price_and_size.spatial", {kConvChannels, 1, 1, 2});
    pairs_one_w_ = resolve("fuse_price_and_size.temporal_one", fusion_temporal);
    pairs_two_w_ = resolve("fuse_price_and_size.temporal_two", fusion_temporal);

    sides_spatial_w_ = resolve("fuse_sides.spatial", {kConvChannels, kConvChannels, 1, 2});
    sides_one_w_ = resolve("fuse_sides.temporal_one", fusion_temporal);
    sides_two_w_ = resolve("fuse_sides.temporal_two", fusion_temporal);

    levels_spatial_w_ =
        resolve("fuse_levels.spatial", {kConvChannels, kConvChannels, 1, kDepthLevels});
    levels_one_w_ = resolve("fuse_levels.temporal_one", fusion_temporal);
    levels_two_w_ = resolve("fuse_levels.temporal_two", fusion_temporal);

    reduce_three_w_ = resolve("inception.reduce_three", {kBranchChannels, kConvChannels, 1, 1});
    conv_three_w_ =
        resolve("inception.conv_three", {kBranchChannels, kBranchChannels, kInceptionThree, 1});
    reduce_five_w_ = resolve("inception.reduce_five", {kBranchChannels, kConvChannels, 1, 1});
    conv_five_w_ =
        resolve("inception.conv_five", {kBranchChannels, kBranchChannels, kInceptionFive, 1});
    pool_project_w_ = resolve("inception.pool_project", {kBranchChannels, kConvChannels, 1, 1});

    project_w_ = resolve_conv1d("project", kTcnChannels, kConcatChannels, 1);
    for (int index = 0; index < kTcnBlocks; ++index) {
        const std::string prefix = "temporal_blocks." + std::to_string(index);
        block_one_w_[index] =
            resolve_conv1d(prefix + ".conv_one", kTcnChannels, kTcnChannels, kTcnKernel);
        block_two_w_[index] =
            resolve_conv1d(prefix + ".conv_two", kTcnChannels, kTcnChannels, kTcnKernel);
    }
    classifier_w_ = resolve("classifier", {kClasses, kTcnChannels});

    configure_rings();

    pairs_out_.reshape({kConvChannels, kWidthAfterPairs});
    sides_out_.reshape({kConvChannels, kWidthAfterSides});
    pooled_.reshape({kConvChannels, kWidthAfterLevels});
    concat_.reshape({kConcatChannels});
    final_column_.reshape({kTcnChannels});
    // Scratch is fully overwritten every tick, so it needs no run-time
    // initialisation - but leaving a malloc'd buffer unread-but-uninitialised is
    // undefined behaviour the moment a future bug reads it, so zero it once.
    pairs_out_.zero();
    sides_out_.zero();
    pooled_.zero();
    concat_.zero();
    final_column_.zero();
}

// Each ring holds exactly the taps its consumer can still reach - four columns
// for a kernel-4 convolution, 2*dilation + 1 for a dilated kernel-3 one. Sizing
// them to the reach rather than to the window is what keeps the whole state
// around 18 KB instead of the full path's 275 KB.
void IncrementalModel::configure_rings() {
    const int pairs_values = kConvChannels * kWidthAfterPairs;
    const int sides_values = kConvChannels * kWidthAfterSides;
    const int levels_values = kConvChannels * kWidthAfterLevels;

    pairs_spatial_.configure(kFusionTemporalKernel, pairs_values);
    pairs_temporal_.configure(kFusionTemporalKernel, pairs_values);
    sides_spatial_.configure(kFusionTemporalKernel, sides_values);
    sides_temporal_.configure(kFusionTemporalKernel, sides_values);
    levels_spatial_.configure(kFusionTemporalKernel, levels_values);
    levels_temporal_.configure(kFusionTemporalKernel, levels_values);

    levels_out_.configure(kInceptionPool, levels_values);
    reduce_three_.configure(kInceptionThree, kBranchChannels * kWidthAfterLevels);
    reduce_five_.configure(kInceptionFive, kBranchChannels * kWidthAfterLevels);

    for (int index = 0; index < kTcnBlocks; ++index) {
        // Taps sit at 0, dilation and 2*dilation ticks back, so the oldest one
        // the block can reach is 2*dilation - hence 2*dilation + 1 slots.
        const int reach = 2 * kDilations[index] + 1;
        block_in_[index].configure(reach, kTcnChannels);
        block_mid_[index].configure(reach, kTcnChannels);
    }
}

void IncrementalModel::reset() {
    pairs_spatial_.reset();
    pairs_temporal_.reset();
    sides_spatial_.reset();
    sides_temporal_.reset();
    levels_spatial_.reset();
    levels_temporal_.reset();
    levels_out_.reset();
    reduce_three_.reset();
    reduce_five_.reset();
    for (int index = 0; index < kTcnBlocks; ++index) {
        block_in_[index].reset();
        block_mid_[index].reset();
    }
}

std::size_t IncrementalModel::state_bytes() const {
    const ColumnRing* rings[] = {&pairs_spatial_, &pairs_temporal_, &sides_spatial_,
                                 &sides_temporal_, &levels_spatial_, &levels_temporal_,
                                 &levels_out_, &reduce_three_, &reduce_five_};
    std::size_t total = 0;
    for (const ColumnRing* ring : rings) total += ring->bytes();
    for (int index = 0; index < kTcnBlocks; ++index) {
        total += block_in_[index].bytes() + block_mid_[index].bytes();
    }
    const Tensor* scratch[] = {&pairs_out_, &sides_out_, &pooled_, &concat_, &final_column_};
    for (const Tensor* buffer : scratch) total += buffer->size() * sizeof(float);
    return total;
}

// A kernel-4 causal convolution whose four taps come out of `history`, newest
// last. Pulled out because both temporal convolutions in every fusion block do
// exactly this and differ only in which ring they read.
void IncrementalModel::apply_temporal(const ColumnRing& history, const ConvWeights& conv, int width,
                                      float* output_column) {
    const float* taps[kFusionTemporalKernel];
    for (int k = 0; k < kFusionTemporalKernel; ++k) {
        taps[k] = history.peek(kFusionTemporalKernel - 1 - k);
    }
    conv2d_column(taps, kConvChannels, width, conv.weight, conv.bias, kConvChannels,
                  kFusionTemporalKernel, 1, 1, output_column);
}

void IncrementalModel::advance_fusion(const ConvWeights& spatial, const ConvWeights& temporal_one,
                                      const ConvWeights& temporal_two, const float* input_column,
                                      int in_channels, int in_width, int spatial_kernel,
                                      int spatial_stride, ColumnRing& after_spatial,
                                      ColumnRing& after_temporal_one, float* output_column) {
    const int out_width = (in_width - spatial_kernel) / spatial_stride + 1;
    const std::size_t out_values = static_cast<std::size_t>(kConvChannels) * out_width;

    // The spatial convolution has kernel height 1: it reads this tick only. It
    // still writes into a ring, because the temporal convolution behind it needs
    // this column again on the next three ticks.
    const float* current[1] = {input_column};
    float* spatial_out = after_spatial.push();
    conv2d_column(current, in_channels, in_width, spatial.weight, spatial.bias, kConvChannels, 1,
                  spatial_kernel, spatial_stride, spatial_out);
    leaky_relu(spatial_out, out_values);

    float* temporal_out = after_temporal_one.push();
    apply_temporal(after_spatial, temporal_one, out_width, temporal_out);
    leaky_relu(temporal_out, out_values);

    apply_temporal(after_temporal_one, temporal_two, out_width, output_column);
    leaky_relu(output_column, out_values);
}

// One of the two convolutional Inception branches: a 1x1 reduction, then a
// causal convolution spanning `kernel` ticks. The branches differ only in that
// kernel height and in which ring holds their reductions.
void IncrementalModel::advance_inception_branch(const ConvWeights& reduce, const ConvWeights& conv,
                                                int kernel, const float* levels_column,
                                                ColumnRing& ring, float* output_column) {
    const float* current[1] = {levels_column};
    float* reduced = ring.push();
    conv2d_column(current, kConvChannels, kWidthAfterLevels, reduce.weight, reduce.bias,
                  kBranchChannels, 1, 1, 1, reduced);
    leaky_relu(reduced, static_cast<std::size_t>(kBranchChannels) * kWidthAfterLevels);

    const float* taps[kLongestTimeKernel];
    for (int k = 0; k < kernel; ++k) taps[k] = ring.peek(kernel - 1 - k);
    conv2d_column(taps, kBranchChannels, kWidthAfterLevels, conv.weight, conv.bias, kBranchChannels,
                  kernel, 1, 1, output_column);
}

void IncrementalModel::advance_inception(const float* levels_column) {
    // The three branches write straight into their slice of `concat_`: with the
    // feature axis already width 1, a channel-major concatenation is three
    // contiguous runs, so there is no concatenation step to perform.
    advance_inception_branch(reduce_three_w_, conv_three_w_, kInceptionThree, levels_column,
                             reduce_three_, concat_.data() + 0 * kBranchChannels);
    advance_inception_branch(reduce_five_w_, conv_five_w_, kInceptionFive, levels_column,
                             reduce_five_, concat_.data() + 1 * kBranchChannels);

    // Branch 3 pools the fusion output itself, not a reduced copy - matching
    // InceptionLite.forward in ml/model.py.
    const float* pool_taps[kInceptionPool];
    for (int k = 0; k < kInceptionPool; ++k) pool_taps[k] = levels_out_.peek(kInceptionPool - 1 - k);
    maxpool_column(pool_taps, kConvChannels, kWidthAfterLevels, kInceptionPool, pooled_.data());

    const float* pooled_now[1] = {pooled_.data()};
    conv2d_column(pooled_now, kConvChannels, kWidthAfterLevels, pool_project_w_.weight,
                  pool_project_w_.bias, kBranchChannels, 1, 1, 1,
                  concat_.data() + 2 * kBranchChannels);

    // The post-concat BatchNorm is already folded into the three branch
    // convolutions, so only the activation remains.
    leaky_relu(concat_.data(), static_cast<std::size_t>(kConcatChannels) * kWidthAfterLevels);
}

void IncrementalModel::advance_temporal_block(int index) {
    const int dilation = kDilations[index];
    const std::size_t count = static_cast<std::size_t>(kTcnChannels);

    const float* taps[kTcnKernel];
    for (int k = 0; k < kTcnKernel; ++k) {
        taps[k] = block_in_[index].peek((kTcnKernel - 1 - k) * dilation);
    }
    float* middle = block_mid_[index].push();
    conv1d_column(taps, kTcnChannels, block_one_w_[index].weight_by_output,
                  block_one_w_[index].bias, kTcnChannels, kTcnKernel, middle);
    leaky_relu(middle, count);

    for (int k = 0; k < kTcnKernel; ++k) {
        taps[k] = block_mid_[index].peek((kTcnKernel - 1 - k) * dilation);
    }
    // The last block feeds the classifier; the others hand their output to the
    // next block's ring. That mirrors model.cpp overwriting `projected_` in
    // place, where each block's output becomes the next block's input.
    float* output = (index + 1 < kTcnBlocks) ? block_in_[index + 1].push() : final_column_.data();
    conv1d_column(taps, kTcnChannels, block_two_w_[index].weight_by_output,
                  block_two_w_[index].bias, kTcnChannels, kTcnKernel, output);
    leaky_relu(output, count);

    // Residual: the block's input at THIS tick, which is its ring's newest slot.
    add_into(output, block_in_[index].peek(0), count);
    leaky_relu(output, count);
}

void IncrementalModel::push_tick(const float* feature_column, float* logits) {
    // 40 -> 20: fuse each (price, size) pair.
    advance_fusion(pairs_spatial_w_, pairs_one_w_, pairs_two_w_, feature_column, 1, kFeatures, 2, 2,
                   pairs_spatial_, pairs_temporal_, pairs_out_.data());
    // 20 -> 10: fuse the bid side against the ask side.
    advance_fusion(sides_spatial_w_, sides_one_w_, sides_two_w_, pairs_out_.data(), kConvChannels,
                   kWidthAfterPairs, 2, 2, sides_spatial_, sides_temporal_, sides_out_.data());
    // 10 -> 1: fuse the ten levels into whole-book shape. The result goes
    // straight into a ring because the Inception max-pool reads three ticks of it.
    advance_fusion(levels_spatial_w_, levels_one_w_, levels_two_w_, sides_out_.data(),
                   kConvChannels, kWidthAfterSides, kDepthLevels, 1, levels_spatial_,
                   levels_temporal_, levels_out_.push());

    advance_inception(levels_out_.peek(0));

    // The 1x1 projection spans a single tick, so its "history" is just now.
    const float* concat_now[1] = {concat_.data()};
    conv1d_column(concat_now, kConcatChannels, project_w_.weight_by_output, project_w_.bias,
                  kTcnChannels, 1, block_in_[0].push());

    for (int index = 0; index < kTcnBlocks; ++index) advance_temporal_block(index);

    // No "read the final timestep" step is needed here: in a streaming model
    // every column IS the final timestep, which is the same property that made
    // the architecture incrementally updatable in the first place.
    linear(final_column_.data(), classifier_w_.weight, classifier_w_.bias, kTcnChannels, kClasses,
           logits);
}

void IncrementalModel::prime(const float* window, float* logits) {
    reset();
    for (int t = 0; t < kWindow; ++t) {
        push_tick(window + static_cast<std::size_t>(t) * kFeatures, logits);
    }
}

}  // namespace tts
