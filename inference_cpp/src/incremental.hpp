// incremental.hpp - the streaming forward pass: one new tick, not one new window.
//
// WHAT
//     `IncrementalModel::push_tick(column, logits)` takes ONE [40] feature
//     column - a single book snapshot - and produces the same three logits that
//     `StudentModel::forward` produces for the [100, 40] window ending at that
//     tick. It does so without recomputing the window, by keeping each layer's
//     recent activations in a small ring buffer.
//
// WHY
//     A tick-to-signal path is called once per book update, and consecutive
//     calls overlap in 99 of their 100 rows. Recomputing all 100 timesteps for
//     each new one does 100x the arithmetic the problem actually requires. This
//     is the single largest speedup available in Stage 7b - larger than SIMD,
//     larger than the loop-order work - because it is an algorithmic change
//     rather than a constant factor.
//
// WHICH LAYERS CAN CACHE, AND WHY
//     Every layer in this network is CAUSAL: its output at time t is a function
//     of inputs at times <= t only. Concretely,
//
//       * the temporal convolutions are left-padded, so output t reads inputs
//         t-(k-1)*d .. t and never t+1;
//       * the max-pool is left-padded with the same convention;
//       * the spatial convolutions (kernel height 1) touch a single timestep;
//       * the residual adds the block input at the SAME t;
//       * the classifier reads the final timestep only.
//
//     Causality is exactly the property that makes caching sound: nothing
//     computed for timestep t can be invalidated by a tick arriving at t+1. So
//     for a new tick this class computes exactly ONE new output column per
//     layer, reading that layer's previous input columns out of a ring buffer
//     instead of recomputing them.
//
//     What could NOT be cached, if the architecture had it:
//       * anything bidirectional - a centred convolution, a BiLSTM, unmasked
//         attention. A new tick would change the output at earlier timesteps.
//       * any normalisation whose statistics span the window - LayerNorm over
//         time, or a per-window standardisation. Appending one tick moves the
//         mean and variance, so every earlier position's normalised value
//         changes. This is why ml/features.py uses a CAUSAL rolling z-score
//         computed from prefix sums: that statistic updates incrementally too,
//         and the whole feature path stays streamable.
//       * a pool with stride > 1 over time, which would make the output grid
//         depend on where the window starts.
//
//     The BatchNorms would have been the awkward case, but Stage 7a already
//     folded them into the convolution weights, so at inference they are per-
//     channel affine constants and no statistic is computed at run time at all.
//
// WHY THE ANSWERS ARE IDENTICAL, NOT MERELY CLOSE
//     Two separate claims, and the second is the interesting one.
//
//     1. Arithmetic. Where the full-window operators *skip* a padded tap, these
//        column operators multiply the ring's zero by the weight and add it.
//        `sum + 0.0f * w == sum` exactly in IEEE-754, and the accumulation order
//        (in_channel, then kernel tap) is deliberately the same as in ops.cpp.
//        So this is not "close enough"; it is the same sequence of roundings.
//
//     2. Horizon. The full-window path sees exactly 100 timesteps and zeros
//        before them. A primed incremental model has seen the entire stream. Those
//        agree only because the RECEPTIVE FIELD IS 83 AND THE WINDOW IS 100:
//        output t cannot depend on anything before t-82, so the extra history
//        the incremental model remembers is unreachable and the 17 timesteps of
//        window the full path recomputes are dead weight. Had the window been
//        64, the two would legitimately disagree and this equivalence test would
//        have to be replaced by a tolerance. `TickToSignalNet.receptive_field()`
//        in ml/model.py is what pins that number.
//
// DESIGN DECISION - one ring per layer input, sized to that layer's taps.
//     Rejected alternative: keep the full [C, 100, W] activation buffers and
//     shift them left by one column each tick. That is simpler to read but
//     copies ~275 KB per tick, which is more memory traffic than the arithmetic
//     it saves. The rings hold only what each layer can still reach - 4 columns
//     for a kernel-4 conv, 2*dilation+1 for a dilated one - so the whole state
//     is ~18 KB and stays in L1.
//
// DESIGN DECISION - weight pointers resolved once, in the constructor.
//     `StudentModel::forward` looks weights up by name on every call, building
//     std::strings to do it. Amortised over 100 timesteps that is invisible;
//     on a path this short it is not. Every pointer this class needs is
//     resolved and shape-checked at construction, so `push_tick` performs no
//     allocation, no string work and no map lookups.

#ifndef TICK_TO_SIGNAL_INCREMENTAL_HPP
#define TICK_TO_SIGNAL_INCREMENTAL_HPP

#include <cstddef>
#include <string>

#include "model.hpp"
#include "tensor.hpp"
#include "weights.hpp"

namespace tts {

// A fixed-capacity circular buffer of activation columns.
//
// "Column" means one timestep of one layer's output, flattened: [C, W] values
// for a 4-D activation, [C] for a 1-D one. The capacity is chosen to be exactly
// the number of taps the consuming layer can reach, so `peek` never returns a
// column too old to be meaningful - it returns either a real past column or,
// before enough ticks have arrived, the zeros left by `reset`. Those zeros are
// the causal padding.
class ColumnRing {
public:
    void configure(int capacity, int values_per_column);

    // Zero every slot. Must be called before the first push, and by
    // IncrementalModel::reset when the stream is restarted.
    void reset();

    // Advance to the next slot and return it for writing. The returned column
    // still holds the values from `capacity` ticks ago, so the caller must
    // overwrite all of it - which every operator here does.
    float* push();

    // `steps_back == 0` is the column written by the most recent push.
    const float* peek(int steps_back) const;

    std::size_t bytes() const { return storage_.size() * sizeof(float); }

private:
    Tensor storage_;
    int capacity_ = 0;
    int values_ = 0;
    int newest_ = 0;
};

class IncrementalModel {
public:
    explicit IncrementalModel(const std::string& weight_path);

    // Forget the stream. Every ring becomes zeros, which is precisely the
    // causal zero-padding a full-window forward assumes before its first row.
    void reset();

    // The hot call. `feature_column` is arch::kFeatures floats - one row of the
    // [100, 40] window, in the layout ml/features.py produces - and `logits`
    // receives arch::kClasses floats. Allocates nothing.
    void push_tick(const float* feature_column, float* logits);

    // Cold start: reset, then feed all arch::kWindow rows of `window` in order.
    // Leaves the logits of the LAST row in `logits`, which makes this
    // interchangeable with StudentModel::forward - and tests/incremental_test.cpp
    // asserts exactly that, on real fixtures.
    void prime(const float* window, float* logits);

    std::size_t weight_values() const { return weights_.total_values(); }
    std::size_t state_bytes() const;

private:
    // A convolution's resolved weight pointers. Held instead of names so the hot
    // path never touches the weight map.
    //
    // `weight` is PyTorch's own [out][in][kh][kw] order, straight out of the
    // file. `weight_by_output` is set only for the 1-D convolutions and points
    // at a transposed [in][k][out] copy this class owns - see
    // transpose_conv1d_weight in the .cpp for why that copy exists.
    struct ConvWeights {
        const float* weight = nullptr;
        const float* bias = nullptr;
        const float* weight_by_output = nullptr;
    };

    ConvWeights resolve(const std::string& name, const std::vector<int>& shape);
    // As `resolve`, plus the transposed copy the streaming 1-D kernel wants.
    ConvWeights resolve_conv1d(const std::string& name, int out_channels, int in_channels,
                               int kernel);
    void configure_rings();

    // A kernel-4 causal convolution reading its taps out of `history`.
    void apply_temporal(const ColumnRing& history, const ConvWeights& conv, int width,
                        float* output_column);
    // One LevelFusionBlock for a single tick: spatial narrowing, then the two
    // causal temporal convolutions, each reading its own ring.
    void advance_fusion(const ConvWeights& spatial, const ConvWeights& temporal_one,
                        const ConvWeights& temporal_two, const float* input_column,
                        int in_channels, int in_width, int spatial_kernel, int spatial_stride,
                        ColumnRing& after_spatial, ColumnRing& after_temporal_one,
                        float* output_column);
    void advance_inception_branch(const ConvWeights& reduce, const ConvWeights& conv, int kernel,
                                  const float* levels_column, ColumnRing& ring,
                                  float* output_column);
    void advance_inception(const float* levels_column);
    // Reads block `index`'s input from block_in_[index] and writes its output
    // into block_in_[index + 1], or into `final_column_` for the last block.
    void advance_temporal_block(int index);

    WeightStore weights_;
    // Transposed copies of the nine 1-D convolution weights (the projection and
    // two per temporal block). Reserved up front so no reallocation happens
    // while ConvWeights members already point into these buffers.
    std::vector<Tensor> transposed_;

    ConvWeights pairs_spatial_w_, pairs_one_w_, pairs_two_w_;
    ConvWeights sides_spatial_w_, sides_one_w_, sides_two_w_;
    ConvWeights levels_spatial_w_, levels_one_w_, levels_two_w_;
    ConvWeights reduce_three_w_, conv_three_w_, reduce_five_w_, conv_five_w_, pool_project_w_;
    ConvWeights project_w_;
    ConvWeights block_one_w_[arch::kTcnBlocks], block_two_w_[arch::kTcnBlocks];
    ConvWeights classifier_w_;

    // ---- ring state: one per layer whose kernel spans more than one tick ----
    ColumnRing pairs_spatial_, pairs_temporal_;    // [8, 20] columns, 4 taps each
    ColumnRing sides_spatial_, sides_temporal_;    // [8, 10] columns, 4 taps each
    ColumnRing levels_spatial_, levels_temporal_;  // [8, 1]  columns, 4 taps each
    ColumnRing levels_out_;                        // [8, 1],  3 taps for the max-pool
    ColumnRing reduce_three_, reduce_five_;        // [16, 1], 3 and 5 taps
    ColumnRing block_in_[arch::kTcnBlocks];        // [32], 2*dilation + 1 taps
    ColumnRing block_mid_[arch::kTcnBlocks];       // [32], 2*dilation + 1 taps

    // ---- scratch columns: values that are consumed within the same tick ----
    Tensor pairs_out_;   // [8, 20]  fuse_price_and_size output
    Tensor sides_out_;   // [8, 10]  fuse_sides output
    Tensor pooled_;      // [8, 1]   the Inception max-pool branch
    Tensor concat_;      // [48]     the three Inception branches, written in place
    Tensor final_column_;// [32]     the last TCN block's output, fed to the classifier
};

}  // namespace tts

#endif  // TICK_TO_SIGNAL_INCREMENTAL_HPP
