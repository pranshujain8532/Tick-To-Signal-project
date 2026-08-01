// ops.hpp - the handful of tensor operators this one model needs.
//
// WHAT
//     Causal 2-D convolution, causal dilated 1-D convolution, causal max-pool,
//     leaky ReLU, a fully-connected layer, an element-wise add, and softmax.
//     Seven functions. That is the entire operator set.
//
// WHY
//     The model in ml/model.py is built from exactly these. Anything more would
//     be code the parity test never executes.
//
// LAYOUT CONVENTION - everything is C-contiguous with the batch dimension gone.
//     4-D activations are [channels][height][width], indexed
//         value = data[c * height * width + h * width + w]
//     3-D activations are [channels][length], indexed
//         value = data[c * length + t]
//     Convolution weights keep PyTorch's own order, [out][in][kh][kw] and
//         [out][in][k], so the exporter can write them without transposing and
//     there is one fewer transformation between the two implementations to get
//     wrong.
//
// CAUSAL PADDING - implemented by skipping, not by padding buffers.
//     Every convolution over the time axis in this model is left-padded with
//     zeros so the output length equals the input length. Rather than
//     materialise a padded copy, the loops compute the source index and skip
//     any that falls before zero. That is arithmetically identical - the padded
//     values are zero and contribute nothing to a sum - and it avoids both an
//     allocation and a copy per layer.
//
//     Max-pool is the exception that proves the rule and it is easy to get
//     wrong: PyTorch pads with *zeros* before pooling, so a window overhanging
//     the start takes the max of the real values **and zero**, not the max of
//     the real values alone. Skipping would give a different answer. See
//     `maxpool2d_causal_time`.
//
// ALIASING CONTRACT - `__restrict` is a promise the CALLER has to keep.
//     Every pointer below is marked `__restrict`, which tells the compiler that
//     within the call no two of them refer to overlapping memory. That is what
//     lets it keep a value in a register across a store instead of reloading it
//     in case the store just changed it, and after the loop reordering in Stage
//     7b - where the output is read-modify-written rather than accumulated in a
//     register - it is worth a great deal.
//
//     It is also UNCHECKED. If a caller ever passes the same buffer as both
//     input and output, the compiler is entitled to produce nonsense and no
//     diagnostic. The rule for this codebase: every operator writes to a buffer
//     that is not one of its inputs. model.cpp ping-pongs between two scratch
//     tensors and incremental.cpp writes into a different ring from the one it
//     reads, so the promise holds - but it holds by convention, not by
//     construction, which is why it is written down here.
//
// CORRECTNESS FIRST - the arithmetic in this file is not up for negotiation.
//     Stage 7a established these as the most obvious loops that produce the
//     right numbers. Stage 7b reorders them for cache and vector friendliness,
//     but only in ways that leave the sequence of floating-point additions for
//     each output element EXACTLY as it was - see the note on conv1d in ops.cpp.
//     The parity and incremental tests are what keep that honest.

#ifndef TICK_TO_SIGNAL_OPS_HPP
#define TICK_TO_SIGNAL_OPS_HPP

#include <cstddef>

namespace tts {

// PyTorch's LeakyReLU default, and what ml/model.py constructs.
constexpr float kLeakySlope = 0.01f;

// 2-D convolution, causal over `height` (time), valid over `width` (features).
//
// Output height equals input height; output width is
// (in_width - kernel_width) / stride_width + 1.
// Weight layout [out_channels][in_channels][kernel_height][kernel_width].
void conv2d_causal_time(const float* __restrict input, int in_channels, int in_height, int in_width,
                        const float* __restrict weight, const float* __restrict bias,
                        int out_channels, int kernel_height, int kernel_width, int stride_width,
                        float* __restrict output);

// 1-D convolution with dilation, causal over the length axis. Output length
// equals input length. Weight layout [out_channels][in_channels][kernel].
void conv1d_causal_dilated(const float* __restrict input, int in_channels, int length,
                           const float* __restrict weight, const float* __restrict bias,
                           int out_channels, int kernel, int dilation, float* __restrict output);

// Max-pool over the time axis with a (kernel_height, 1) window, stride 1, and
// causal ZERO padding - see the header note.
void maxpool2d_causal_time(const float* __restrict input, int channels, int height, int width,
                           int kernel_height, float* __restrict output);

void leaky_relu(float* data, std::size_t count, float slope = kLeakySlope);

// out = weight * input + bias, with weight laid out [out_features][in_features]
// exactly as torch.nn.Linear stores it.
void linear(const float* __restrict input, const float* __restrict weight,
            const float* __restrict bias, int in_features, int out_features,
            float* __restrict output);

void add_into(float* __restrict accumulator, const float* __restrict addend, std::size_t count);

// Numerically stable softmax (subtracts the max before exponentiating).
void softmax(const float* __restrict input, int count, float* __restrict output);

}  // namespace tts

#endif  // TICK_TO_SIGNAL_OPS_HPP
