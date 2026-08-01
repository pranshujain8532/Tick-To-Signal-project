#include "ops.hpp"

#include <cmath>
#include <limits>

namespace tts {

// LOOP ORDER - the output width is the innermost loop, and that is the point.
//
//     Stage 7a wrote the obvious nesting: for each output element, sum over
//     (in_channel, kernel_height, kernel_width). That has two problems, and the
//     second is the expensive one.
//
//     1. Locality. Consecutive `in_channel` values are `in_height * in_width`
//        floats apart - 8 KB for the widest layer here - so the innermost loop
//        touched a new cache line on almost every iteration, and re-read the
//        whole input plane once per output element.
//     2. Dependency chain. `sum` is a single float accumulator, so the 32
//        additions for one output element are 32 dependent floating-point adds.
//        At ~4 cycles of latency each that is ~128 cycles per output, and no
//        amount of instruction-level parallelism helps because each add needs
//        the previous one's result.
//
//     Hoisting the weight to a scalar and running over `ow` innermost turns the
//     inner loop into `out[ow] += in[ow * stride] * coefficient` - contiguous in
//     both arrays, and, crucially, ELEMENT-WISE rather than a reduction. Every
//     output element gets its own independent accumulation, so the chain is gone
//     and the loop is vectorisable.
//
// WHY THIS IS SAFE WITHOUT -ffast-math, which is the interesting part.
//     Vectorising a reduction requires reassociating it: `(a+b)+(c+d)` instead
//     of `((a+b)+c)+d`, which changes the result. That is why the compiler
//     refuses to do it without permission, and why this project refuses to give
//     that permission.
//
//     This transformation is different. For each individual output element the
//     additions still happen in exactly the original order - bias, then
//     (ic, kh, kw) - because the loop nest over those axes is unchanged. What
//     moved is which output element is being worked on, and that is not a
//     floating-point operation at all. So the compiler may vectorise across
//     `ow` freely: eight independent accumulators, each doing its own additions
//     in its own order. The parity and incremental tests confirm it - both stay
//     bit-identical after this change.
void conv2d_causal_time(const float* __restrict input, int in_channels, int in_height, int in_width,
                        const float* __restrict weight, const float* __restrict bias,
                        int out_channels, int kernel_height, int kernel_width, int stride_width,
                        float* __restrict output) {
    const int out_width = (in_width - kernel_width) / stride_width + 1;
    // Left pad on the time axis, so output position t sees inputs t-(kh-1)..t.
    const int time_pad = kernel_height - 1;
    const int plane = in_height * out_width;

    for (int oc = 0; oc < out_channels; ++oc) {
        float* __restrict out_plane = output + oc * plane;
        for (int i = 0; i < plane; ++i) out_plane[i] = bias[oc];

        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_height; ++kh) {
                for (int kw = 0; kw < kernel_width; ++kw) {
                    const float coefficient =
                        weight[((oc * in_channels + ic) * kernel_height + kh) * kernel_width + kw];
                    // Output rows before this one would read input above row 0.
                    // Those taps are the causal zero padding, so instead of a
                    // branch inside the loop the row range simply starts later.
                    const int first_row = (time_pad - kh > 0) ? (time_pad - kh) : 0;
                    for (int oh = first_row; oh < in_height; ++oh) {
                        const float* __restrict in_row =
                            input + (ic * in_height + (oh + kh - time_pad)) * in_width + kw;
                        float* __restrict out_row = out_plane + oh * out_width;
                        for (int ow = 0; ow < out_width; ++ow) {
                            out_row[ow] += in_row[ow * stride_width] * coefficient;
                        }
                    }
                }
            }
        }
    }
}

void conv1d_causal_dilated(const float* __restrict input, int in_channels, int length,
                           const float* __restrict weight, const float* __restrict bias,
                           int out_channels, int kernel, int dilation, float* __restrict output) {
    // PyTorch left-pads by (kernel - 1) * dilation, so tap k of output t reads
    // input position t + k*dilation - (kernel-1)*dilation. The last tap
    // (k = kernel-1) lands exactly on t, which is what makes it causal.
    const int offset = (kernel - 1) * dilation;

    // Same transformation as conv2d above, and it matters most here: the TCN is
    // 53% of this model's multiply-accumulates. The time axis is innermost, so
    // both `in_row` and `out_row` are walked contiguously and each output
    // timestep accumulates independently of its neighbours. See the note on
    // conv2d_causal_time for why this is vectorisable without -ffast-math.
    for (int oc = 0; oc < out_channels; ++oc) {
        float* __restrict out_row = output + oc * length;
        for (int t = 0; t < length; ++t) out_row[t] = bias[oc];

        for (int ic = 0; ic < in_channels; ++ic) {
            const float* __restrict in_row = input + ic * length;
            for (int k = 0; k < kernel; ++k) {
                // shift <= 0 always, so the outputs that would read before the
                // start of the sequence are exactly the first (-shift) of them.
                // A loop bound, not a branch on the hot path.
                const int shift = k * dilation - offset;
                const float coefficient = weight[(oc * in_channels + ic) * kernel + k];
                for (int t = -shift; t < length; ++t) {
                    out_row[t] += in_row[t + shift] * coefficient;
                }
            }
        }
    }
}

void maxpool2d_causal_time(const float* __restrict input, int channels, int height, int width,
                           int kernel_height, float* __restrict output) {
    const int time_pad = kernel_height - 1;

    for (int c = 0; c < channels; ++c) {
        for (int oh = 0; oh < height; ++oh) {
            for (int w = 0; w < width; ++w) {
                float best = -std::numeric_limits<float>::infinity();
                for (int kh = 0; kh < kernel_height; ++kh) {
                    const int ih = oh + kh - time_pad;
                    // A padded tap contributes a literal 0.0f - it is NOT
                    // skipped. PyTorch zero-pads before pooling, so a window
                    // overhanging the start competes against those zeros: where
                    // every real activation in reach is negative, the correct
                    // answer is 0.0, not the largest negative value. Skipping
                    // padded taps would disagree only in the first few
                    // timesteps and only for negative inputs, which is exactly
                    // the kind of bug that survives a careless test.
                    const float value = (ih < 0) ? 0.0f : input[(c * height + ih) * width + w];
                    if (value > best) best = value;
                }
                output[(c * height + oh) * width + w] = best;
            }
        }
    }
}

void leaky_relu(float* data, std::size_t count, float slope) {
    for (std::size_t i = 0; i < count; ++i) {
        if (data[i] < 0.0f) data[i] *= slope;
    }
}

void linear(const float* __restrict input, const float* __restrict weight,
            const float* __restrict bias, int in_features, int out_features,
            float* __restrict output) {
    for (int o = 0; o < out_features; ++o) {
        float sum = bias[o];
        for (int i = 0; i < in_features; ++i) {
            sum += input[i] * weight[o * in_features + i];
        }
        output[o] = sum;
    }
}

void add_into(float* __restrict accumulator, const float* __restrict addend, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) accumulator[i] += addend[i];
}

void softmax(const float* __restrict input, int count, float* __restrict output) {
    float largest = input[0];
    for (int i = 1; i < count; ++i) {
        if (input[i] > largest) largest = input[i];
    }
    // Subtracting the max before exp is the standard guard: without it a logit
    // of a few hundred overflows float32 and the whole distribution becomes NaN.
    float total = 0.0f;
    for (int i = 0; i < count; ++i) {
        output[i] = std::exp(input[i] - largest);
        total += output[i];
    }
    for (int i = 0; i < count; ++i) output[i] /= total;
}

}  // namespace tts
