// ops_test.cpp - each operator checked against a PyTorch fixture, on its own.
//
// WHY THIS EXISTS SEPARATELY FROM THE PARITY TEST.
//     The parity test says the network disagrees with PyTorch. It does not say
//     which of seven operators is responsible, and bisecting a 40-layer forward
//     pass by hand is exactly the tedium worth spending a file to avoid. These
//     cases are small enough to print in full and shaped to hit the same code
//     paths the model uses - the strides, dilations and kernel heights below
//     are the ones that actually occur in it.
//
//     The max-pool case is the one that earns its keep: its inputs are shifted
//     negative, so an implementation that *skips* padded taps rather than
//     treating them as zero passes every other test and fails this one.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "ops.hpp"
#include "weights.hpp"

namespace {

constexpr float kTolerance = 1e-5f;
int failures = 0;

// Compares against the PyTorch expectation and reports the worst element.
void check(const std::string& name, const std::vector<float>& actual, const tts::Tensor& expected) {
    if (actual.size() != expected.size()) {
        std::printf("  %-18s FAIL  produced %zu values, expected %zu\n", name.c_str(), actual.size(),
                    expected.size());
        ++failures;
        return;
    }
    double worst = 0.0;
    std::size_t worst_index = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double difference = std::fabs(static_cast<double>(actual[i]) - expected.data()[i]);
        if (difference > worst) {
            worst = difference;
            worst_index = i;
        }
    }
    const bool ok = worst < kTolerance;
    std::printf("  %-18s %s  max |diff| %.3e", name.c_str(), ok ? "pass" : "FAIL", worst);
    if (!ok) {
        std::printf("  (element %zu: got %.7f, expected %.7f)", worst_index, actual[worst_index],
                    expected.data()[worst_index]);
        ++failures;
    }
    std::printf("\n");
}

std::vector<float> make_output(const tts::Tensor& expected) {
    return std::vector<float>(expected.size(), 0.0f);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string path = (argc > 1) ? argv[1] : "artifacts/op_fixtures.ttsw";
    tts::WeightStore fixtures(path);
    std::printf("operator fixtures from %s\n\n", path.c_str());

    {   // Spatial convolution: [2, 7, 8] -> [3, 7, 4] with a width stride of 2.
        const tts::Tensor& input = fixtures.get("conv2d_spatial.input", {2, 7, 8});
        const tts::Tensor& weight = fixtures.get("conv2d_spatial.weight", {3, 2, 1, 2});
        const tts::Tensor& bias = fixtures.get("conv2d_spatial.bias", {3});
        const tts::Tensor& expected = fixtures.get("conv2d_spatial.expected", {3, 7, 4});
        std::vector<float> output = make_output(expected);
        tts::conv2d_causal_time(input.data(), 2, 7, 8, weight.data(), bias.data(), 3, 1, 2, 2,
                                output.data());
        check("conv2d spatial", output, expected);
    }

    {   // Temporal convolution of height 4 with causal padding.
        const tts::Tensor& input = fixtures.get("conv2d_temporal.input", {3, 7, 4});
        const tts::Tensor& weight = fixtures.get("conv2d_temporal.weight", {3, 3, 4, 1});
        const tts::Tensor& bias = fixtures.get("conv2d_temporal.bias", {3});
        const tts::Tensor& expected = fixtures.get("conv2d_temporal.expected", {3, 7, 4});
        std::vector<float> output = make_output(expected);
        tts::conv2d_causal_time(input.data(), 3, 7, 4, weight.data(), bias.data(), 3, 4, 1, 1,
                                output.data());
        check("conv2d temporal", output, expected);
    }

    {   // Height-5 kernel, the Inception block's widest branch.
        const tts::Tensor& input = fixtures.get("conv2d_five.input", {2, 9, 1});
        const tts::Tensor& weight = fixtures.get("conv2d_five.weight", {4, 2, 5, 1});
        const tts::Tensor& bias = fixtures.get("conv2d_five.bias", {4});
        const tts::Tensor& expected = fixtures.get("conv2d_five.expected", {4, 9, 1});
        std::vector<float> output = make_output(expected);
        tts::conv2d_causal_time(input.data(), 2, 9, 1, weight.data(), bias.data(), 4, 5, 1, 1,
                                output.data());
        check("conv2d kernel 5", output, expected);
    }

    {   // Dilated causal 1-D convolution at the deepest dilation the TCN uses.
        const tts::Tensor& input = fixtures.get("conv1d_dilated.input", {3, 12});
        const tts::Tensor& weight = fixtures.get("conv1d_dilated.weight", {4, 3, 3});
        const tts::Tensor& bias = fixtures.get("conv1d_dilated.bias", {4});
        const tts::Tensor& expected = fixtures.get("conv1d_dilated.expected", {4, 12});
        std::vector<float> output = make_output(expected);
        tts::conv1d_causal_dilated(input.data(), 3, 12, weight.data(), bias.data(), 4, 3, 4,
                                   output.data());
        check("conv1d dilation 4", output, expected);
    }

    {   // The case that catches zero-padding done wrong. All inputs negative.
        const tts::Tensor& input = fixtures.get("maxpool_negative.input", {2, 7, 3});
        const tts::Tensor& expected = fixtures.get("maxpool_negative.expected", {2, 7, 3});
        std::vector<float> output = make_output(expected);
        tts::maxpool2d_causal_time(input.data(), 2, 7, 3, 3, output.data());
        check("maxpool negative", output, expected);
    }

    {
        const tts::Tensor& input = fixtures.get("leaky_relu.input", {10});
        const tts::Tensor& expected = fixtures.get("leaky_relu.expected", {10});
        std::vector<float> output(input.data(), input.data() + input.size());
        tts::leaky_relu(output.data(), output.size());
        check("leaky relu", output, expected);
    }

    {
        const tts::Tensor& input = fixtures.get("linear.input", {6});
        const tts::Tensor& weight = fixtures.get("linear.weight", {4, 6});
        const tts::Tensor& bias = fixtures.get("linear.bias", {4});
        const tts::Tensor& expected = fixtures.get("linear.expected", {4});
        std::vector<float> output = make_output(expected);
        tts::linear(input.data(), weight.data(), bias.data(), 6, 4, output.data());
        check("linear", output, expected);
    }

    {   // A logit of 12 against one of -11 would overflow without the
        // max-subtraction guard in softmax.
        const tts::Tensor& input = fixtures.get("softmax.input", {5});
        const tts::Tensor& expected = fixtures.get("softmax.expected", {5});
        std::vector<float> output = make_output(expected);
        tts::softmax(input.data(), 5, output.data());
        check("softmax", output, expected);
    }

    std::printf("\n");
    if (failures == 0) {
        std::printf("OPS PASS: every operator matches PyTorch within %.0e\n",
                    static_cast<double>(kTolerance));
        return 0;
    }
    std::printf("OPS FAIL: %d operator(s) disagree with PyTorch\n", failures);
    return 1;
}
