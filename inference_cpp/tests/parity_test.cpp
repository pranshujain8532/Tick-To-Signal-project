// parity_test.cpp - the mandatory C++/PyTorch parity test.
//
// WHAT
//     Replays 1,000 (input, output) fixtures dumped from PyTorch through the
//     hand-written forward pass and asserts the largest absolute logit
//     difference stays under 1e-4. Prints the actual maximum either way.
//
// WHY FIXTURE-DRIVEN TESTING RATHER THAN HAND-WRITTEN EXPECTATIONS.
//     There is no useful way to write down, by hand, what a 32,155-parameter
//     network should output for a given input. Any expectation this file could
//     contain would have been produced by running *something*, so the honest
//     move is to say what that something was and pin it: the fixtures come from
//     the same PyTorch model object that exported the weights, in the same
//     process, so weights and expectations cannot drift apart.
//
//     It also makes the test a genuine oracle rather than a change detector. If
//     someone rewrites conv2d with SIMD in Stage 7b and gets the accumulation
//     order subtly wrong, this fails - and it fails against the definition of
//     correct (what PyTorch does), not against a previous run of the same
//     possibly-wrong code.
//
//     The fixtures are real held-out windows, not random noise. Stage 6
//     measured the feature distribution as heavy-tailed (bulk near +/-1.4,
//     extremes near +/-22), and large magnitudes are exactly where two
//     implementations first disagree numerically.
//
// WHY 1e-4 AND NOT BIT-EXACT.
//     Bit-exactness between PyTorch and a hand-written loop is not achievable
//     and not worth chasing: PyTorch's convolutions dispatch to vectorised
//     kernels that accumulate in a different order, and float addition is not
//     associative. The question that matters is whether the difference is
//     rounding or a bug, and rounding on this network sits three orders of
//     magnitude below 1e-4 - so the threshold discriminates cleanly while
//     leaving room for a legitimately different summation order.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "fixture_io.hpp"
#include "model.hpp"

namespace {
constexpr float kTolerance = 1e-4f;
}  // namespace

int run(const std::string& weight_path, const std::string& fixture_path) {
    tts::StudentModel model(weight_path);
    std::printf("loaded %zu weight values, %zu bytes of activation buffers\n",
                model.weight_values(), model.activation_bytes());

    const tts::FixtureSet fixtures = tts::load_fixtures(fixture_path);
    if (fixtures.window != tts::arch::kWindow || fixtures.features != tts::arch::kFeatures ||
        fixtures.classes != tts::arch::kClasses) {
        std::printf("fixture shape [%u, %u] -> %u disagrees with the compiled model\n",
                    fixtures.window, fixtures.features, fixtures.classes);
        return 1;
    }
    std::printf("replaying %u fixtures from %s\n\n", fixtures.count, fixture_path.c_str());

    std::vector<float> actual(fixtures.classes);

    double worst_absolute = 0.0;
    double sum_absolute = 0.0;
    double worst_relative = 0.0;
    std::uint32_t worst_index = 0;
    std::uint32_t argmax_matches = 0;
    float largest_logit = 0.0f;

    for (std::uint32_t index = 0; index < fixtures.count; ++index) {
        const float* expected = fixtures.output(index);
        model.forward(fixtures.input(index), actual.data());

        double fixture_worst = 0.0;
        int expected_argmax = 0;
        int actual_argmax = 0;
        for (std::uint32_t c = 0; c < fixtures.classes; ++c) {
            const double difference = std::fabs(static_cast<double>(actual[c]) - expected[c]);
            if (difference > fixture_worst) fixture_worst = difference;
            if (std::fabs(expected[c]) > largest_logit) largest_logit = std::fabs(expected[c]);
            if (expected[c] > expected[expected_argmax]) expected_argmax = static_cast<int>(c);
            if (actual[c] > actual[actual_argmax]) actual_argmax = static_cast<int>(c);
        }
        sum_absolute += fixture_worst;
        if (fixture_worst > worst_absolute) {
            worst_absolute = fixture_worst;
            worst_index = index;
        }
        if (expected_argmax == actual_argmax) ++argmax_matches;
    }

    worst_relative = (largest_logit > 0.0f) ? worst_absolute / largest_logit : 0.0;
    const double mean_absolute = sum_absolute / fixtures.count;
    const double agreement = static_cast<double>(argmax_matches) / fixtures.count;

    std::printf("max |diff|      : %.6e   (worst at fixture %u)\n", worst_absolute, worst_index);
    std::printf("mean |diff|     : %.6e\n", mean_absolute);
    std::printf("largest |logit| : %.4f\n", largest_logit);
    std::printf("relative        : %.6e\n", worst_relative);
    std::printf("argmax agreement: %.4f  (%u of %u)\n", agreement, argmax_matches, fixtures.count);
    std::printf("tolerance       : %.1e\n\n", static_cast<double>(kTolerance));

    // Argmax agreement is checked as well as the numeric difference because
    // they answer different questions: the difference is the numerical claim,
    // while a flipped argmax is the only thing that changes a prediction. A
    // failure of either is a failure.
    const bool numeric_ok = worst_absolute < kTolerance;
    const bool decisions_ok = argmax_matches == fixtures.count;
    if (numeric_ok && decisions_ok) {
        std::printf("PARITY PASS: C++ matches PyTorch on all %u fixtures\n", fixtures.count);
        return 0;
    }
    if (!numeric_ok) {
        std::printf("PARITY FAIL: max |diff| %.6e exceeds tolerance %.1e\n", worst_absolute,
                    static_cast<double>(kTolerance));
    }
    if (!decisions_ok) {
        std::printf("PARITY FAIL: %u of %u fixtures disagree on argmax\n",
                    fixtures.count - argmax_matches, fixtures.count);
    }
    return 1;
}

int main(int argc, char** argv) {
    const std::string weight_path = (argc > 1) ? argv[1] : "artifacts/student_weights.ttsw";
    const std::string fixture_path = (argc > 2) ? argv[2] : "artifacts/student_fixtures.ttsf";
    try {
        return run(weight_path, fixture_path);
    } catch (const std::exception& error) {
        std::printf("PARITY FAIL: %s\n", error.what());
        return 1;
    }
}
