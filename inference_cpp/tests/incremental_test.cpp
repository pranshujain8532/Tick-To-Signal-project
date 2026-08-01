// incremental_test.cpp - the streaming path must equal the full recompute.
//
// WHAT
//     Runs IncrementalModel and StudentModel over the same real data two ways
//     and requires their logits to agree:
//
//       1. COLD START - prime() the streaming model with a fixture window and
//          compare against forward() on that window. This exercises the
//          zero-padding path, where the rings still hold their reset values.
//       2. STREAMING - concatenate four fixture windows into 400 consecutive
//          ticks, push them one at a time, and compare every tick from the
//          100th onwards against forward() on the sliding window ending there.
//          This is the check that matters: it runs long enough for every ring
//          to wrap (the widest holds 17 columns) and it compares against a
//          fresh full recompute each time, so accumulated state cannot drift
//          without being caught.
//
// WHY THIS IS A REAL ORACLE AND NOT A CHANGE DETECTOR
//     StudentModel is not "the other implementation" - it is the one already
//     pinned to PyTorch by parity_test on 1,000 fixtures. Chaining the two
//     tests means the streaming path is transitively verified against PyTorch,
//     which is the only definition of correct that counts. Neither test alone
//     would do: parity_test never calls push_tick, and this test would happily
//     agree with a StudentModel that had drifted.
//
// WHY EQUALITY IS EXPECTED TO BE EXACT
//     The column operators in incremental.cpp accumulate in the same order as
//     their full-window counterparts in ops.cpp, and where the full version
//     skips a zero-padded tap the column version multiplies by the ring's zero
//     and adds - which is exact in IEEE-754. So the correct expectation is
//     bit-identical output, and the test reports how many comparisons actually
//     were. The tolerance is kept at 1e-6 rather than zero only so that a
//     future compiler that contracts a multiply-add in one path but not the
//     other reports a number instead of an unexplained failure; if the exact
//     count ever stops being 100%, that is worth understanding before the
//     tolerance is leaned on.
//
// WHY THE HORIZONS AGREE AT ALL
//     After 400 ticks the streaming model has state reflecting the whole
//     stream, while forward() sees exactly the last 100 rows and zeros before
//     them. Those are different inputs. They give the same answer because the
//     receptive field is 83 and the window is 100: nothing before t-82 can
//     reach the output. If ml/model.py ever grows the receptive field past 100,
//     this test is what will fail, and it should - the equivalence would no
//     longer hold.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "fixture_io.hpp"
#include "incremental.hpp"
#include "model.hpp"

namespace {

constexpr float kTolerance = 1e-6f;
constexpr std::uint32_t kColdStartWindows = 32;
constexpr std::uint32_t kStreamWindows = 4;

struct Comparison {
    double worst = 0.0;
    std::size_t identical = 0;
    std::size_t compared = 0;
};

void accumulate(Comparison& result, const float* left, const float* right, int count) {
    double worst = 0.0;
    bool same_bits = true;
    for (int i = 0; i < count; ++i) {
        const double difference = std::fabs(static_cast<double>(left[i]) - right[i]);
        if (difference > worst) worst = difference;
        if (left[i] != right[i]) same_bits = false;
    }
    if (worst > result.worst) result.worst = worst;
    if (same_bits) ++result.identical;
    ++result.compared;
}

void report(const char* label, const Comparison& result) {
    const double exact_fraction =
        result.compared > 0 ? static_cast<double>(result.identical) / result.compared : 0.0;
    std::printf("%-14s %5zu comparisons   max |diff| %.3e   bit-identical %5.1f%% (%zu)\n", label,
                result.compared, result.worst, 100.0 * exact_fraction, result.identical);
}

}  // namespace

int run(const std::string& weight_path, const std::string& fixture_path) {
    tts::StudentModel full(weight_path);
    tts::IncrementalModel streaming(weight_path);
    const tts::FixtureSet fixtures = tts::load_fixtures(fixture_path, kColdStartWindows);
    if (fixtures.window != tts::arch::kWindow || fixtures.features != tts::arch::kFeatures) {
        std::printf("fixture shape [%u, %u] disagrees with the compiled model\n", fixtures.window,
                    fixtures.features);
        return 1;
    }

    std::printf("full path  : %zu bytes of activation buffers\n", full.activation_bytes());
    std::printf("streaming  : %zu bytes of ring + scratch state\n\n", streaming.state_bytes());

    const int classes = static_cast<int>(fixtures.classes);
    std::vector<float> from_full(classes);
    std::vector<float> from_stream(classes);

    // --- 1. cold start -------------------------------------------------------
    Comparison cold;
    for (std::uint32_t index = 0; index < kColdStartWindows && index < fixtures.count; ++index) {
        full.forward(fixtures.input(index), from_full.data());
        streaming.prime(fixtures.input(index), from_stream.data());
        accumulate(cold, from_full.data(), from_stream.data(), classes);
    }
    report("cold start", cold);

    // --- 2. streaming --------------------------------------------------------
    // The fixtures are already contiguous in memory, so the first four windows
    // are 400 consecutive rows without a copy. They are four unrelated windows
    // rather than one continuous capture, which makes the joins large jumps -
    // useful here, because a stale ring slot shows up loudest after a jump.
    const std::size_t total_rows = static_cast<std::size_t>(kStreamWindows) * fixtures.window;
    const float* stream = fixtures.input(0);

    Comparison streamed;
    std::size_t moved = 0;
    std::vector<float> previous(classes, 0.0f);
    streaming.reset();
    for (std::size_t t = 0; t < total_rows; ++t) {
        streaming.push_tick(stream + t * fixtures.features, from_stream.data());
        if (t + 1 < fixtures.window) continue;  // not yet a full window to compare against

        const std::size_t window_start = t + 1 - fixtures.window;
        full.forward(stream + window_start * fixtures.features, from_full.data());
        accumulate(streamed, from_full.data(), from_stream.data(), classes);

        // Vacuity guard: a push_tick that ignored its argument would agree with
        // itself forever and pass everything above. Real windows must produce
        // moving logits.
        if (streamed.compared > 1 && from_stream != previous) ++moved;
        previous = from_stream;
    }
    report("streaming", streamed);
    std::printf("%-14s %5zu of %zu ticks changed the logits\n\n", "responsive", moved,
                streamed.compared - 1);

    const bool cold_ok = cold.compared > 0 && cold.worst < kTolerance;
    const bool stream_ok = streamed.compared > 0 && streamed.worst < kTolerance;
    const bool responsive = moved > (streamed.compared - 1) / 2;

    if (cold_ok && stream_ok && responsive) {
        std::printf("INCREMENTAL PASS: streaming output equals the full recompute\n");
        return 0;
    }
    if (!cold_ok) std::printf("INCREMENTAL FAIL: cold start diverges (%.3e)\n", cold.worst);
    if (!stream_ok) std::printf("INCREMENTAL FAIL: streaming diverges (%.3e)\n", streamed.worst);
    if (!responsive) {
        std::printf("INCREMENTAL FAIL: logits barely moved (%zu of %zu) - the test is vacuous\n",
                    moved, streamed.compared - 1);
    }
    return 1;
}

int main(int argc, char** argv) {
    const std::string weight_path = (argc > 1) ? argv[1] : "artifacts/student_weights.ttsw";
    const std::string fixture_path = (argc > 2) ? argv[2] : "artifacts/student_fixtures.ttsf";
    try {
        return run(weight_path, fixture_path);
    } catch (const std::exception& error) {
        std::printf("INCREMENTAL FAIL: %s\n", error.what());
        return 1;
    }
}
