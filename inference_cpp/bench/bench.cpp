// bench.cpp - the latency harness for the hand-written inference path.
//
// WHAT
//     Times one forward pass, a million times, and writes the whole latency
//     distribution to JSON under benchmarks/. Two variants: `full` recomputes a
//     [100, 40] window through StudentModel, `incremental` advances one tick
//     through IncrementalModel.
//
// WHAT "TICK-TO-SIGNAL" MEANS HERE - the boundary, stated rather than implied.
//
//     INCLUDED in the timed region:
//       * the model forward pass, from a prepared [40] feature column (or a
//         [100, 40] window) to three logits;
//       * for the `full` variant, the 16 KB input memcpy that
//         StudentModel::forward performs. The incremental variant has no such
//         copy, and that difference is real, not an accounting trick.
//
//     EXCLUDED, deliberately and explicitly:
//       * network round-trip to the exchange. It is 4-5 orders of magnitude
//         larger than anything measured here and would swamp the signal; it is
//         also not what this code is responsible for. Quoting a latency that
//         includes it would say nothing about the model.
//       * order-book maintenance (data_engine/book.py's apply_diff).
//       * FEATURE CONSTRUCTION. This is the honest gap: ml/features.py builds
//         the mid-relative prices, the log1p sizes and the causal rolling
//         z-score, and that work is still Python-side. It is genuinely part of
//         tick-to-signal and it is NOT in these numbers. Its cost in C++ is
//         TODO(measure) - see docs/benchmark_methodology.md. The arithmetic is
//         O(40) per tick against the forward pass's tens of thousands of
//         multiply-accumulates, so it is expected to be small, but "expected to
//         be small" is not a measurement and is not reported as one.
//       * softmax. The parity fixtures compare logits, so the timed path stops
//           where the verified path stops.
//
// WHY REAL RECORDED INPUTS AND NOT ZEROS
//     A harness fed zeros measures a machine that does not exist:
//       * zeros never produce denormals, and a denormal multiply can cost
//         over a hundred cycles on x86 when it is not flushed to zero. This
//         build does not enable flush-to-zero (that would need -ffast-math,
//         which is refused project-wide), so denormals are a real tail risk and
//         must be in the input distribution if they are in production.
//       * every data-dependent branch takes the same direction every time.
//         `leaky_relu` branches on sign and `maxpool` branches on comparison;
//         with zeros the predictor is right 100% of the time, which it will
//         not be on real books.
//       * the numbers stop meaning anything for the tails, which are the whole
//         point of measuring p99.9.
//     The inputs here are the same real held-out windows the parity test uses:
//     Stage 6 measured that distribution as heavy-tailed, bulk near +/-1.4 with
//     extremes near +/-22.
//
// WHY BOTH INPUT MODES ARE MEASURED
//     `stream` walks a rotating ~1 MB working set of real windows, so the input
//     is cold-ish every iteration. `hot` replays one input forever, so it stays
//     in L1. Neither is "the" answer: production sits between them, because the
//     feature updater writes the new column immediately before the forward pass
//     (hot) while the model weights and activations are shared with everything
//     else on the box (cold). Measuring both bounds the effect instead of
//     assuming it away.
//
// WHY A CLOSED LOOP IS NOT COORDINATED OMISSION HERE
//     Coordinated omission is what happens when a load generator that should
//     issue a request every T stalls because the previous request was slow, and
//     so never records the queueing delay its own slowness caused. It is a
//     property of OPEN-loop systems measured with a closed-loop harness.
//     This path has no queue: it is a synchronous function call made by the
//     thread that just received a book update, and if it is slow the next tick
//     simply has not arrived yet. There is no request that "should" have been
//     issued and was not, so there is nothing to omit. What this harness
//     therefore measures is service time, and it says so - it is NOT a claim
//     about end-to-end latency under a tick arrival rate. If ticks ever arrive
//     faster than the model can serve them, the right instrument is an
//     open-loop harness with a fixed arrival schedule, and that is a different
//     program.
//
// CORE PINNING
//     The thread is pinned to one core before warmup, because a migration
//     mid-run empties L1/L2 and shows up as a spurious tail sample. On Windows
//     that is SetThreadAffinityMask; on Linux the same is achieved externally
//     with `taskset -c 3 ./bench ...` and internally with pthread_setaffinity_np,
//     both of which are implemented below. The mask actually applied is recorded
//     in the JSON, so a run where pinning silently failed is identifiable -
//     Stage 6 had exactly that bug in its Python harness, where a ctypes call
//     returned success and pinned nothing.
//
// THERMAL STATE
//     A fixed-work calibration kernel runs before and after the timed loop and
//     both durations go into the JSON. If the CPU down-clocked during the run,
//     the same arithmetic takes measurably longer the second time and the ratio
//     says by how much. This needs no OS-specific temperature API and answers
//     the question that actually matters: did the machine that produced the
//     tail run at the same speed as the machine that produced the median?

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "bench_clock.hpp"  // brings in windows.h on Windows
#include "bench_report.hpp"
#include "fixture_io.hpp"
#include "incremental.hpp"
#include "model.hpp"

#ifndef _WIN32
#include <pthread.h>
#include <sched.h>
#endif

namespace {

// 64 real windows is a ~1 MB rotating working set: larger than L2 on the parts
// this targets, small enough that the benchmark is not simply a memory test.
constexpr std::uint32_t kWorkingSetWindows = 64;

struct Options {
    std::string weight_path = "artifacts/student_weights.ttsw";
    std::string fixture_path = "artifacts/student_fixtures.ttsf";
    std::string output_path;
    std::string variant = "incremental";    // "full" or "incremental"
    std::string input_mode = "stream";      // "stream" or "hot"
    std::string label;
    std::uint64_t warmup = 10000;
    std::uint64_t iterations = 1000000;
    double settle_seconds = 3.0;
    int core = 0;
};

// ------------------------------------------------------------------ machinery

bool pin_to_core(int core, std::uint64_t& applied_mask) {
    applied_mask = static_cast<std::uint64_t>(1) << core;
#ifdef _WIN32
    const DWORD_PTR mask = static_cast<DWORD_PTR>(1) << core;
    return SetThreadAffinityMask(GetCurrentThread(), mask) != 0;
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#endif
}

// Raise scheduling priority so the timed loop is preempted less often.
//
// WHY: with default priority, repeated runs of the identical binary reported
// p99 of 124, 187 and 143 us while p50 held at 31.8, 31.9 and 31.9. The median
// is the model; that tail spread is other processes taking the core. Raising
// priority removes some of it, and what remains is reported rather than hidden.
//
// NOT realtime priority. REALTIME_PRIORITY_CLASS can starve the input and
// storage stacks badly enough to require a hard reset, which is too high a
// price for a slightly cleaner histogram.
bool raise_priority() {
#ifdef _WIN32
    return SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS) != 0;
#else
    // On Linux the equivalent needs privileges the benchmark should not assume;
    // `chrt`/`nice` from the shell is the documented route. See
    // docs/benchmark_methodology.md.
    return false;
#endif
}

// The smallest non-zero gap two back-to-back reads ever report. A clock's
// nominal period is a compile-time ratio and the standard does not require it to
// describe the hardware: on this toolchain std::chrono::steady_clock advertises
// one nanosecond and delivers about a millisecond, which is exactly what this
// function exists to expose.
std::uint64_t measured_tick_ns(const MonotonicClock& clock) {
    std::int64_t smallest = 0;
    for (int i = 0; i < 200000; ++i) {
        const std::int64_t first = clock.now();
        const std::int64_t second = clock.now();
        const std::int64_t delta = second - first;
        if (delta > 0 && (smallest == 0 || delta < smallest)) smallest = delta;
    }
    return static_cast<std::uint64_t>(clock.to_nanoseconds(smallest));
}

// Kept because it is the finding that motivated bench_clock.hpp. Recording it
// in every run means the JSON carries the evidence for why the harness does not
// use the standard clock, instead of a comment asserting it.
std::uint64_t measured_steady_clock_tick_ns() {
    std::int64_t smallest = 0;
    for (int i = 0; i < 200000; ++i) {
        const auto first = std::chrono::steady_clock::now();
        const auto second = std::chrono::steady_clock::now();
        const std::int64_t delta =
            std::chrono::duration_cast<std::chrono::nanoseconds>(second - first).count();
        if (delta > 0 && (smallest == 0 || delta < smallest)) smallest = delta;
    }
    return static_cast<std::uint64_t>(smallest);
}

// What a pair of clock reads costs: the noise floor under every sample below.
// If this were comparable to the kernel being timed, the harness would be
// measuring itself and would have to be restructured to time batches instead.
std::uint64_t timer_overhead_ns(const MonotonicClock& clock) {
    std::vector<std::int64_t> samples;
    samples.reserve(100000);
    for (int i = 0; i < 100000; ++i) {
        const std::int64_t start = clock.now();
        const std::int64_t end = clock.now();
        samples.push_back(end - start);
    }
    std::sort(samples.begin(), samples.end());
    return static_cast<std::uint64_t>(clock.to_nanoseconds(samples[samples.size() / 2]));
}

// A fixed quantity of arithmetic, timed identically before and after the run.
//
// The recurrence is STRICTLY SERIAL on purpose: each step consumes the previous
// step's result, so the compiler cannot vectorise it, cannot reassociate it, and
// cannot hoist it out of the round loop. That is the whole requirement for a
// fixed-work instrument, and the first version failed it.
//
//     The original was `total += buffer[i] * c`. The inner sum is identical on
//     every round, so it is loop-invariant, and at -O3 -march=native GCC
//     exploited that: the same nominal 20.5M operations took 0.0201 s in the
//     release-O3 build and 0.0016 s in the native build. A "fixed workload"
//     that moves 12x with a compiler flag measures the compiler, not the
//     machine.
//
// The 0.9999/0.0001 weights keep `total` bounded. An unbounded recurrence
// reaches infinity partway through, and arithmetic on infinities is not
// necessarily the same speed as arithmetic on normals - which would put a
// second variable inside the instrument.
//
// SCOPE, stated because it is easy to over-read: this compares the machine to
// ITSELF, within one run of one binary. It is not flag-invariant and cannot be -
// x87 and SSE scalar arithmetic are different instruction streams, and after the
// fix above the same source still takes 0.047 s under release-O3 and 0.027 s
// under release-native. Comparing the absolute value across builds says nothing.
// Comparing before against after, inside one run, is the entire purpose.
double calibration_seconds(const MonotonicClock& clock, double& sink) {
    constexpr int kValues = 1024;
    constexpr int kRounds = 20000;
    static float buffer[kValues];
    for (int i = 0; i < kValues; ++i) buffer[i] = 1.0f + 1e-6f * static_cast<float>(i);

    const std::int64_t start = clock.now();
    float total = 1.0f;
    for (int round = 0; round < kRounds; ++round) {
        for (int i = 0; i < kValues; ++i) total = total * 0.9999f + buffer[i] * 0.0001f;
    }
    const std::int64_t end = clock.now();
    sink += static_cast<double>(total);
    return clock.to_seconds(end - start);
}

// Burn the CPU before measuring anything, so every run starts from the same
// power state.
//
// WHY, with the numbers that forced it: on the laptop this was developed on the
// identical binary reports p50 22.7 us when the machine has been idle and p50
// 31.8 us once it has been busy for a few seconds - a 1.40x spread caused
// entirely by boost clocks decaying, and the calibration kernel independently
// measured the same 1.40x. A "before" taken in boost compared against an
// "after" taken in steady state would manufacture or erase an improvement of
// that size, which is larger than most of the optimisations in Stage 7b.
//
// This makes runs comparable TO EACH OTHER on one machine. It does not make
// them comparable to a different machine, and no amount of warmup would.
void settle(const MonotonicClock& clock, double seconds, double& sink) {
    if (seconds <= 0.0) return;
    const std::int64_t deadline =
        clock.now() + static_cast<std::int64_t>(seconds * clock.ticks_per_second());
    while (clock.now() < deadline) calibration_seconds(clock, sink);
}

// Matches the filename convention the Python benchmarks already use, e.g.
// benchmarks/python_variants_20260731T175351Z.json. `std::gmtime` returns a
// pointer to a shared static buffer, which would be a problem in threaded code;
// this is called twice, at startup and at exit, on one thread. The reentrant
// spellings are not portable between MinGW and glibc, so the simple one wins.
std::string utc_timestamp() {
    const std::time_t now = std::time(nullptr);
    const std::tm* parts = std::gmtime(&now);
    char text[32];
    std::strftime(text, sizeof(text), "%Y%m%dT%H%M%SZ", parts);
    return text;
}

// --------------------------------------------------------------- the two loops

// Both loops store RAW TICKS, not nanoseconds: converting a performance-counter
// tick to a duration costs a divide, and a divide between the two clock reads
// would be harness arithmetic charged to the model. The conversion happens once
// per sample after the run, in `run`.
//
// Each returns a sink value so the optimiser cannot delete the calls it is
// being asked to time.
double time_full(tts::StudentModel& model, const tts::FixtureSet& fixtures, const Options& options,
                 const MonotonicClock& clock, std::vector<std::uint32_t>& ticks) {
    const std::uint32_t pool = std::min(kWorkingSetWindows, fixtures.count);
    float logits[tts::arch::kClasses];
    double sink = 0.0;

    for (std::uint64_t i = 0; i < options.warmup; ++i) {
        model.forward(fixtures.input(options.input_mode == "hot" ? 0 : i % pool), logits);
        sink += logits[0];
    }
    for (std::uint64_t i = 0; i < options.iterations; ++i) {
        // Chosen outside the timed region: selecting the input is the harness's
        // work, not the model's.
        const float* input = fixtures.input(options.input_mode == "hot" ? 0 : i % pool);
        const std::int64_t start = clock.now();
        model.forward(input, logits);
        const std::int64_t end = clock.now();
        ticks[i] = static_cast<std::uint32_t>(end - start);
        sink += logits[0];
    }
    return sink;
}

double time_incremental(tts::IncrementalModel& model, const tts::FixtureSet& fixtures,
                        const Options& options, const MonotonicClock& clock,
                        std::vector<std::uint32_t>& ticks) {
    // The fixtures are contiguous, so the first 64 windows are 6,400 consecutive
    // [40] columns - a real stream to walk, wrapping at the end.
    const std::uint32_t pool = std::min(kWorkingSetWindows, fixtures.count);
    const std::size_t rows = static_cast<std::size_t>(pool) * fixtures.window;
    const float* stream = fixtures.input(0);
    float logits[tts::arch::kClasses];
    double sink = 0.0;

    model.reset();
    for (std::uint64_t i = 0; i < options.warmup; ++i) {
        model.push_tick(stream + (options.input_mode == "hot" ? 0 : i % rows) * fixtures.features,
                        logits);
        sink += logits[0];
    }
    for (std::uint64_t i = 0; i < options.iterations; ++i) {
        const std::size_t row = (options.input_mode == "hot") ? 0 : i % rows;
        const float* column = stream + row * fixtures.features;
        const std::int64_t start = clock.now();
        model.push_tick(column, logits);
        const std::int64_t end = clock.now();
        ticks[i] = static_cast<std::uint32_t>(end - start);
        sink += logits[0];
    }
    return sink;
}

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        const bool has_value = (i + 1 < argc);
        if (flag == "--weights" && has_value) options.weight_path = argv[++i];
        else if (flag == "--fixtures" && has_value) options.fixture_path = argv[++i];
        else if (flag == "--out" && has_value) options.output_path = argv[++i];
        else if (flag == "--variant" && has_value) options.variant = argv[++i];
        else if (flag == "--input-mode" && has_value) options.input_mode = argv[++i];
        else if (flag == "--label" && has_value) options.label = argv[++i];
        else if (flag == "--warmup" && has_value) options.warmup = std::strtoull(argv[++i], nullptr, 10);
        else if (flag == "--iterations" && has_value) options.iterations = std::strtoull(argv[++i], nullptr, 10);
        else if (flag == "--settle" && has_value) options.settle_seconds = std::atof(argv[++i]);
        else if (flag == "--core" && has_value) options.core = std::atoi(argv[++i]);
        else {
            std::printf("unknown or incomplete option: %s\n", flag.c_str());
            std::exit(2);
        }
    }
    if (options.label.empty()) options.label = "cpp_" + options.variant;
    if (options.output_path.empty()) {
        options.output_path = "../benchmarks/" + options.label + "_" + utc_timestamp() + ".json";
    }
    return options;
}

int run(const Options& options) {
    std::uint64_t affinity = 0;
    const bool pinned = pin_to_core(options.core, affinity);
    const bool elevated = raise_priority();
    std::printf("pinned to core %d: %s   high priority: %s\n", options.core,
                pinned ? "yes" : "NO - results are noisy", elevated ? "yes" : "no");

    // Clock characterisation FIRST, before any claim rests on it. This is the
    // check that caught std::chrono::steady_clock resolving at ~1 ms on this
    // toolchain; both clocks are measured every run so the comparison is in the
    // record rather than in a comment.
    const MonotonicClock clock;
    const std::uint64_t tick_ns = measured_tick_ns(clock);
    const std::uint64_t overhead_ns = timer_overhead_ns(clock);
    const std::uint64_t steady_tick_ns = measured_steady_clock_tick_ns();
    std::printf("clock: QPC/CLOCK_MONOTONIC at %lld ticks/s\n",
                static_cast<long long>(clock.ticks_per_second()));
    std::printf("       measured tick %llu ns, timer pair overhead %llu ns (median)\n",
                static_cast<unsigned long long>(tick_ns),
                static_cast<unsigned long long>(overhead_ns));
    std::printf("       std::chrono::steady_clock claims %ld/%ld s, resolves %llu ns\n",
                static_cast<long>(std::chrono::steady_clock::period::num),
                static_cast<long>(std::chrono::steady_clock::period::den),
                static_cast<unsigned long long>(steady_tick_ns));

    const tts::FixtureSet fixtures = tts::load_fixtures(options.fixture_path, kWorkingSetWindows);
    if (fixtures.window != tts::arch::kWindow || fixtures.features != tts::arch::kFeatures) {
        std::printf("fixture shape disagrees with the compiled model\n");
        return 1;
    }

    // Sized and touched before warmup so the timed loop never takes a page
    // fault writing a sample - that fault would be recorded as model latency.
    std::vector<std::uint32_t> samples(options.iterations, 0);

    double sink = 0.0;
    std::printf("settling for %.1f s so boost clocks are not part of the result\n",
                options.settle_seconds);
    settle(clock, options.settle_seconds, sink);
    const double calibration_before = calibration_seconds(clock, sink);

    std::printf("\nvariant %s, input-mode %s, %llu warmup + %llu timed iterations\n",
                options.variant.c_str(), options.input_mode.c_str(),
                static_cast<unsigned long long>(options.warmup),
                static_cast<unsigned long long>(options.iterations));

    const std::int64_t wall_start = clock.now();
    if (options.variant == "full") {
        tts::StudentModel model(options.weight_path);
        sink += time_full(model, fixtures, options, clock, samples);
    } else if (options.variant == "incremental") {
        tts::IncrementalModel model(options.weight_path);
        sink += time_incremental(model, fixtures, options, clock, samples);
    } else {
        std::printf("unknown variant: %s (expected full or incremental)\n", options.variant.c_str());
        return 1;
    }
    const double wall_seconds = clock.to_seconds(clock.now() - wall_start);
    const double calibration_after = calibration_seconds(clock, sink);

    // Ticks become nanoseconds only now that nothing is being timed.
    for (std::uint64_t i = 0; i < options.iterations; ++i) {
        samples[i] = static_cast<std::uint32_t>(clock.to_nanoseconds(samples[i]));
    }

    BenchResult result;
    result.label = options.label;
    result.variant = options.variant;
    result.input_mode = options.input_mode;
    result.warmup = options.warmup;
    result.iterations = options.iterations;
    result.wall_seconds = wall_seconds;
    result.clock_tick_ns = tick_ns;
    result.timer_overhead_ns = overhead_ns;
    result.steady_clock_tick_ns = steady_tick_ns;
    result.pinned = pinned;
    result.high_priority = elevated;
    result.affinity_mask = affinity;
    result.calibration_before_s = calibration_before;
    result.calibration_after_s = calibration_after;
    result.settle_seconds = options.settle_seconds;
    result.timestamp = utc_timestamp();
    result.checksum = sink;
    summarise(samples, result);

    print_summary(result);
    write_json(result, options.output_path);
    std::printf("\nwrote %s\n", options.output_path.c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(parse(argc, argv));
    } catch (const std::exception& error) {
        std::printf("BENCH FAILED: %s\n", error.what());
        return 1;
    }
}
