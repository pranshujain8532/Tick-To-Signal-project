// bench_report.hpp - summarising and serialising a latency run.
//
// WHAT
//     The statistics half of bench.cpp: turn a vector of per-iteration
//     nanosecond samples into percentiles and a histogram, print them, and
//     write a JSON file that notebooks/07_latency_results_analysis.ipynb reads.
//
// WHY IT IS A SEPARATE HEADER
//     bench.cpp is about *how the measurement is taken* - pinning, warmup,
//     the clock, what is inside the timed region. This file is about *what is
//     reported*. Keeping them apart means the reporting can be read and
//     criticised without wading through the timing loop, and the timing loop
//     stays short enough to audit line by line.
//
// WHY NEVER A MEAN ALONE
//     The mean is reported here, but never on its own and never first. A
//     latency distribution with a heavy tail has a mean that describes no
//     actual call: Stage 6 measured the PyTorch eager path at p50 10.6 ms and
//     p99.9 74.1 ms, where the mean sits above 70% of the samples and hides the
//     tail entirely. The tail is the interesting part - it is where the
//     allocator, the scheduler and the page fault live - so p99 and p99.9 lead.
//
// WHY p99.9 AND NOT JUST p99
//     A tick-to-signal path runs on every book update. Binance BTCUSDT depth
//     updates arrive on the order of 10 per second per stream, so p99 is
//     something like a once-every-ten-seconds event and p99.9 is a
//     once-every-two-minutes event. Both happen constantly over a trading day,
//     and the p99.9 is where the mechanisms that a p50 optimisation cannot
//     touch show up: a page fault, a scheduler preemption, an allocator slow
//     path. Reporting only p99 lets a change that trades a rare 10x spike for a
//     small median win look like an improvement.
//
//     p99.9 needs samples to exist at all: at 1M iterations it is the average
//     of the top 1,000, at 100k it is the top 100, and below about 10k it is a
//     single observation and should not be quoted. The iteration count is in
//     every record for exactly this reason.

#ifndef TICK_TO_SIGNAL_BENCH_REPORT_HPP
#define TICK_TO_SIGNAL_BENCH_REPORT_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

// Passed in by CMake so the record says which flags produced it, rather than
// relying on whoever reads the JSON to remember.
#ifndef TTS_BUILD_FLAGS
#define TTS_BUILD_FLAGS "unknown"
#endif
#ifndef TTS_BUILD_NAME
#define TTS_BUILD_NAME "unknown"
#endif

struct Quantile {
    double quantile;
    std::uint32_t nanoseconds;
};

struct BenchResult {
    std::string label, variant, input_mode, timestamp;
    std::uint64_t warmup = 0;
    std::uint64_t iterations = 0;
    double wall_seconds = 0.0;

    // Clock characterisation, recorded so a reader can judge whether the
    // measurement was even possible at this magnitude.
    std::uint64_t clock_tick_ns = 0;
    std::uint64_t timer_overhead_ns = 0;
    // What std::chrono::steady_clock resolves on this toolchain. Recorded every
    // run because it is the evidence for why bench_clock.hpp exists.
    std::uint64_t steady_clock_tick_ns = 0;
    std::uint64_t zero_samples = 0;  // calls that appeared to take no time at all

    bool pinned = false;
    bool high_priority = false;
    std::uint64_t affinity_mask = 0;
    double calibration_before_s = 0.0;
    double calibration_after_s = 0.0;
    double settle_seconds = 0.0;
    double checksum = 0.0;  // keeps the optimiser from deleting the timed calls

    std::uint32_t min_ns = 0, max_ns = 0;
    std::uint32_t p50_ns = 0, p90_ns = 0, p99_ns = 0, p999_ns = 0, p9999_ns = 0;
    double mean_ns = 0.0;

    std::vector<Quantile> curve;            // for the CDF plot
    std::uint32_t hist_low_ns = 0;
    std::uint32_t hist_width_ns = 1;
    std::vector<std::uint64_t> hist_counts;  // 256 buckets spanning min..p99.9
    std::uint64_t hist_overflow = 0;         // everything above the last bucket
};

// Nearest-rank, no interpolation. Interpolating between two observed latencies
// reports a duration the machine never produced, which is the kind of invented
// number this project refuses everywhere else.
inline std::uint32_t percentile(const std::vector<std::uint32_t>& sorted, double quantile) {
    if (sorted.empty()) return 0;
    std::size_t rank = static_cast<std::size_t>(std::ceil(quantile / 100.0 * sorted.size()));
    if (rank == 0) rank = 1;
    if (rank > sorted.size()) rank = sorted.size();
    return sorted[rank - 1];
}

namespace detail {

// 0.5% steps through the body, then a fine grid through the tail where the
// interesting behaviour is. A uniform grid would spend 199 of its 206 points
// describing the part of the distribution nobody argues about.
inline std::vector<double> quantile_grid() {
    std::vector<double> grid;
    for (int step = 1; step <= 199; ++step) grid.push_back(0.5 * step);
    const double tail[] = {99.6, 99.7, 99.8, 99.9, 99.95, 99.99, 100.0};
    for (double q : tail) grid.push_back(q);
    return grid;
}

}  // namespace detail

// Sorts `samples` in place - the caller has no further use for their order.
inline void summarise(std::vector<std::uint32_t>& samples, BenchResult& result) {
    if (samples.empty()) throw std::runtime_error("no samples to summarise");

    result.zero_samples = 0;
    long double total = 0.0L;
    for (std::uint32_t sample : samples) {
        total += sample;
        if (sample == 0) ++result.zero_samples;
    }
    result.mean_ns = static_cast<double>(total / samples.size());

    std::sort(samples.begin(), samples.end());
    result.min_ns = samples.front();
    result.max_ns = samples.back();
    result.p50_ns = percentile(samples, 50.0);
    result.p90_ns = percentile(samples, 90.0);
    result.p99_ns = percentile(samples, 99.0);
    result.p999_ns = percentile(samples, 99.9);
    result.p9999_ns = percentile(samples, 99.99);

    result.curve.clear();
    for (double quantile : detail::quantile_grid()) {
        result.curve.push_back({quantile, percentile(samples, quantile)});
    }

    // The histogram spans min..p99.9 rather than min..max: a single 40x outlier
    // would otherwise put every real sample in bucket zero and the plot would
    // show nothing. Everything above the top bucket is counted, not discarded.
    const std::uint32_t span = (result.p999_ns > result.min_ns) ? result.p999_ns - result.min_ns : 1;
    result.hist_low_ns = result.min_ns;
    result.hist_width_ns = std::max<std::uint32_t>(1, (span + 255) / 256);
    result.hist_counts.assign(256, 0);
    result.hist_overflow = 0;
    for (std::uint32_t sample : samples) {
        const std::size_t bucket = (sample - result.hist_low_ns) / result.hist_width_ns;
        if (bucket < result.hist_counts.size()) ++result.hist_counts[bucket];
        else ++result.hist_overflow;
    }
}

inline std::string simd_features() {
    // These reflect the flags this translation unit was compiled with, which
    // CMake keeps identical to the library's.
    std::string features;
#ifdef __AVX2__
    features += "avx2 ";
#endif
#ifdef __FMA__
    features += "fma ";
#endif
#ifdef __AVX__
    features += "avx ";
#endif
#ifdef __SSE2__
    features += "sse2 ";
#endif
    if (features.empty()) features = "baseline";
    return features;
}

inline void print_summary(const BenchResult& result) {
    const double drift = (result.calibration_before_s > 0.0)
                             ? result.calibration_after_s / result.calibration_before_s
                             : 0.0;
    std::printf("\n--- %s (%s, %s) ---\n", result.label.c_str(), result.variant.c_str(),
                result.input_mode.c_str());
    std::printf("  p50    %9.3f us\n", result.p50_ns / 1000.0);
    std::printf("  p90    %9.3f us\n", result.p90_ns / 1000.0);
    std::printf("  p99    %9.3f us\n", result.p99_ns / 1000.0);
    std::printf("  p99.9  %9.3f us\n", result.p999_ns / 1000.0);
    std::printf("  p99.99 %9.3f us\n", result.p9999_ns / 1000.0);
    std::printf("  max    %9.3f us\n", result.max_ns / 1000.0);
    std::printf("  min    %9.3f us      mean %9.3f us (reported last, on purpose)\n",
                result.min_ns / 1000.0, result.mean_ns / 1000.0);
    std::printf("  zero-duration samples: %llu  (must be 0, or the clock is too coarse)\n",
                static_cast<unsigned long long>(result.zero_samples));
    std::printf("  calibration kernel %.4f s before, %.4f s after  (ratio %.4f)\n",
                result.calibration_before_s, result.calibration_after_s, drift);
    std::printf("  wall %.1f s   build %s [%s]   simd %s\n", result.wall_seconds, TTS_BUILD_NAME,
                TTS_BUILD_FLAGS, simd_features().c_str());
}

inline void write_json(const BenchResult& result, const std::string& path) {
    std::FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) throw std::runtime_error("cannot write " + path);

    std::fprintf(file, "{\n");
    std::fprintf(file, "  \"label\": \"%s\",\n", result.label.c_str());
    std::fprintf(file, "  \"variant\": \"%s\",\n", result.variant.c_str());
    std::fprintf(file, "  \"input_mode\": \"%s\",\n", result.input_mode.c_str());
    std::fprintf(file, "  \"timestamp_utc\": \"%s\",\n", result.timestamp.c_str());
    std::fprintf(file, "  \"build_name\": \"%s\",\n", TTS_BUILD_NAME);
    std::fprintf(file, "  \"build_flags\": \"%s\",\n", TTS_BUILD_FLAGS);
    std::fprintf(file, "  \"compiler\": \"%s\",\n", __VERSION__);
    std::fprintf(file, "  \"pointer_bits\": %d,\n", static_cast<int>(sizeof(void*) * 8));
    std::fprintf(file, "  \"simd\": \"%s\",\n", simd_features().c_str());
    std::fprintf(file, "  \"avx2_kernel_compiled_in\": %s,\n",
#ifdef TTS_USE_AVX2_KERNEL
                 "true"
#else
                 "false"
#endif
    );
    std::fprintf(file, "  \"warmup\": %llu,\n", static_cast<unsigned long long>(result.warmup));
    std::fprintf(file, "  \"iterations\": %llu,\n",
                 static_cast<unsigned long long>(result.iterations));
    std::fprintf(file, "  \"wall_seconds\": %.4f,\n", result.wall_seconds);
    std::fprintf(file, "  \"pinned\": %s,\n", result.pinned ? "true" : "false");
    std::fprintf(file, "  \"high_priority\": %s,\n", result.high_priority ? "true" : "false");
    std::fprintf(file, "  \"affinity_mask\": %llu,\n",
                 static_cast<unsigned long long>(result.affinity_mask));
    std::fprintf(file, "  \"clock_tick_ns\": %llu,\n",
                 static_cast<unsigned long long>(result.clock_tick_ns));
    std::fprintf(file, "  \"timer_overhead_ns\": %llu,\n",
                 static_cast<unsigned long long>(result.timer_overhead_ns));
    std::fprintf(file, "  \"steady_clock_tick_ns\": %llu,\n",
                 static_cast<unsigned long long>(result.steady_clock_tick_ns));
    std::fprintf(file, "  \"zero_samples\": %llu,\n",
                 static_cast<unsigned long long>(result.zero_samples));
    std::fprintf(file, "  \"calibration_before_s\": %.6f,\n", result.calibration_before_s);
    std::fprintf(file, "  \"calibration_after_s\": %.6f,\n", result.calibration_after_s);
    std::fprintf(file, "  \"settle_seconds\": %.2f,\n", result.settle_seconds);
    std::fprintf(file, "  \"checksum\": %.6f,\n", result.checksum);
    std::fprintf(file, "  \"min_ns\": %u,\n", result.min_ns);
    std::fprintf(file, "  \"p50_ns\": %u,\n", result.p50_ns);
    std::fprintf(file, "  \"p90_ns\": %u,\n", result.p90_ns);
    std::fprintf(file, "  \"p99_ns\": %u,\n", result.p99_ns);
    std::fprintf(file, "  \"p999_ns\": %u,\n", result.p999_ns);
    std::fprintf(file, "  \"p9999_ns\": %u,\n", result.p9999_ns);
    std::fprintf(file, "  \"max_ns\": %u,\n", result.max_ns);
    std::fprintf(file, "  \"mean_ns\": %.3f,\n", result.mean_ns);

    std::fprintf(file, "  \"quantiles\": [");
    for (std::size_t i = 0; i < result.curve.size(); ++i) {
        std::fprintf(file, "%s[%.4f, %u]", i ? ", " : "", result.curve[i].quantile,
                     result.curve[i].nanoseconds);
    }
    std::fprintf(file, "],\n");

    std::fprintf(file, "  \"histogram\": {\n");
    std::fprintf(file, "    \"low_ns\": %u,\n", result.hist_low_ns);
    std::fprintf(file, "    \"width_ns\": %u,\n", result.hist_width_ns);
    std::fprintf(file, "    \"overflow\": %llu,\n",
                 static_cast<unsigned long long>(result.hist_overflow));
    std::fprintf(file, "    \"counts\": [");
    for (std::size_t i = 0; i < result.hist_counts.size(); ++i) {
        std::fprintf(file, "%s%llu", i ? ", " : "",
                     static_cast<unsigned long long>(result.hist_counts[i]));
    }
    std::fprintf(file, "]\n  }\n}\n");
    std::fclose(file);
}

#endif  // TICK_TO_SIGNAL_BENCH_REPORT_HPP
