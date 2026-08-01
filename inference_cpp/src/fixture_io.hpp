// fixture_io.hpp - reader for the TTSF fixture file that ml/export_weights.py writes.
//
// WHAT
//     Loads the (input, output) pairs dumped from PyTorch: `count` windows of
//     [window, features] float32, each followed by the `classes` logits that
//     model produced for it.
//
// WHY IT IS HEADER-ONLY, AND WHY IT LIVES HERE
//     Three translation units need it - the parity test, the incremental
//     equivalence test, and the benchmark - and none of them is the inference
//     library. `tts_inference` must not grow a file-format dependency it never
//     uses at run time, so this is never compiled into it; it is a header those
//     three include, and `src/` is simply the include directory they share.
//
//     It exists as shared code only because the second consumer appeared. Stage
//     7a had one, and the reader was correctly inlined into parity_test.cpp then.
//
// WHY THE BENCHMARK NEEDS REAL FIXTURES RATHER THAN ZEROS
//     A harness fed zeros measures a machine that does not exist. Zeros never
//     produce denormals, take the same branch every time in max-pool and leaky
//     ReLU, and compress into cache in a way real data does not. Stage 6
//     measured this feature distribution as heavy-tailed - bulk near +/-1.4,
//     extremes near +/-22 - and those tails are where the arithmetic gets slow.
//     bench/bench.cpp cycles through these recorded windows for that reason.

#ifndef TICK_TO_SIGNAL_FIXTURE_IO_HPP
#define TICK_TO_SIGNAL_FIXTURE_IO_HPP

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace tts {

constexpr char kFixtureMagic[4] = {'T', 'T', 'S', 'F'};
constexpr std::uint32_t kFixtureVersion = 1;

struct FixtureSet {
    std::uint32_t count = 0;
    std::uint32_t window = 0;
    std::uint32_t features = 0;
    std::uint32_t classes = 0;
    std::vector<float> inputs;   // count * window * features, row-major [t][f]
    std::vector<float> outputs;  // count * classes

    const float* input(std::size_t index) const {
        return inputs.data() + index * window * features;
    }
    const float* output(std::size_t index) const { return outputs.data() + index * classes; }
    // One row of one window: the [features] column a streaming model is fed.
    const float* row(std::size_t index, std::size_t step) const {
        return input(index) + step * features;
    }
};

namespace detail {

inline void read_or_throw(std::FILE* file, void* destination, std::size_t bytes, const char* what) {
    if (std::fread(destination, 1, bytes, file) != bytes) {
        std::fclose(file);
        throw std::runtime_error(std::string("fixture file ended while reading ") + what);
    }
}

}  // namespace detail

// `limit` of 0 means "every fixture in the file". A smaller limit is useful for
// a fast test; the parity test deliberately uses all of them.
inline FixtureSet load_fixtures(const std::string& path, std::uint32_t limit = 0) {
    std::FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) throw std::runtime_error("cannot open fixtures: " + path);

    char magic[4];
    detail::read_or_throw(file, magic, sizeof(magic), "magic");
    if (std::memcmp(magic, kFixtureMagic, 4) != 0) {
        std::fclose(file);
        throw std::runtime_error(path + " is not a TTSF fixture file");
    }
    std::uint32_t header[5];
    detail::read_or_throw(file, header, sizeof(header), "header");
    if (header[0] != kFixtureVersion) {
        std::fclose(file);
        throw std::runtime_error("unsupported fixture version " + std::to_string(header[0]));
    }

    FixtureSet set;
    set.count = (limit > 0 && limit < header[1]) ? limit : header[1];
    set.window = header[2];
    set.features = header[3];
    set.classes = header[4];

    const std::size_t stride = static_cast<std::size_t>(set.window) * set.features;
    set.inputs.resize(static_cast<std::size_t>(set.count) * stride);
    set.outputs.resize(static_cast<std::size_t>(set.count) * set.classes);
    for (std::uint32_t index = 0; index < set.count; ++index) {
        detail::read_or_throw(file, set.inputs.data() + index * stride, stride * sizeof(float),
                              "input");
        detail::read_or_throw(file, set.outputs.data() + index * set.classes,
                              set.classes * sizeof(float), "output");
    }
    std::fclose(file);
    return set;
}

}  // namespace tts

#endif  // TICK_TO_SIGNAL_FIXTURE_IO_HPP
