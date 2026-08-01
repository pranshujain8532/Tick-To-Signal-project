// bench_clock.hpp - a monotonic clock that actually resolves microseconds.
//
// WHAT
//     A thin wrapper over QueryPerformanceCounter on Windows and
//     clock_gettime(CLOCK_MONOTONIC) on POSIX, returning raw ticks plus the
//     conversion to nanoseconds.
//
// WHY IT EXISTS - a measured defect, not a preference.
//     The first version of this harness used std::chrono::steady_clock. On this
//     toolchain (MinGW GCC 6.3) that clock advertises `period = 1/1000000000`,
//     i.e. one nanosecond, and DELIVERS about one millisecond: it is backed by a
//     coarse OS tick rather than the performance counter. The harness's own
//     resolution check is what caught it, on the very first run:
//
//         measured tick 997000 ns, timer pair overhead 0 ns
//         zero-duration samples: 19518 of 20000
//
//     Nineteen thousand forward passes had apparently taken no time at all, and
//     the "p50 5,000 us" the full variant reported was a 1 ms grid, not a
//     measurement. Every quoted latency would have been a rounding artefact.
//
//     This is the entire reason the methodology says to measure and print the
//     clock's real resolution BEFORE any number that depends on it. A nominal
//     period is a compile-time ratio and the standard does not require it to
//     bear any relation to what the hardware can distinguish.
//
// DESIGN DECISION - raw ticks in the hot loop, nanoseconds afterwards.
//     QueryPerformanceCounter hands back a tick count; converting it to
//     nanoseconds needs a divide. Doing that divide between the two clock reads
//     would put ~20-40 cycles of harness arithmetic inside the measured region.
//     The loop therefore stores `end - start` in ticks and the conversion runs
//     once per sample after the run is over.
//
// DESIGN DECISION - not __rdtsc().
//     Rejected because it needs its own calibration to become a duration, it is
//     only invariant with respect to frequency scaling on parts that advertise
//     invariant TSC, and QueryPerformanceCounter is already built on that same
//     TSC when it is available. Using rdtsc directly would mean reimplementing
//     what the OS already got right, and getting the calibration wrong is
//     silent.

#ifndef TICK_TO_SIGNAL_BENCH_CLOCK_HPP
#define TICK_TO_SIGNAL_BENCH_CLOCK_HPP

#include <cstdint>
#include <stdexcept>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <ctime>
#endif

class MonotonicClock {
public:
    MonotonicClock() {
#ifdef _WIN32
        LARGE_INTEGER frequency;
        if (!QueryPerformanceFrequency(&frequency)) {
            throw std::runtime_error("QueryPerformanceFrequency failed");
        }
        ticks_per_second_ = frequency.QuadPart;
#else
        // clock_gettime reports nanoseconds directly, so a tick IS a nanosecond
        // and the conversion below is the identity.
        ticks_per_second_ = 1000000000LL;
#endif
        origin_ = read();
    }

    // Ticks since construction. Small enough that differences never overflow.
    std::int64_t now() const { return read() - origin_; }

    std::int64_t ticks_per_second() const { return ticks_per_second_; }

    // Exact integer conversion: splitting into whole seconds and a remainder
    // avoids the overflow that `ticks * 1000000000 / frequency` would hit on a
    // long-running counter.
    std::int64_t to_nanoseconds(std::int64_t ticks) const {
        const std::int64_t whole = ticks / ticks_per_second_;
        const std::int64_t rest = ticks % ticks_per_second_;
        return whole * 1000000000LL + rest * 1000000000LL / ticks_per_second_;
    }

    double to_seconds(std::int64_t ticks) const {
        return static_cast<double>(ticks) / static_cast<double>(ticks_per_second_);
    }

private:
    std::int64_t read() const {
#ifdef _WIN32
        LARGE_INTEGER counter;
        QueryPerformanceCounter(&counter);
        return counter.QuadPart;
#else
        std::timespec moment;
        clock_gettime(CLOCK_MONOTONIC, &moment);
        return static_cast<std::int64_t>(moment.tv_sec) * 1000000000LL + moment.tv_nsec;
#endif
    }

    std::int64_t ticks_per_second_ = 1;
    std::int64_t origin_ = 0;
};

#endif  // TICK_TO_SIGNAL_BENCH_CLOCK_HPP
