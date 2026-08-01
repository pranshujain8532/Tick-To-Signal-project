# Build for the hand-written inference path.
#
# Two presets, and the pair is the point of Stage 7a:
#
#   debug-O0    -O0, no fast math. The reference build. Correctness is
#               established here first, where the compiler has done almost
#               nothing to the arithmetic and a disagreement with PyTorch can
#               only be the code's fault.
#   release-O3  -O3, still without -ffast-math. The parity test must pass here
#               too, which is what proves the optimiser did not change the
#               answer.
#
# WHY -ffast-math IS NEVER ENABLED.
#     It permits the compiler to reassociate floating-point sums, contract
#     multiply-adds, and assume no NaNs or infinities. Reassociation alone can
#     move a reduction's result by more than this project's parity tolerance,
#     and it would do so invisibly - the build would simply start failing the
#     parity test with no code change to blame. Chasing a numerical
#     disagreement that turns out to be a compiler flag is a genuinely
#     miserable way to spend a day, so the flag stays off and any future
#     vectorisation in Stage 7b will be written explicitly rather than begged
#     for.
#
# No dependencies: no libtorch, no BLAS, no onnxruntime. Only the C++ standard
# library, which is what makes this artefact worth building at all.

cmake_minimum_required(VERSION 3.15)
project(tick_to_signal_inference CXX)

# C++14 rather than 17: the toolchain this was developed against is GCC 6.3,
# whose C++17 library support is incomplete. Nothing here needs 17.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

# EVERY build computes in true float32. This is not a micro-optimisation; it is
# what makes the presets comparable at all, and it was arrived at by being wrong
# twice.
#
#     32-bit MinGW defaults to -mfpmath=387: float arithmetic happens in the x87
#     unit's 80-bit registers and is rounded to 32 bits only when stored to
#     memory. The consequence is that two loops performing the SAME additions in
#     the SAME order can still produce different float32 results, depending on
#     whether the compiler kept an intermediate in a register or spilled it, and
#     on whether it vectorised one of them with SSE - which does round at 32.
#
#     This surfaced twice in Stage 7b, both times as tests/incremental_test
#     losing bit-identity between the streaming and full-recompute paths:
#
#       * adding -march=native (drift 4.5e-06). The obvious suspect was FMA
#         contraction, so -ffp-contract=off went on first. It changed nothing.
#         Three isolating builds found the real cause:
#             -msse2 -mfpmath=sse         bit-identical 100%
#             -msse2 -mfpmath=387         bit-identical   0%
#             -mavx2 -mfma -mfpmath=387   bit-identical   0%
#       * reordering the convolution loops (drift 1.2e-05), which failed only in
#         release-O3 - the one preset still left on the x87 default at that
#         point, on the reasoning that a reference build should keep the
#         toolchain's own semantics. That reasoning was wrong. A reference build
#         whose float32 results depend on register allocation is not a reference.
#
#     -mfpmath=sse makes every float operation a true float32 operation. It is
#     what any x86-64 build does by default - the ABI mandates SSE2 for floating
#     point - and it is what the model was trained and exported under. SSE2 is
#     present on every x86 part since 2003, so this costs no portability worth
#     having.
#
# Note what this is NOT. It changes which unit performs the arithmetic, not what
# the arithmetic is allowed to be: no reassociation, no denormal flushing, no
# assumed-finite operands. -ffast-math remains refused (see the note below).
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -mfpmath=sse")

# Explicit, so the presets differ in exactly the flags named here.
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# -march=native lets the compiler emit AVX2/FMA for this exact CPU instead of
# the 1990s baseline that a bare -O3 targets on 32-bit MinGW. It is a SEPARATE
# preset rather than a change to the release flags, for two reasons:
#
#   1. The resulting binary does not run on an older CPU. A portable -O3 build
#      has to keep existing, or the project quietly stops being buildable
#      anywhere but this laptop.
#   2. Parity passing on BOTH is the evidence that matters. `release-O3` and
#      `release-native` compile identical source with different instruction
#      selection; if only one of them agreed with PyTorch, the disagreement
#      would be the compiler's vectoriser, and that is worth being able to
#      isolate.
#
# Note this is still not -ffast-math. -march=native changes which instructions
# are used; -ffast-math would change what they are allowed to compute.
#
option(TTS_NATIVE "append -march=native to the release flags" OFF)
if(TTS_NATIVE)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
endif()

# Pinned rather than inherited, because Stage 7b made it load-bearing.
#
# GCC's default for C++ is already -ffp-contract=fast, and the assembly proves
# it: the reordered inner loop under -march=native compiles to 16 vfmadd
# instructions and zero vmulps/vaddps. incremental.cpp's hand-written AVX2
# kernel therefore has to use _mm256_fmadd_ps to stay bit-identical to the
# compiler-generated code around it. Leaving that agreement resting on a default
# means a toolchain that defaults to `off` would silently break the equivalence
# test with no source change to blame.
#
# Contraction is NOT the thing -ffast-math would enable. Contracting a*b+c into
# one instruction removes a rounding step and never changes which numbers are
# added together; reassociation does exactly that, which is why it stays
# refused. See the -ffast-math note at the top of this file.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffp-contract=fast")

# The hand-written AVX2 kernel for the widest inner loop in the streaming path.
# OFF by default: the scalar path is the reference, it is what any non-x86
# target compiles, and ctest runs both so the intrinsics are never trusted
# without a parity check behind them.
option(TTS_AVX2 "use the hand-written AVX2 kernel in incremental.cpp" OFF)

# A human-readable name for the build, stamped into every benchmark record so a
# JSON file cannot be misattributed to the wrong flags six weeks later. The
# presets set it; a bare `cmake -S . -B build` falls back to the build type.
set(TTS_BUILD_LABEL "${CMAKE_BUILD_TYPE}" CACHE STRING "name recorded in benchmark output")
string(TOUPPER "${CMAKE_BUILD_TYPE}" TTS_BUILD_TYPE_UPPER)
set(TTS_ACTIVE_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${TTS_BUILD_TYPE_UPPER}}")
string(STRIP "${TTS_ACTIVE_FLAGS}" TTS_ACTIVE_FLAGS)

add_library(tts_inference STATIC
    src/weights.cpp
    src/ops.cpp
    src/model.cpp
    src/incremental.cpp
)
target_include_directories(tts_inference PUBLIC src)
target_compile_options(tts_inference PRIVATE -Wall -Wextra)
if(TTS_AVX2)
    # PUBLIC so the benchmark records which kernel it actually measured.
    target_compile_definitions(tts_inference PUBLIC TTS_USE_AVX2_KERNEL)
endif()

add_executable(parity_test tests/parity_test.cpp)
target_link_libraries(parity_test PRIVATE tts_inference)

add_executable(ops_test tests/ops_test.cpp)
target_link_libraries(ops_test PRIVATE tts_inference)

add_executable(incremental_test tests/incremental_test.cpp)
target_link_libraries(incremental_test PRIVATE tts_inference)

# The latency harness. It is NOT a ctest target: a benchmark that runs for
# minutes and reports a machine-dependent number is not a pass/fail check, and
# putting it in the test suite would mean every `ctest` invocation either takes
# ten minutes or measures nothing useful.
add_executable(bench bench/bench.cpp)
target_link_libraries(bench PRIVATE tts_inference)
target_include_directories(bench PRIVATE bench)
target_compile_definitions(bench PRIVATE
    TTS_BUILD_NAME="${TTS_BUILD_LABEL}"
    TTS_BUILD_FLAGS="${TTS_ACTIVE_FLAGS}")

enable_testing()
# Artefacts are written by `python -m ml.export_weights` into inference_cpp/artifacts,
# so the tests run from the project directory and use relative paths.
set(TTS_WEIGHTS "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/student_weights.ttsw")
set(TTS_FIXTURES "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/student_fixtures.ttsf")

add_test(NAME ops COMMAND ops_test "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/op_fixtures.ttsw")
add_test(NAME parity COMMAND parity_test "${TTS_WEIGHTS}" "${TTS_FIXTURES}")
add_test(NAME incremental COMMAND incremental_test "${TTS_WEIGHTS}" "${TTS_FIXTURES}")
