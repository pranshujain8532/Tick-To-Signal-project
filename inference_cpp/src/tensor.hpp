// tensor.hpp - a minimal owning float buffer with aligned storage.
//
// WHAT
//     One class: an owning, move-only block of float32 with a shape attached
//     and 64-byte aligned storage. Nothing else.
//
// WHY
//     The forward pass needs somewhere to put intermediate activations, and it
//     needs those buffers allocated once at construction so the hot path never
//     touches the allocator. This is the smallest type that does that job.
//
// DESIGN DECISION - 64-byte alignment.
//     64 bytes is the cache-line size on every x86-64 part this will run on.
//     Aligning each buffer to a line boundary means a tensor never shares a
//     cache line with an unrelated one (no false sharing if this is ever
//     threaded) and, more importantly for Stage 7b, it is the precondition for
//     aligned SIMD loads: `_mm256_load_ps` requires 32-byte alignment and
//     `_mm512_load_ps` requires 64. Doing it now costs nothing and means the
//     vectorisation work later is a change to the inner loop rather than a
//     change to every allocation.
//
//     This stage does NOT use SIMD - see the note on correctness below.
//
// DESIGN DECISION - no templates, no broadcasting, no general shapes.
//     Rejected alternative: a templated `Tensor<T, Rank>` with strides and
//     numpy-style broadcasting. That is what a library would need. This is not
//     a library: it runs exactly one model, whose every tensor is float32 and
//     whose every shape is known at compile time. A broadcasting engine would
//     be several hundred lines of code that the parity test cannot exercise,
//     which makes it several hundred lines that could be wrong without anyone
//     noticing. `float` only, C-contiguous only, and the ops take explicit
//     dimensions rather than inferring them.
//
// CORRECTNESS FIRST - this whole stage (7a) contains no performance work.
//     Every loop here is the most obvious one that computes the right answer.
//     No blocking, no unrolling, no intrinsics, no fast-math. The point of this
//     stage is a bit-comparable reference implementation; optimising before
//     that exists means debugging arithmetic and performance simultaneously,
//     and chasing a numerical disagreement that turns out to be `-ffast-math`
//     reassociating a sum is a genuinely miserable way to spend a day.

#ifndef TICK_TO_SIGNAL_TENSOR_HPP
#define TICK_TO_SIGNAL_TENSOR_HPP

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace tts {

// Cache line on x86-64, and the alignment AVX-512 loads want.
constexpr std::size_t kAlignment = 64;

// The model is small and fixed; four dimensions covers every tensor it has
// (conv weights are [out, in, kh, kw]).
constexpr int kMaxDims = 4;

class Tensor {
public:
    Tensor() : data_(nullptr), raw_(nullptr), size_(0), ndim_(0) {
        for (int i = 0; i < kMaxDims; ++i) dims_[i] = 0;
    }

    explicit Tensor(const std::vector<int>& dims) : Tensor() { reshape(dims); }

    ~Tensor() { release(); }

    // Move-only. Copying a buffer of activations is never something this code
    // wants to do by accident, so the copy constructor is deleted rather than
    // written - a silent deep copy on the hot path is exactly the kind of cost
    // that hides in a profile.
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& other) noexcept : Tensor() { swap(other); }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            release();
            swap(other);
        }
        return *this;
    }

    // Allocate (or reallocate) to hold `dims`. Called once per buffer during
    // model construction and never again.
    void reshape(const std::vector<int>& dims) {
        if (dims.empty() || dims.size() > static_cast<std::size_t>(kMaxDims)) {
            throw std::runtime_error("tensor rank must be between 1 and 4, got " +
                                     std::to_string(dims.size()));
        }
        std::size_t count = 1;
        for (std::size_t i = 0; i < dims.size(); ++i) {
            if (dims[i] <= 0) throw std::runtime_error("tensor dimensions must be positive");
            count *= static_cast<std::size_t>(dims[i]);
        }
        release();
        allocate(count);
        ndim_ = static_cast<int>(dims.size());
        for (int i = 0; i < kMaxDims; ++i) dims_[i] = (i < ndim_) ? dims[i] : 1;
        size_ = count;
    }

    float* data() { return data_; }
    const float* data() const { return data_; }
    std::size_t size() const { return size_; }
    int ndim() const { return ndim_; }
    int dim(int index) const { return dims_[index]; }

    void zero() {
        if (data_ != nullptr) std::memset(data_, 0, size_ * sizeof(float));
    }

    std::string shape_string() const {
        std::string text = "[";
        for (int i = 0; i < ndim_; ++i) {
            if (i > 0) text += ", ";
            text += std::to_string(dims_[i]);
        }
        return text + "]";
    }

private:
    // Over-allocate and align by hand rather than calling `std::aligned_alloc`
    // (C++17, and absent from this MinGW libc) or `_aligned_malloc` (Windows
    // only). The arithmetic is four lines and works everywhere, which matters
    // for a file whose entire selling point is having no dependencies.
    void allocate(std::size_t count) {
        const std::size_t bytes = count * sizeof(float);
        void* raw = std::malloc(bytes + kAlignment);
        if (raw == nullptr) throw std::runtime_error("out of memory allocating a tensor");
        const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(raw);
        const std::uintptr_t aligned = (address + kAlignment - 1) & ~(std::uintptr_t)(kAlignment - 1);
        raw_ = raw;
        data_ = reinterpret_cast<float*>(aligned);
    }

    void release() {
        if (raw_ != nullptr) std::free(raw_);
        raw_ = nullptr;
        data_ = nullptr;
        size_ = 0;
        ndim_ = 0;
    }

    void swap(Tensor& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(raw_, other.raw_);
        std::swap(size_, other.size_);
        std::swap(ndim_, other.ndim_);
        for (int i = 0; i < kMaxDims; ++i) std::swap(dims_[i], other.dims_[i]);
    }

    float* data_;       // aligned view into raw_, what callers use
    void* raw_;         // what malloc returned, what free needs
    std::size_t size_;  // element count
    int dims_[kMaxDims];
    int ndim_;
};

}  // namespace tts

#endif  // TICK_TO_SIGNAL_TENSOR_HPP
