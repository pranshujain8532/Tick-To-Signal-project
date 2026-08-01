// weights.hpp - reader for the TTSW weight file written by ml/export_weights.py.
//
// WHAT
//     Loads a name-keyed set of float32 tensors from disk and hands them out by
//     name, with the shape the caller expects checked at lookup time.
//
// WHY
//     The C++ side has no dependencies, so it cannot read a torch checkpoint, a
//     .npz or an ONNX graph. `ml/export_weights.py` writes a format that needs
//     nothing but `fread` and some length checks; this is the other half.
//
// DESIGN DECISION - validate everything, and fail with the offending value.
//     A weight file is read once at startup, so validation is free, and the
//     failure modes it catches are the worst kind: a truncated file or a
//     shape mismatch produces a model that runs and returns plausible logits
//     that are wrong. Magic, version, tensor count, name lengths, rank, and
//     every dimension are checked, and `get()` refuses a tensor whose shape is
//     not what the caller asked for. Silent success on bad input is the one
//     outcome this file is written to prevent.

#ifndef TICK_TO_SIGNAL_WEIGHTS_HPP
#define TICK_TO_SIGNAL_WEIGHTS_HPP

#include <map>
#include <string>
#include <vector>

#include "tensor.hpp"

namespace tts {

// Must match ml/export_weights.py.
constexpr char kWeightMagic[4] = {'T', 'T', 'S', 'W'};
constexpr std::uint32_t kWeightVersion = 1;
constexpr std::uint32_t kDtypeFloat32 = 0;

class WeightStore {
public:
    // Reads the whole file into memory. Throws std::runtime_error with a
    // specific message on any inconsistency.
    explicit WeightStore(const std::string& path);

    // Fetch a tensor and assert its shape. `expected` must match exactly;
    // a mismatch means the C++ architecture and the exported model have
    // diverged, which is a bug to stop on rather than to work around.
    const Tensor& get(const std::string& name, const std::vector<int>& expected) const;

    std::size_t tensor_count() const { return tensors_.size(); }
    std::size_t total_values() const;
    std::vector<std::string> names() const;

private:
    std::map<std::string, Tensor> tensors_;
};

}  // namespace tts

#endif  // TICK_TO_SIGNAL_WEIGHTS_HPP
