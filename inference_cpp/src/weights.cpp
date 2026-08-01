#include "weights.hpp"

#include <cstdio>
#include <stdexcept>

namespace tts {
namespace {

// Every read goes through these two helpers so that a short file is caught at
// the point it happens, with the field name, rather than surfacing later as a
// nonsensical dimension.
void read_exact(std::FILE* file, void* destination, std::size_t bytes, const char* what) {
    if (std::fread(destination, 1, bytes, file) != bytes) {
        throw std::runtime_error(std::string("weight file ended while reading ") + what);
    }
}

std::uint32_t read_u32(std::FILE* file, const char* what) {
    std::uint32_t value = 0;
    read_exact(file, &value, sizeof(value), what);
    return value;
}

// Guards against a corrupt length field turning into a huge allocation. The
// real model's largest tensor is a few thousand values and its longest name is
// well under 100 characters, so these bounds are generous by orders of
// magnitude while still refusing nonsense.
constexpr std::uint32_t kMaxNameLength = 256;
constexpr std::uint32_t kMaxValues = 100u * 1000u * 1000u;

}  // namespace

WeightStore::WeightStore(const std::string& path) {
    std::FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        throw std::runtime_error("cannot open weight file: " + path);
    }

    try {
        char magic[4];
        read_exact(file, magic, sizeof(magic), "magic");
        for (int i = 0; i < 4; ++i) {
            if (magic[i] != kWeightMagic[i]) {
                throw std::runtime_error("bad magic in " + path + ": this is not a TTSW weight file");
            }
        }

        const std::uint32_t version = read_u32(file, "version");
        if (version != kWeightVersion) {
            throw std::runtime_error("weight file is version " + std::to_string(version) +
                                     "; this build understands version " + std::to_string(kWeightVersion));
        }

        const std::uint32_t count = read_u32(file, "tensor count");
        for (std::uint32_t index = 0; index < count; ++index) {
            const std::uint32_t name_length = read_u32(file, "name length");
            if (name_length == 0 || name_length > kMaxNameLength) {
                throw std::runtime_error("implausible tensor name length " + std::to_string(name_length));
            }
            std::string name(name_length, '\0');
            read_exact(file, &name[0], name_length, "tensor name");

            const std::uint32_t dtype = read_u32(file, "dtype");
            if (dtype != kDtypeFloat32) {
                throw std::runtime_error("tensor '" + name + "' has dtype tag " + std::to_string(dtype) +
                                         "; only float32 (0) is supported");
            }

            const std::uint32_t ndim = read_u32(file, "ndim");
            if (ndim == 0 || ndim > static_cast<std::uint32_t>(kMaxDims)) {
                throw std::runtime_error("tensor '" + name + "' has rank " + std::to_string(ndim) +
                                         ", outside the supported 1-" + std::to_string(kMaxDims));
            }

            std::vector<int> dims(ndim);
            std::size_t values = 1;
            for (std::uint32_t d = 0; d < ndim; ++d) {
                const std::uint32_t extent = read_u32(file, "dimension");
                if (extent == 0) throw std::runtime_error("tensor '" + name + "' has a zero dimension");
                dims[d] = static_cast<int>(extent);
                values *= extent;
                if (values > kMaxValues) {
                    throw std::runtime_error("tensor '" + name + "' claims more than 100M values");
                }
            }

            Tensor tensor(dims);
            read_exact(file, tensor.data(), values * sizeof(float), ("data for '" + name + "'").c_str());
            if (!tensors_.emplace(name, std::move(tensor)).second) {
                throw std::runtime_error("duplicate tensor name '" + name + "' in " + path);
            }
        }

        // Anything after the last tensor means the writer and reader disagree
        // about the layout, which is worth failing on even though the data we
        // needed is already in hand.
        char trailing = 0;
        if (std::fread(&trailing, 1, 1, file) != 0) {
            throw std::runtime_error("unexpected trailing bytes in " + path);
        }
    } catch (...) {
        std::fclose(file);
        throw;
    }
    std::fclose(file);
}

const Tensor& WeightStore::get(const std::string& name, const std::vector<int>& expected) const {
    const auto found = tensors_.find(name);
    if (found == tensors_.end()) {
        throw std::runtime_error("weight file has no tensor named '" + name + "'");
    }
    const Tensor& tensor = found->second;

    bool matches = tensor.ndim() == static_cast<int>(expected.size());
    for (std::size_t i = 0; matches && i < expected.size(); ++i) {
        matches = tensor.dim(static_cast<int>(i)) == expected[i];
    }
    if (!matches) {
        std::string wanted = "[";
        for (std::size_t i = 0; i < expected.size(); ++i) {
            if (i > 0) wanted += ", ";
            wanted += std::to_string(expected[i]);
        }
        wanted += "]";
        throw std::runtime_error("tensor '" + name + "' has shape " + tensor.shape_string() +
                                 " but the model expects " + wanted);
    }
    return tensor;
}

std::size_t WeightStore::total_values() const {
    std::size_t total = 0;
    for (const auto& entry : tensors_) total += entry.second.size();
    return total;
}

std::vector<std::string> WeightStore::names() const {
    std::vector<std::string> result;
    result.reserve(tensors_.size());
    for (const auto& entry : tensors_) result.push_back(entry.first);
    return result;
}

}  // namespace tts
