#pragma once

#include "tfs/tensor.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

namespace tfs {

class RandomInitializer {
public:
    explicit RandomInitializer(const std::uint32_t seed)
        : generator(seed) {
    }

    void fillUniform(Tensor& tensor, const TensorValue minValue, const TensorValue maxValue) {
        std::uniform_real_distribution<TensorValue> distribution(minValue, maxValue);

        TensorValue* const data = tensor.data();
        for (std::size_t i = 0; i < tensor.size(); ++i) {
            data[i] = distribution(generator);
        }
    }

    void fillXavier(Tensor& tensor, const std::size_t fanIn, const std::size_t fanOut) {
        const TensorValue limit = std::sqrt(6.0f / static_cast<TensorValue>(fanIn + fanOut));
        fillUniform(tensor, -limit, limit);
    }

private:
    std::mt19937 generator;
};

} // namespace tfs
