#pragma once

#include "tfs/linear_layer.h"
#include "tfs/tensor.h"

#include <cstddef>
#include <utility>
#include <vector>

namespace tfs {

class AttentionOutputProjection {
public:
    AttentionOutputProjection(const std::size_t inputSize, const std::size_t outputSize)
        : layer(inputSize, outputSize) {
    }

    AttentionOutputProjection(
        const std::size_t inputSize,
        const std::size_t outputSize,
        std::vector<TensorValue> weightValues,
        std::vector<TensorValue> biasValues
    )
        : layer(inputSize, outputSize, std::move(weightValues), std::move(biasValues)) {
    }

    std::size_t inputSize() const {
        return layer.inputSize();
    }

    std::size_t outputSize() const {
        return layer.outputSize();
    }

    const LinearLayer& getLayer() const {
        return layer;
    }

    Tensor forward(const Tensor& attentionOutput) const {
        return layer.forward(attentionOutput);
    }

private:
    LinearLayer layer;
};

} // namespace tfs
