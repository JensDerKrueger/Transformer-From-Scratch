#pragma once

#include "tfs/linear_layer.h"
#include "tfs/tensor.h"

#include <cstddef>
#include <utility>
#include <vector>

namespace tfs {

struct AttentionProjectionResult {
    Tensor queries;
    Tensor keys;
    Tensor values;
};

class AttentionProjections {
public:
    AttentionProjections(const std::size_t inputSize, const std::size_t projectionSize)
        : queryLayer(inputSize, projectionSize),
          keyLayer(inputSize, projectionSize),
          valueLayer(inputSize, projectionSize) {
    }

    AttentionProjections(
        const std::size_t inputSize,
        const std::size_t projectionSize,
        std::vector<TensorValue> queryWeightValues,
        std::vector<TensorValue> queryBiasValues,
        std::vector<TensorValue> keyWeightValues,
        std::vector<TensorValue> keyBiasValues,
        std::vector<TensorValue> valueWeightValues,
        std::vector<TensorValue> valueBiasValues
    )
        : queryLayer(inputSize, projectionSize, std::move(queryWeightValues), std::move(queryBiasValues)),
          keyLayer(inputSize, projectionSize, std::move(keyWeightValues), std::move(keyBiasValues)),
          valueLayer(inputSize, projectionSize, std::move(valueWeightValues), std::move(valueBiasValues)) {
    }

    std::size_t inputSize() const {
        return queryLayer.inputSize();
    }

    std::size_t projectionSize() const {
        return queryLayer.outputSize();
    }

    const LinearLayer& getQueryLayer() const {
        return queryLayer;
    }

    const LinearLayer& getKeyLayer() const {
        return keyLayer;
    }

    const LinearLayer& getValueLayer() const {
        return valueLayer;
    }

    AttentionProjectionResult forward(const Tensor& input) const {
        return AttentionProjectionResult{
            queryLayer.forward(input),
            keyLayer.forward(input),
            valueLayer.forward(input)
        };
    }

private:
    LinearLayer queryLayer;
    LinearLayer keyLayer;
    LinearLayer valueLayer;
};

} // namespace tfs
