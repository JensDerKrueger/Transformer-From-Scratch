#pragma once

#include "tfs/linear_layer.h"
#include "tfs/tensor.h"

#include <cstddef>
#include <utility>
#include <vector>

namespace tfs {

class LanguageModelHead {
public:
    LanguageModelHead(const std::size_t modelSize, const std::size_t vocabSize)
        : layer(modelSize, vocabSize) {
    }

    LanguageModelHead(
        const std::size_t modelSize,
        const std::size_t vocabSize,
        std::vector<TensorValue> weightValues,
        std::vector<TensorValue> biasValues
    )
        : layer(modelSize, vocabSize, std::move(weightValues), std::move(biasValues)) {
    }

    std::size_t modelSize() const {
        return layer.inputSize();
    }

    std::size_t vocabSize() const {
        return layer.outputSize();
    }

    const LinearLayer& getLayer() const {
        return layer;
    }

    Tensor forward(const Tensor& input) const {
        return layer.forward(input);
    }

private:
    LinearLayer layer;
};

} // namespace tfs
