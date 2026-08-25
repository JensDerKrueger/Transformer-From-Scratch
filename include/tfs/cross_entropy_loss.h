#pragma once

#include "tfs/softmax.h"
#include "tfs/tensor.h"
#include "tfs/tensor_operations.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tfs {

struct CrossEntropyResult {
    TensorValue loss = 0.0f;
    Tensor probabilities;
    Tensor gradient;
};

inline CrossEntropyResult crossEntropyLoss(
    const Tensor& logits,
    const std::vector<std::size_t>& targetTokenIds
) {
    requireMatrix(logits, "logits");

    const TensorShape& shape = logits.getShape();
    const std::size_t rows = shape[0];
    const std::size_t columns = shape[1];

    if (targetTokenIds.size() != rows) {
        throw std::runtime_error("Cross-entropy target count must match row count");
    }

    Tensor probabilities = softmaxRows(logits);
    Tensor gradient = probabilities;
    TensorValue loss = 0.0f;

    const std::vector<std::size_t>& strides = probabilities.getStrides();
    const TensorValue* const probabilityData = probabilities.data();
    TensorValue* const gradientData = gradient.data();
    const TensorValue scale = 1.0f / static_cast<TensorValue>(rows);

    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t target = targetTokenIds[row];
        if (target >= columns) {
            throw std::runtime_error("Cross-entropy target token is out of bounds");
        }

        const std::size_t targetIndex = row * strides[0] + target * strides[1];
        loss -= std::log(probabilityData[targetIndex]);
        gradientData[targetIndex] -= 1.0f;
    }

    for (std::size_t i = 0; i < gradient.size(); ++i) {
        gradientData[i] *= scale;
    }

    loss *= scale;

    return CrossEntropyResult{loss, std::move(probabilities), std::move(gradient)};
}

} // namespace tfs
