#pragma once

#include "tfs/tensor.h"
#include "tfs/tensor_operations.h"

#include <cstddef>
#include <limits>

namespace tfs {

inline Tensor applyCausalMask(const Tensor& scores) {
    requireMatrix(scores, "scores");

    const TensorShape& shape = scores.getShape();
    const std::size_t rows = shape[0];
    const std::size_t columns = shape[1];

    Tensor result = scores;

    const std::vector<std::size_t>& resultStrides = result.getStrides();
    TensorValue* const resultData = result.data();

    const std::size_t resultRowStride = resultStrides[0];
    const std::size_t resultColumnStride = resultStrides[1];
    const TensorValue maskedValue = -std::numeric_limits<TensorValue>::infinity();

    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t rowOffset = row * resultRowStride;

        for (std::size_t column = row + 1; column < columns; ++column) {
            resultData[rowOffset + column * resultColumnStride] = maskedValue;
        }
    }

    return result;
}

inline Tensor attentionWeights(const Tensor& scores) {
    return softmaxRows(applyCausalMask(scores));
}

} // namespace tfs
