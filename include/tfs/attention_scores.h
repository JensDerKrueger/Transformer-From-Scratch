#pragma once

#include "tfs/tensor.h"
#include "tfs/tensor_operations.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace tfs {

inline Tensor attentionScores(const Tensor& queries, const Tensor& keys) {
    requireMatrix(queries, "queries");
    requireMatrix(keys, "keys");

    const TensorShape& queryShape = queries.getShape();
    const TensorShape& keyShape = keys.getShape();

    const std::size_t projectionSize = queryShape[1];
    if (projectionSize != keyShape[1]) {
        throw std::runtime_error("Query and key projection sizes do not match");
    }

    Tensor scores = matrixMultiply(queries, transposeMatrix(keys));
    const TensorValue scale = 1.0f / std::sqrt(static_cast<TensorValue>(projectionSize));

    return multiply(scores, scale);
}

} // namespace tfs
