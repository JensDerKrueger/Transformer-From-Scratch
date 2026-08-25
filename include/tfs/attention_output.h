#pragma once

#include "tfs/tensor.h"
#include "tfs/tensor_operations.h"

#include <cstddef>
#include <stdexcept>

namespace tfs {

inline Tensor attentionOutput(const Tensor& weights, const Tensor& values) {
    requireMatrix(weights, "weights");
    requireMatrix(values, "values");

    const TensorShape& weightShape = weights.getShape();
    const TensorShape& valueShape = values.getShape();

    if (weightShape[1] != valueShape[0]) {
        throw std::runtime_error("Attention weight columns must match value rows");
    }

    return matrixMultiply(weights, values);
}

} // namespace tfs
