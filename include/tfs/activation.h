#pragma once

#include "tfs/tensor.h"

#include <cmath>
#include <cstddef>

namespace tfs {

inline TensorValue geluValue(const TensorValue value) {
    const TensorValue inverseSqrtTwo = 0.7071067811865475f;
    return 0.5f * value * (1.0f + std::erf(value * inverseSqrtTwo));
}

inline Tensor gelu(const Tensor& input) {
    Tensor result(input.getShape());

    const TensorValue* const inputData = input.data();
    TensorValue* const resultData = result.data();

    for (std::size_t i = 0; i < input.size(); ++i) {
        resultData[i] = geluValue(inputData[i]);
    }

    return result;
}

} // namespace tfs
