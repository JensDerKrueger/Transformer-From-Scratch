#pragma once

#include "tfs/tensor.h"

#include <cmath>
#include <cstddef>

namespace tfs {

inline TensorValue geluDerivative(const TensorValue x) {
    constexpr TensorValue invSqrtTwo = 0.70710678118654752440f;
    constexpr TensorValue invSqrtTwoPi = 0.39894228040143267794f;

    const TensorValue cdf = 0.5f * (1.0f + std::erf(x * invSqrtTwo));
    const TensorValue pdf = invSqrtTwoPi * std::exp(-0.5f * x * x);
    return cdf + x * pdf;
}

inline Tensor geluBackward(const Tensor& input, const Tensor& outputGradient) {
    requireSameShape(input, outputGradient);

    Tensor result(input.getShape());
    const TensorValue* const inputData = input.data();
    const TensorValue* const outputGradientData = outputGradient.data();
    TensorValue* const resultData = result.data();

    for (std::size_t i = 0; i < result.size(); ++i) {
        resultData[i] = outputGradientData[i] * geluDerivative(inputData[i]);
    }

    return result;
}

} // namespace tfs
