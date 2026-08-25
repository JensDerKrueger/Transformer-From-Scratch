#pragma once

#include "tfs/parameter.h"

#include <cstddef>
#include <vector>

namespace tfs {

inline void sgdStep(const std::vector<Parameter*>& parameters, const TensorValue learningRate) {
    for (Parameter* const parameter : parameters) {
        Tensor& value = parameter->getValue();
        const Tensor& gradient = parameter->getGradient();

        for (std::size_t i = 0; i < value.size(); ++i) {
            value[i] -= learningRate * gradient[i];
        }
    }
}

inline void zeroGradients(const std::vector<Parameter*>& parameters) {
    for (Parameter* const parameter : parameters) {
        parameter->zeroGradient();
    }
}

} // namespace tfs
