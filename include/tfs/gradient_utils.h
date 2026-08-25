#pragma once

#include "tfs/tensor.h"

#include <cstddef>

namespace tfs {

inline void addInPlace(Tensor& destination, const Tensor& source) {
    requireSameShape(destination, source);

    TensorValue* const destinationData = destination.data();
    const TensorValue* const sourceData = source.data();

    for (std::size_t i = 0; i < destination.size(); ++i) {
        destinationData[i] += sourceData[i];
    }
}

inline Tensor added(const Tensor& left, const Tensor& right) {
    Tensor result = left;
    addInPlace(result, right);
    return result;
}

} // namespace tfs
