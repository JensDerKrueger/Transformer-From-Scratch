#pragma once

#include "tfs/tensor.h"

#include <utility>
#include <vector>

namespace tfs {

class Parameter {
public:
    explicit Parameter(const TensorShape& shape)
        : value(shape),
          gradient(shape) {
    }

    Parameter(const TensorShape& shape, std::vector<TensorValue> values)
        : value(shape, std::move(values)),
          gradient(shape) {
    }

    const Tensor& getValue() const {
        return value;
    }

    Tensor& getValue() {
        return value;
    }

    const Tensor& getGradient() const {
        return gradient;
    }

    Tensor& getGradient() {
        return gradient;
    }

    void zeroGradient() {
        gradient.fill(0.0f);
    }

private:
    Tensor value;
    Tensor gradient;
};

} // namespace tfs
