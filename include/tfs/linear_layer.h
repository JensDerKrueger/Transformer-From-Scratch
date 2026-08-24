#pragma once

#include "tfs/tensor.h"
#include "tfs/tensor_operations.h"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tfs {

class LinearLayer {
public:
    LinearLayer(const std::size_t inputSize, const std::size_t outputSize)
        : weights(TensorShape({inputSize, outputSize})),
          bias(TensorShape({outputSize})) {
    }

    LinearLayer(
        const std::size_t inputSize,
        const std::size_t outputSize,
        std::vector<TensorValue> weightValues,
        std::vector<TensorValue> biasValues
    )
        : weights(TensorShape({inputSize, outputSize}), std::move(weightValues)),
          bias(TensorShape({outputSize}), std::move(biasValues)) {
    }

    std::size_t inputSize() const {
        return weights.getShape()[0];
    }

    std::size_t outputSize() const {
        return weights.getShape()[1];
    }

    const Tensor& getWeights() const {
        return weights;
    }

    Tensor& getWeights() {
        return weights;
    }

    const Tensor& getBias() const {
        return bias;
    }

    Tensor& getBias() {
        return bias;
    }

    Tensor forward(const Tensor& input) const {
        requireMatrix(input, "input");

        const TensorShape& inputShape = input.getShape();
        if (inputShape[1] != inputSize()) {
            throw std::runtime_error("Linear layer input size does not match tensor shape");
        }

        Tensor result = matrixMultiply(input, weights);

        const std::vector<std::size_t>& resultStrides = result.getStrides();
        TensorValue* const resultData = result.data();
        const TensorValue* const biasData = bias.data();

        const std::size_t rows = result.getShape()[0];
        const std::size_t columns = result.getShape()[1];
        const std::size_t resultRowStride = resultStrides[0];
        const std::size_t resultColumnStride = resultStrides[1];

        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t rowOffset = row * resultRowStride;

            for (std::size_t column = 0; column < columns; ++column) {
                resultData[rowOffset + column * resultColumnStride] += biasData[column];
            }
        }

        return result;
    }

private:
    Tensor weights;
    Tensor bias;
};

} // namespace tfs
