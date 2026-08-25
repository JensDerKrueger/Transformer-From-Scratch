#pragma once

#include "tfs/tensor.h"
#include "tfs/tensor_operations.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tfs {

class LayerNorm {
public:
    explicit LayerNorm(const std::size_t featureSize, const TensorValue epsilon = 1.0e-5f)
        : gamma(TensorShape({featureSize}), 1.0f),
          beta(TensorShape({featureSize})),
          epsilon(epsilon) {
    }

    LayerNorm(
        const std::size_t featureSize,
        std::vector<TensorValue> gammaValues,
        std::vector<TensorValue> betaValues,
        const TensorValue epsilon = 1.0e-5f
    )
        : gamma(TensorShape({featureSize}), std::move(gammaValues)),
          beta(TensorShape({featureSize}), std::move(betaValues)),
          epsilon(epsilon) {
    }

    std::size_t featureSize() const {
        return gamma.getShape()[0];
    }

    TensorValue getEpsilon() const {
        return epsilon;
    }

    const Tensor& getGamma() const {
        return gamma;
    }

    Tensor& getGamma() {
        return gamma;
    }

    const Tensor& getBeta() const {
        return beta;
    }

    Tensor& getBeta() {
        return beta;
    }

    Tensor forward(const Tensor& input) const {
        requireMatrix(input, "input");

        const TensorShape& inputShape = input.getShape();
        const std::size_t rows = inputShape[0];
        const std::size_t columns = inputShape[1];

        if (columns != featureSize()) {
            throw std::runtime_error("LayerNorm feature size does not match tensor shape");
        }

        Tensor result(input.getShape());

        const std::vector<std::size_t>& inputStrides = input.getStrides();
        const std::vector<std::size_t>& resultStrides = result.getStrides();

        const TensorValue* const inputData = input.data();
        const TensorValue* const gammaData = gamma.data();
        const TensorValue* const betaData = beta.data();
        TensorValue* const resultData = result.data();

        const std::size_t inputRowStride = inputStrides[0];
        const std::size_t inputColumnStride = inputStrides[1];
        const std::size_t resultRowStride = resultStrides[0];
        const std::size_t resultColumnStride = resultStrides[1];

        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t inputRowOffset = row * inputRowStride;
            const std::size_t resultRowOffset = row * resultRowStride;

            TensorValue mean = 0.0f;
            for (std::size_t column = 0; column < columns; ++column) {
                mean += inputData[inputRowOffset + column * inputColumnStride];
            }
            mean /= static_cast<TensorValue>(columns);

            TensorValue variance = 0.0f;
            for (std::size_t column = 0; column < columns; ++column) {
                const TensorValue centered = inputData[inputRowOffset + column * inputColumnStride] - mean;
                variance += centered * centered;
            }
            variance /= static_cast<TensorValue>(columns);

            const TensorValue inverseStddev = 1.0f / std::sqrt(variance + epsilon);
            for (std::size_t column = 0; column < columns; ++column) {
                const TensorValue value = inputData[inputRowOffset + column * inputColumnStride];
                const TensorValue normalized = (value - mean) * inverseStddev;
                resultData[resultRowOffset + column * resultColumnStride] =
                    normalized * gammaData[column] + betaData[column];
            }
        }

        return result;
    }

private:
    Tensor gamma;
    Tensor beta;
    TensorValue epsilon;
};

} // namespace tfs
