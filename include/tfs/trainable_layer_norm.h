#pragma once

#include "tfs/parameter.h"
#include "tfs/tensor.h"
#include "tfs/tensor_operations.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tfs {

struct LayerNormForwardResult {
    Tensor output;
    Tensor normalized;
    std::vector<TensorValue> inverseStddev;
};

class TrainableLayerNorm {
public:
    explicit TrainableLayerNorm(const std::size_t featureSize, const TensorValue epsilon = 1.0e-5f)
        : gamma(TensorShape({featureSize}), std::vector<TensorValue>(featureSize, 1.0f)),
          beta(TensorShape({featureSize})),
          epsilon(epsilon) {
    }

    std::size_t featureSize() const {
        return gamma.getValue().getShape()[0];
    }

    TensorValue getEpsilon() const {
        return epsilon;
    }

    Parameter& getGamma() {
        return gamma;
    }

    Parameter& getBeta() {
        return beta;
    }

    std::vector<Parameter*> parameters() {
        return {&gamma, &beta};
    }

    LayerNormForwardResult forwardDetailed(const Tensor& input) const {
        requireMatrix(input, "input");

        const std::size_t rows = input.getShape()[0];
        const std::size_t columns = input.getShape()[1];
        if (columns != featureSize()) {
            throw std::runtime_error("LayerNorm feature size does not match tensor shape");
        }

        Tensor output(input.getShape());
        Tensor normalized(input.getShape());
        std::vector<TensorValue> inverseStddev(rows, 0.0f);

        const TensorValue* const inputData = input.data();
        const TensorValue* const gammaData = gamma.getValue().data();
        const TensorValue* const betaData = beta.getValue().data();
        TensorValue* const outputData = output.data();
        TensorValue* const normalizedData = normalized.data();

        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t rowOffset = row * columns;
            TensorValue mean = 0.0f;

            for (std::size_t column = 0; column < columns; ++column) {
                mean += inputData[rowOffset + column];
            }
            mean /= static_cast<TensorValue>(columns);

            TensorValue variance = 0.0f;
            for (std::size_t column = 0; column < columns; ++column) {
                const TensorValue centered = inputData[rowOffset + column] - mean;
                variance += centered * centered;
            }
            variance /= static_cast<TensorValue>(columns);

            const TensorValue invStd = 1.0f / std::sqrt(variance + epsilon);
            inverseStddev[row] = invStd;

            for (std::size_t column = 0; column < columns; ++column) {
                const TensorValue norm = (inputData[rowOffset + column] - mean) * invStd;
                normalizedData[rowOffset + column] = norm;
                outputData[rowOffset + column] = norm * gammaData[column] + betaData[column];
            }
        }

        return LayerNormForwardResult{
            std::move(output),
            std::move(normalized),
            std::move(inverseStddev)
        };
    }

    Tensor forward(const Tensor& input) const {
        return forwardDetailed(input).output;
    }

    Tensor backward(
        const Tensor& input,
        const LayerNormForwardResult& forwardResult,
        const Tensor& outputGradient
    ) {
        requireSameShape(input, outputGradient);
        requireSameShape(input, forwardResult.normalized);

        const std::size_t rows = input.getShape()[0];
        const std::size_t columns = input.getShape()[1];
        Tensor inputGradient(input.getShape());

        const TensorValue* const outputGradientData = outputGradient.data();
        const TensorValue* const normalizedData = forwardResult.normalized.data();
        const TensorValue* const gammaData = gamma.getValue().data();
        TensorValue* const gammaGradientData = gamma.getGradient().data();
        TensorValue* const betaGradientData = beta.getGradient().data();
        TensorValue* const inputGradientData = inputGradient.data();

        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t rowOffset = row * columns;
            TensorValue sumDNormalized = 0.0f;
            TensorValue sumDNormalizedTimesNormalized = 0.0f;

            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t index = rowOffset + column;
                gammaGradientData[column] += outputGradientData[index] * normalizedData[index];
                betaGradientData[column] += outputGradientData[index];

                const TensorValue dNormalized = outputGradientData[index] * gammaData[column];
                sumDNormalized += dNormalized;
                sumDNormalizedTimesNormalized += dNormalized * normalizedData[index];
            }

            const TensorValue invStd = forwardResult.inverseStddev[row];
            const TensorValue invColumns = 1.0f / static_cast<TensorValue>(columns);

            for (std::size_t column = 0; column < columns; ++column) {
                const std::size_t index = rowOffset + column;
                const TensorValue dNormalized = outputGradientData[index] * gammaData[column];
                inputGradientData[index] =
                    invStd * (dNormalized
                        - sumDNormalized * invColumns
                        - normalizedData[index] * sumDNormalizedTimesNormalized * invColumns);
            }
        }

        return inputGradient;
    }

private:
    Parameter gamma;
    Parameter beta;
    TensorValue epsilon;
};

} // namespace tfs
