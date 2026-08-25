#pragma once

#include "tfs/parameter.h"
#include "tfs/tensor.h"
#include "tfs/tensor_operations.h"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tfs {

class TrainableLinearLayer {
public:
    TrainableLinearLayer(const std::size_t inputSize, const std::size_t outputSize)
        : weights(TensorShape({inputSize, outputSize})),
          bias(TensorShape({outputSize})) {
    }

    TrainableLinearLayer(
        const std::size_t inputSize,
        const std::size_t outputSize,
        std::vector<TensorValue> weightValues,
        std::vector<TensorValue> biasValues
    )
        : weights(TensorShape({inputSize, outputSize}), std::move(weightValues)),
          bias(TensorShape({outputSize}), std::move(biasValues)) {
    }

    std::size_t inputSize() const {
        return weights.getValue().getShape()[0];
    }

    std::size_t outputSize() const {
        return weights.getValue().getShape()[1];
    }

    Parameter& getWeights() {
        return weights;
    }

    const Parameter& getWeights() const {
        return weights;
    }

    Parameter& getBias() {
        return bias;
    }

    const Parameter& getBias() const {
        return bias;
    }

    std::vector<Parameter*> parameters() {
        return {&weights, &bias};
    }

    void zeroGradient() {
        weights.zeroGradient();
        bias.zeroGradient();
    }

    Tensor forward(const Tensor& input) const {
        requireMatrix(input, "input");

        if (input.getShape()[1] != inputSize()) {
            throw std::runtime_error("TrainableLinearLayer input size does not match tensor shape");
        }

        Tensor result = matrixMultiply(input, weights.getValue());
        TensorValue* const outputData = result.data();
        const TensorValue* const biasData = bias.getValue().data();
        const std::vector<std::size_t>& strides = result.getStrides();

        for (std::size_t row = 0; row < result.getShape()[0]; ++row) {
            const std::size_t rowOffset = row * strides[0];

            for (std::size_t column = 0; column < result.getShape()[1]; ++column) {
                outputData[rowOffset + column * strides[1]] += biasData[column];
            }
        }

        return result;
    }

    Tensor backward(const Tensor& input, const Tensor& outputGradient) {
        requireMatrix(input, "input");
        requireMatrix(outputGradient, "outputGradient");

        const TensorShape& inputShape = input.getShape();
        const TensorShape& gradientShape = outputGradient.getShape();

        if (inputShape[0] != gradientShape[0] || gradientShape[1] != outputSize()) {
            throw std::runtime_error("TrainableLinearLayer backward shape mismatch");
        }

        const std::size_t rows = inputShape[0];
        const std::size_t inputColumns = inputShape[1];
        const std::size_t outputColumns = gradientShape[1];

        Tensor inputGradient(inputShape);
        const TensorValue* const inputData = input.data();
        const TensorValue* const outputGradientData = outputGradient.data();
        const TensorValue* const weightData = weights.getValue().data();
        TensorValue* const inputGradientData = inputGradient.data();
        TensorValue* const weightGradientData = weights.getGradient().data();
        TensorValue* const biasGradientData = bias.getGradient().data();

        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t outputColumn = 0; outputColumn < outputColumns; ++outputColumn) {
                const TensorValue upstream = outputGradientData[row * outputColumns + outputColumn];
                biasGradientData[outputColumn] += upstream;

                for (std::size_t inputColumn = 0; inputColumn < inputColumns; ++inputColumn) {
                    weightGradientData[inputColumn * outputColumns + outputColumn] +=
                        inputData[row * inputColumns + inputColumn] * upstream;
                    inputGradientData[row * inputColumns + inputColumn] +=
                        upstream * weightData[inputColumn * outputColumns + outputColumn];
                }
            }
        }

        return inputGradient;
    }

private:
    Parameter weights;
    Parameter bias;
};

} // namespace tfs
