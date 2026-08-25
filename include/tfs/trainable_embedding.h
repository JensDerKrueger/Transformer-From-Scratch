#pragma once

#include "tfs/parameter.h"
#include "tfs/random_initializer.h"
#include "tfs/tensor.h"
#include "tfs/tensor_operations.h"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tfs {

class TrainableEmbedding {
public:
    TrainableEmbedding(const std::size_t vocabSize, const std::size_t embeddingSize)
        : weights(TensorShape({vocabSize, embeddingSize})) {
    }

    TrainableEmbedding(
        const std::size_t vocabSize,
        const std::size_t embeddingSize,
        std::vector<TensorValue> values
    )
        : weights(TensorShape({vocabSize, embeddingSize}), std::move(values)) {
    }

    std::size_t vocabSize() const {
        return weights.getValue().getShape()[0];
    }

    std::size_t embeddingSize() const {
        return weights.getValue().getShape()[1];
    }

    Parameter& getWeights() {
        return weights;
    }

    const Parameter& getWeights() const {
        return weights;
    }

    std::vector<Parameter*> parameters() {
        return {&weights};
    }

    void initialize(RandomInitializer& initializer) {
        initializer.fillUniform(weights.getValue(), -0.05f, 0.05f);
    }

    Tensor forward(const std::vector<std::size_t>& tokenIds) const {
        Tensor result(TensorShape({tokenIds.size(), embeddingSize()}));

        const TensorValue* const weightData = weights.getValue().data();
        TensorValue* const resultData = result.data();
        const std::size_t width = embeddingSize();

        for (std::size_t row = 0; row < tokenIds.size(); ++row) {
            const std::size_t token = tokenIds[row];
            if (token >= vocabSize()) {
                throw std::runtime_error("Token id is out of embedding vocabulary");
            }

            for (std::size_t column = 0; column < width; ++column) {
                resultData[row * width + column] = weightData[token * width + column];
            }
        }

        return result;
    }

    void backward(const std::vector<std::size_t>& tokenIds, const Tensor& outputGradient) {
        requireMatrix(outputGradient, "outputGradient");

        if (outputGradient.getShape()[0] != tokenIds.size() || outputGradient.getShape()[1] != embeddingSize()) {
            throw std::runtime_error("Embedding backward shape mismatch");
        }

        TensorValue* const gradientData = weights.getGradient().data();
        const TensorValue* const outputGradientData = outputGradient.data();
        const std::size_t width = embeddingSize();

        for (std::size_t row = 0; row < tokenIds.size(); ++row) {
            const std::size_t token = tokenIds[row];
            if (token >= vocabSize()) {
                throw std::runtime_error("Token id is out of embedding vocabulary");
            }

            for (std::size_t column = 0; column < width; ++column) {
                gradientData[token * width + column] += outputGradientData[row * width + column];
            }
        }
    }

private:
    Parameter weights;
};

class TrainablePositionEmbedding {
public:
    TrainablePositionEmbedding(const std::size_t maxSequenceLength, const std::size_t embeddingSize)
        : weights(TensorShape({maxSequenceLength, embeddingSize})) {
    }

    std::size_t maxSequenceLength() const {
        return weights.getValue().getShape()[0];
    }

    std::size_t embeddingSize() const {
        return weights.getValue().getShape()[1];
    }

    Parameter& getWeights() {
        return weights;
    }

    const Parameter& getWeights() const {
        return weights;
    }

    std::vector<Parameter*> parameters() {
        return {&weights};
    }

    void initialize(RandomInitializer& initializer) {
        initializer.fillUniform(weights.getValue(), -0.05f, 0.05f);
    }

    Tensor forward(const std::size_t sequenceLength) const {
        if (sequenceLength > maxSequenceLength()) {
            throw std::runtime_error("Sequence is longer than position embedding table");
        }

        Tensor result(TensorShape({sequenceLength, embeddingSize()}));
        const TensorValue* const weightData = weights.getValue().data();
        TensorValue* const resultData = result.data();
        const std::size_t width = embeddingSize();

        for (std::size_t row = 0; row < sequenceLength; ++row) {
            for (std::size_t column = 0; column < width; ++column) {
                resultData[row * width + column] = weightData[row * width + column];
            }
        }

        return result;
    }

    void backward(const Tensor& outputGradient) {
        requireMatrix(outputGradient, "outputGradient");

        if (outputGradient.getShape()[0] > maxSequenceLength() || outputGradient.getShape()[1] != embeddingSize()) {
            throw std::runtime_error("Position embedding backward shape mismatch");
        }

        TensorValue* const gradientData = weights.getGradient().data();
        const TensorValue* const outputGradientData = outputGradient.data();
        const std::size_t width = embeddingSize();

        for (std::size_t row = 0; row < outputGradient.getShape()[0]; ++row) {
            for (std::size_t column = 0; column < width; ++column) {
                gradientData[row * width + column] += outputGradientData[row * width + column];
            }
        }
    }

private:
    Parameter weights;
};

} // namespace tfs
