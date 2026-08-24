#pragma once

#include "tfs/tensor.h"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tfs {

class PositionEmbedding {
public:
    PositionEmbedding(const std::size_t maxSequenceLength, const std::size_t embeddingSize)
        : weights(TensorShape({maxSequenceLength, embeddingSize})) {
    }

    PositionEmbedding(
        const std::size_t maxSequenceLength,
        const std::size_t embeddingSize,
        std::vector<TensorValue> values
    )
        : weights(TensorShape({maxSequenceLength, embeddingSize}), std::move(values)) {
    }

    std::size_t maxSequenceLength() const {
        return weights.getShape()[0];
    }

    std::size_t embeddingSize() const {
        return weights.getShape()[1];
    }

    const Tensor& getWeights() const {
        return weights;
    }

    Tensor& getWeights() {
        return weights;
    }

    Tensor embed(const std::size_t sequenceLength) const {
        if (sequenceLength == 0) {
            throw std::runtime_error("Sequence length must be greater than zero");
        }

        if (sequenceLength > maxSequenceLength()) {
            throw std::runtime_error("Sequence length exceeds maximum positional embedding length");
        }

        Tensor result(TensorShape({sequenceLength, embeddingSize()}));

        const std::vector<std::size_t>& weightStrides = weights.getStrides();
        const std::vector<std::size_t>& resultStrides = result.getStrides();

        const TensorValue* const weightData = weights.data();
        TensorValue* const resultData = result.data();

        const std::size_t weightRowStride = weightStrides[0];
        const std::size_t weightColumnStride = weightStrides[1];
        const std::size_t resultRowStride = resultStrides[0];
        const std::size_t resultColumnStride = resultStrides[1];

        for (std::size_t position = 0; position < sequenceLength; ++position) {
            const std::size_t weightRowOffset = position * weightRowStride;
            const std::size_t resultRowOffset = position * resultRowStride;

            for (std::size_t dimension = 0; dimension < embeddingSize(); ++dimension) {
                resultData[resultRowOffset + dimension * resultColumnStride] =
                    weightData[weightRowOffset + dimension * weightColumnStride];
            }
        }

        return result;
    }

    Tensor addTo(const Tensor& tokenEmbeddings) const {
        if (tokenEmbeddings.rank() != 2) {
            throw std::runtime_error("Token embeddings must be a matrix");
        }

        const TensorShape& tokenShape = tokenEmbeddings.getShape();
        const std::size_t sequenceLength = tokenShape[0];
        const std::size_t tokenEmbeddingSize = tokenShape[1];

        if (tokenEmbeddingSize != embeddingSize()) {
            throw std::runtime_error("Token embedding size does not match positional embedding size");
        }

        if (sequenceLength > maxSequenceLength()) {
            throw std::runtime_error("Sequence length exceeds maximum positional embedding length");
        }

        Tensor result(tokenEmbeddings.getShape());

        const std::vector<std::size_t>& tokenStrides = tokenEmbeddings.getStrides();
        const std::vector<std::size_t>& weightStrides = weights.getStrides();
        const std::vector<std::size_t>& resultStrides = result.getStrides();

        const TensorValue* const tokenData = tokenEmbeddings.data();
        const TensorValue* const weightData = weights.data();
        TensorValue* const resultData = result.data();

        const std::size_t tokenRowStride = tokenStrides[0];
        const std::size_t tokenColumnStride = tokenStrides[1];
        const std::size_t weightRowStride = weightStrides[0];
        const std::size_t weightColumnStride = weightStrides[1];
        const std::size_t resultRowStride = resultStrides[0];
        const std::size_t resultColumnStride = resultStrides[1];

        for (std::size_t position = 0; position < sequenceLength; ++position) {
            const std::size_t tokenRowOffset = position * tokenRowStride;
            const std::size_t weightRowOffset = position * weightRowStride;
            const std::size_t resultRowOffset = position * resultRowStride;

            for (std::size_t dimension = 0; dimension < embeddingSize(); ++dimension) {
                resultData[resultRowOffset + dimension * resultColumnStride] =
                    tokenData[tokenRowOffset + dimension * tokenColumnStride]
                    + weightData[weightRowOffset + dimension * weightColumnStride];
            }
        }

        return result;
    }

private:
    Tensor weights;
};

} // namespace tfs
