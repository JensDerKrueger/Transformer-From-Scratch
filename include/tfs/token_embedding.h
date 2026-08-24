#pragma once

#include "tfs/tensor.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tfs {

using TokenId = std::uint32_t;

class TokenEmbedding {
public:
    TokenEmbedding(const std::size_t vocabSize, const std::size_t embeddingSize)
        : weights(TensorShape({vocabSize, embeddingSize})) {
    }

    TokenEmbedding(
        const std::size_t vocabSize,
        const std::size_t embeddingSize,
        std::vector<TensorValue> values
    )
        : weights(TensorShape({vocabSize, embeddingSize}), std::move(values)) {
    }

    std::size_t vocabSize() const {
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

    Tensor embed(const std::vector<TokenId>& tokens) const {
        if (tokens.empty()) {
            throw std::runtime_error("Token sequence must not be empty");
        }

        Tensor result(TensorShape({tokens.size(), embeddingSize()}));

        const std::vector<std::size_t>& weightStrides = weights.getStrides();
        const std::vector<std::size_t>& resultStrides = result.getStrides();

        const TensorValue* const weightData = weights.data();
        TensorValue* const resultData = result.data();

        const std::size_t weightRowStride = weightStrides[0];
        const std::size_t weightColumnStride = weightStrides[1];
        const std::size_t resultRowStride = resultStrides[0];
        const std::size_t resultColumnStride = resultStrides[1];

        for (std::size_t position = 0; position < tokens.size(); ++position) {
            const TokenId token = tokens[position];
            if (token >= vocabSize()) {
                throw std::runtime_error("Token id is outside the embedding vocabulary");
            }

            const std::size_t weightRowOffset = static_cast<std::size_t>(token) * weightRowStride;
            const std::size_t resultRowOffset = position * resultRowStride;

            for (std::size_t dimension = 0; dimension < embeddingSize(); ++dimension) {
                resultData[resultRowOffset + dimension * resultColumnStride] =
                    weightData[weightRowOffset + dimension * weightColumnStride];
            }
        }

        return result;
    }

private:
    Tensor weights;
};

} // namespace tfs
