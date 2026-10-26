#pragma once

#include "tfs/tiny_language_model.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tfs {

struct ResponseLossResult {
    TensorValue loss;
    Tensor gradient;
};

// Teacher forcing: the prompt is context; only response predictions incur loss.
inline ResponseLossResult responseLoss(
    const Tensor& logits,
    const std::size_t promptLength,
    const std::vector<std::size_t>& responseTokens
) {
    requireMatrix(logits, "response logits");
    if (promptLength == 0 || responseTokens.empty()
        || logits.getShape()[0] != promptLength + responseTokens.size() - 1) {
        throw std::runtime_error("Response logits must cover prompt and response except the last token");
    }
    const std::size_t columns = logits.getShape()[1];
    const std::size_t firstRow = promptLength - 1;
    Tensor responseLogits({responseTokens.size(), columns});
    std::copy_n(logits.data() + firstRow * columns, responseLogits.size(), responseLogits.data());
    const CrossEntropyResult loss = crossEntropyLoss(responseLogits, responseTokens);
    Tensor fullGradient(logits.getShape());
    std::copy_n(loss.gradient.data(), loss.gradient.size(), fullGradient.data() + firstRow * columns);
    return ResponseLossResult{loss.loss, std::move(fullGradient)};
}

inline TensorValue trainResponse(
    TinyLanguageModel& model,
    const std::vector<std::size_t>& promptTokens,
    const std::vector<std::size_t>& responseTokens,
    const TensorValue learningRate
) {
    if (promptTokens.empty() || responseTokens.empty()) {
        throw std::runtime_error("Training needs a prompt and a response");
    }
    std::vector<std::size_t> inputTokens = promptTokens;
    inputTokens.insert(inputTokens.end(), responseTokens.begin(), responseTokens.end() - 1);
    const std::vector<Parameter*> parameters = model.parameters();
    zeroGradients(parameters);
    const TinyLanguageModelForwardResult forwardResult = model.forwardDetailed(inputTokens);
    const ResponseLossResult loss = responseLoss(forwardResult.logits, promptTokens.size(), responseTokens);
    model.backward(forwardResult, loss.gradient);
    sgdStep(parameters, learningRate);
    return loss.loss;
}

inline std::vector<std::size_t> generateResponse(
    const TinyLanguageModel& model,
    std::vector<std::size_t> context,
    const std::size_t endToken,
    const std::size_t maxTokens
) {
    std::vector<std::size_t> response;
    for (std::size_t step = 0; step < maxTokens; ++step) {
        if (context.size() > model.getConfig().contextLength) {
            throw std::runtime_error("Generation exceeded the context length");
        }
        const Tensor logits = model.forward(context);
        const std::size_t columns = logits.getShape()[1];
        const TensorValue* const lastRow = logits.data() + (context.size() - 1) * columns;
        const std::size_t nextToken = static_cast<std::size_t>(
            std::max_element(lastRow, lastRow + columns) - lastRow);
        response.push_back(nextToken);
        if (nextToken == endToken) { break; }
        context.push_back(nextToken);
    }
    return response;
}

} // namespace tfs
