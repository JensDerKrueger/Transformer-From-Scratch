#pragma once

#include "tfs/cross_entropy_loss.h"
#include "tfs/gradient_utils.h"
#include "tfs/random_initializer.h"
#include "tfs/sgd_optimizer.h"
#include "tfs/softmax.h"
#include "tfs/tensor.h"
#include "tfs/trainable_decoder_block.h"
#include "tfs/trainable_embedding.h"
#include "tfs/trainable_linear_layer.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tfs {

struct TinyLanguageModelConfig {
    std::size_t vocabSize = 256;
    std::size_t contextLength = 16;
    std::size_t modelSize = 16;
    std::size_t attentionSize = 16;
    std::size_t hiddenSize = 32;
};

struct TinyLanguageModelForwardResult {
    std::vector<std::size_t> inputTokens;
    Tensor tokenVectors;
    Tensor positionVectors;
    Tensor transformerInput;
    TrainableDecoderBlockForwardResult blockResult;
    Tensor logits;
};

class TinyLanguageModel {
public:
    explicit TinyLanguageModel(TinyLanguageModelConfig config)
        : config(config),
          tokenEmbedding(config.vocabSize, config.modelSize),
          positionEmbedding(config.contextLength, config.modelSize),
          block(config.modelSize, config.attentionSize, config.hiddenSize),
          head(config.modelSize, config.vocabSize) {
    }

    const TinyLanguageModelConfig& getConfig() const {
        return config;
    }

    std::vector<Parameter*> parameters() {
        std::vector<Parameter*> result;
        appendParameters(result, tokenEmbedding.parameters());
        appendParameters(result, positionEmbedding.parameters());
        appendParameters(result, block.parameters());
        appendParameters(result, head.parameters());
        return result;
    }

    void initialize(RandomInitializer& initializer) {
        tokenEmbedding.initialize(initializer);
        positionEmbedding.initialize(initializer);
        initializeLinear(initializer, block.getAttention().getQueryLayer());
        initializeLinear(initializer, block.getAttention().getKeyLayer());
        initializeLinear(initializer, block.getAttention().getValueLayer());
        initializeLinear(initializer, block.getAttention().getOutputLayer());
        initializeLinear(initializer, block.getFeedForward().getExpandLayer());
        initializeLinear(initializer, block.getFeedForward().getProjectLayer());
        initializeLinear(initializer, head);
    }

    TinyLanguageModelForwardResult forwardDetailed(const std::vector<std::size_t>& inputTokens) const {
        if (inputTokens.empty() || inputTokens.size() > config.contextLength) {
            throw std::runtime_error("Input token count must be in range 1..contextLength");
        }

        Tensor tokenVectors = tokenEmbedding.forward(inputTokens);
        Tensor positionVectors = positionEmbedding.forward(inputTokens.size());
        Tensor transformerInput = add(tokenVectors, positionVectors);
        TrainableDecoderBlockForwardResult blockResult = block.forwardDetailed(transformerInput);
        Tensor logits = head.forward(blockResult.output);

        return TinyLanguageModelForwardResult{
            inputTokens,
            std::move(tokenVectors),
            std::move(positionVectors),
            std::move(transformerInput),
            std::move(blockResult),
            std::move(logits)
        };
    }

    Tensor forward(const std::vector<std::size_t>& inputTokens) const {
        return forwardDetailed(inputTokens).logits;
    }

    void backward(
        const TinyLanguageModelForwardResult& forwardResult,
        const Tensor& logitsGradient
    ) {
        Tensor blockOutputGradient = head.backward(forwardResult.blockResult.output, logitsGradient);
        Tensor transformerInputGradient = block.backward(
            forwardResult.transformerInput,
            forwardResult.blockResult,
            blockOutputGradient
        );
        tokenEmbedding.backward(forwardResult.inputTokens, transformerInputGradient);
        positionEmbedding.backward(transformerInputGradient);
    }

    TensorValue trainOneWindow(
        const std::vector<std::size_t>& inputTokens,
        const std::vector<std::size_t>& targetTokens,
        const TensorValue learningRate
    ) {
        if (inputTokens.size() != targetTokens.size()) {
            throw std::runtime_error("Input and target token counts must match");
        }

        const std::vector<Parameter*> params = parameters();
        zeroGradients(params);
        const TinyLanguageModelForwardResult forwardResult = forwardDetailed(inputTokens);
        const CrossEntropyResult loss = crossEntropyLoss(forwardResult.logits, targetTokens);
        backward(forwardResult, loss.gradient);
        sgdStep(params, learningRate);
        return loss.loss;
    }

    TensorValue trainNextToken(
        const std::vector<std::size_t>& inputTokens,
        const std::size_t targetToken,
        const TensorValue learningRate
    ) {
        const std::vector<Parameter*> params = parameters();
        zeroGradients(params);

        const TinyLanguageModelForwardResult forwardResult = forwardDetailed(inputTokens);
        const Tensor probabilities = softmaxRows(forwardResult.logits);
        Tensor logitsGradient(forwardResult.logits.getShape());

        const std::size_t row = probabilities.getShape()[0] - 1;
        const std::size_t columns = probabilities.getShape()[1];
        if (targetToken >= columns) {
            throw std::runtime_error("Target token is out of vocabulary");
        }

        const TensorValue* const probabilityData = probabilities.data();
        TensorValue* const gradientData = logitsGradient.data();
        const std::size_t rowOffset = row * columns;

        for (std::size_t column = 0; column < columns; ++column) {
            gradientData[rowOffset + column] = probabilityData[rowOffset + column];
        }
        gradientData[rowOffset + targetToken] -= 1.0f;

        const TensorValue loss = -std::log(probabilityData[rowOffset + targetToken]);
        backward(forwardResult, logitsGradient);
        sgdStep(params, learningRate);
        return loss;
    }

private:
    static void appendParameters(std::vector<Parameter*>& destination, const std::vector<Parameter*>& source) {
        destination.insert(destination.end(), source.begin(), source.end());
    }

    static void initializeLinear(RandomInitializer& initializer, TrainableLinearLayer& layer) {
        initializer.fillXavier(layer.getWeights().getValue(), layer.inputSize(), layer.outputSize());
        layer.getBias().getValue().fill(0.0f);
    }

    TinyLanguageModelConfig config;
    TrainableEmbedding tokenEmbedding;
    TrainablePositionEmbedding positionEmbedding;
    TrainableDecoderBlock block;
    TrainableLinearLayer head;
};

} // namespace tfs
