#pragma once

#include "tfs/gradient_utils.h"
#include "tfs/tensor.h"
#include "tfs/trainable_attention.h"
#include "tfs/trainable_feed_forward_block.h"
#include "tfs/trainable_layer_norm.h"

#include <cstddef>
#include <utility>
#include <vector>

namespace tfs {

struct TrainableDecoderBlockForwardResult {
    TrainableAttentionForwardResult attentionResult;
    Tensor afterAttentionResidual;
    LayerNormForwardResult attentionNormResult;
    Tensor afterAttentionNorm;
    TrainableFeedForwardForwardResult feedForwardResult;
    Tensor output;
};

class TrainableDecoderBlock {
public:
    TrainableDecoderBlock(
        const std::size_t modelSize,
        const std::size_t attentionSize,
        const std::size_t hiddenSize
    )
        : attention(modelSize, attentionSize),
          attentionNorm(modelSize),
          feedForward(modelSize, hiddenSize) {
    }

    std::vector<Parameter*> parameters() {
        std::vector<Parameter*> result;
        appendParameters(result, attention.parameters());
        appendParameters(result, attentionNorm.parameters());
        appendParameters(result, feedForward.parameters());
        return result;
    }

    TrainableSelfAttention& getAttention() {
        return attention;
    }

    TrainableLayerNorm& getAttentionNorm() {
        return attentionNorm;
    }

    TrainableFeedForwardBlock& getFeedForward() {
        return feedForward;
    }

    void initialize(RandomInitializer& initializer) {
        attention.initialize(initializer);
        feedForward.initialize(initializer);
    }

    TrainableDecoderBlockForwardResult forwardDetailed(const Tensor& input) const {
        TrainableAttentionForwardResult attentionResult = attention.forwardDetailed(input);
        Tensor afterAttentionResidual = add(input, attentionResult.output);
        LayerNormForwardResult attentionNormResult = attentionNorm.forwardDetailed(afterAttentionResidual);
        Tensor afterAttentionNorm = attentionNormResult.output;
        TrainableFeedForwardForwardResult feedForwardResult = feedForward.forwardDetailed(afterAttentionNorm);
        Tensor output = feedForwardResult.output;

        return TrainableDecoderBlockForwardResult{
            std::move(attentionResult),
            std::move(afterAttentionResidual),
            std::move(attentionNormResult),
            std::move(afterAttentionNorm),
            std::move(feedForwardResult),
            std::move(output)
        };
    }

    Tensor forward(const Tensor& input) const {
        return forwardDetailed(input).output;
    }

    Tensor backward(
        const Tensor& input,
        const TrainableDecoderBlockForwardResult& forwardResult,
        const Tensor& outputGradient
    ) {
        Tensor afterAttentionNormGradient = feedForward.backward(
            forwardResult.afterAttentionNorm,
            forwardResult.feedForwardResult,
            outputGradient
        );
        Tensor afterAttentionResidualGradient = attentionNorm.backward(
            forwardResult.afterAttentionResidual,
            forwardResult.attentionNormResult,
            afterAttentionNormGradient
        );
        Tensor inputGradient = afterAttentionResidualGradient;
        addInPlace(inputGradient, attention.backward(input, forwardResult.attentionResult, afterAttentionResidualGradient));

        return inputGradient;
    }

private:
    static void appendParameters(std::vector<Parameter*>& destination, const std::vector<Parameter*>& source) {
        destination.insert(destination.end(), source.begin(), source.end());
    }

    TrainableSelfAttention attention;
    TrainableLayerNorm attentionNorm;
    TrainableFeedForwardBlock feedForward;
};

} // namespace tfs
