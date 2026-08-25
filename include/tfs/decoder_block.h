#pragma once

#include "tfs/attention_output.h"
#include "tfs/attention_output_projection.h"
#include "tfs/attention_projections.h"
#include "tfs/attention_scores.h"
#include "tfs/attention_weights.h"
#include "tfs/feed_forward_block.h"
#include "tfs/layer_norm.h"
#include "tfs/residual_connection.h"
#include "tfs/tensor.h"

#include <utility>

namespace tfs {

struct DecoderBlockResult {
    AttentionProjectionResult projections;
    Tensor scores;
    Tensor weights;
    Tensor attentionContext;
    Tensor projectedAttention;
    Tensor afterAttentionResidual;
    Tensor afterAttentionNorm;
    FeedForwardBlockResult feedForwardBlock;
    Tensor output;
};

class DecoderBlock {
public:
    DecoderBlock(
        AttentionProjections attentionProjections,
        AttentionOutputProjection attentionOutputProjection,
        LayerNorm attentionNorm,
        FeedForwardBlock feedForwardBlock
    )
        : attentionProjections(std::move(attentionProjections)),
          attentionOutputProjection(std::move(attentionOutputProjection)),
          attentionNorm(std::move(attentionNorm)),
          feedForwardBlock(std::move(feedForwardBlock)) {
    }

    DecoderBlockResult forwardDetailed(const Tensor& input) const {
        AttentionProjectionResult projections = attentionProjections.forward(input);
        Tensor scores = attentionScores(projections.queries, projections.keys);
        Tensor weights = attentionWeights(scores);
        Tensor attentionContext = attentionOutput(weights, projections.values);
        Tensor projectedAttention = attentionOutputProjection.forward(attentionContext);
        Tensor afterAttentionResidual = residualAdd(input, projectedAttention);
        Tensor afterAttentionNorm = attentionNorm.forward(afterAttentionResidual);
        FeedForwardBlockResult feedForwardBlockResult = feedForwardBlock.forwardDetailed(afterAttentionNorm);
        Tensor output = feedForwardBlockResult.output;

        return DecoderBlockResult{
            std::move(projections),
            std::move(scores),
            std::move(weights),
            std::move(attentionContext),
            std::move(projectedAttention),
            std::move(afterAttentionResidual),
            std::move(afterAttentionNorm),
            std::move(feedForwardBlockResult),
            std::move(output)
        };
    }

    Tensor forward(const Tensor& input) const {
        return forwardDetailed(input).output;
    }

private:
    AttentionProjections attentionProjections;
    AttentionOutputProjection attentionOutputProjection;
    LayerNorm attentionNorm;
    FeedForwardBlock feedForwardBlock;
};

} // namespace tfs
