#pragma once

#include "demo_helpers.h"

#include "tfs/attention_output.h"
#include "tfs/attention_output_projection.h"
#include "tfs/attention_projections.h"
#include "tfs/attention_scores.h"
#include "tfs/attention_weights.h"
#include "tfs/decoder_block.h"
#include "tfs/feed_forward.h"
#include "tfs/feed_forward_block.h"
#include "tfs/layer_norm.h"
#include "tfs/language_model_head.h"
#include "tfs/position_embedding.h"
#include "tfs/residual_connection.h"
#include "tfs/token_embedding.h"

#include <limits>
#include <stdexcept>
#include <vector>

namespace demo {

inline tfs::TokenEmbedding makeDemoTokenEmbedding() {
    return tfs::TokenEmbedding(
        6,
        3,
        {
            0.00f, 0.01f, 0.02f,
            0.10f, 0.11f, 0.12f,
            0.20f, 0.21f, 0.22f,
            0.30f, 0.31f, 0.32f,
            0.40f, 0.41f, 0.42f,
            0.50f, 0.51f, 0.52f
        }
    );
}

inline tfs::PositionEmbedding makeDemoPositionEmbedding() {
    return tfs::PositionEmbedding(
        4,
        3,
        {
            0.00f, 1.00f, 2.00f,
            0.01f, 1.01f, 2.01f,
            0.02f, 1.02f, 2.02f,
            0.03f, 1.03f, 2.03f
        }
    );
}

inline tfs::AttentionProjections makeDemoAttentionProjections() {
    return tfs::AttentionProjections(
        3,
        2,
        {
            0.50f, -0.25f,
            1.00f, 0.00f,
            -0.50f, 0.75f
        },
        {0.10f, -0.20f},
        {
            0.25f, 0.50f,
            -0.50f, 0.25f,
            1.00f, -0.75f
        },
        {0.00f, 0.10f},
        {
            1.00f, 0.00f,
            0.00f, 1.00f,
            0.50f, 0.50f
        },
        {-0.10f, 0.20f}
    );
}

inline tfs::AttentionOutputProjection makeDemoAttentionOutputProjection() {
    return tfs::AttentionOutputProjection(
        2,
        3,
        {
            0.25f, 0.50f, -0.25f,
            -0.50f, 0.25f, 0.75f
        },
        {0.05f, -0.10f, 0.20f}
    );
}

inline tfs::FeedForwardNetwork makeDemoFeedForwardNetwork() {
    return tfs::FeedForwardNetwork(
        3,
        5,
        {
            0.30f, -0.20f, 0.10f, 0.40f, -0.30f,
            -0.10f, 0.50f, 0.20f, -0.40f, 0.10f,
            0.25f, 0.10f, -0.35f, 0.15f, 0.45f
        },
        {0.05f, -0.05f, 0.10f, 0.00f, -0.10f},
        {
            0.20f, -0.30f, 0.10f,
            -0.40f, 0.10f, 0.25f,
            0.15f, 0.35f, -0.20f,
            0.30f, -0.25f, 0.40f,
            -0.10f, 0.20f, 0.30f
        },
        {0.02f, -0.03f, 0.04f}
    );
}

inline tfs::FeedForwardBlock makeDemoFeedForwardBlock() {
    return tfs::FeedForwardBlock(
        makeDemoFeedForwardNetwork(),
        tfs::LayerNorm(3)
    );
}

inline tfs::DecoderBlock makeDemoDecoderBlock() {
    return tfs::DecoderBlock(
        makeDemoAttentionProjections(),
        makeDemoAttentionOutputProjection(),
        tfs::LayerNorm(3),
        makeDemoFeedForwardBlock()
    );
}

inline tfs::LanguageModelHead makeDemoLanguageModelHead() {
    return tfs::LanguageModelHead(
        3,
        6,
        {
            0.10f, -0.20f, 0.30f, 0.00f, 0.25f, -0.15f,
            -0.05f, 0.40f, -0.10f, 0.20f, -0.30f, 0.35f,
            0.30f, 0.10f, -0.25f, 0.45f, 0.05f, -0.20f
        },
        {0.00f, 0.05f, -0.02f, 0.03f, 0.01f, -0.04f}
    );
}

inline std::vector<tfs::TokenId> makeDemoTokens() {
    return {3, 1, 4, 1};
}

inline tfs::Tensor makeDemoTransformerInput() {
    const tfs::TokenEmbedding tokenEmbedding = makeDemoTokenEmbedding();
    const tfs::PositionEmbedding positionEmbedding = makeDemoPositionEmbedding();
    const std::vector<tfs::TokenId> tokens = makeDemoTokens();
    const tfs::Tensor tokenVectors = tokenEmbedding.embed(tokens);

    return positionEmbedding.addTo(tokenVectors);
}

inline tfs::Tensor makeDemoAttentionNormOutput() {
    const tfs::Tensor transformerInput = makeDemoTransformerInput();
    const tfs::AttentionProjections projections = makeDemoAttentionProjections();
    const tfs::AttentionOutputProjection outputProjection = makeDemoAttentionOutputProjection();
    const tfs::LayerNorm layerNorm(3);
    const tfs::AttentionProjectionResult projectionsResult = projections.forward(transformerInput);
    const tfs::Tensor scores = tfs::attentionScores(projectionsResult.queries, projectionsResult.keys);
    const tfs::Tensor weights = tfs::attentionWeights(scores);
    const tfs::Tensor attentionContext = tfs::attentionOutput(weights, projectionsResult.values);
    const tfs::Tensor projected = outputProjection.forward(attentionContext);
    const tfs::Tensor afterResidual = tfs::residualAdd(transformerInput, projected);

    return layerNorm.forward(afterResidual);
}

inline std::size_t argmaxRow(const tfs::Tensor& matrix, const std::size_t row) {
    tfs::requireMatrix(matrix, "matrix");

    const tfs::TensorShape& shape = matrix.getShape();
    if (row >= shape[0]) {
        throw std::runtime_error("Row is out of bounds");
    }

    const std::vector<std::size_t>& strides = matrix.getStrides();
    const tfs::TensorValue* const values = matrix.data();
    const std::size_t rowOffset = row * strides[0];

    std::size_t bestColumn = 0;
    tfs::TensorValue bestValue = -std::numeric_limits<tfs::TensorValue>::infinity();
    for (std::size_t column = 0; column < shape[1]; ++column) {
        const tfs::TensorValue value = values[rowOffset + column * strides[1]];
        if (value > bestValue) {
            bestValue = value;
            bestColumn = column;
        }
    }

    return bestColumn;
}

} // namespace demo
