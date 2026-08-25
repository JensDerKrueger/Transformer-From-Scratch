#pragma once

#include "tfs/feed_forward.h"
#include "tfs/layer_norm.h"
#include "tfs/residual_connection.h"
#include "tfs/tensor.h"

#include <utility>

namespace tfs {

struct FeedForwardBlockResult {
    FeedForwardResult feedForward;
    Tensor afterResidual;
    Tensor output;
};

class FeedForwardBlock {
public:
    FeedForwardBlock(FeedForwardNetwork feedForward, LayerNorm norm)
        : feedForward(std::move(feedForward)),
          norm(std::move(norm)) {
    }

    const FeedForwardNetwork& getFeedForward() const {
        return feedForward;
    }

    const LayerNorm& getNorm() const {
        return norm;
    }

    FeedForwardBlockResult forwardDetailed(const Tensor& input) const {
        FeedForwardResult feedForwardResult = feedForward.forwardDetailed(input);
        Tensor afterResidual = residualAdd(input, feedForwardResult.output);
        Tensor output = norm.forward(afterResidual);

        return FeedForwardBlockResult{
            std::move(feedForwardResult),
            std::move(afterResidual),
            std::move(output)
        };
    }

    Tensor forward(const Tensor& input) const {
        return forwardDetailed(input).output;
    }

private:
    FeedForwardNetwork feedForward;
    LayerNorm norm;
};

} // namespace tfs
