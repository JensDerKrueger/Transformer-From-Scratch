#pragma once

#include "tfs/activation.h"
#include "tfs/linear_layer.h"
#include "tfs/tensor.h"

#include <cstddef>
#include <utility>
#include <vector>

namespace tfs {

struct FeedForwardResult {
    Tensor hidden;
    Tensor activated;
    Tensor output;
};

class FeedForwardNetwork {
public:
    FeedForwardNetwork(const std::size_t modelSize, const std::size_t hiddenSize)
        : expandLayer(modelSize, hiddenSize),
          projectLayer(hiddenSize, modelSize) {
    }

    FeedForwardNetwork(
        const std::size_t modelSize,
        const std::size_t hiddenSize,
        std::vector<TensorValue> expandWeightValues,
        std::vector<TensorValue> expandBiasValues,
        std::vector<TensorValue> projectWeightValues,
        std::vector<TensorValue> projectBiasValues
    )
        : expandLayer(modelSize, hiddenSize, std::move(expandWeightValues), std::move(expandBiasValues)),
          projectLayer(hiddenSize, modelSize, std::move(projectWeightValues), std::move(projectBiasValues)) {
    }

    std::size_t modelSize() const {
        return expandLayer.inputSize();
    }

    std::size_t hiddenSize() const {
        return expandLayer.outputSize();
    }

    const LinearLayer& getExpandLayer() const {
        return expandLayer;
    }

    const LinearLayer& getProjectLayer() const {
        return projectLayer;
    }

    FeedForwardResult forwardDetailed(const Tensor& input) const {
        Tensor hidden = expandLayer.forward(input);
        Tensor activated = gelu(hidden);
        Tensor output = projectLayer.forward(activated);

        return FeedForwardResult{
            std::move(hidden),
            std::move(activated),
            std::move(output)
        };
    }

    Tensor forward(const Tensor& input) const {
        return forwardDetailed(input).output;
    }

private:
    LinearLayer expandLayer;
    LinearLayer projectLayer;
};

} // namespace tfs
