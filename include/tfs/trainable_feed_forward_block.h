#pragma once

#include "tfs/activation.h"
#include "tfs/activation_backward.h"
#include "tfs/gradient_utils.h"
#include "tfs/random_initializer.h"
#include "tfs/tensor.h"
#include "tfs/trainable_layer_norm.h"
#include "tfs/trainable_linear_layer.h"

#include <cstddef>
#include <utility>
#include <vector>

namespace tfs {

struct TrainableFeedForwardForwardResult {
    Tensor hidden;
    Tensor activated;
    Tensor branchOutput;
    Tensor afterResidual;
    LayerNormForwardResult normResult;
    Tensor output;
};

class TrainableFeedForwardBlock {
public:
    TrainableFeedForwardBlock(const std::size_t modelSize, const std::size_t hiddenSize)
        : expandLayer(modelSize, hiddenSize),
          projectLayer(hiddenSize, modelSize),
          norm(modelSize) {
    }

    std::vector<Parameter*> parameters() {
        std::vector<Parameter*> result;
        appendParameters(result, expandLayer.parameters());
        appendParameters(result, projectLayer.parameters());
        appendParameters(result, norm.parameters());
        return result;
    }

    TrainableLinearLayer& getExpandLayer() {
        return expandLayer;
    }

    TrainableLinearLayer& getProjectLayer() {
        return projectLayer;
    }

    TrainableLayerNorm& getNorm() {
        return norm;
    }

    void initialize(RandomInitializer& initializer) {
        initializeLinear(initializer, expandLayer);
        initializeLinear(initializer, projectLayer);
    }

    TrainableFeedForwardForwardResult forwardDetailed(const Tensor& input) const {
        Tensor hidden = expandLayer.forward(input);
        Tensor activated = gelu(hidden);
        Tensor branchOutput = projectLayer.forward(activated);
        Tensor afterResidual = add(input, branchOutput);
        LayerNormForwardResult normResult = norm.forwardDetailed(afterResidual);
        Tensor output = normResult.output;

        return TrainableFeedForwardForwardResult{
            std::move(hidden),
            std::move(activated),
            std::move(branchOutput),
            std::move(afterResidual),
            std::move(normResult),
            std::move(output)
        };
    }

    Tensor forward(const Tensor& input) const {
        return forwardDetailed(input).output;
    }

    Tensor backward(
        const Tensor& input,
        const TrainableFeedForwardForwardResult& forwardResult,
        const Tensor& outputGradient
    ) {
        const Tensor afterResidualGradient = norm.backward(
            forwardResult.afterResidual,
            forwardResult.normResult,
            outputGradient
        );
        Tensor inputGradient = afterResidualGradient;
        Tensor branchGradient = afterResidualGradient;
        Tensor activatedGradient = projectLayer.backward(forwardResult.activated, branchGradient);
        Tensor hiddenGradient = geluBackward(forwardResult.hidden, activatedGradient);
        addInPlace(inputGradient, expandLayer.backward(input, hiddenGradient));

        return inputGradient;
    }

private:
    static void appendParameters(std::vector<Parameter*>& destination, const std::vector<Parameter*>& source) {
        destination.insert(destination.end(), source.begin(), source.end());
    }

    static void initializeLinear(RandomInitializer& initializer, TrainableLinearLayer& layer) {
        initializer.fillXavier(layer.getWeights().getValue(), layer.inputSize(), layer.outputSize());
        layer.getBias().getValue().fill(0.0f);
    }

    TrainableLinearLayer expandLayer;
    TrainableLinearLayer projectLayer;
    TrainableLayerNorm norm;
};

} // namespace tfs
