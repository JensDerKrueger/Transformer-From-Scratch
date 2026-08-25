#pragma once

#include "tfs/attention_scores.h"
#include "tfs/attention_weights.h"
#include "tfs/gradient_utils.h"
#include "tfs/random_initializer.h"
#include "tfs/tensor.h"
#include "tfs/tensor_operations.h"
#include "tfs/trainable_linear_layer.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tfs {

struct TrainableAttentionForwardResult {
    Tensor queries;
    Tensor keys;
    Tensor values;
    Tensor scores;
    Tensor weights;
    Tensor context;
    Tensor output;
};

class TrainableSelfAttention {
public:
    TrainableSelfAttention(const std::size_t modelSize, const std::size_t attentionSize)
        : queryLayer(modelSize, attentionSize),
          keyLayer(modelSize, attentionSize),
          valueLayer(modelSize, attentionSize),
          outputLayer(attentionSize, modelSize) {
    }

    std::size_t modelSize() const {
        return queryLayer.inputSize();
    }

    std::size_t attentionSize() const {
        return queryLayer.outputSize();
    }

    std::vector<Parameter*> parameters() {
        std::vector<Parameter*> result;
        appendParameters(result, queryLayer.parameters());
        appendParameters(result, keyLayer.parameters());
        appendParameters(result, valueLayer.parameters());
        appendParameters(result, outputLayer.parameters());
        return result;
    }

    TrainableLinearLayer& getQueryLayer() {
        return queryLayer;
    }

    TrainableLinearLayer& getKeyLayer() {
        return keyLayer;
    }

    TrainableLinearLayer& getValueLayer() {
        return valueLayer;
    }

    TrainableLinearLayer& getOutputLayer() {
        return outputLayer;
    }

    void initialize(RandomInitializer& initializer) {
        initializeLinear(initializer, queryLayer);
        initializeLinear(initializer, keyLayer);
        initializeLinear(initializer, valueLayer);
        initializeLinear(initializer, outputLayer);
    }

    TrainableAttentionForwardResult forwardDetailed(const Tensor& input) const {
        Tensor queries = queryLayer.forward(input);
        Tensor keys = keyLayer.forward(input);
        Tensor values = valueLayer.forward(input);
        Tensor scores = attentionScores(queries, keys);
        Tensor weights = attentionWeights(scores);
        Tensor context = matrixMultiply(weights, values);
        Tensor output = outputLayer.forward(context);

        return TrainableAttentionForwardResult{
            std::move(queries),
            std::move(keys),
            std::move(values),
            std::move(scores),
            std::move(weights),
            std::move(context),
            std::move(output)
        };
    }

    Tensor forward(const Tensor& input) const {
        return forwardDetailed(input).output;
    }

    Tensor backward(
        const Tensor& input,
        const TrainableAttentionForwardResult& forwardResult,
        const Tensor& outputGradient
    ) {
        Tensor contextGradient = outputLayer.backward(forwardResult.context, outputGradient);
        Tensor weightGradient(forwardResult.weights.getShape());
        Tensor valueGradient(forwardResult.values.getShape());

        attentionOutputBackward(
            forwardResult.weights,
            forwardResult.values,
            contextGradient,
            weightGradient,
            valueGradient
        );

        const Tensor scoreGradient = attentionWeightBackward(forwardResult.weights, weightGradient);
        Tensor queryGradient(forwardResult.queries.getShape());
        Tensor keyGradient(forwardResult.keys.getShape());

        attentionScoreBackward(
            forwardResult.queries,
            forwardResult.keys,
            scoreGradient,
            queryGradient,
            keyGradient
        );

        Tensor inputGradient = queryLayer.backward(input, queryGradient);
        addInPlace(inputGradient, keyLayer.backward(input, keyGradient));
        addInPlace(inputGradient, valueLayer.backward(input, valueGradient));

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

    static void attentionOutputBackward(
        const Tensor& weights,
        const Tensor& values,
        const Tensor& outputGradient,
        Tensor& weightGradient,
        Tensor& valueGradient
    ) {
        const std::size_t rows = weights.getShape()[0];
        const std::size_t columns = weights.getShape()[1];
        const std::size_t valueColumns = values.getShape()[1];

        const TensorValue* const weightData = weights.data();
        const TensorValue* const valueData = values.data();
        const TensorValue* const outputGradientData = outputGradient.data();
        TensorValue* const weightGradientData = weightGradient.data();
        TensorValue* const valueGradientData = valueGradient.data();

        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                TensorValue weightGrad = 0.0f;

                for (std::size_t valueColumn = 0; valueColumn < valueColumns; ++valueColumn) {
                    const TensorValue upstream = outputGradientData[row * valueColumns + valueColumn];
                    weightGrad += upstream * valueData[column * valueColumns + valueColumn];
                    valueGradientData[column * valueColumns + valueColumn] +=
                        weightData[row * columns + column] * upstream;
                }

                weightGradientData[row * columns + column] += weightGrad;
            }
        }
    }

    static Tensor attentionWeightBackward(const Tensor& weights, const Tensor& weightGradient) {
        Tensor scoreGradient(weights.getShape());
        const std::size_t rows = weights.getShape()[0];
        const std::size_t columns = weights.getShape()[1];
        const TensorValue* const weightData = weights.data();
        const TensorValue* const weightGradientData = weightGradient.data();
        TensorValue* const scoreGradientData = scoreGradient.data();

        for (std::size_t row = 0; row < rows; ++row) {
            const std::size_t rowOffset = row * columns;
            TensorValue dot = 0.0f;

            for (std::size_t column = 0; column < columns; ++column) {
                dot += weightGradientData[rowOffset + column] * weightData[rowOffset + column];
            }

            for (std::size_t column = 0; column < columns; ++column) {
                if (column > row) {
                    scoreGradientData[rowOffset + column] = 0.0f;
                } else {
                    scoreGradientData[rowOffset + column] =
                        weightData[rowOffset + column] * (weightGradientData[rowOffset + column] - dot);
                }
            }
        }

        return scoreGradient;
    }

    static void attentionScoreBackward(
        const Tensor& queries,
        const Tensor& keys,
        const Tensor& scoreGradient,
        Tensor& queryGradient,
        Tensor& keyGradient
    ) {
        const std::size_t rows = queries.getShape()[0];
        const std::size_t width = queries.getShape()[1];
        const TensorValue scale = 1.0f / std::sqrt(static_cast<TensorValue>(width));

        const TensorValue* const queryData = queries.data();
        const TensorValue* const keyData = keys.data();
        const TensorValue* const scoreGradientData = scoreGradient.data();
        TensorValue* const queryGradientData = queryGradient.data();
        TensorValue* const keyGradientData = keyGradient.data();

        for (std::size_t queryRow = 0; queryRow < rows; ++queryRow) {
            for (std::size_t keyRow = 0; keyRow < rows; ++keyRow) {
                const TensorValue upstream = scoreGradientData[queryRow * rows + keyRow] * scale;

                for (std::size_t column = 0; column < width; ++column) {
                    queryGradientData[queryRow * width + column] += upstream * keyData[keyRow * width + column];
                    keyGradientData[keyRow * width + column] += upstream * queryData[queryRow * width + column];
                }
            }
        }
    }

    TrainableLinearLayer queryLayer;
    TrainableLinearLayer keyLayer;
    TrainableLinearLayer valueLayer;
    TrainableLinearLayer outputLayer;
};

} // namespace tfs
