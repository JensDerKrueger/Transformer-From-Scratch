#include "demo_model.h"

#include <exception>
#include <iomanip>
#include <iostream>

int main() {
    try {
        const tfs::AttentionProjections projections = demo::makeDemoAttentionProjections();
        const tfs::AttentionOutputProjection outputProjection = demo::makeDemoAttentionOutputProjection();
        const tfs::LayerNorm layerNorm(3);
        const tfs::Tensor transformerInput = demo::makeDemoTransformerInput();
        const tfs::AttentionProjectionResult projectionsResult = projections.forward(transformerInput);
        const tfs::Tensor scores = tfs::attentionScores(projectionsResult.queries, projectionsResult.keys);
        const tfs::Tensor weights = tfs::attentionWeights(scores);
        const tfs::Tensor attentionContext = tfs::attentionOutput(weights, projectionsResult.values);
        const tfs::Tensor projected = outputProjection.forward(attentionContext);
        const tfs::Tensor residualOutput = tfs::residualAdd(transformerInput, projected);
        const tfs::Tensor normalized = layerNorm.forward(residualOutput);

        std::cout << "Residual output shape: ";
        demo::printList(residualOutput.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "LayerNorm feature size: " << layerNorm.featureSize() << '\n';
        std::cout << "LayerNorm epsilon: " << layerNorm.getEpsilon() << '\n';
        std::cout << "Normalized shape: ";
        demo::printList(normalized.getShape().getDimensions());
        std::cout << '\n';

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Before LayerNorm:\n";
        demo::printMatrix(residualOutput);
        std::cout << "After LayerNorm:\n";
        demo::printMatrix(normalized);
        std::cout << "Row mean and variance after LayerNorm:\n";
        demo::printRowMeanVariance(normalized);
        std::cout << std::defaultfloat;

        std::cout << "Each token row is normalized independently\n";
        std::cout << "Gamma and beta can later learn scale and shift\n";
        std::cout << "The tensor shape is unchanged\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
