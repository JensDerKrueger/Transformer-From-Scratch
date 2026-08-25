#include "demo_model.h"

#include <exception>
#include <iomanip>
#include <iostream>

int main() {
    try {
        const tfs::AttentionProjections projections = demo::makeDemoAttentionProjections();
        const tfs::AttentionOutputProjection outputProjection = demo::makeDemoAttentionOutputProjection();
        const tfs::Tensor transformerInput = demo::makeDemoTransformerInput();
        const tfs::AttentionProjectionResult projectionsResult = projections.forward(transformerInput);
        const tfs::Tensor scores = tfs::attentionScores(projectionsResult.queries, projectionsResult.keys);
        const tfs::Tensor weights = tfs::attentionWeights(scores);
        const tfs::Tensor attentionContext = tfs::attentionOutput(weights, projectionsResult.values);
        const tfs::Tensor projected = outputProjection.forward(attentionContext);
        const tfs::Tensor residualOutput = tfs::residualAdd(transformerInput, projected);

        std::cout << "Transformer input shape: ";
        demo::printList(transformerInput.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Projected attention shape: ";
        demo::printList(projected.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Residual output shape: ";
        demo::printList(residualOutput.getShape().getDimensions());
        std::cout << '\n';

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Transformer input:\n";
        demo::printMatrix(transformerInput);
        std::cout << "Projected attention:\n";
        demo::printMatrix(projected);
        std::cout << "After residual add:\n";
        demo::printMatrix(residualOutput);
        std::cout << std::defaultfloat;

        std::cout << "Residual add keeps the original path visible\n";
        std::cout << "Input and branch output must have identical shapes\n";
        std::cout << "Layer normalization comes next in the transformer block\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
