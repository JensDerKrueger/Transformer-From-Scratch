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

        std::cout << "Transformer input shape: ";
        demo::printList(transformerInput.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Attention context shape: ";
        demo::printList(attentionContext.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Output projection weight shape: ";
        demo::printList(outputProjection.getLayer().getWeights().getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Projected shape: ";
        demo::printList(projected.getShape().getDimensions());
        std::cout << '\n';

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Attention context:\n";
        demo::printMatrix(attentionContext);
        std::cout << "Projected vectors:\n";
        demo::printMatrix(projected);
        std::cout << std::defaultfloat;

        std::cout << "The projection maps value size 2 back to model size 3\n";
        std::cout << "Now the tensor has the same shape as the transformer input\n";
        std::cout << "Residual connections and the next block can use this shape\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
