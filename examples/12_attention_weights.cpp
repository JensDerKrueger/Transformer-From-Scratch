#include "demo_model.h"

#include <exception>
#include <iomanip>
#include <iostream>

int main() {
    try {
        const tfs::AttentionProjections projections = demo::makeDemoAttentionProjections();
        const tfs::Tensor transformerInput = demo::makeDemoTransformerInput();
        const tfs::AttentionProjectionResult projectionsResult = projections.forward(transformerInput);
        const tfs::Tensor scores = tfs::attentionScores(projectionsResult.queries, projectionsResult.keys);
        const tfs::Tensor maskedScores = tfs::applyCausalMask(scores);
        const tfs::Tensor weights = tfs::attentionWeights(scores);

        std::cout << "Score shape: ";
        demo::printList(scores.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Weight shape: ";
        demo::printList(weights.getShape().getDimensions());
        std::cout << '\n';

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Attention scores:\n";
        demo::printMatrix(scores);
        std::cout << "Masked scores:\n";
        demo::printMatrix(maskedScores);
        std::cout << "Attention weights:\n";
        demo::printMatrix(weights);
        demo::printRowSums(weights);
        std::cout << std::defaultfloat;

        std::cout << "Future positions have weight 0 after masking\n";
        std::cout << "Each row is a probability distribution\n";
        std::cout << "Values are mixed in the next lesson\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
