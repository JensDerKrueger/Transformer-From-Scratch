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
        const tfs::Tensor weights = tfs::attentionWeights(scores);
        const tfs::Tensor output = tfs::attentionOutput(weights, projectionsResult.values);

        std::cout << "Weight shape: ";
        demo::printList(weights.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Value shape: ";
        demo::printList(projectionsResult.values.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Output shape: ";
        demo::printList(output.getShape().getDimensions());
        std::cout << '\n';

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Attention weights:\n";
        demo::printMatrix(weights);
        std::cout << "Values:\n";
        demo::printMatrix(projectionsResult.values);
        std::cout << "Attention output:\n";
        demo::printMatrix(output);
        std::cout << std::defaultfloat;

        std::cout << "Each output row is a weighted sum of value rows\n";
        std::cout << "The output has one context vector per input token\n";
        std::cout << "This is single-head masked self-attention without the final projection\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
