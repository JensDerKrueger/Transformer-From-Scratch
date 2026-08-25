#include "demo_model.h"

#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>

int main() {
    try {
        const tfs::AttentionProjections projections = demo::makeDemoAttentionProjections();
        const tfs::Tensor transformerInput = demo::makeDemoTransformerInput();
        const tfs::AttentionProjectionResult projectionsResult = projections.forward(transformerInput);
        const tfs::Tensor transposedKeys = tfs::transposeMatrix(projectionsResult.keys);
        const tfs::Tensor scores = tfs::attentionScores(projectionsResult.queries, projectionsResult.keys);
        const tfs::TensorValue scale =
            1.0f / std::sqrt(static_cast<tfs::TensorValue>(projections.projectionSize()));

        std::cout << "Query shape: ";
        demo::printList(projectionsResult.queries.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Key shape: ";
        demo::printList(projectionsResult.keys.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Transposed key shape: ";
        demo::printList(transposedKeys.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Score shape: ";
        demo::printList(scores.getShape().getDimensions());
        std::cout << '\n';

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Scale factor: " << scale << '\n';
        std::cout << "Queries:\n";
        demo::printMatrix(projectionsResult.queries);
        std::cout << "Keys transposed:\n";
        demo::printMatrix(transposedKeys);
        std::cout << "Attention scores:\n";
        demo::printMatrix(scores);
        std::cout << std::defaultfloat;

        std::cout << "Rows are querying tokens\n";
        std::cout << "Columns are key tokens\n";
        std::cout << "Masking and softmax come next\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
