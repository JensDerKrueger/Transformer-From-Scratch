#include "demo_model.h"

#include <exception>
#include <iomanip>
#include <iostream>

int main() {
    try {
        const tfs::AttentionProjections projections = demo::makeDemoAttentionProjections();
        const tfs::Tensor transformerInput = demo::makeDemoTransformerInput();
        const tfs::AttentionProjectionResult result = projections.forward(transformerInput);

        std::cout << "Input shape: ";
        demo::printList(transformerInput.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Projection size: " << projections.projectionSize() << '\n';
        std::cout << "Query shape: ";
        demo::printList(result.queries.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Key shape: ";
        demo::printList(result.keys.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Value shape: ";
        demo::printList(result.values.getShape().getDimensions());
        std::cout << '\n';

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Queries:\n";
        demo::printMatrix(result.queries);
        std::cout << "Keys:\n";
        demo::printMatrix(result.keys);
        std::cout << "Values:\n";
        demo::printMatrix(result.values);
        std::cout << std::defaultfloat;

        std::cout << "All three tensors came from the same input\n";
        std::cout << "Each projection has its own weights and bias\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
