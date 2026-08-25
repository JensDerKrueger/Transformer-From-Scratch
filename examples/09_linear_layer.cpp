#include "demo_model.h"

#include "tfs/linear_layer.h"

#include <exception>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    try {
        const tfs::LinearLayer projection(
            3,
            2,
            {
                0.50f, -0.25f,
                1.00f, 0.00f,
                -0.50f, 0.75f
            },
            {0.10f, -0.20f}
        );
        const tfs::Tensor transformerInput = demo::makeDemoTransformerInput();
        const tfs::Tensor projected = projection.forward(transformerInput);

        std::cout << "Input shape: ";
        demo::printList(transformerInput.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Weight shape: ";
        demo::printList(projection.getWeights().getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Bias shape: ";
        demo::printList(projection.getBias().getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Output shape: ";
        demo::printList(projected.getShape().getDimensions());
        std::cout << '\n';

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Transformer input:\n";
        demo::printMatrix(transformerInput);
        std::cout << "Projected vectors:\n";
        demo::printMatrix(projected);
        std::cout << std::defaultfloat;

        std::cout << "Each token row used the same weights and bias\n";
        std::cout << "Later we will use linear layers for query, key, value and logits\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
