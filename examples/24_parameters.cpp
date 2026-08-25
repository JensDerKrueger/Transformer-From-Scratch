#include "demo_helpers.h"

#include "tfs/parameter.h"

#include <exception>
#include <iomanip>
#include <iostream>

int main() {
    try {
        tfs::Parameter weights(
            tfs::TensorShape({2, 3}),
            {0.10f, -0.20f, 0.30f,
             0.40f, 0.00f, -0.10f}
        );

        tfs::Tensor& gradient = weights.getGradient();
        gradient[0] = 0.50f;
        gradient[3] = -0.25f;
        gradient[5] = 0.75f;

        std::cout << "Value shape: ";
        demo::printList(weights.getValue().getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Gradient shape: ";
        demo::printList(weights.getGradient().getShape().getDimensions());
        std::cout << '\n';

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Parameter values:\n";
        demo::printMatrix(weights.getValue());
        std::cout << "Gradient before zeroGradient:\n";
        demo::printMatrix(weights.getGradient());

        weights.zeroGradient();

        std::cout << "Gradient after zeroGradient:\n";
        demo::printMatrix(weights.getGradient());
        std::cout << std::defaultfloat;

        std::cout << "A trainable parameter stores values and gradients side by side\n";
        std::cout << "Gradients must be cleared before the next backward pass\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
