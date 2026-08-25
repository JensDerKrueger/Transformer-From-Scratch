#include "demo_helpers.h"

#include "tfs/activation.h"
#include "tfs/tensor.h"

#include <exception>
#include <iomanip>
#include <iostream>

int main() {
    try {
        const tfs::Tensor input = tfs::Tensor::fromValues(
            tfs::TensorShape({2, 4}),
            {-2.0f, -1.0f, 0.0f, 1.0f,
             2.0f, 3.0f, -0.5f, 0.5f}
        );
        const tfs::Tensor output = tfs::gelu(input);

        std::cout << "Input shape: ";
        demo::printList(input.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Output shape: ";
        demo::printList(output.getShape().getDimensions());
        std::cout << '\n';

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Input:\n";
        demo::printMatrix(input);
        std::cout << "GELU output:\n";
        demo::printMatrix(output);
        std::cout << std::defaultfloat;

        std::cout << "GELU is applied independently to each value\n";
        std::cout << "Large positive values mostly pass through\n";
        std::cout << "Negative values are damped but not hard-clipped\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
