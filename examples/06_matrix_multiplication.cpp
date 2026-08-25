#include "demo_helpers.h"

#include "tfs/tensor.h"
#include "tfs/tensor_operations.h"

#include <exception>
#include <iomanip>
#include <iostream>

int main() {
    try {
        const tfs::Tensor left = tfs::Tensor::fromValues(
            tfs::TensorShape({2, 3}),
            {1.0f, 2.0f, 3.0f,
             4.0f, 5.0f, 6.0f}
        );
        const tfs::Tensor right = tfs::Tensor::fromValues(
            tfs::TensorShape({3, 2}),
            {7.0f, 8.0f,
             9.0f, 10.0f,
             11.0f, 12.0f}
        );
        const tfs::Tensor result = tfs::matrixMultiply(left, right);

        std::cout << "Left shape: ";
        demo::printList(left.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Right shape: ";
        demo::printList(right.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Result shape: ";
        demo::printList(result.getShape().getDimensions());
        std::cout << '\n';

        std::cout << "Left matrix:\n";
        demo::printMatrix(left);
        std::cout << "Right matrix:\n";
        demo::printMatrix(right);
        std::cout << "Result matrix:\n";
        demo::printMatrix(result);
        std::cout << "result[1, 0] = 4*7 + 5*9 + 6*11 = "
                  << result.at({1, 0}) << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
