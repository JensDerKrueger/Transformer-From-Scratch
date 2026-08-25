#include "demo_helpers.h"

#include "tfs/tensor.h"

#include <exception>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    try {
        const tfs::TensorShape shape({2, 3, 4});
        tfs::Tensor tensor(shape);

        for (std::size_t i = 0; i < tensor.size(); ++i) {
            tensor[i] = static_cast<tfs::TensorValue>(i);
        }

        const std::vector<std::size_t> index = {1, 2, 3};
        tensor.at(index) = 42.0f;

        const tfs::Tensor bias = tfs::Tensor::filled(shape, 1.0f);
        const tfs::Tensor shifted = tfs::add(tensor, bias);
        const tfs::Tensor scaled = tfs::multiply(shifted, 0.5f);

        std::cout << "Shape: ";
        demo::printList(shape.getDimensions());
        std::cout << '\n';
        std::cout << "Rank: " << shape.rank() << '\n';
        std::cout << "Element count: " << shape.elementCount() << '\n';
        std::cout << "Strides: ";
        demo::printList(tensor.getStrides());
        std::cout << '\n';
        std::cout << "Index: ";
        demo::printList(index);
        std::cout << '\n';
        std::cout << "Flat offset: " << tensor.offset(index) << '\n';
        std::cout << "Value at index: " << tensor.at(index) << '\n';

        std::cout << "First row after add and scale:";
        std::cout << std::fixed << std::setprecision(1);
        for (std::size_t i = 0; i < 4; ++i) {
            std::cout << ' ' << scaled[i];
        }
        std::cout << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
