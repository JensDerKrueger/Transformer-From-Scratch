#include "demo_helpers.h"

#include "tfs/trainable_linear_layer.h"

#include <exception>
#include <iomanip>
#include <iostream>

int main() {
    try {
        tfs::TrainableLinearLayer layer(
            2,
            3,
            {0.10f, -0.20f, 0.30f,
             0.40f, 0.00f, -0.10f},
            {0.01f, 0.02f, -0.03f}
        );
        const tfs::Tensor input = tfs::Tensor::fromValues(
            tfs::TensorShape({2, 2}),
            {1.00f, 2.00f,
             -1.00f, 0.50f}
        );
        const tfs::Tensor output = layer.forward(input);
        const tfs::Tensor outputGradient = tfs::Tensor::fromValues(
            tfs::TensorShape({2, 3}),
            {0.20f, -0.10f, 0.05f,
             -0.30f, 0.40f, 0.10f}
        );
        const tfs::Tensor inputGradient = layer.backward(input, outputGradient);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Input:\n";
        demo::printMatrix(input);
        std::cout << "Weights:\n";
        demo::printMatrix(layer.getWeights().getValue());
        std::cout << "Output:\n";
        demo::printMatrix(output);
        std::cout << "Incoming gradient dLoss/dOutput:\n";
        demo::printMatrix(outputGradient);
        std::cout << "Gradient dLoss/dInput:\n";
        demo::printMatrix(inputGradient);
        std::cout << "Gradient dLoss/dWeights:\n";
        demo::printMatrix(layer.getWeights().getGradient());
        std::cout << "Gradient dLoss/dBias: ";
        demo::printList(layer.getBias().getGradient().getValues());
        std::cout << '\n';
        std::cout << std::defaultfloat;

        std::cout << "Backward computes gradients for input, weights and bias\n";
        std::cout << "The layer accumulates parameter gradients until they are cleared\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
