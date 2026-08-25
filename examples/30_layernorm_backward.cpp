#include "demo_helpers.h"

#include "tfs/trainable_layer_norm.h"

#include <exception>
#include <iomanip>
#include <iostream>

int main() {
    try {
        tfs::TrainableLayerNorm norm(3);
        const tfs::Tensor input = tfs::Tensor::fromValues(
            tfs::TensorShape({2, 3}),
            {1.0f, 2.0f, 4.0f,
             -1.0f, 0.0f, 1.0f}
        );
        const tfs::LayerNormForwardResult forward = norm.forwardDetailed(input);
        const tfs::Tensor outputGradient = tfs::Tensor::fromValues(
            tfs::TensorShape({2, 3}),
            {0.3f, -0.2f, 0.1f,
             -0.1f, 0.4f, 0.2f}
        );
        const tfs::Tensor inputGradient = norm.backward(input, forward, outputGradient);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Input:\n";
        demo::printMatrix(input);
        std::cout << "Normalized:\n";
        demo::printMatrix(forward.normalized);
        std::cout << "Output:\n";
        demo::printMatrix(forward.output);
        std::cout << "Incoming gradient:\n";
        demo::printMatrix(outputGradient);
        std::cout << "Gradient dLoss/dInput:\n";
        demo::printMatrix(inputGradient);
        std::cout << "Gradient dLoss/dGamma: ";
        demo::printList(norm.getGamma().getGradient().getValues());
        std::cout << '\n';
        std::cout << "Gradient dLoss/dBeta: ";
        demo::printList(norm.getBeta().getGradient().getValues());
        std::cout << '\n';
        std::cout << std::defaultfloat;

        std::cout << "LayerNorm backward works row by row\n";
        std::cout << "Gamma and beta receive one accumulated gradient per feature\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
