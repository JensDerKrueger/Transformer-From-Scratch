#include "demo_helpers.h"

#include "tfs/random_initializer.h"
#include "tfs/trainable_decoder_block.h"

#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>

namespace {

tfs::TensorValue l1Norm(const tfs::Tensor& tensor) {
    tfs::TensorValue sum = 0.0f;
    for (const tfs::TensorValue value : tensor.getValues()) {
        sum += std::fabs(value);
    }
    return sum;
}

} // namespace

int main() {
    try {
        tfs::RandomInitializer initializer(11);
        tfs::TrainableDecoderBlock block(3, 2, 5);
        block.initialize(initializer);

        const tfs::Tensor input = tfs::Tensor::fromValues(
            tfs::TensorShape({4, 3}),
            {0.30f, 1.31f, 2.32f,
             0.11f, 1.12f, 2.13f,
             0.42f, 1.43f, 2.44f,
             0.13f, 1.14f, 2.15f}
        );
        const tfs::TrainableDecoderBlockForwardResult forward = block.forwardDetailed(input);
        const tfs::Tensor outputGradient = tfs::Tensor::fromValues(
            forward.output.getShape(),
            {0.05f, -0.02f, 0.03f,
             -0.01f, 0.04f, 0.02f,
             0.03f, 0.01f, -0.04f,
             0.02f, -0.03f, 0.05f}
        );
        const tfs::Tensor inputGradient = block.backward(input, forward, outputGradient);

        std::cout << "Input shape: ";
        demo::printList(input.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Block output shape: ";
        demo::printList(forward.output.getShape().getDimensions());
        std::cout << '\n';

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Block output:\n";
        demo::printMatrix(forward.output);
        std::cout << "Incoming gradient:\n";
        demo::printMatrix(outputGradient);
        std::cout << "Gradient dLoss/dInput:\n";
        demo::printMatrix(inputGradient);
        std::cout << "Attention query gradient L1: "
                  << l1Norm(block.getAttention().getQueryLayer().getWeights().getGradient()) << '\n';
        std::cout << "Feed-forward expand gradient L1: "
                  << l1Norm(block.getFeedForward().getExpandLayer().getWeights().getGradient()) << '\n';
        std::cout << std::defaultfloat;

        std::cout << "A decoder block backward combines attention, residual paths, norms and feed-forward gradients\n";
        std::cout << "The shape stays unchanged: one vector per token in, one vector per token out\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
