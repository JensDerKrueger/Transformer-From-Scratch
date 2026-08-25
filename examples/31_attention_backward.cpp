#include "demo_helpers.h"

#include "tfs/random_initializer.h"
#include "tfs/trainable_attention.h"

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
        tfs::RandomInitializer initializer(7);
        tfs::TrainableSelfAttention attention(3, 2);
        attention.initialize(initializer);

        const tfs::Tensor input = tfs::Tensor::fromValues(
            tfs::TensorShape({4, 3}),
            {0.30f, 1.31f, 2.32f,
             0.11f, 1.12f, 2.13f,
             0.42f, 1.43f, 2.44f,
             0.13f, 1.14f, 2.15f}
        );
        const tfs::TrainableAttentionForwardResult forward = attention.forwardDetailed(input);
        const tfs::Tensor outputGradient = tfs::Tensor::filled(forward.output.getShape(), 0.10f);
        const tfs::Tensor inputGradient = attention.backward(input, forward, outputGradient);

        std::cout << "Input shape: ";
        demo::printList(input.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Attention output shape: ";
        demo::printList(forward.output.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Attention weight shape: ";
        demo::printList(forward.weights.getShape().getDimensions());
        std::cout << '\n';

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Attention weights:\n";
        demo::printMatrix(forward.weights);
        std::cout << "Gradient dLoss/dInput:\n";
        demo::printMatrix(inputGradient);
        std::cout << "Query weight gradient L1: " << l1Norm(attention.getQueryLayer().getWeights().getGradient()) << '\n';
        std::cout << "Key weight gradient L1: " << l1Norm(attention.getKeyLayer().getWeights().getGradient()) << '\n';
        std::cout << "Value weight gradient L1: " << l1Norm(attention.getValueLayer().getWeights().getGradient()) << '\n';
        std::cout << "Output weight gradient L1: " << l1Norm(attention.getOutputLayer().getWeights().getGradient()) << '\n';
        std::cout << std::defaultfloat;

        std::cout << "Attention backward sends gradients through output, values, weights, scores and QKV\n";
        std::cout << "Masked future positions do not receive score gradients\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
