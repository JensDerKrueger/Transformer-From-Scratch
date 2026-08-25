#include "demo_model.h"

#include <exception>
#include <iomanip>
#include <iostream>

int main() {
    try {
        const tfs::Tensor input = demo::makeDemoAttentionNormOutput();
        const tfs::FeedForwardBlock block = demo::makeDemoFeedForwardBlock();
        const tfs::FeedForwardBlockResult result = block.forwardDetailed(input);

        std::cout << "Input shape: ";
        demo::printList(input.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Feed-forward output shape: ";
        demo::printList(result.feedForward.output.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Block output shape: ";
        demo::printList(result.output.getShape().getDimensions());
        std::cout << '\n';

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Input to feed-forward block:\n";
        demo::printMatrix(input);
        std::cout << "Feed-forward branch:\n";
        demo::printMatrix(result.feedForward.output);
        std::cout << "After residual add:\n";
        demo::printMatrix(result.afterResidual);
        std::cout << "After LayerNorm:\n";
        demo::printMatrix(result.output);
        std::cout << "Row mean and variance after LayerNorm:\n";
        demo::printRowMeanVariance(result.output);
        std::cout << std::defaultfloat;

        std::cout << "The second sublayer also uses residual add and LayerNorm\n";
        std::cout << "The block keeps one vector per input token\n";
        std::cout << "A decoder block can now combine attention and feed-forward\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
