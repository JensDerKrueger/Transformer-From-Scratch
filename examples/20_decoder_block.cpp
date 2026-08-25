#include "demo_model.h"

#include <exception>
#include <iomanip>
#include <iostream>

int main() {
    try {
        const tfs::Tensor input = demo::makeDemoTransformerInput();
        const tfs::DecoderBlock block = demo::makeDemoDecoderBlock();
        const tfs::DecoderBlockResult result = block.forwardDetailed(input);

        std::cout << "Input shape: ";
        demo::printList(input.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Attention weight shape: ";
        demo::printList(result.weights.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Decoder output shape: ";
        demo::printList(result.output.getShape().getDimensions());
        std::cout << '\n';

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Transformer input:\n";
        demo::printMatrix(input);
        std::cout << "Attention weights:\n";
        demo::printMatrix(result.weights);
        std::cout << "After attention norm:\n";
        demo::printMatrix(result.afterAttentionNorm);
        std::cout << "Feed-forward branch:\n";
        demo::printMatrix(result.feedForwardBlock.feedForward.output);
        std::cout << "Decoder block output:\n";
        demo::printMatrix(result.output);
        std::cout << std::defaultfloat;

        std::cout << "The decoder block keeps sequence length and model size unchanged\n";
        std::cout << "Masked self-attention mixes tokens across the sequence\n";
        std::cout << "The feed-forward network transforms each token row independently\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
