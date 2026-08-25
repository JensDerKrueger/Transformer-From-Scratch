#include "demo_model.h"

#include <exception>
#include <iomanip>
#include <iostream>

int main() {
    try {
        const tfs::Tensor input = demo::makeDemoTransformerInput();
        const tfs::DecoderBlock block = demo::makeDemoDecoderBlock();
        const tfs::LanguageModelHead head = demo::makeDemoLanguageModelHead();
        const tfs::Tensor blockOutput = block.forward(input);
        const tfs::Tensor logits = head.forward(blockOutput);
        const std::size_t finalRow = logits.getShape()[0] - 1;
        const std::size_t bestToken = demo::argmaxRow(logits, finalRow);

        std::cout << "Decoder output shape: ";
        demo::printList(blockOutput.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Vocabulary size: " << head.vocabSize() << '\n';
        std::cout << "Logit shape: ";
        demo::printList(logits.getShape().getDimensions());
        std::cout << '\n';

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Decoder block output:\n";
        demo::printMatrix(blockOutput);
        std::cout << "Logits:\n";
        demo::printMatrix(logits);
        std::cout << std::defaultfloat;

        std::cout << "Best next token id at final position: " << bestToken << '\n';
        std::cout << "Logits are raw scores, not probabilities\n";
        std::cout << "Softmax over logits comes next\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
