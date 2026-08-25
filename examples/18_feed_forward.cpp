#include "demo_model.h"

#include <exception>
#include <iomanip>
#include <iostream>

int main() {
    try {
        const tfs::Tensor input = demo::makeDemoAttentionNormOutput();
        const tfs::FeedForwardNetwork feedForward = demo::makeDemoFeedForwardNetwork();
        const tfs::FeedForwardResult result = feedForward.forwardDetailed(input);

        std::cout << "Input shape: ";
        demo::printList(input.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Hidden shape: ";
        demo::printList(result.hidden.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Output shape: ";
        demo::printList(result.output.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Model size: " << feedForward.modelSize() << '\n';
        std::cout << "Hidden size: " << feedForward.hiddenSize() << '\n';

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Input after attention norm:\n";
        demo::printMatrix(input);
        std::cout << "Hidden before GELU:\n";
        demo::printMatrix(result.hidden);
        std::cout << "Hidden after GELU:\n";
        demo::printMatrix(result.activated);
        std::cout << "Feed-forward output:\n";
        demo::printMatrix(result.output);
        std::cout << std::defaultfloat;

        std::cout << "Each token row is processed independently\n";
        std::cout << "The hidden dimension expands from 3 to 5\n";
        std::cout << "The output returns to model size 3\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
