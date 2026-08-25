#include "demo_helpers.h"

#include "tfs/softmax.h"
#include "tfs/tensor.h"

#include <exception>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    try {
        const tfs::Tensor logits = tfs::Tensor::fromValues(
            tfs::TensorShape({3, 6}),
            {
                0.10f, 0.80f, -0.40f, 1.20f, 0.00f, -0.20f,
                1.50f, 0.20f, 0.10f, -0.50f, 0.90f, 0.30f,
                0.17f, 0.53f, -0.71f, 0.56f, -0.35f, 0.04f
            }
        );
        const tfs::Tensor probabilities = tfs::softmaxRows(logits);
        const std::vector<tfs::TopKEntry> top = tfs::topKRow(probabilities, 2, 3);

        std::cout << "Logit shape: ";
        demo::printList(logits.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Probability shape: ";
        demo::printList(probabilities.getShape().getDimensions());
        std::cout << '\n';

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Logits:\n";
        demo::printMatrix(logits);
        std::cout << "Softmax probabilities:\n";
        demo::printMatrix(probabilities);
        demo::printRowSums(probabilities);

        std::cout << "Top 3 tokens for final row:\n";
        for (const tfs::TopKEntry& entry : top) {
            std::cout << "  token " << entry.index << " probability " << entry.value << '\n';
        }
        std::cout << std::defaultfloat;

        std::cout << "Softmax turns logits into probabilities per row\n";
        std::cout << "Top-k keeps the best candidates for prediction\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
