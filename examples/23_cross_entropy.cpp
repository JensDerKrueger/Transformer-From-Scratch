#include "demo_helpers.h"

#include "tfs/cross_entropy_loss.h"
#include "tfs/tensor.h"

#include <exception>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    try {
        const tfs::Tensor logits = tfs::Tensor::fromValues(
            tfs::TensorShape({3, 4}),
            {
                0.20f, 1.40f, -0.30f, 0.10f,
                1.10f, 0.30f, 0.00f, -0.20f,
                -0.40f, 0.20f, 0.50f, 1.30f
            }
        );
        const std::vector<std::size_t> targets = {1, 0, 3};
        const tfs::CrossEntropyResult result = tfs::crossEntropyLoss(logits, targets);

        std::cout << "Logit shape: ";
        demo::printList(logits.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Target token ids: ";
        demo::printList(targets);
        std::cout << '\n';

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Probabilities:\n";
        demo::printMatrix(result.probabilities);
        std::cout << "Loss: " << result.loss << '\n';
        std::cout << "Gradient dLoss/dLogits:\n";
        demo::printMatrix(result.gradient);
        std::cout << std::defaultfloat;

        std::cout << "Each target selects one column per row\n";
        std::cout << "The gradient is probability minus one-hot target, averaged over rows\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
