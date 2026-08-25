#include "demo_helpers.h"

#include "tfs/trainable_embedding.h"

#include <exception>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    try {
        tfs::TrainableEmbedding embedding(
            4,
            3,
            {
                0.00f, 0.01f, 0.02f,
                0.10f, 0.11f, 0.12f,
                0.20f, 0.21f, 0.22f,
                0.30f, 0.31f, 0.32f
            }
        );
        const std::vector<std::size_t> tokens = {2, 1, 2};
        const tfs::Tensor vectors = embedding.forward(tokens);
        const tfs::Tensor vectorGradient = tfs::Tensor::fromValues(
            tfs::TensorShape({3, 3}),
            {
                1.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f,
                0.5f, 0.5f, 1.0f
            }
        );

        embedding.backward(tokens, vectorGradient);

        std::cout << "Token ids: ";
        demo::printList(tokens);
        std::cout << '\n';

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Selected vectors:\n";
        demo::printMatrix(vectors);
        std::cout << "Incoming gradient:\n";
        demo::printMatrix(vectorGradient);
        std::cout << "Embedding weight gradient:\n";
        demo::printMatrix(embedding.getWeights().getGradient());
        std::cout << std::defaultfloat;

        std::cout << "Token 2 appears twice, so its gradient is the sum of two rows\n";
        std::cout << "Embedding backward does not compute dLoss/dTokenId\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
