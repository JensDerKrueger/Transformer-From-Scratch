#include "demo_helpers.h"

#include "tfs/token_embedding.h"

#include <exception>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    try {
        const tfs::TokenEmbedding embedding(
            6,
            3,
            {
                0.00f, 0.01f, 0.02f,
                0.10f, 0.11f, 0.12f,
                0.20f, 0.21f, 0.22f,
                0.30f, 0.31f, 0.32f,
                0.40f, 0.41f, 0.42f,
                0.50f, 0.51f, 0.52f
            }
        );
        const std::vector<tfs::TokenId> tokens = {3, 1, 4, 1};
        const tfs::Tensor vectors = embedding.embed(tokens);

        std::cout << "Vocab size: " << embedding.vocabSize() << '\n';
        std::cout << "Embedding size: " << embedding.embeddingSize() << '\n';
        std::cout << "Token ids: ";
        demo::printList(tokens);
        std::cout << '\n';
        std::cout << "Embedding shape: ";
        demo::printList(vectors.getShape().getDimensions());
        std::cout << '\n';

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Selected vectors:\n";
        demo::printMatrix(vectors);
        std::cout << std::defaultfloat;

        std::cout << "First token vector came from weight row 3\n";
        std::cout << "Repeated token id 1 produces the same vector twice\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
