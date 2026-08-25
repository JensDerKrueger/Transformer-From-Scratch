#include "demo_model.h"

#include <exception>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    try {
        const tfs::TokenEmbedding tokenEmbedding = demo::makeDemoTokenEmbedding();
        const tfs::PositionEmbedding positionEmbedding = demo::makeDemoPositionEmbedding();
        const std::vector<tfs::TokenId> tokens = demo::makeDemoTokens();
        const tfs::Tensor tokenVectors = tokenEmbedding.embed(tokens);
        const tfs::Tensor positionVectors = positionEmbedding.embed(tokens.size());
        const tfs::Tensor transformerInput = positionEmbedding.addTo(tokenVectors);

        std::cout << "Token ids: ";
        demo::printList(tokens);
        std::cout << '\n';
        std::cout << "Token embedding shape: ";
        demo::printList(tokenVectors.getShape().getDimensions());
        std::cout << '\n';
        std::cout << "Position embedding shape: ";
        demo::printList(positionVectors.getShape().getDimensions());
        std::cout << '\n';

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Token vectors:\n";
        demo::printMatrix(tokenVectors);
        std::cout << "Position vectors:\n";
        demo::printMatrix(positionVectors);
        std::cout << "Transformer input:\n";
        demo::printMatrix(transformerInput);
        std::cout << std::defaultfloat;

        std::cout << "Same token id 1 appears at positions 1 and 3\n";
        std::cout << "After adding positions, these rows are no longer identical\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
