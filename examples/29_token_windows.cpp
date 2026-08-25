#include "demo_helpers.h"

#include "tfs/byte_tokenizer.h"
#include "tfs/token_dataset.h"

#include <exception>
#include <iostream>
#include <string>
#include <vector>

int main() {
    try {
        const std::string text = "der film war gut";
        const tfs::ByteTokenizer tokenizer;
        const std::vector<tfs::ByteTokenizer::TokenId> byteTokens = tokenizer.encode(text);

        std::vector<std::size_t> tokens;
        tokens.reserve(byteTokens.size());
        for (const tfs::ByteTokenizer::TokenId token : byteTokens) {
            tokens.push_back(token);
        }

        const tfs::TokenWindowDataset dataset(tokens, 6);
        const tfs::TokenWindow first = dataset.windowAt(0);
        const tfs::TokenWindow second = dataset.windowAt(1);

        std::cout << "Text: " << text << '\n';
        std::cout << "Token count: " << tokens.size() << '\n';
        std::cout << "Context length: " << dataset.getContextLength() << '\n';
        std::cout << "Window count: " << dataset.size() << '\n';

        std::cout << "Window 0 input: ";
        demo::printList(first.inputTokens);
        std::cout << '\n';
        std::cout << "Window 0 target: ";
        demo::printList(first.targetTokens);
        std::cout << '\n';

        std::cout << "Window 1 input: ";
        demo::printList(second.inputTokens);
        std::cout << '\n';
        std::cout << "Window 1 target: ";
        demo::printList(second.targetTokens);
        std::cout << '\n';

        std::cout << "Targets are inputs shifted by one token\n";
        std::cout << "This is the data layout for next-token prediction\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
