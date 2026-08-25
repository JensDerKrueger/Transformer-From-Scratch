#include "tfs/byte_tokenizer.h"

#include <exception>
#include <iostream>
#include <string>
#include <vector>

int main(const int argc, char** argv) {
    try {
        const std::string text = argc >= 2 ? argv[1] : "Hallo Transformer";
        const tfs::ByteTokenizer tokenizer;
        const std::vector<tfs::ByteTokenizer::TokenId> tokens = tokenizer.encode(text);
        const std::string decoded = tokenizer.decode(tokens);

        std::cout << "Input: " << text << '\n';
        std::cout << "Token ids:";
        for (const tfs::ByteTokenizer::TokenId token : tokens) {
            std::cout << ' ' << token;
        }
        std::cout << '\n';
        std::cout << "Decoded: " << decoded << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
