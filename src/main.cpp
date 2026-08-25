#include "tfs/byte_tokenizer.h"

#include <iostream>
#include <string>
#include <vector>

int main() {
    const tfs::ByteTokenizer tokenizer;
    const std::string text = "Hallo Transformer";
    const std::vector<tfs::ByteTokenizer::TokenId> tokens = tokenizer.encode(text);
    const std::string decoded = tokenizer.decode(tokens);

    std::cout << "Input: " << text << '\n';
    std::cout << "Token ids:";
    for (const tfs::ByteTokenizer::TokenId token : tokens) {
        std::cout << ' ' << token;
    }
    std::cout << '\n';
    std::cout << "Decoded: " << decoded << '\n';

    return 0;
}
