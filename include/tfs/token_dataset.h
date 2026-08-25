#pragma once

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tfs {

struct TokenWindow {
    std::vector<std::size_t> inputTokens;
    std::vector<std::size_t> targetTokens;
};

inline std::vector<std::size_t> readByteTokensFromFile(
    const std::string& path,
    const std::size_t maxBytes
) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open token dataset file: " + path);
    }

    std::vector<std::size_t> tokens;
    if (maxBytes != 0) {
        tokens.reserve(maxBytes);
    }

    char value = 0;
    while (file.get(value)) {
        tokens.push_back(static_cast<unsigned char>(value));
        if (maxBytes != 0 && tokens.size() >= maxBytes) {
            break;
        }
    }

    return tokens;
}

class TokenWindowDataset {
public:
    TokenWindowDataset(std::vector<std::size_t> tokens, const std::size_t contextLength)
        : tokens(std::move(tokens)),
          contextLength(contextLength) {
        if (contextLength == 0) {
            throw std::runtime_error("Context length must be greater than zero");
        }
        if (this->tokens.size() <= contextLength) {
            throw std::runtime_error("Token dataset is too small for the requested context length");
        }
    }

    std::size_t size() const {
        return tokens.size() - contextLength;
    }

    std::size_t getContextLength() const {
        return contextLength;
    }

    TokenWindow windowAt(const std::size_t index) const {
        if (index >= size()) {
            throw std::runtime_error("Token window index is out of bounds");
        }

        std::vector<std::size_t> inputTokens;
        std::vector<std::size_t> targetTokens;
        inputTokens.reserve(contextLength);
        targetTokens.reserve(contextLength);

        for (std::size_t i = 0; i < contextLength; ++i) {
            inputTokens.push_back(tokens[index + i]);
            targetTokens.push_back(tokens[index + i + 1]);
        }

        return TokenWindow{std::move(inputTokens), std::move(targetTokens)};
    }

private:
    std::vector<std::size_t> tokens;
    std::size_t contextLength = 0;
};

} // namespace tfs
