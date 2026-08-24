#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace tfs {

class ByteTokenizer {
public:
    using TokenId = std::uint16_t;

    static constexpr std::size_t vocabSize() {
        return 256;
    }

    std::vector<TokenId> encode(std::string_view text) const {
        std::vector<TokenId> tokens;
        tokens.reserve(text.size());

        for (const unsigned char byte : text) {
            tokens.push_back(static_cast<TokenId>(byte));
        }

        return tokens;
    }

    std::string decode(const std::vector<TokenId>& tokens) const {
        std::string text;
        text.reserve(tokens.size());

        for (const TokenId token : tokens) {
            text.push_back(static_cast<char>(token & 0xff));
        }

        return text;
    }
};

} // namespace tfs
