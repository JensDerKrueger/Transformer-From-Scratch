#pragma once

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace tfs {

// A deliberately small, closed vocabulary for controlled learning experiments.
class WordTokenizer {
public:
    explicit WordTokenizer(const std::vector<std::string>& texts) {
        std::vector<std::string> vocabulary;
        for (const std::string& text : texts) {
            const std::vector<std::string> words = split(text);
            vocabulary.insert(vocabulary.end(), words.begin(), words.end());
        }
        std::sort(vocabulary.begin(), vocabulary.end());
        vocabulary.erase(std::unique(vocabulary.begin(), vocabulary.end()), vocabulary.end());
        tokens.push_back("<eos>");
        tokens.insert(tokens.end(), vocabulary.begin(), vocabulary.end());
        for (std::size_t index = 0; index < tokens.size(); ++index) {
            ids.emplace(tokens[index], index);
        }
    }

    std::size_t size() const { return tokens.size(); }
    std::size_t endToken() const { return 0; }

    std::vector<std::size_t> encode(const std::string& text) const {
        std::vector<std::size_t> result;
        for (const std::string& word : split(text)) {
            const auto entry = ids.find(word);
            if (entry == ids.end()) {
                throw std::runtime_error("Unknown word in this closed vocabulary: " + word);
            }
            result.push_back(entry->second);
        }
        return result;
    }

    std::string decode(const std::vector<std::size_t>& sequence) const {
        std::string result;
        for (const std::size_t id : sequence) {
            if (id >= tokens.size()) {
                throw std::runtime_error("Word token out of range");
            }
            if (id == endToken()) { break; }
            const std::string& word = tokens[id];
            if (!result.empty() && !isPunctuation(word[0])) { result += ' '; }
            result += word;
        }
        return result;
    }

private:
    static bool isPunctuation(const char character) {
        return character == '.' || character == ',' || character == '?' || character == ':' || character == '!';
    }

    static std::vector<std::string> split(const std::string& text) {
        std::string separated;
        for (const char character : text) {
            if (isPunctuation(character)) {
                separated += ' ';
                separated += character;
                separated += ' ';
            } else {
                separated += character;
            }
        }
        std::istringstream input(separated);
        std::vector<std::string> result;
        std::string word;
        while (input >> word) { result.push_back(word); }
        return result;
    }

    std::vector<std::string> tokens;
    std::unordered_map<std::string, std::size_t> ids;
};

} // namespace tfs
