#pragma once

#include "tfs/corpus_reader.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tfs {

using BpeTokenId = std::uint32_t;

struct BpeMerge {
    BpeTokenId left = 0;
    BpeTokenId right = 0;
    BpeTokenId token = 0;
    std::uint64_t count = 0;
};

class BpeModel {
public:
    static constexpr BpeTokenId baseVocabSize() {
        return 256;
    }

    BpeTokenId addMerge(const BpeTokenId left, const BpeTokenId right, const std::uint64_t count) {
        const BpeTokenId token = static_cast<BpeTokenId>(baseVocabSize() + merges.size());
        merges.push_back(BpeMerge{left, right, token, count});
        return token;
    }

    std::size_t vocabSize() const {
        return static_cast<std::size_t>(baseVocabSize()) + merges.size();
    }

    const std::vector<BpeMerge>& getMerges() const {
        return merges;
    }

    std::vector<BpeTokenId> encode(const std::string_view text) const {
        std::vector<BpeTokenId> tokens;
        tokens.reserve(text.size());

        for (const unsigned char byte : text) {
            tokens.push_back(static_cast<BpeTokenId>(byte));
        }

        for (const BpeMerge& merge : merges) {
            tokens = applyMerge(tokens, merge.left, merge.right, merge.token);
        }

        return tokens;
    }

    std::string decode(const std::vector<BpeTokenId>& tokens) const {
        std::string text;

        for (const BpeTokenId token : tokens) {
            decodeTokenInto(token, text);
        }

        return text;
    }

    std::string decodeToken(const BpeTokenId token) const {
        std::string text;
        decodeTokenInto(token, text);
        return text;
    }

private:
    static std::vector<BpeTokenId> applyMerge(
        const std::vector<BpeTokenId>& tokens,
        const BpeTokenId left,
        const BpeTokenId right,
        const BpeTokenId newToken
    ) {
        std::vector<BpeTokenId> merged;
        merged.reserve(tokens.size());

        for (std::size_t i = 0; i < tokens.size(); ++i) {
            if (i + 1 < tokens.size() && tokens[i] == left && tokens[i + 1] == right) {
                merged.push_back(newToken);
                i += 1;
            } else {
                merged.push_back(tokens[i]);
            }
        }

        return merged;
    }

    void decodeTokenInto(const BpeTokenId token, std::string& text) const {
        if (token < baseVocabSize()) {
            text.push_back(static_cast<char>(token));
            return;
        }

        const BpeTokenId mergeIndex = token - baseVocabSize();
        if (mergeIndex >= merges.size()) {
            throw std::runtime_error("Invalid BPE token id");
        }

        const BpeMerge& merge = merges[mergeIndex];
        decodeTokenInto(merge.left, text);
        decodeTokenInto(merge.right, text);
    }

    std::vector<BpeMerge> merges;
};

struct BpeTrainingResult {
    BpeModel model;
    std::uint64_t lineCount = 0;
    std::uint64_t initialTokenCount = 0;
    std::uint64_t finalTokenCount = 0;
};

class BpeTrainer {
public:
    BpeTrainingResult trainFromCorpus(
        const std::string& path,
        const std::uint64_t maxLines,
        const std::size_t mergeCount
    ) const {
        BpeTrainingResult result;
        std::vector<std::vector<BpeTokenId>> sequences;

        const CorpusReader reader(path);
        reader.forEachLine([&](const std::string& line) {
            if (line.empty()) {
                return;
            }

            result.lineCount += 1;
            result.initialTokenCount += static_cast<std::uint64_t>(line.size());
            sequences.push_back(bytesToTokens(line));
        }, maxLines);

        for (std::size_t i = 0; i < mergeCount; ++i) {
            const PairCount bestPair = findBestPair(sequences);
            if (bestPair.count < 2) {
                break;
            }

            const BpeTokenId newToken = result.model.addMerge(bestPair.left, bestPair.right, bestPair.count);
            mergePairInAllSequences(sequences, bestPair.left, bestPair.right, newToken);
        }

        result.finalTokenCount = countTokens(sequences);
        return result;
    }

private:
    struct PairCount {
        BpeTokenId left = 0;
        BpeTokenId right = 0;
        std::uint64_t count = 0;
    };

    static std::vector<BpeTokenId> bytesToTokens(const std::string& text) {
        std::vector<BpeTokenId> tokens;
        tokens.reserve(text.size());

        for (const unsigned char byte : text) {
            tokens.push_back(static_cast<BpeTokenId>(byte));
        }

        return tokens;
    }

    static std::uint64_t pairKey(const BpeTokenId left, const BpeTokenId right) {
        return (static_cast<std::uint64_t>(left) << 32) | static_cast<std::uint64_t>(right);
    }

    static PairCount pairFromKey(const std::uint64_t key, const std::uint64_t count) {
        return PairCount{
            static_cast<BpeTokenId>(key >> 32),
            static_cast<BpeTokenId>(key & 0xffffffffu),
            count
        };
    }

    static bool isBetterPair(const PairCount& candidate, const PairCount& best) {
        if (candidate.count != best.count) {
            return candidate.count > best.count;
        }

        if (candidate.left != best.left) {
            return candidate.left < best.left;
        }

        return candidate.right < best.right;
    }

    static PairCount findBestPair(const std::vector<std::vector<BpeTokenId>>& sequences) {
        std::unordered_map<std::uint64_t, std::uint64_t> counts;

        for (const std::vector<BpeTokenId>& sequence : sequences) {
            if (sequence.size() < 2) {
                continue;
            }

            for (std::size_t i = 0; i + 1 < sequence.size(); ++i) {
                counts[pairKey(sequence[i], sequence[i + 1])] += 1;
            }
        }

        PairCount best;
        for (const auto& entry : counts) {
            const PairCount candidate = pairFromKey(entry.first, entry.second);
            if (isBetterPair(candidate, best)) {
                best = candidate;
            }
        }

        return best;
    }

    static void mergePairInAllSequences(
        std::vector<std::vector<BpeTokenId>>& sequences,
        const BpeTokenId left,
        const BpeTokenId right,
        const BpeTokenId newToken
    ) {
        for (std::vector<BpeTokenId>& sequence : sequences) {
            sequence = mergePair(sequence, left, right, newToken);
        }
    }

    static std::vector<BpeTokenId> mergePair(
        const std::vector<BpeTokenId>& sequence,
        const BpeTokenId left,
        const BpeTokenId right,
        const BpeTokenId newToken
    ) {
        std::vector<BpeTokenId> merged;
        merged.reserve(sequence.size());

        for (std::size_t i = 0; i < sequence.size(); ++i) {
            if (i + 1 < sequence.size() && sequence[i] == left && sequence[i + 1] == right) {
                merged.push_back(newToken);
                i += 1;
            } else {
                merged.push_back(sequence[i]);
            }
        }

        return merged;
    }

    static std::uint64_t countTokens(const std::vector<std::vector<BpeTokenId>>& sequences) {
        std::uint64_t count = 0;

        for (const std::vector<BpeTokenId>& sequence : sequences) {
            count += static_cast<std::uint64_t>(sequence.size());
        }

        return count;
    }
};

} // namespace tfs
