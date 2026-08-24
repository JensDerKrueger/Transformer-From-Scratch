#pragma once

#include "tfs/corpus_reader.h"
#include "tfs/indexed_priority_queue.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
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

    BpeTrainingResult trainFromCorpusWithQueue(
        const std::string& path,
        const std::uint64_t maxLines,
        const std::size_t mergeCount
    ) const {
        BpeTrainingResult result;
        FastTrainingState state;

        const CorpusReader reader(path);
        reader.forEachLine([&](const std::string& line) {
            if (line.empty()) {
                return;
            }

            result.lineCount += 1;
            result.initialTokenCount += static_cast<std::uint64_t>(line.size());
            state.addSequence(bytesToTokens(line));
        }, maxLines);

        state.buildInitialQueue();

        for (std::size_t i = 0; i < mergeCount; ++i) {
            const PairCount bestPair = state.bestPair();
            if (bestPair.count < 2) {
                break;
            }

            const BpeTokenId newToken = result.model.addMerge(bestPair.left, bestPair.right, bestPair.count);
            state.mergePair(pairKey(bestPair.left, bestPair.right), newToken);
        }

        result.finalTokenCount = state.countActiveTokens();
        return result;
    }

private:
    static constexpr std::size_t noIndex = std::numeric_limits<std::size_t>::max();

    struct PairCount {
        BpeTokenId left = 0;
        BpeTokenId right = 0;
        std::uint64_t count = 0;
    };

    struct PairPriority {
        std::uint64_t count = 0;
        std::uint64_t inverseKey = 0;

        bool operator<(const PairPriority& other) const {
            if (count != other.count) {
                return count < other.count;
            }

            return inverseKey < other.inverseKey;
        }
    };

    struct TokenNode {
        BpeTokenId token = 0;
        std::size_t previous = noIndex;
        std::size_t next = noIndex;
        bool active = true;
    };

    struct TokenSequence {
        std::vector<TokenNode> nodes;
        std::size_t first = noIndex;
    };

    struct PairOccurrence {
        std::size_t sequenceIndex = 0;
        std::size_t nodeIndex = 0;

        bool operator==(const PairOccurrence& other) const {
            return sequenceIndex == other.sequenceIndex && nodeIndex == other.nodeIndex;
        }
    };

    struct PairOccurrenceHash {
        std::size_t operator()(const PairOccurrence& occurrence) const {
            const std::size_t first = occurrence.sequenceIndex * 0x9e3779b97f4a7c15ull;
            return first ^ (occurrence.nodeIndex + 0x9e3779b97f4a7c15ull + (first << 6) + (first >> 2));
        }
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

    static PairPriority pairPriority(const std::uint64_t key, const std::uint64_t count) {
        return PairPriority{count, std::numeric_limits<std::uint64_t>::max() - key};
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

    class FastTrainingState {
    public:
        void addSequence(const std::vector<BpeTokenId>& tokens) {
            TokenSequence sequence;
            sequence.nodes.reserve(tokens.size());

            for (std::size_t i = 0; i < tokens.size(); ++i) {
                sequence.nodes.push_back(TokenNode{
                    tokens[i],
                    i == 0 ? noIndex : i - 1,
                    i + 1 < tokens.size() ? i + 1 : noIndex,
                    true
                });
            }

            if (!sequence.nodes.empty()) {
                sequence.first = 0;
            }

            sequences.push_back(std::move(sequence));
        }

        void buildInitialQueue() {
            for (std::size_t sequenceIndex = 0; sequenceIndex < sequences.size(); ++sequenceIndex) {
                const TokenSequence& sequence = sequences[sequenceIndex];
                for (std::size_t nodeIndex = sequence.first;
                     nodeIndex != noIndex;
                     nodeIndex = sequence.nodes[nodeIndex].next) {
                    addAdjacentOccurrence(sequenceIndex, nodeIndex);
                }
            }
        }

        PairCount bestPair() {
            while (!pairQueue.empty()) {
                const std::uint64_t key = pairQueue.topKey();
                cleanPair(key);

                if (!pairQueue.empty() && pairQueue.topKey() == key) {
                    const PairPriority priority = pairQueue.topPriority();
                    return pairFromKey(key, priority.count);
                }
            }

            return PairCount{};
        }

        void mergePair(const std::uint64_t key, const BpeTokenId newToken) {
            const auto existing = pairOccurrences.find(key);
            if (existing == pairOccurrences.end()) {
                return;
            }

            std::vector<PairOccurrence> candidates(existing->second.begin(), existing->second.end());
            std::sort(candidates.begin(), candidates.end(), [](const PairOccurrence& lhs, const PairOccurrence& rhs) {
                if (lhs.sequenceIndex != rhs.sequenceIndex) {
                    return lhs.sequenceIndex < rhs.sequenceIndex;
                }

                return lhs.nodeIndex < rhs.nodeIndex;
            });

            for (const PairOccurrence& occurrence : candidates) {
                if (isValidOccurrence(key, occurrence)) {
                    mergeOccurrence(occurrence, newToken);
                } else {
                    removeOccurrence(key, occurrence);
                }
            }
        }

        std::uint64_t countActiveTokens() const {
            std::uint64_t count = 0;

            for (const TokenSequence& sequence : sequences) {
                for (std::size_t nodeIndex = sequence.first;
                     nodeIndex != noIndex;
                     nodeIndex = sequence.nodes[nodeIndex].next) {
                    count += 1;
                }
            }

            return count;
        }

    private:
        using OccurrenceSet = std::unordered_set<PairOccurrence, PairOccurrenceHash>;
        using PairQueue = IndexedPriorityQueue<std::uint64_t, PairPriority>;

        bool hasRightNeighbor(const TokenSequence& sequence, const std::size_t nodeIndex) const {
            return nodeIndex != noIndex
                && nodeIndex < sequence.nodes.size()
                && sequence.nodes[nodeIndex].active
                && sequence.nodes[nodeIndex].next != noIndex
                && sequence.nodes[sequence.nodes[nodeIndex].next].active;
        }

        std::uint64_t adjacentPairKey(const TokenSequence& sequence, const std::size_t nodeIndex) const {
            const TokenNode& left = sequence.nodes[nodeIndex];
            const TokenNode& right = sequence.nodes[left.next];
            return pairKey(left.token, right.token);
        }

        bool isValidOccurrence(const std::uint64_t key, const PairOccurrence& occurrence) const {
            if (occurrence.sequenceIndex >= sequences.size()) {
                return false;
            }

            const TokenSequence& sequence = sequences[occurrence.sequenceIndex];
            if (!hasRightNeighbor(sequence, occurrence.nodeIndex)) {
                return false;
            }

            return adjacentPairKey(sequence, occurrence.nodeIndex) == key;
        }

        void updateQueueForPair(const std::uint64_t key) {
            const auto existing = pairOccurrences.find(key);
            if (existing == pairOccurrences.end() || existing->second.empty()) {
                pairQueue.erase(key);
                return;
            }

            pairQueue.pushOrAssign(key, pairPriority(key, static_cast<std::uint64_t>(existing->second.size())));
        }

        void cleanPair(const std::uint64_t key) {
            const auto existing = pairOccurrences.find(key);
            if (existing == pairOccurrences.end()) {
                pairQueue.erase(key);
                return;
            }

            OccurrenceSet& occurrences = existing->second;
            for (auto occurrence = occurrences.begin(); occurrence != occurrences.end(); ) {
                if (isValidOccurrence(key, *occurrence)) {
                    ++occurrence;
                } else {
                    occurrence = occurrences.erase(occurrence);
                }
            }

            if (occurrences.empty()) {
                pairOccurrences.erase(key);
            }

            updateQueueForPair(key);
        }

        void addOccurrence(const std::uint64_t key, const PairOccurrence occurrence) {
            OccurrenceSet& occurrences = pairOccurrences[key];
            const bool inserted = occurrences.insert(occurrence).second;
            if (inserted) {
                updateQueueForPair(key);
            }
        }

        void removeOccurrence(const std::uint64_t key, const PairOccurrence occurrence) {
            const auto existing = pairOccurrences.find(key);
            if (existing == pairOccurrences.end()) {
                return;
            }

            OccurrenceSet& occurrences = existing->second;
            const std::size_t removed = occurrences.erase(occurrence);
            if (removed == 0) {
                return;
            }

            if (occurrences.empty()) {
                pairOccurrences.erase(key);
            }

            updateQueueForPair(key);
        }

        void addAdjacentOccurrence(const std::size_t sequenceIndex, const std::size_t nodeIndex) {
            const TokenSequence& sequence = sequences[sequenceIndex];
            if (!hasRightNeighbor(sequence, nodeIndex)) {
                return;
            }

            addOccurrence(adjacentPairKey(sequence, nodeIndex), PairOccurrence{sequenceIndex, nodeIndex});
        }

        void removeAdjacentOccurrence(const std::size_t sequenceIndex, const std::size_t nodeIndex) {
            const TokenSequence& sequence = sequences[sequenceIndex];
            if (!hasRightNeighbor(sequence, nodeIndex)) {
                return;
            }

            removeOccurrence(adjacentPairKey(sequence, nodeIndex), PairOccurrence{sequenceIndex, nodeIndex});
        }

        void mergeOccurrence(const PairOccurrence occurrence, const BpeTokenId newToken) {
            TokenSequence& sequence = sequences[occurrence.sequenceIndex];
            TokenNode& leftNode = sequence.nodes[occurrence.nodeIndex];
            const std::size_t rightIndex = leftNode.next;
            TokenNode& rightNode = sequence.nodes[rightIndex];
            const std::size_t previousIndex = leftNode.previous;
            const std::size_t nextIndex = rightNode.next;

            if (previousIndex != noIndex) {
                removeAdjacentOccurrence(occurrence.sequenceIndex, previousIndex);
            }
            removeAdjacentOccurrence(occurrence.sequenceIndex, occurrence.nodeIndex);
            if (nextIndex != noIndex) {
                removeAdjacentOccurrence(occurrence.sequenceIndex, rightIndex);
            }

            leftNode.token = newToken;
            leftNode.next = nextIndex;
            rightNode.active = false;
            rightNode.previous = noIndex;
            rightNode.next = noIndex;

            if (nextIndex != noIndex) {
                sequence.nodes[nextIndex].previous = occurrence.nodeIndex;
            }

            if (previousIndex != noIndex) {
                addAdjacentOccurrence(occurrence.sequenceIndex, previousIndex);
            }
            addAdjacentOccurrence(occurrence.sequenceIndex, occurrence.nodeIndex);
        }

        std::vector<TokenSequence> sequences;
        std::unordered_map<std::uint64_t, OccurrenceSet> pairOccurrences;
        PairQueue pairQueue;
    };
};

} // namespace tfs
