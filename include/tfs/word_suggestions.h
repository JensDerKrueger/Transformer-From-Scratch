#pragma once

#include "tfs/byte_tokenizer.h"
#include "tfs/softmax.h"
#include "tfs/tiny_language_model.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cctype>
#include <string>
#include <vector>

namespace tfs {

struct WordSuggestion {
    std::string word;
    TensorValue score = 0.0f;
};

inline std::vector<std::size_t> makeContextTokens(
    const std::string& text,
    const std::size_t contextLength
) {
    const ByteTokenizer tokenizer;
    const std::vector<ByteTokenizer::TokenId> byteTokens = tokenizer.encode(text.empty() ? " " : text);
    std::vector<std::size_t> tokens;
    tokens.reserve(byteTokens.size());

    for (const ByteTokenizer::TokenId token : byteTokens) {
        tokens.push_back(token);
    }

    if (tokens.size() > contextLength) {
        tokens.erase(tokens.begin(), tokens.end() - static_cast<std::ptrdiff_t>(contextLength));
    }
    while (tokens.size() < contextLength) {
        tokens.insert(tokens.begin(), static_cast<std::size_t>(' '));
    }

    return tokens;
}

inline std::vector<WordSuggestion> suggestNextWords(
    const TinyLanguageModel& model,
    const std::string& prompt,
    const std::size_t suggestionCount,
    const std::size_t maxCharacters
) {
    struct Beam {
        std::string text;
        std::vector<std::size_t> context;
        TensorValue score = 0.0f;
        bool complete = false;
    };

    const std::size_t contextLength = model.getConfig().contextLength;
    std::vector<Beam> beams = {Beam{"", makeContextTokens(prompt, contextLength), 0.0f, false}};
    std::vector<WordSuggestion> completed;

    for (std::size_t step = 0; step < maxCharacters; ++step) {
        std::vector<Beam> nextBeams;

        for (const Beam& beam : beams) {
            if (beam.complete) {
                nextBeams.push_back(beam);
                continue;
            }

            const Tensor logits = model.forward(beam.context);
            const Tensor probabilities = softmaxRows(logits);
            const std::size_t row = probabilities.getShape()[0] - 1;
            const std::vector<TopKEntry> top = topKRow(probabilities, row, suggestionCount * 24);

            for (const TopKEntry& entry : top) {
                const unsigned char byte = static_cast<unsigned char>(entry.index);
                const char character = static_cast<char>(byte);
                const bool isWordEnd = character == ' ' || character == '\n' || character == '\t';

                if (isWordEnd && beam.text.size() < 3) {
                    continue;
                }

                Beam next = beam;
                next.score += std::log(std::max(entry.value, 1.0e-12f));
                next.context.push_back(entry.index);
                if (next.context.size() > contextLength) {
                    next.context.erase(next.context.begin());
                }

                if (isWordEnd) {
                    next.complete = true;
                    const TensorValue normalizedScore =
                        next.score / static_cast<TensorValue>(next.text.size() + 1);
                    completed.push_back(WordSuggestion{next.text, normalizedScore});
                } else if (std::isprint(byte) != 0) {
                    next.text.push_back(character);
                    nextBeams.push_back(std::move(next));
                }
            }
        }

        std::sort(nextBeams.begin(), nextBeams.end(), [](const Beam& left, const Beam& right) {
            return left.score > right.score;
        });

        if (nextBeams.size() > suggestionCount * 16) {
            nextBeams.resize(suggestionCount * 16);
        }
        beams = std::move(nextBeams);

        if (completed.size() >= suggestionCount) {
            break;
        }
    }

    if (completed.empty()) {
        for (const Beam& beam : beams) {
            if (!beam.text.empty()) {
                completed.push_back(WordSuggestion{
                    beam.text,
                    beam.score / static_cast<TensorValue>(beam.text.size())
                });
            }
        }
    }

    std::sort(completed.begin(), completed.end(), [](const WordSuggestion& left, const WordSuggestion& right) {
        return left.score > right.score;
    });

    std::vector<WordSuggestion> unique;
    for (const WordSuggestion& suggestion : completed) {
        const auto found = std::find_if(unique.begin(), unique.end(), [&](const WordSuggestion& item) {
            return item.word == suggestion.word;
        });
        if (found == unique.end()) {
            unique.push_back(suggestion);
        }
        if (unique.size() == suggestionCount) {
            break;
        }
    }

    return unique;
}

inline TensorValue scoreContinuation(
    const TinyLanguageModel& model,
    const std::string& prompt,
    const std::string& continuation
) {
    std::vector<std::size_t> context = makeContextTokens(prompt, model.getConfig().contextLength);
    const ByteTokenizer tokenizer;
    const std::vector<ByteTokenizer::TokenId> byteTokens = tokenizer.encode(continuation + " ");
    TensorValue score = 0.0f;

    for (const ByteTokenizer::TokenId token : byteTokens) {
        const Tensor logits = model.forward(context);
        const Tensor probabilities = softmaxRows(logits);
        const std::size_t row = probabilities.getShape()[0] - 1;
        const std::size_t columns = probabilities.getShape()[1];
        const TensorValue probability = probabilities[row * columns + token];

        score += std::log(std::max(probability, 1.0e-12f));
        context.push_back(token);
        if (context.size() > model.getConfig().contextLength) {
            context.erase(context.begin());
        }
    }

    return score / static_cast<TensorValue>(byteTokens.size());
}

inline std::vector<WordSuggestion> rankCandidateWords(
    const TinyLanguageModel& model,
    const std::string& prompt,
    const std::vector<std::string>& candidates,
    const std::size_t suggestionCount
) {
    std::vector<WordSuggestion> suggestions;
    suggestions.reserve(candidates.size());

    for (const std::string& candidate : candidates) {
        if (!candidate.empty()) {
            suggestions.push_back(WordSuggestion{
                candidate,
                scoreContinuation(model, prompt, candidate)
            });
        }
    }

    std::sort(suggestions.begin(), suggestions.end(), [](const WordSuggestion& left, const WordSuggestion& right) {
        return left.score > right.score;
    });

    if (suggestions.size() > suggestionCount) {
        suggestions.resize(suggestionCount);
    }

    return suggestions;
}

} // namespace tfs
