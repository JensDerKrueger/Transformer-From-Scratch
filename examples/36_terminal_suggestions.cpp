#include "demo_helpers.h"

#include "tfs/checkpoint.h"
#include "tfs/random_initializer.h"
#include "tfs/tiny_language_model.h"
#include "tfs/word_suggestions.h"

#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<std::size_t> byteTokensFromText(const std::string& text) {
    std::vector<std::size_t> tokens;
    tokens.reserve(text.size());
    for (const unsigned char value : text) {
        tokens.push_back(value);
    }
    return tokens;
}

void trainPhrase(tfs::TinyLanguageModel& model, const std::string& prompt, const std::string& continuation) {
    std::vector<std::size_t> context = tfs::makeContextTokens(prompt, model.getConfig().contextLength);
    const std::vector<std::size_t> continuationTokens = byteTokensFromText(continuation + " ");

    for (const std::size_t token : continuationTokens) {
        model.trainNextToken(context, token, 0.02f);
        context.push_back(token);
        if (context.size() > model.getConfig().contextLength) {
            context.erase(context.begin());
        }
    }
}

void trainToyModel(tfs::TinyLanguageModel& model) {
    for (std::size_t epoch = 0; epoch < 420; ++epoch) {
        trainPhrase(model, "der film war ", "gut");
        trainPhrase(model, "der film war ", "schlecht");
        trainPhrase(model, "der film war ", "spannend");
        trainPhrase(model, "ueberhaupt nicht ", "gut");
        trainPhrase(model, "ueberhaupt nicht ", "schlecht");
    }
}

void printSuggestions(
    const tfs::TinyLanguageModel& model,
    const std::string& prompt,
    const std::vector<std::string>& candidates
) {
    const std::vector<tfs::WordSuggestion> suggestions = tfs::rankCandidateWords(model, prompt, candidates, 3);

    std::cout << "Prompt: " << prompt << '\n';
    for (std::size_t i = 0; i < suggestions.size(); ++i) {
        std::cout << (i + 1) << ") " << suggestions[i].word << '\n';
    }
}

void runLineMode(const tfs::TinyLanguageModel& model, const std::vector<std::string>& candidates) {
    std::string text;

    while (true) {
        printSuggestions(model, text, candidates);
        std::cout << "> ";

        std::string input;
        if (!std::getline(std::cin, input) || input.empty()) {
            break;
        }

        const std::vector<tfs::WordSuggestion> suggestions = tfs::rankCandidateWords(model, text, candidates, 3);
        if (input.size() == 1 && input[0] >= '1' && input[0] <= '3') {
            const std::size_t index = static_cast<std::size_t>(input[0] - '1');
            if (index < suggestions.size()) {
                text += suggestions[index].word;
                text += ' ';
            }
        } else {
            text += input;
        }

        std::cout << '\n';
    }
}

std::vector<std::string> toyCandidates() {
    return {"gut", "schlecht", "spannend", "langweilig"};
}

void loadCheckpoint(tfs::TinyLanguageModel& model, const std::string& checkpointPath) {
    tfs::loadParameters(checkpointPath, model.parameters());
}

void printUsage(const char* const programName) {
    std::cerr << "Usage:\n";
    std::cerr << "  " << programName << " --suggest checkpoint corpus prompt maxBytes candidateCount\n";
    std::cerr << "  " << programName << " --interactive checkpoint corpus maxBytes candidateCount\n";
    std::cerr << "  " << programName << " --toy [prompt]\n";
}

} // namespace

int main(const int argc, char** argv) {
    try {
        tfs::TinyLanguageModelConfig config;
        config.contextLength = 16;
        config.modelSize = 16;
        config.attentionSize = 16;
        config.hiddenSize = 32;

        tfs::RandomInitializer initializer(41);
        tfs::TinyLanguageModel model(config);
        model.initialize(initializer);

        if (argc >= 2 && std::string(argv[1]) == "--toy") {
            trainToyModel(model);
            const std::string prompt = argc >= 3 ? argv[2] : "der film war ";
            printSuggestions(model, prompt, toyCandidates());
            std::cout << "Toy mode trains on fixed phrases inside this demo\n";
            return 0;
        }

        if (argc >= 2 && std::string(argv[1]) == "--interactive") {
            if (argc != 6) {
                printUsage(argv[0]);
                return 1;
            }

            const std::string checkpointPath = argv[2];
            const std::string corpusPath = argv[3];
            const std::size_t maxBytes = static_cast<std::size_t>(demo::parseU64(argv[4]));
            const std::size_t candidateCount = static_cast<std::size_t>(demo::parseU64(argv[5]));
            loadCheckpoint(model, checkpointPath);
            const std::vector<std::string> candidates =
                tfs::readFrequentWordsFromFile(corpusPath, maxBytes, candidateCount, 3);
            runLineMode(model, candidates);
            return 0;
        }

        if (argc != 7 || std::string(argv[1]) != "--suggest") {
            printUsage(argv[0]);
            return 1;
        }

        const std::string checkpointPath = argv[2];
        const std::string corpusPath = argv[3];
        const std::string prompt = argv[4];
        const std::size_t maxBytes = static_cast<std::size_t>(demo::parseU64(argv[5]));
        const std::size_t candidateCount = static_cast<std::size_t>(demo::parseU64(argv[6]));

        loadCheckpoint(model, checkpointPath);
        const std::vector<std::string> candidates =
            tfs::readFrequentWordsFromFile(corpusPath, maxBytes, candidateCount, 3);

        printSuggestions(model, prompt, candidates);
        std::cout << "Checkpoint: " << checkpointPath << '\n';
        std::cout << "Candidate corpus: " << corpusPath << '\n';
        std::cout << "Candidate words: " << candidates.size() << '\n';
        std::cout << "Type 1, 2 or 3 in interactive mode to accept a suggestion\n";
        std::cout << "Any other text continues the prompt manually\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
