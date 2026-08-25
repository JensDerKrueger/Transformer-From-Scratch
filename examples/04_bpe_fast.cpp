#include "demo_helpers.h"

#include "tfs/bpe_tokenizer.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

void printBpeTokens(const std::vector<tfs::BpeTokenId>& tokens) {
    std::cout << "Token ids:";
    for (const tfs::BpeTokenId token : tokens) {
        std::cout << ' ' << token;
    }
    std::cout << '\n';
}

int main(const int argc, char** argv) {
    try {
        const int firstArg = (argc >= 2 && std::string(argv[1]) == "--bpe-fast") ? 2 : 1;

        if (argc != firstArg + 4) {
            std::cerr << "Usage: " << argv[0] << " [--bpe-fast] path/to/corpus.txt maxLines mergeCount sampleText\n";
            return 1;
        }

        const std::string path = argv[firstArg];
        const std::uint64_t maxLines = demo::parseU64(argv[firstArg + 1]);
        const auto mergeCount = static_cast<std::size_t>(demo::parseU64(argv[firstArg + 2]));
        const std::string sampleText = argv[firstArg + 3];

        const tfs::BpeTrainer trainer;
        const tfs::BpeTrainingResult result = trainer.trainFromCorpusWithQueue(path, maxLines, mergeCount);
        const tfs::BpeModel& model = result.model;
        const std::vector<tfs::BpeMerge>& merges = model.getMerges();

        std::cout << "Corpus: " << path << '\n';
        std::cout << "Trainer: indexed priority queue\n";
        std::cout << "Training lines: " << result.lineCount << '\n';
        std::cout << "Requested merges: " << mergeCount << '\n';
        std::cout << "Learned merges: " << merges.size() << '\n';
        std::cout << "Vocabulary size: " << model.vocabSize() << '\n';
        std::cout << "Training token count before/after: "
                  << result.initialTokenCount << " / " << result.finalTokenCount << '\n';

        std::cout << "First merges:\n";
        const std::size_t displayedMerges = std::min<std::size_t>(merges.size(), 16);
        for (std::size_t i = 0; i < displayedMerges; ++i) {
            const tfs::BpeMerge& merge = merges[i];
            std::cout << "  "
                      << merge.token << " = "
                      << merge.left << " + " << merge.right
                      << "  count " << merge.count
                      << "  text \"" << demo::escapeText(model.decodeToken(merge.token)) << "\"\n";
        }

        const std::vector<tfs::BpeTokenId> tokens = model.encode(sampleText);
        const std::string decoded = model.decode(tokens);

        std::cout << "Sample: " << sampleText << '\n';
        printBpeTokens(tokens);
        std::cout << "Sample bytes/tokens: " << sampleText.size() << " / " << tokens.size() << '\n';
        std::cout << "Decoded: " << decoded << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
