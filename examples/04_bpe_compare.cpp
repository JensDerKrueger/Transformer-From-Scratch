#include "demo_helpers.h"

#include "tfs/bpe_tokenizer.h"

#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main(const int argc, char** argv) {
    try {
        const int firstArg = (argc >= 2 && std::string(argv[1]) == "--bpe-compare") ? 2 : 1;

        if (argc != firstArg + 3) {
            std::cerr << "Usage: " << argv[0] << " [--bpe-compare] path/to/corpus.txt maxLines mergeCount\n";
            return 1;
        }

        const std::string path = argv[firstArg];
        const std::uint64_t maxLines = demo::parseU64(argv[firstArg + 1]);
        const auto mergeCount = static_cast<std::size_t>(demo::parseU64(argv[firstArg + 2]));
        const tfs::BpeTrainer trainer;

        tfs::BpeTrainingResult naiveResult;
        const double naiveMilliseconds = demo::measureMilliseconds([&]() {
            naiveResult = trainer.trainFromCorpus(path, maxLines, mergeCount);
        });

        tfs::BpeTrainingResult queueResult;
        const double queueMilliseconds = demo::measureMilliseconds([&]() {
            queueResult = trainer.trainFromCorpusWithQueue(path, maxLines, mergeCount);
        });

        const std::vector<tfs::BpeMerge>& naiveMerges = naiveResult.model.getMerges();
        const std::vector<tfs::BpeMerge>& queueMerges = queueResult.model.getMerges();
        bool sameMerges = naiveMerges.size() == queueMerges.size();

        for (std::size_t i = 0; i < naiveMerges.size() && i < queueMerges.size(); ++i) {
            const tfs::BpeMerge& naiveMerge = naiveMerges[i];
            const tfs::BpeMerge& queueMerge = queueMerges[i];
            if (naiveMerge.left != queueMerge.left
                || naiveMerge.right != queueMerge.right
                || naiveMerge.token != queueMerge.token
                || naiveMerge.count != queueMerge.count) {
                sameMerges = false;
                break;
            }
        }

        std::cout << "Corpus: " << path << '\n';
        std::cout << "Training lines: " << naiveResult.lineCount << '\n';
        std::cout << "Requested merges: " << mergeCount << '\n';
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Naive trainer: " << naiveMilliseconds << " ms\n";
        std::cout << "IndexedPQ trainer: " << queueMilliseconds << " ms\n";
        std::cout << "Speedup: " << naiveMilliseconds / queueMilliseconds << "x\n";
        std::cout << "Naive learned/final tokens: "
                  << naiveMerges.size() << " / " << naiveResult.finalTokenCount << '\n';
        std::cout << "IndexedPQ learned/final tokens: "
                  << queueMerges.size() << " / " << queueResult.finalTokenCount << '\n';
        std::cout << "Same merges: " << (sameMerges ? "yes" : "no") << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
