#include "demo_helpers.h"

#include "tfs/corpus_reader.h"

#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

int main(const int argc, char** argv) {
    try {
        const int firstArg = (argc >= 2 && std::string(argv[1]) == "--corpus") ? 2 : 1;

        if (argc != firstArg + 2) {
            std::cerr << "Usage: " << argv[0] << " [--corpus] path/to/corpus.txt maxLines\n";
            return 1;
        }

        const std::string path = argv[firstArg];
        const std::uint64_t maxLines = demo::parseU64(argv[firstArg + 1]);
        const tfs::CorpusReader reader(path);
        const tfs::CorpusStats stats = reader.analyze(maxLines);

        std::cout << "Corpus: " << path << '\n';
        std::cout << "Lines: " << stats.lineCount << '\n';
        std::cout << "Bytes: " << stats.byteCount << '\n';
        std::cout << "Line bytes min/avg/max: "
                  << stats.safeMinLineBytes() << " / "
                  << std::fixed << std::setprecision(2) << stats.averageLineBytes() << " / "
                  << stats.maxLineBytes << '\n';

        std::cout << "Most frequent bytes:\n";
        for (const tfs::ByteFrequency& entry : tfs::topBytes(stats, 16)) {
            if (entry.count == 0) {
                break;
            }

            std::cout << "  "
                      << std::setw(3) << static_cast<int>(entry.byte)
                      << "  " << std::setw(13) << entry.count
                      << "  " << demo::describeByte(entry.byte)
                      << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
