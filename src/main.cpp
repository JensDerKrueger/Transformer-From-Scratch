#include "tfs/bpe_tokenizer.h"
#include "tfs/byte_tokenizer.h"
#include "tfs/corpus_reader.h"
#include "tfs/indexed_priority_queue.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void printUsage(const char* const executable) {
    std::cout << "Usage:\n"
              << "  " << executable << "\n"
              << "  " << executable << " --text \"Hallo Transformer\"\n"
              << "  " << executable << " --corpus path/to/corpus.txt [maxLines]\n"
              << "  " << executable << " --bpe path/to/corpus.txt [maxLines] [mergeCount] [sampleText]\n"
              << "  " << executable << " --ipq\n"
              << "  " << executable << " --ipq-benchmark path/to/corpus.txt [maxLines] [candidateCount] [iterations]\n";
}

void runTextDemo(const std::string& text) {
    const tfs::ByteTokenizer tokenizer;
    const auto tokens = tokenizer.encode(text);
    const auto roundtrip = tokenizer.decode(tokens);

    std::cout << "Input: " << text << '\n';
    std::cout << "Token ids:";
    for (const auto token : tokens) {
        std::cout << ' ' << token;
    }
    std::cout << '\n';
    std::cout << "Decoded: " << roundtrip << '\n';
}

std::uint64_t parseU64(const char* const text) {
    char* end = nullptr;
    const auto value = std::strtoull(text, &end, 10);

    if (end == text || *end != '\0') {
        throw std::runtime_error(std::string("Expected an unsigned integer, got: ") + text);
    }

    return static_cast<std::uint64_t>(value);
}

std::string describeByte(const unsigned char byte) {
    if (byte == ' ') {
        return "space";
    }

    if (byte == '\t') {
        return "tab";
    }

    if (byte == '\n') {
        return "newline";
    }

    if (std::isprint(byte) != 0) {
        return std::string("'") + static_cast<char>(byte) + "'";
    }

    return "non-printable";
}

std::string escapeText(const std::string& text) {
    std::string escaped;

    for (const char value : text) {
        const unsigned char byte = static_cast<unsigned char>(value);

        if (value == '\\') {
            escaped += "\\\\";
        } else if (value == '\t') {
            escaped += "\\t";
        } else if (value == '\n') {
            escaped += "\\n";
        } else if (std::isprint(byte) != 0) {
            escaped.push_back(value);
        } else {
            escaped += "?";
        }
    }

    return escaped;
}

void runCorpusDemo(const std::string& path, const std::uint64_t maxLines) {
    const tfs::CorpusReader reader(path);
    const auto stats = reader.analyze(maxLines);

    std::cout << "Corpus: " << path << '\n';
    std::cout << "Lines: " << stats.lineCount << '\n';
    std::cout << "Bytes: " << stats.byteCount << '\n';
    std::cout << "Line bytes min/avg/max: "
              << stats.safeMinLineBytes() << " / "
              << std::fixed << std::setprecision(2) << stats.averageLineBytes() << " / "
              << stats.maxLineBytes << '\n';

    std::cout << "Most frequent bytes:\n";
    for (const auto& entry : tfs::topBytes(stats, 16)) {
        if (entry.count == 0) {
            break;
        }

        std::cout << "  "
                  << std::setw(3) << static_cast<int>(entry.byte)
                  << "  " << std::setw(13) << entry.count
                  << "  " << describeByte(entry.byte)
                  << '\n';
    }
}

void printBpeTokens(const std::vector<tfs::BpeTokenId>& tokens) {
    std::cout << "Token ids:";
    for (const tfs::BpeTokenId token : tokens) {
        std::cout << ' ' << token;
    }
    std::cout << '\n';
}

void printQueueTop(
    const tfs::IndexedPriorityQueue<std::string, std::uint64_t>& queue,
    const std::string& label
) {
    std::cout << label << ": \""
              << queue.topKey() << "\" count "
              << queue.topPriority() << '\n';
}

void runIndexedPriorityQueueDemo() {
    tfs::IndexedPriorityQueue<std::string, std::uint64_t> queue;

    queue.pushOrAssign("e ", 6823);
    queue.pushOrAssign("s ", 4926);
    queue.pushOrAssign("th", 4225);
    queue.pushOrAssign("d ", 3931);
    queue.pushOrAssign("in", 3905);

    std::cout << "Initial queue size: " << queue.size() << '\n';
    printQueueTop(queue, "Best pair");

    queue.changePriority("th", 7200);
    printQueueTop(queue, "After updating th");

    queue.erase("th");
    printQueueTop(queue, "After removing th");

    std::cout << "Pop order:\n";
    while (!queue.empty()) {
        const auto item = queue.popTop();
        std::cout << "  \"" << item.first << "\" -> " << item.second << '\n';
    }
}

struct PairBenchmarkSeed {
    std::uint64_t lineCount = 0;
    std::uint64_t byteCount = 0;
    std::array<std::uint64_t, 256 * 256> bytePairCounts = {};
};

struct PairCandidate {
    std::uint64_t key = 0;
    std::uint64_t priority = 0;
};

std::uint64_t benchmarkPairKey(const std::uint32_t left, const std::uint32_t right) {
    return (static_cast<std::uint64_t>(left) << 32) | static_cast<std::uint64_t>(right);
}

std::size_t bytePairIndex(const unsigned char left, const unsigned char right) {
    return static_cast<std::size_t>(left) * 256 + static_cast<std::size_t>(right);
}

PairBenchmarkSeed readPairBenchmarkSeed(const std::string& path, const std::uint64_t maxLines) {
    PairBenchmarkSeed seed;
    const tfs::CorpusReader reader(path);

    reader.forEachLine([&](const std::string& line) {
        seed.lineCount += 1;
        seed.byteCount += static_cast<std::uint64_t>(line.size());

        for (std::size_t i = 0; i + 1 < line.size(); ++i) {
            const unsigned char left = static_cast<unsigned char>(line[i]);
            const unsigned char right = static_cast<unsigned char>(line[i + 1]);
            seed.bytePairCounts[bytePairIndex(left, right)] += 1;
        }
    }, maxLines);

    return seed;
}

std::vector<PairCandidate> makePairCandidates(
    const PairBenchmarkSeed& seed,
    const std::size_t candidateCount
) {
    std::vector<PairCandidate> candidates;
    candidates.reserve(candidateCount);

    for (std::size_t i = 0; i < candidateCount; ++i) {
        const auto left = static_cast<std::uint32_t>(256 + i / 65536);
        const auto right = static_cast<std::uint32_t>(i % 65536);
        const auto leftByte = static_cast<unsigned char>(left & 0xffu);
        const auto rightByte = static_cast<unsigned char>(right & 0xffu);
        const std::uint64_t corpusCount = seed.bytePairCounts[bytePairIndex(leftByte, rightByte)];
        const std::uint64_t tieBreaker = static_cast<std::uint64_t>(candidateCount - i);

        candidates.push_back(PairCandidate{
            benchmarkPairKey(left, right),
            corpusCount * 1000000ull + tieBreaker
        });
    }

    return candidates;
}

std::size_t findBestCandidate(const std::vector<PairCandidate>& candidates) {
    std::size_t best = 0;

    for (std::size_t i = 1; i < candidates.size(); ++i) {
        if (candidates[best].priority < candidates[i].priority) {
            best = i;
        }
    }

    return best;
}

std::uint64_t demotePriority(const std::uint64_t priority, const std::size_t iteration) {
    return priority / 2 + static_cast<std::uint64_t>(iteration % 97);
}

template <typename Function>
double measureMilliseconds(Function function) {
    const auto start = std::chrono::steady_clock::now();
    function();
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

void runIndexedPriorityQueueBenchmark(
    const std::string& path,
    const std::uint64_t maxLines,
    const std::size_t candidateCount,
    const std::size_t iterations
) {
    if (candidateCount == 0) {
        throw std::runtime_error("candidateCount must be greater than zero");
    }

    const PairBenchmarkSeed seed = readPairBenchmarkSeed(path, maxLines);
    const std::vector<PairCandidate> originalCandidates = makePairCandidates(seed, candidateCount);

    std::vector<PairCandidate> naiveCandidates = originalCandidates;
    std::uint64_t naiveChecksum = 0;
    const double naiveMilliseconds = measureMilliseconds([&]() {
        for (std::size_t i = 0; i < iterations; ++i) {
            const std::size_t best = findBestCandidate(naiveCandidates);
            naiveChecksum += naiveCandidates[best].key ^ naiveCandidates[best].priority ^ static_cast<std::uint64_t>(i);
            naiveCandidates[best].priority = demotePriority(naiveCandidates[best].priority, i);
        }
    });

    tfs::IndexedPriorityQueue<std::uint64_t, std::uint64_t> queue;
    const double queueBuildMilliseconds = measureMilliseconds([&]() {
        for (const PairCandidate& candidate : originalCandidates) {
            queue.pushOrAssign(candidate.key, candidate.priority);
        }
    });

    std::uint64_t queueChecksum = 0;
    const double queueUpdateMilliseconds = measureMilliseconds([&]() {
        for (std::size_t i = 0; i < iterations; ++i) {
            const std::uint64_t key = queue.topKey();
            const std::uint64_t priority = queue.topPriority();
            queueChecksum += key ^ priority ^ static_cast<std::uint64_t>(i);
            queue.changePriority(key, demotePriority(priority, i));
        }
    });

    std::cout << "Corpus: " << path << '\n';
    std::cout << "Corpus lines: " << seed.lineCount << '\n';
    std::cout << "Corpus bytes: " << seed.byteCount << '\n';
    std::cout << "Candidate pairs: " << candidateCount << '\n';
    std::cout << "Iterations: " << iterations << '\n';
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Naive linear scan: " << naiveMilliseconds << " ms\n";
    std::cout << "Indexed priority queue build: " << queueBuildMilliseconds << " ms\n";
    std::cout << "Indexed priority queue updates: " << queueUpdateMilliseconds << " ms\n";
    std::cout << "Speedup for repeated max/update: "
              << naiveMilliseconds / queueUpdateMilliseconds << "x\n";
    std::cout << "Checksums: " << naiveChecksum << " / " << queueChecksum << '\n';
}

void runBpeDemo(
    const std::string& path,
    const std::uint64_t maxLines,
    const std::size_t mergeCount,
    const std::string& sampleText
) {
    const tfs::BpeTrainer trainer;
    const auto result = trainer.trainFromCorpus(path, maxLines, mergeCount);
    const auto& model = result.model;
    const auto& merges = model.getMerges();

    std::cout << "Corpus: " << path << '\n';
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
                  << "  text \"" << escapeText(model.decodeToken(merge.token)) << "\"\n";
    }

    const auto tokens = model.encode(sampleText);
    const auto decoded = model.decode(tokens);

    std::cout << "Sample: " << sampleText << '\n';
    printBpeTokens(tokens);
    std::cout << "Sample bytes/tokens: " << sampleText.size() << " / " << tokens.size() << '\n';
    std::cout << "Decoded: " << decoded << '\n';
}

} // namespace

int main(const int argc, char** argv) {
    try {
        if (argc == 1) {
            runTextDemo("Hallo Transformer");
            return 0;
        }

        const std::string mode = argv[1];

        if (mode == "--help" || mode == "-h") {
            printUsage(argv[0]);
            return 0;
        }

        if (mode == "--text") {
            if (argc != 3) {
                printUsage(argv[0]);
                return 1;
            }

            runTextDemo(argv[2]);
            return 0;
        }

        if (mode == "--corpus") {
            if (argc != 3 && argc != 4) {
                printUsage(argv[0]);
                return 1;
            }

            const auto maxLines = argc == 4 ? parseU64(argv[3]) : 0;
            runCorpusDemo(argv[2], maxLines);
            return 0;
        }

        if (mode == "--bpe") {
            if (argc < 3 || argc > 6) {
                printUsage(argv[0]);
                return 1;
            }

            const std::uint64_t maxLines = argc >= 4 ? parseU64(argv[3]) : 1000;
            const std::size_t mergeCount = argc >= 5 ? static_cast<std::size_t>(parseU64(argv[4])) : 32;
            const std::string sampleText = argc >= 6 ? argv[5] : "the transformer learns from text";

            runBpeDemo(argv[2], maxLines, mergeCount, sampleText);
            return 0;
        }

        if (mode == "--ipq") {
            if (argc != 2) {
                printUsage(argv[0]);
                return 1;
            }

            runIndexedPriorityQueueDemo();
            return 0;
        }

        if (mode == "--ipq-benchmark") {
            if (argc < 3 || argc > 6) {
                printUsage(argv[0]);
                return 1;
            }

            const std::uint64_t maxLines = argc >= 4 ? parseU64(argv[3]) : 1000000;
            const auto candidateCount = argc >= 5 ? static_cast<std::size_t>(parseU64(argv[4])) : 1000000;
            const auto iterations = argc >= 6 ? static_cast<std::size_t>(parseU64(argv[5])) : 1000;

            runIndexedPriorityQueueBenchmark(argv[2], maxLines, candidateCount, iterations);
            return 0;
        }

        runTextDemo(argv[1]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
