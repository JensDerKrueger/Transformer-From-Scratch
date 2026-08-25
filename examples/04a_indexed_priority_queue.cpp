#include "demo_helpers.h"

#include "tfs/corpus_reader.h"
#include "tfs/indexed_priority_queue.h"

#include <array>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct PairBenchmarkSeed {
    std::uint64_t lineCount = 0;
    std::uint64_t byteCount = 0;
    std::array<std::uint64_t, 256 * 256> bytePairCounts = {};
};

struct PairCandidate {
    std::uint64_t key = 0;
    std::uint64_t priority = 0;
};

void printQueueTop(
    const tfs::IndexedPriorityQueue<std::string, std::uint64_t>& queue,
    const std::string& label
) {
    std::cout << label << ": \""
              << queue.topKey() << "\" count "
              << queue.topPriority() << '\n';
}

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

void runQueueDemo() {
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

void runBenchmark(
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
    const double naiveMilliseconds = demo::measureMilliseconds([&]() {
        for (std::size_t i = 0; i < iterations; ++i) {
            const std::size_t best = findBestCandidate(naiveCandidates);
            naiveChecksum += naiveCandidates[best].key ^ naiveCandidates[best].priority ^ static_cast<std::uint64_t>(i);
            naiveCandidates[best].priority = demotePriority(naiveCandidates[best].priority, i);
        }
    });

    tfs::IndexedPriorityQueue<std::uint64_t, std::uint64_t> queue;
    const double queueBuildMilliseconds = demo::measureMilliseconds([&]() {
        for (const PairCandidate& candidate : originalCandidates) {
            queue.pushOrAssign(candidate.key, candidate.priority);
        }
    });

    std::uint64_t queueChecksum = 0;
    const double queueUpdateMilliseconds = demo::measureMilliseconds([&]() {
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

int main(const int argc, char** argv) {
    try {
        if (argc == 1 || (argc == 2 && std::string(argv[1]) == "--ipq")) {
            runQueueDemo();
            return 0;
        }

        const int firstArg = (argc >= 2 && std::string(argv[1]) == "--ipq-benchmark") ? 2 : 1;

        if (argc != firstArg + 4) {
            std::cerr << "Usage: " << argv[0]
                      << " [--ipq-benchmark] path/to/corpus.txt maxLines candidateCount iterations\n";
            return 1;
        }

        runBenchmark(
            argv[firstArg],
            demo::parseU64(argv[firstArg + 1]),
            static_cast<std::size_t>(demo::parseU64(argv[firstArg + 2])),
            static_cast<std::size_t>(demo::parseU64(argv[firstArg + 3]))
        );
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
