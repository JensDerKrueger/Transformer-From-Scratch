#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tfs {

struct CorpusStats {
    std::uint64_t lineCount = 0;
    std::uint64_t byteCount = 0;
    std::uint64_t minLineBytes = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t maxLineBytes = 0;
    std::array<std::uint64_t, 256> byteHistogram = {};

    double averageLineBytes() const {
        if (lineCount == 0) {
            return 0.0;
        }

        return static_cast<double>(byteCount) / static_cast<double>(lineCount);
    }

    std::uint64_t safeMinLineBytes() const {
        if (lineCount == 0) {
            return 0;
        }

        return minLineBytes;
    }
};

class CorpusReader {
public:
    explicit CorpusReader(std::string pathValue)
        : path(std::move(pathValue)) {}

    CorpusStats analyze(const std::uint64_t maxLines = 0) const {
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            throw std::runtime_error("Could not open corpus file: " + path);
        }

        CorpusStats stats;
        std::string line;

        while (std::getline(input, line)) {
            if (maxLines != 0 && stats.lineCount >= maxLines) {
                break;
            }

            updateStats(stats, line);
        }

        return stats;
    }

    template <typename LineCallback>
    void forEachLine(LineCallback callback, const std::uint64_t maxLines = 0) const {
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            throw std::runtime_error("Could not open corpus file: " + path);
        }

        std::uint64_t lineCount = 0;
        std::string line;

        while (std::getline(input, line)) {
            if (maxLines != 0 && lineCount >= maxLines) {
                break;
            }

            callback(line);
            lineCount += 1;
        }
    }

private:
    static void updateStats(CorpusStats& stats, const std::string& line) {
        const auto lineBytes = static_cast<std::uint64_t>(line.size());

        stats.lineCount += 1;
        stats.byteCount += lineBytes;
        stats.minLineBytes = std::min(stats.minLineBytes, lineBytes);
        stats.maxLineBytes = std::max(stats.maxLineBytes, lineBytes);

        for (const unsigned char byte : line) {
            stats.byteHistogram[byte] += 1;
        }
    }

    const std::string path;
};

struct ByteFrequency {
    unsigned char byte = 0;
    std::uint64_t count = 0;
};

inline std::vector<ByteFrequency> topBytes(const CorpusStats& stats, const std::size_t limit) {
    std::vector<ByteFrequency> frequencies;
    frequencies.reserve(stats.byteHistogram.size());

    for (std::size_t i = 0; i < stats.byteHistogram.size(); ++i) {
        frequencies.push_back(ByteFrequency{
            static_cast<unsigned char>(i),
            stats.byteHistogram[i]
        });
    }

    std::sort(frequencies.begin(), frequencies.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.count > rhs.count;
    });

    if (frequencies.size() > limit) {
        frequencies.resize(limit);
    }

    return frequencies;
}

} // namespace tfs
