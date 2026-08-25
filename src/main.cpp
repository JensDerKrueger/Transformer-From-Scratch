#include "tfs/attention_projections.h"
#include "tfs/attention_scores.h"
#include "tfs/attention_weights.h"
#include "tfs/bpe_tokenizer.h"
#include "tfs/byte_tokenizer.h"
#include "tfs/corpus_reader.h"
#include "tfs/indexed_priority_queue.h"
#include "tfs/linear_layer.h"
#include "tfs/position_embedding.h"
#include "tfs/tensor.h"
#include "tfs/tensor_operations.h"
#include "tfs/token_embedding.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
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
              << "  " << executable << " --bpe-fast path/to/corpus.txt [maxLines] [mergeCount] [sampleText]\n"
              << "  " << executable << " --bpe-compare path/to/corpus.txt [maxLines] [mergeCount]\n"
              << "  " << executable << " --tensor\n"
              << "  " << executable << " --matmul\n"
              << "  " << executable << " --embedding\n"
              << "  " << executable << " --position\n"
              << "  " << executable << " --linear\n"
              << "  " << executable << " --qkv\n"
              << "  " << executable << " --attention-scores\n"
              << "  " << executable << " --attention-weights\n"
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

template <typename Value>
void printList(const std::vector<Value>& values) {
    std::cout << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << values[i];
    }
    std::cout << ']';
}

void runTensorDemo() {
    const tfs::TensorShape shape({2, 3, 4});
    tfs::Tensor tensor(shape);

    for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<tfs::TensorValue>(i);
    }

    const std::vector<std::size_t> index = {1, 2, 3};
    tensor.at(index) = 42.0f;

    const tfs::Tensor bias = tfs::Tensor::filled(shape, 1.0f);
    const tfs::Tensor shifted = tfs::add(tensor, bias);
    const tfs::Tensor scaled = tfs::multiply(shifted, 0.5f);

    std::cout << "Shape: ";
    printList(shape.getDimensions());
    std::cout << '\n';

    std::cout << "Rank: " << shape.rank() << '\n';
    std::cout << "Element count: " << shape.elementCount() << '\n';

    std::cout << "Strides: ";
    printList(tensor.getStrides());
    std::cout << '\n';

    std::cout << "Index: ";
    printList(index);
    std::cout << '\n';

    std::cout << "Flat offset: " << tensor.offset(index) << '\n';
    std::cout << "Value at index: " << tensor.at(index) << '\n';

    std::cout << "First row after add and scale:";
    std::cout << std::fixed << std::setprecision(1);
    for (std::size_t i = 0; i < 4; ++i) {
        std::cout << ' ' << scaled[i];
    }
    std::cout << '\n';
}

void printMatrix(const tfs::Tensor& matrix) {
    const auto& shape = matrix.getShape();
    const auto& strides = matrix.getStrides();
    const tfs::TensorValue* const values = matrix.data();

    for (std::size_t row = 0; row < shape[0]; ++row) {
        const std::size_t rowOffset = row * strides[0];
        std::cout << "  ";

        for (std::size_t column = 0; column < shape[1]; ++column) {
            if (column != 0) {
                std::cout << ' ';
            }

            std::cout << values[rowOffset + column * strides[1]];
        }

        std::cout << '\n';
    }
}

void runMatrixMultiplyDemo() {
    const tfs::Tensor left = tfs::Tensor::fromValues(
        tfs::TensorShape({2, 3}),
        {1.0f, 2.0f, 3.0f,
         4.0f, 5.0f, 6.0f}
    );
    const tfs::Tensor right = tfs::Tensor::fromValues(
        tfs::TensorShape({3, 2}),
        {7.0f, 8.0f,
         9.0f, 10.0f,
         11.0f, 12.0f}
    );
    const tfs::Tensor result = tfs::matrixMultiply(left, right);

    std::cout << "Left shape: ";
    printList(left.getShape().getDimensions());
    std::cout << '\n';

    std::cout << "Right shape: ";
    printList(right.getShape().getDimensions());
    std::cout << '\n';

    std::cout << "Result shape: ";
    printList(result.getShape().getDimensions());
    std::cout << '\n';

    std::cout << "Left matrix:\n";
    printMatrix(left);

    std::cout << "Right matrix:\n";
    printMatrix(right);

    std::cout << "Result matrix:\n";
    printMatrix(result);

    std::cout << "result[1, 0] = 4*7 + 5*9 + 6*11 = "
              << result.at({1, 0}) << '\n';
}

void runEmbeddingDemo() {
    const tfs::TokenEmbedding embedding(
        6,
        3,
        {
            0.00f, 0.01f, 0.02f,
            0.10f, 0.11f, 0.12f,
            0.20f, 0.21f, 0.22f,
            0.30f, 0.31f, 0.32f,
            0.40f, 0.41f, 0.42f,
            0.50f, 0.51f, 0.52f
        }
    );
    const std::vector<tfs::TokenId> tokens = {3, 1, 4, 1};
    const tfs::Tensor vectors = embedding.embed(tokens);

    std::cout << "Vocab size: " << embedding.vocabSize() << '\n';
    std::cout << "Embedding size: " << embedding.embeddingSize() << '\n';

    std::cout << "Token ids: ";
    printList(tokens);
    std::cout << '\n';

    std::cout << "Embedding shape: ";
    printList(vectors.getShape().getDimensions());
    std::cout << '\n';

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Selected vectors:\n";
    printMatrix(vectors);
    std::cout << std::defaultfloat;

    std::cout << "First token vector came from weight row 3\n";
    std::cout << "Repeated token id 1 produces the same vector twice\n";
}

void runPositionEmbeddingDemo() {
    const tfs::TokenEmbedding tokenEmbedding(
        6,
        3,
        {
            0.00f, 0.01f, 0.02f,
            0.10f, 0.11f, 0.12f,
            0.20f, 0.21f, 0.22f,
            0.30f, 0.31f, 0.32f,
            0.40f, 0.41f, 0.42f,
            0.50f, 0.51f, 0.52f
        }
    );
    const tfs::PositionEmbedding positionEmbedding(
        4,
        3,
        {
            0.00f, 1.00f, 2.00f,
            0.01f, 1.01f, 2.01f,
            0.02f, 1.02f, 2.02f,
            0.03f, 1.03f, 2.03f
        }
    );
    const std::vector<tfs::TokenId> tokens = {3, 1, 4, 1};
    const tfs::Tensor tokenVectors = tokenEmbedding.embed(tokens);
    const tfs::Tensor positionVectors = positionEmbedding.embed(tokens.size());
    const tfs::Tensor transformerInput = positionEmbedding.addTo(tokenVectors);

    std::cout << "Token ids: ";
    printList(tokens);
    std::cout << '\n';

    std::cout << "Token embedding shape: ";
    printList(tokenVectors.getShape().getDimensions());
    std::cout << '\n';

    std::cout << "Position embedding shape: ";
    printList(positionVectors.getShape().getDimensions());
    std::cout << '\n';

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Token vectors:\n";
    printMatrix(tokenVectors);
    std::cout << "Position vectors:\n";
    printMatrix(positionVectors);
    std::cout << "Transformer input:\n";
    printMatrix(transformerInput);
    std::cout << std::defaultfloat;

    std::cout << "Same token id 1 appears at positions 1 and 3\n";
    std::cout << "After adding positions, these rows are no longer identical\n";
}

void runLinearLayerDemo() {
    const tfs::TokenEmbedding tokenEmbedding(
        6,
        3,
        {
            0.00f, 0.01f, 0.02f,
            0.10f, 0.11f, 0.12f,
            0.20f, 0.21f, 0.22f,
            0.30f, 0.31f, 0.32f,
            0.40f, 0.41f, 0.42f,
            0.50f, 0.51f, 0.52f
        }
    );
    const tfs::PositionEmbedding positionEmbedding(
        4,
        3,
        {
            0.00f, 1.00f, 2.00f,
            0.01f, 1.01f, 2.01f,
            0.02f, 1.02f, 2.02f,
            0.03f, 1.03f, 2.03f
        }
    );
    const tfs::LinearLayer projection(
        3,
        2,
        {
            0.50f, -0.25f,
            1.00f, 0.00f,
            -0.50f, 0.75f
        },
        {0.10f, -0.20f}
    );
    const std::vector<tfs::TokenId> tokens = {3, 1, 4, 1};
    const tfs::Tensor tokenVectors = tokenEmbedding.embed(tokens);
    const tfs::Tensor transformerInput = positionEmbedding.addTo(tokenVectors);
    const tfs::Tensor projected = projection.forward(transformerInput);

    std::cout << "Input shape: ";
    printList(transformerInput.getShape().getDimensions());
    std::cout << '\n';

    std::cout << "Weight shape: ";
    printList(projection.getWeights().getShape().getDimensions());
    std::cout << '\n';

    std::cout << "Bias shape: ";
    printList(projection.getBias().getShape().getDimensions());
    std::cout << '\n';

    std::cout << "Output shape: ";
    printList(projected.getShape().getDimensions());
    std::cout << '\n';

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Transformer input:\n";
    printMatrix(transformerInput);
    std::cout << "Projected vectors:\n";
    printMatrix(projected);
    std::cout << std::defaultfloat;

    std::cout << "Each token row used the same weights and bias\n";
    std::cout << "Later we will use linear layers for query, key, value and logits\n";
}

void runAttentionProjectionDemo() {
    const tfs::TokenEmbedding tokenEmbedding(
        6,
        3,
        {
            0.00f, 0.01f, 0.02f,
            0.10f, 0.11f, 0.12f,
            0.20f, 0.21f, 0.22f,
            0.30f, 0.31f, 0.32f,
            0.40f, 0.41f, 0.42f,
            0.50f, 0.51f, 0.52f
        }
    );
    const tfs::PositionEmbedding positionEmbedding(
        4,
        3,
        {
            0.00f, 1.00f, 2.00f,
            0.01f, 1.01f, 2.01f,
            0.02f, 1.02f, 2.02f,
            0.03f, 1.03f, 2.03f
        }
    );
    const tfs::AttentionProjections projections(
        3,
        2,
        {
            0.50f, -0.25f,
            1.00f, 0.00f,
            -0.50f, 0.75f
        },
        {0.10f, -0.20f},
        {
            0.25f, 0.50f,
            -0.50f, 0.25f,
            1.00f, -0.75f
        },
        {0.00f, 0.10f},
        {
            1.00f, 0.00f,
            0.00f, 1.00f,
            0.50f, 0.50f
        },
        {-0.10f, 0.20f}
    );
    const std::vector<tfs::TokenId> tokens = {3, 1, 4, 1};
    const tfs::Tensor tokenVectors = tokenEmbedding.embed(tokens);
    const tfs::Tensor transformerInput = positionEmbedding.addTo(tokenVectors);
    const tfs::AttentionProjectionResult result = projections.forward(transformerInput);

    std::cout << "Input shape: ";
    printList(transformerInput.getShape().getDimensions());
    std::cout << '\n';

    std::cout << "Projection size: " << projections.projectionSize() << '\n';

    std::cout << "Query shape: ";
    printList(result.queries.getShape().getDimensions());
    std::cout << '\n';

    std::cout << "Key shape: ";
    printList(result.keys.getShape().getDimensions());
    std::cout << '\n';

    std::cout << "Value shape: ";
    printList(result.values.getShape().getDimensions());
    std::cout << '\n';

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Queries:\n";
    printMatrix(result.queries);
    std::cout << "Keys:\n";
    printMatrix(result.keys);
    std::cout << "Values:\n";
    printMatrix(result.values);
    std::cout << std::defaultfloat;

    std::cout << "All three tensors came from the same input\n";
    std::cout << "Each projection has its own weights and bias\n";
}

void runAttentionScoreDemo() {
    const tfs::TokenEmbedding tokenEmbedding(
        6,
        3,
        {
            0.00f, 0.01f, 0.02f,
            0.10f, 0.11f, 0.12f,
            0.20f, 0.21f, 0.22f,
            0.30f, 0.31f, 0.32f,
            0.40f, 0.41f, 0.42f,
            0.50f, 0.51f, 0.52f
        }
    );
    const tfs::PositionEmbedding positionEmbedding(
        4,
        3,
        {
            0.00f, 1.00f, 2.00f,
            0.01f, 1.01f, 2.01f,
            0.02f, 1.02f, 2.02f,
            0.03f, 1.03f, 2.03f
        }
    );
    const tfs::AttentionProjections projections(
        3,
        2,
        {
            0.50f, -0.25f,
            1.00f, 0.00f,
            -0.50f, 0.75f
        },
        {0.10f, -0.20f},
        {
            0.25f, 0.50f,
            -0.50f, 0.25f,
            1.00f, -0.75f
        },
        {0.00f, 0.10f},
        {
            1.00f, 0.00f,
            0.00f, 1.00f,
            0.50f, 0.50f
        },
        {-0.10f, 0.20f}
    );
    const std::vector<tfs::TokenId> tokens = {3, 1, 4, 1};
    const tfs::Tensor tokenVectors = tokenEmbedding.embed(tokens);
    const tfs::Tensor transformerInput = positionEmbedding.addTo(tokenVectors);
    const tfs::AttentionProjectionResult projectionsResult = projections.forward(transformerInput);
    const tfs::Tensor transposedKeys = tfs::transposeMatrix(projectionsResult.keys);
    const tfs::Tensor scores = tfs::attentionScores(projectionsResult.queries, projectionsResult.keys);
    const tfs::TensorValue scale =
        1.0f / std::sqrt(static_cast<tfs::TensorValue>(projections.projectionSize()));

    std::cout << "Query shape: ";
    printList(projectionsResult.queries.getShape().getDimensions());
    std::cout << '\n';

    std::cout << "Key shape: ";
    printList(projectionsResult.keys.getShape().getDimensions());
    std::cout << '\n';

    std::cout << "Transposed key shape: ";
    printList(transposedKeys.getShape().getDimensions());
    std::cout << '\n';

    std::cout << "Score shape: ";
    printList(scores.getShape().getDimensions());
    std::cout << '\n';

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Scale factor: " << scale << '\n';
    std::cout << "Queries:\n";
    printMatrix(projectionsResult.queries);
    std::cout << "Keys transposed:\n";
    printMatrix(transposedKeys);
    std::cout << "Attention scores:\n";
    printMatrix(scores);
    std::cout << std::defaultfloat;

    std::cout << "Rows are querying tokens\n";
    std::cout << "Columns are key tokens\n";
    std::cout << "Masking and softmax come next\n";
}

void printRowSums(const tfs::Tensor& matrix) {
    const auto& shape = matrix.getShape();
    const auto& strides = matrix.getStrides();
    const tfs::TensorValue* const values = matrix.data();

    const std::size_t rows = shape[0];
    const std::size_t columns = shape[1];
    const std::size_t rowStride = strides[0];
    const std::size_t columnStride = strides[1];

    std::cout << "Row sums:";
    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t rowOffset = row * rowStride;
        tfs::TensorValue sum = 0.0f;

        for (std::size_t column = 0; column < columns; ++column) {
            sum += values[rowOffset + column * columnStride];
        }

        std::cout << ' ' << sum;
    }
    std::cout << '\n';
}

void runAttentionWeightDemo() {
    const tfs::TokenEmbedding tokenEmbedding(
        6,
        3,
        {
            0.00f, 0.01f, 0.02f,
            0.10f, 0.11f, 0.12f,
            0.20f, 0.21f, 0.22f,
            0.30f, 0.31f, 0.32f,
            0.40f, 0.41f, 0.42f,
            0.50f, 0.51f, 0.52f
        }
    );
    const tfs::PositionEmbedding positionEmbedding(
        4,
        3,
        {
            0.00f, 1.00f, 2.00f,
            0.01f, 1.01f, 2.01f,
            0.02f, 1.02f, 2.02f,
            0.03f, 1.03f, 2.03f
        }
    );
    const tfs::AttentionProjections projections(
        3,
        2,
        {
            0.50f, -0.25f,
            1.00f, 0.00f,
            -0.50f, 0.75f
        },
        {0.10f, -0.20f},
        {
            0.25f, 0.50f,
            -0.50f, 0.25f,
            1.00f, -0.75f
        },
        {0.00f, 0.10f},
        {
            1.00f, 0.00f,
            0.00f, 1.00f,
            0.50f, 0.50f
        },
        {-0.10f, 0.20f}
    );
    const std::vector<tfs::TokenId> tokens = {3, 1, 4, 1};
    const tfs::Tensor tokenVectors = tokenEmbedding.embed(tokens);
    const tfs::Tensor transformerInput = positionEmbedding.addTo(tokenVectors);
    const tfs::AttentionProjectionResult projectionsResult = projections.forward(transformerInput);
    const tfs::Tensor scores = tfs::attentionScores(projectionsResult.queries, projectionsResult.keys);
    const tfs::Tensor maskedScores = tfs::applyCausalMask(scores);
    const tfs::Tensor weights = tfs::attentionWeights(scores);

    std::cout << "Score shape: ";
    printList(scores.getShape().getDimensions());
    std::cout << '\n';

    std::cout << "Weight shape: ";
    printList(weights.getShape().getDimensions());
    std::cout << '\n';

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Attention scores:\n";
    printMatrix(scores);
    std::cout << "Masked scores:\n";
    printMatrix(maskedScores);
    std::cout << "Attention weights:\n";
    printMatrix(weights);
    printRowSums(weights);
    std::cout << std::defaultfloat;

    std::cout << "Future positions have weight 0 after masking\n";
    std::cout << "Each row is a probability distribution\n";
    std::cout << "Values are mixed in the next lesson\n";
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
    const std::string& sampleText,
    const bool useIndexedQueue
) {
    const tfs::BpeTrainer trainer;
    const auto result = useIndexedQueue
        ? trainer.trainFromCorpusWithQueue(path, maxLines, mergeCount)
        : trainer.trainFromCorpus(path, maxLines, mergeCount);
    const auto& model = result.model;
    const auto& merges = model.getMerges();

    std::cout << "Corpus: " << path << '\n';
    if (useIndexedQueue) {
        std::cout << "Trainer: indexed priority queue\n";
    }
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

void runBpeCompareDemo(
    const std::string& path,
    const std::uint64_t maxLines,
    const std::size_t mergeCount
) {
    const tfs::BpeTrainer trainer;

    tfs::BpeTrainingResult naiveResult;
    const double naiveMilliseconds = measureMilliseconds([&]() {
        naiveResult = trainer.trainFromCorpus(path, maxLines, mergeCount);
    });

    tfs::BpeTrainingResult queueResult;
    const double queueMilliseconds = measureMilliseconds([&]() {
        queueResult = trainer.trainFromCorpusWithQueue(path, maxLines, mergeCount);
    });

    const auto& naiveMerges = naiveResult.model.getMerges();
    const auto& queueMerges = queueResult.model.getMerges();
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

            runBpeDemo(argv[2], maxLines, mergeCount, sampleText, false);
            return 0;
        }

        if (mode == "--bpe-fast") {
            if (argc < 3 || argc > 6) {
                printUsage(argv[0]);
                return 1;
            }

            const std::uint64_t maxLines = argc >= 4 ? parseU64(argv[3]) : 1000;
            const std::size_t mergeCount = argc >= 5 ? static_cast<std::size_t>(parseU64(argv[4])) : 32;
            const std::string sampleText = argc >= 6 ? argv[5] : "the transformer learns from text";

            runBpeDemo(argv[2], maxLines, mergeCount, sampleText, true);
            return 0;
        }

        if (mode == "--bpe-compare") {
            if (argc < 3 || argc > 5) {
                printUsage(argv[0]);
                return 1;
            }

            const std::uint64_t maxLines = argc >= 4 ? parseU64(argv[3]) : 2000;
            const std::size_t mergeCount = argc >= 5 ? static_cast<std::size_t>(parseU64(argv[4])) : 128;

            runBpeCompareDemo(argv[2], maxLines, mergeCount);
            return 0;
        }

        if (mode == "--tensor") {
            if (argc != 2) {
                printUsage(argv[0]);
                return 1;
            }

            runTensorDemo();
            return 0;
        }

        if (mode == "--matmul") {
            if (argc != 2) {
                printUsage(argv[0]);
                return 1;
            }

            runMatrixMultiplyDemo();
            return 0;
        }

        if (mode == "--embedding") {
            if (argc != 2) {
                printUsage(argv[0]);
                return 1;
            }

            runEmbeddingDemo();
            return 0;
        }

        if (mode == "--position") {
            if (argc != 2) {
                printUsage(argv[0]);
                return 1;
            }

            runPositionEmbeddingDemo();
            return 0;
        }

        if (mode == "--linear") {
            if (argc != 2) {
                printUsage(argv[0]);
                return 1;
            }

            runLinearLayerDemo();
            return 0;
        }

        if (mode == "--qkv") {
            if (argc != 2) {
                printUsage(argv[0]);
                return 1;
            }

            runAttentionProjectionDemo();
            return 0;
        }

        if (mode == "--attention-scores") {
            if (argc != 2) {
                printUsage(argv[0]);
                return 1;
            }

            runAttentionScoreDemo();
            return 0;
        }

        if (mode == "--attention-weights") {
            if (argc != 2) {
                printUsage(argv[0]);
                return 1;
            }

            runAttentionWeightDemo();
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
