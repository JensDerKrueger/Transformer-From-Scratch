#pragma once

#include "tfs/tensor.h"
#include "tfs/tensor_operations.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace tfs {

struct TopKEntry {
    std::size_t index = 0;
    TensorValue value = 0.0f;
};

inline Tensor logSoftmaxRows(const Tensor& logits) {
    requireMatrix(logits, "logits");

    const TensorShape& shape = logits.getShape();
    const std::size_t rows = shape[0];
    const std::size_t columns = shape[1];
    const std::vector<std::size_t>& strides = logits.getStrides();
    const TensorValue* const input = logits.data();

    Tensor result(shape);
    TensorValue* const output = result.data();

    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t rowOffset = row * strides[0];
        TensorValue maxValue = input[rowOffset];

        for (std::size_t column = 1; column < columns; ++column) {
            maxValue = std::max(maxValue, input[rowOffset + column * strides[1]]);
        }

        TensorValue sum = 0.0f;
        for (std::size_t column = 0; column < columns; ++column) {
            sum += std::exp(input[rowOffset + column * strides[1]] - maxValue);
        }

        const TensorValue logSumExp = maxValue + std::log(sum);
        for (std::size_t column = 0; column < columns; ++column) {
            const std::size_t index = rowOffset + column * strides[1];
            output[index] = input[index] - logSumExp;
        }
    }

    return result;
}

inline std::vector<TopKEntry> topKRow(const Tensor& values, const std::size_t row, const std::size_t k) {
    requireMatrix(values, "values");

    const TensorShape& shape = values.getShape();
    if (row >= shape[0]) {
        throw std::runtime_error("topK row is out of bounds");
    }

    std::vector<TopKEntry> entries;
    entries.reserve(shape[1]);

    const std::vector<std::size_t>& strides = values.getStrides();
    const TensorValue* const data = values.data();
    const std::size_t rowOffset = row * strides[0];

    for (std::size_t column = 0; column < shape[1]; ++column) {
        entries.push_back(TopKEntry{column, data[rowOffset + column * strides[1]]});
    }

    const std::size_t limitedK = std::min(k, entries.size());
    std::partial_sort(
        entries.begin(),
        entries.begin() + static_cast<std::ptrdiff_t>(limitedK),
        entries.end(),
        [](const TopKEntry& left, const TopKEntry& right) {
            return left.value > right.value;
        }
    );
    entries.resize(limitedK);

    return entries;
}

} // namespace tfs
