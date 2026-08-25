#pragma once

#include "tfs/tensor.h"
#include "tfs/tensor_operations.h"

#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace demo {

inline std::uint64_t parseU64(const char* const text) {
    char* end = nullptr;
    const auto value = std::strtoull(text, &end, 10);

    if (end == text || *end != '\0') {
        throw std::runtime_error(std::string("Expected an unsigned integer, got: ") + text);
    }

    return static_cast<std::uint64_t>(value);
}

inline float parseF32(const char* const text) {
    char* end = nullptr;
    const float value = std::strtof(text, &end);

    if (end == text || *end != '\0') {
        throw std::runtime_error(std::string("Expected a floating point value, got: ") + text);
    }

    return value;
}

inline std::string describeByte(const unsigned char byte) {
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

inline std::string escapeText(const std::string& text) {
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

inline void printMatrix(const tfs::Tensor& matrix) {
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

inline void printRowSums(const tfs::Tensor& matrix) {
    const auto& shape = matrix.getShape();
    const auto& strides = matrix.getStrides();
    const tfs::TensorValue* const values = matrix.data();

    std::cout << "Row sums:";
    for (std::size_t row = 0; row < shape[0]; ++row) {
        const std::size_t rowOffset = row * strides[0];
        tfs::TensorValue sum = 0.0f;

        for (std::size_t column = 0; column < shape[1]; ++column) {
            sum += values[rowOffset + column * strides[1]];
        }

        std::cout << ' ' << sum;
    }
    std::cout << '\n';
}

inline void printRowMeanVariance(const tfs::Tensor& matrix) {
    const auto& shape = matrix.getShape();
    const auto& strides = matrix.getStrides();
    const tfs::TensorValue* const values = matrix.data();

    for (std::size_t row = 0; row < shape[0]; ++row) {
        const std::size_t rowOffset = row * strides[0];

        tfs::TensorValue mean = 0.0f;
        for (std::size_t column = 0; column < shape[1]; ++column) {
            mean += values[rowOffset + column * strides[1]];
        }
        mean /= static_cast<tfs::TensorValue>(shape[1]);

        tfs::TensorValue variance = 0.0f;
        for (std::size_t column = 0; column < shape[1]; ++column) {
            const tfs::TensorValue centered = values[rowOffset + column * strides[1]] - mean;
            variance += centered * centered;
        }
        variance /= static_cast<tfs::TensorValue>(shape[1]);

        std::cout << "  row " << row << ": mean " << mean << ", variance " << variance << '\n';
    }
}

template <typename Function>
double measureMilliseconds(Function function) {
    const auto start = std::chrono::steady_clock::now();
    function();
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

} // namespace demo
