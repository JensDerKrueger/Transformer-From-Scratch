#pragma once

#include "tfs/tensor.h"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace tfs {

inline void requireMatrix(const Tensor& tensor, const char* const name) {
    if (tensor.rank() != 2) {
        throw std::runtime_error(std::string(name) + " must be a matrix");
    }
}

inline Tensor matrixMultiply(const Tensor& left, const Tensor& right) {
    requireMatrix(left, "left");
    requireMatrix(right, "right");

    const TensorShape& leftShape = left.getShape();
    const TensorShape& rightShape = right.getShape();

    const std::size_t rows = leftShape[0];
    const std::size_t inner = leftShape[1];
    const std::size_t rightRows = rightShape[0];
    const std::size_t columns = rightShape[1];

    if (inner != rightRows) {
        throw std::runtime_error("matrix dimensions do not match");
    }

    Tensor result(TensorShape({rows, columns}));

    const std::vector<std::size_t>& leftStrides = left.getStrides();
    const std::vector<std::size_t>& rightStrides = right.getStrides();
    const std::vector<std::size_t>& resultStrides = result.getStrides();

    const TensorValue* const leftData = left.data();
    const TensorValue* const rightData = right.data();
    TensorValue* const resultData = result.data();

    const std::size_t leftRowStride = leftStrides[0];
    const std::size_t leftColumnStride = leftStrides[1];
    const std::size_t rightRowStride = rightStrides[0];
    const std::size_t rightColumnStride = rightStrides[1];
    const std::size_t resultRowStride = resultStrides[0];
    const std::size_t resultColumnStride = resultStrides[1];

    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t leftRowOffset = row * leftRowStride;
        const std::size_t resultRowOffset = row * resultRowStride;

        for (std::size_t column = 0; column < columns; ++column) {
            const std::size_t rightColumnOffset = column * rightColumnStride;
            TensorValue sum = 0.0f;

            for (std::size_t innerIndex = 0; innerIndex < inner; ++innerIndex) {
                const TensorValue leftValue = leftData[leftRowOffset + innerIndex * leftColumnStride];
                const TensorValue rightValue = rightData[innerIndex * rightRowStride + rightColumnOffset];
                sum += leftValue * rightValue;
            }

            resultData[resultRowOffset + column * resultColumnStride] = sum;
        }
    }

    return result;
}

inline Tensor transposeMatrix(const Tensor& matrix) {
    requireMatrix(matrix, "matrix");

    const TensorShape& shape = matrix.getShape();
    const std::size_t rows = shape[0];
    const std::size_t columns = shape[1];

    Tensor result(TensorShape({columns, rows}));

    const std::vector<std::size_t>& matrixStrides = matrix.getStrides();
    const std::vector<std::size_t>& resultStrides = result.getStrides();

    const TensorValue* const matrixData = matrix.data();
    TensorValue* const resultData = result.data();

    const std::size_t matrixRowStride = matrixStrides[0];
    const std::size_t matrixColumnStride = matrixStrides[1];
    const std::size_t resultRowStride = resultStrides[0];
    const std::size_t resultColumnStride = resultStrides[1];

    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t matrixRowOffset = row * matrixRowStride;

        for (std::size_t column = 0; column < columns; ++column) {
            resultData[column * resultRowStride + row * resultColumnStride] =
                matrixData[matrixRowOffset + column * matrixColumnStride];
        }
    }

    return result;
}

} // namespace tfs
