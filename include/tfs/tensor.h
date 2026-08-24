#pragma once

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tfs {

using TensorValue = float;

class TensorShape {
public:
    TensorShape() = default;

    TensorShape(const std::initializer_list<std::size_t> dimensions)
        : dimensions(dimensions) {
        validate();
    }

    explicit TensorShape(std::vector<std::size_t> dimensions)
        : dimensions(std::move(dimensions)) {
        validate();
    }

    std::size_t rank() const {
        return dimensions.size();
    }

    std::size_t operator[](const std::size_t axis) const {
        return dimensions.at(axis);
    }

    const std::vector<std::size_t>& getDimensions() const {
        return dimensions;
    }

    std::size_t elementCount() const {
        if (dimensions.empty()) {
            return 0;
        }

        std::size_t count = 1;
        for (const std::size_t dimension : dimensions) {
            count *= dimension;
        }

        return count;
    }

    bool operator==(const TensorShape& other) const {
        return dimensions == other.dimensions;
    }

    bool operator!=(const TensorShape& other) const {
        return !(*this == other);
    }

private:
    void validate() const {
        for (const std::size_t dimension : dimensions) {
            if (dimension == 0) {
                throw std::runtime_error("Tensor dimensions must be greater than zero");
            }
        }
    }

    std::vector<std::size_t> dimensions;
};

class Tensor {
public:
    Tensor() = default;

    explicit Tensor(const TensorShape& shape)
        : shape(shape),
          strides(makeStrides(shape)),
          values(shape.elementCount(), 0.0f) {
    }

    Tensor(const TensorShape& shape, const TensorValue fillValue)
        : shape(shape),
          strides(makeStrides(shape)),
          values(shape.elementCount(), fillValue) {
    }

    Tensor(const TensorShape& shape, std::vector<TensorValue> values)
        : shape(shape),
          strides(makeStrides(shape)),
          values(std::move(values)) {
        if (this->values.size() != shape.elementCount()) {
            throw std::runtime_error("Tensor value count does not match shape");
        }
    }

    static Tensor zeros(const TensorShape& shape) {
        return Tensor(shape);
    }

    static Tensor filled(const TensorShape& shape, const TensorValue value) {
        return Tensor(shape, value);
    }

    static Tensor fromValues(const TensorShape& shape, std::vector<TensorValue> values) {
        return Tensor(shape, std::move(values));
    }

    std::size_t rank() const {
        return shape.rank();
    }

    std::size_t size() const {
        return values.size();
    }

    const TensorShape& getShape() const {
        return shape;
    }

    const std::vector<std::size_t>& getStrides() const {
        return strides;
    }

    const std::vector<TensorValue>& getValues() const {
        return values;
    }

    std::vector<TensorValue>& getValues() {
        return values;
    }

    TensorValue* data() {
        return values.data();
    }

    const TensorValue* data() const {
        return values.data();
    }

    TensorValue& operator[](const std::size_t flatIndex) {
        return values.at(flatIndex);
    }

    TensorValue operator[](const std::size_t flatIndex) const {
        return values.at(flatIndex);
    }

    TensorValue& at(const std::vector<std::size_t>& indices) {
        return values.at(offset(indices));
    }

    TensorValue at(const std::vector<std::size_t>& indices) const {
        return values.at(offset(indices));
    }

    std::size_t offset(const std::vector<std::size_t>& indices) const {
        if (indices.size() != shape.rank()) {
            throw std::runtime_error("Tensor index rank does not match shape rank");
        }

        std::size_t flatIndex = 0;
        for (std::size_t axis = 0; axis < indices.size(); ++axis) {
            if (indices[axis] >= shape[axis]) {
                throw std::runtime_error("Tensor index is out of bounds");
            }

            flatIndex += indices[axis] * strides[axis];
        }

        return flatIndex;
    }

    void fill(const TensorValue value) {
        std::fill(values.begin(), values.end(), value);
    }

private:
    static std::vector<std::size_t> makeStrides(const TensorShape& shape) {
        std::vector<std::size_t> strides(shape.rank(), 1);

        for (std::size_t axis = shape.rank(); axis > 1; --axis) {
            strides[axis - 2] = strides[axis - 1] * shape[axis - 1];
        }

        return strides;
    }

    TensorShape shape;
    std::vector<std::size_t> strides;
    std::vector<TensorValue> values;
};

inline void requireSameShape(const Tensor& left, const Tensor& right) {
    if (left.getShape() != right.getShape()) {
        throw std::runtime_error("Tensor shapes do not match");
    }
}

inline Tensor add(const Tensor& left, const Tensor& right) {
    requireSameShape(left, right);

    Tensor result(left.getShape());
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = left[i] + right[i];
    }

    return result;
}

inline Tensor multiply(const Tensor& tensor, const TensorValue factor) {
    Tensor result(tensor.getShape());
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = tensor[i] * factor;
    }

    return result;
}

} // namespace tfs
