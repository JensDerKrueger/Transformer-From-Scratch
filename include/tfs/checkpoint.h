#pragma once

#include "tfs/parameter.h"

#include <cstddef>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

namespace tfs {

inline void saveParameters(const std::string& path, const std::vector<Parameter*>& parameters) {
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("Could not open checkpoint for writing: " + path);
    }

    file << "TFS_PARAMETERS_V1\n";
    file << parameters.size() << '\n';
    file << std::setprecision(9);

    for (const Parameter* const parameter : parameters) {
        const Tensor& value = parameter->getValue();
        const std::vector<std::size_t>& dimensions = value.getShape().getDimensions();

        file << dimensions.size();
        for (const std::size_t dimension : dimensions) {
            file << ' ' << dimension;
        }
        file << '\n';

        file << value.size();
        for (const TensorValue item : value.getValues()) {
            file << ' ' << item;
        }
        file << '\n';
    }
}

inline void loadParameters(const std::string& path, const std::vector<Parameter*>& parameters) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Could not open checkpoint for reading: " + path);
    }

    std::string magic;
    std::size_t parameterCount = 0;
    file >> magic >> parameterCount;

    if (magic != "TFS_PARAMETERS_V1" || parameterCount != parameters.size()) {
        throw std::runtime_error("Checkpoint does not match parameter list");
    }

    for (Parameter* const parameter : parameters) {
        Tensor& value = parameter->getValue();
        const std::vector<std::size_t>& expectedDimensions = value.getShape().getDimensions();

        std::size_t rank = 0;
        file >> rank;
        if (rank != expectedDimensions.size()) {
            throw std::runtime_error("Checkpoint tensor rank mismatch");
        }

        for (const std::size_t expectedDimension : expectedDimensions) {
            std::size_t dimension = 0;
            file >> dimension;
            if (dimension != expectedDimension) {
                throw std::runtime_error("Checkpoint tensor shape mismatch");
            }
        }

        std::size_t valueCount = 0;
        file >> valueCount;
        if (valueCount != value.size()) {
            throw std::runtime_error("Checkpoint tensor value count mismatch");
        }

        for (TensorValue& item : value.getValues()) {
            file >> item;
        }
    }
}

} // namespace tfs
