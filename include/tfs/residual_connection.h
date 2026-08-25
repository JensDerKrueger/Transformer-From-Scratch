#pragma once

#include "tfs/tensor.h"

namespace tfs {

inline Tensor residualAdd(const Tensor& input, const Tensor& branchOutput) {
    return add(input, branchOutput);
}

} // namespace tfs
