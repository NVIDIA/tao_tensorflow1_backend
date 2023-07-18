// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#ifndef MAGLEV_SDK_LIB_SRC_INSERT_INSERT_OP_HELPERS_H_
#define MAGLEV_SDK_LIB_SRC_INSERT_INSERT_OP_HELPERS_H_

#include <vector>

// Returns vector containing permutation that sorts input vector.
//
// Example: Input [3, 1, 2] returns [1, 2, 0].
std::vector<std::size_t> ArgSort(const std::vector<std::size_t>& vec);

// Computes index mapping when scattering along an input dimension.
//
// Args:
//   indices: List of indices along axis dimension before which we want to
//     insert a constant.
//   input_size: Size of dimension of input tensor along insert axis.
// Returns:
//   List of indices along insert axis of output tensor, where we should
//   scatter the input tensor. Length of vector matches input_size.
//
// Example: Suppose a rank 3 tensor is scattered into a rank 5 tensor as follows
//
// [3, 1, 2] -> [0, 3, 0, 1, 2]
//
// In this case the input tensor is [3, 1, 2] and indices is [0, 1]. The
// function will then return [1, 3, 4].
std::vector<std::size_t> ComputeScatterIndices(const std::vector<std::size_t>& indices,
                                               std::size_t input_size);

#endif  // MAGLEV_SDK_LIB_SRC_INSERT_INSERT_OP_HELPERS_H_
