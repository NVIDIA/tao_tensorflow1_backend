// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#include "insert_op_helpers.h"

#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

std::vector<std::size_t> ArgSort(const std::vector<std::size_t>& vec) {
    std::vector<std::size_t> result;
    result.reserve(vec.size());

    for (std::size_t i = 0; i < vec.size(); ++i) {
        result.push_back(i);
    }

    auto comparator = [&vec](std::size_t a, std::size_t b) { return vec[a] < vec[b]; };
    std::stable_sort(result.begin(), result.end(), comparator);

    return result;
}

std::vector<std::size_t> ComputeScatterIndices(const std::vector<std::size_t>& indices,
                                               std::size_t input_size) {
    // Compute indices along scatter axis that should contain value constant.
    std::vector<std::size_t> scatter_indices(input_size);
    std::vector<std::size_t> perms = ArgSort(indices);
    std::vector<std::size_t> non_scatter_indices(indices.size(), indices[0]);
    for (std::size_t i = 0; i < indices.size(); ++i) {
        non_scatter_indices[i] = indices[i] + perms[i];
    }

    // Compute indices along scatter axis containing original input tensor.
    std::size_t non_scatter_slot_cursor = 0;
    std::size_t scatter_slot_cursor = 0;
    std::size_t output_size = input_size + indices.size();

    for (std::size_t next_available_slot = 0; next_available_slot < output_size;
         ++next_available_slot) {
        if (non_scatter_slot_cursor < non_scatter_indices.size() &&
            next_available_slot == non_scatter_indices[non_scatter_slot_cursor]) {
            non_scatter_slot_cursor++;
            continue;
        }

        scatter_indices[scatter_slot_cursor++] = next_available_slot;
    }
    return scatter_indices;
}
