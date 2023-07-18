// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

#ifndef CUDA_HELPER_H_
#define CUDA_HELPER_H_

#include <cfloat>
#include <fstream>
#include <iostream>
#include <string>

#include "tensorflow/core/framework/op_kernel.h"

// Macro for checking cuda errors following a cuda launch or api call.
// example: CHECK_CUDA_ERROR(cudaMalloc(....));
#define CHECK_CUDA_ERROR(err)                                                                  \
    {                                                                                          \
        if (err != cudaSuccess) {                                                              \
            printf("Cuda Failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(0);                                                                           \
        }                                                                                      \
    }

// Use after a kernel launched.
// example: CHECK_LAST_CUDA_ERROR();
#define CHECK_LAST_CUDA_ERROR()                                                                \
    {                                                                                          \
        cudaError_t err = cudaGetLastError();                                                  \
        if (err != cudaSuccess) {                                                              \
            printf("Cuda Failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(0);                                                                           \
        }                                                                                      \
    }

#endif /* CUDA_HELPER_H_ */
