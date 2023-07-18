// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
// This header file implements common functionality of rasterize_polygon maglev op

#ifndef PUBLIC_MAGLEV_SDK_LIB_SRC_BINARY_TO_DISTANCE_BINARY_TO_DISTANCE_H_
#define PUBLIC_MAGLEV_SDK_LIB_SRC_BINARY_TO_DISTANCE_BINARY_TO_DISTANCE_H_

#include <float.h>
#include <math.h>

#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/logging.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#ifdef __CUDACC__
#define CUDA_HOSTDEV __device__
#else
#define CUDA_HOSTDEV
#endif

/**
 * compute distance to left and right.
 * also normalization mask is computed.
 * @param output_distance: compuated output distance mask.
 * @param output_Z: compuated output normalization mask.
 * @param input: given binary input.
 * @param width: input width.
 * @param y: y th row to be processed on.
 * @param to_left: if we are computing distance to left mask,
                   if not compuate distance to right mask.
 * @param defined_inf: threashold at we are defining the infinity distance.
 */
static __inline__ CUDA_HOSTDEV void _BinaryToDistanceHorizontalKernel(
    float* output_distance, float* output_Z, const float* input, const int width, const int y,
    const bool to_left, const float defined_inf) {
    // Code common to CPU and GPU kernel
    // first pass
    for (int x = 0; x < width; x++) {
        const int index = y * width + x;
        if (input[index] > 0) {
            output_distance[index] = 0;
        } else {
            output_distance[index] = defined_inf;
        }
    }
    // second pass
    if (to_left) {
        for (int x = 0; x < width; x++) {
            const int index = y * width + x;
            output_distance[index] =
                MIN(output_distance[index], output_distance[y * width + MAX(0, x - 1)] + 1);
        }
        // To get normalization Z.....we do backward scan....
        float Z = 0;
        float prev_Z = -1.0;
        int zero_index = -1;
        for (int x = 0; x < width; x++) {
            const int index = y * width + width - 1 - x;
            float dist = output_distance[index];
            if (dist > FLT_MIN) {  // we can keep update
                if (dist > Z) {
                    Z = dist;
                }
                if (zero_index > 0) {
                    output_Z[zero_index] = prev_Z;
                    zero_index = -1;
                }
            } else {
                if (dist == 0) {
                    // consecutive zeros and boundary zeros
                    if (zero_index > 0 && x < width - 1) {
                        output_Z[zero_index] = prev_Z;
                    }
                    zero_index = index;
                    prev_Z = Z;
                    if (x == width - 1) {
                        output_Z[index] = Z;
                        continue;
                    }
                }
                Z = 0;
            }
            output_Z[index] = Z;
        }

    } else {
        for (int x = 0; x < width; x++) {
            const int index = y * width + width - 1 - x;
            output_distance[index] = MIN(
                output_distance[index], output_distance[y * width + MIN(width - 1, width - x)] + 1);
        }
        // To get normalization Z.....we do backward scan....
        float Z = 0;
        float prev_Z = -1.0;
        int zero_index = -1;
        for (int x = 0; x < width; x++) {
            const int index = y * width + x;
            float dist = output_distance[index];
            if (dist > FLT_MIN) {  // we can keep update
                if (dist > Z) {
                    Z = dist;
                }
                if (zero_index > 0) {
                    output_Z[zero_index] = prev_Z;
                    zero_index = -1;
                }
            } else {
                if (dist == 0) {
                    if (zero_index > 0) {
                        output_Z[zero_index] = prev_Z;
                    }
                    zero_index = index;
                    prev_Z = Z;
                    if (x == width - 1) {
                        output_Z[index] = Z;
                        continue;
                    }
                }
                Z = 0;
            }
            output_Z[index] = Z;
        }
    }
}

static __inline__ CUDA_HOSTDEV void _DistanceToBinaryHorizontalKernel(
    float* output, const float* input, const float* input_mask, const int width, const int y,
    const bool to_left, const float defined_inf, const int scale) {
    for (uint32_t x = 0; x < static_cast<uint32_t>(width); x++) {
        if (input_mask[x] < 0.5) {
            continue;
        }

        int newx;
        if (to_left) {
            newx = static_cast<int>(floor(0.5f + scale * x - input[x]));
        } else {
            newx = static_cast<int>(floor(scale * x + input[x]));
        }

        if (newx >= 0 && newx < width * scale) {
            if (input[x] > 80) {
                continue;
            }
            output[newx]++;
        }
    }
}

/**
 * compute distance to up and down
 * also normalization mask is computed.
 * @param output: compuated output distance mask.
 * @param input: given input.
 * @param height: input height.
 * @param x: x th column computed on.
 * @param to_down: if we are computing distance to down.
                   if not we are computing distance to up.
 * @param defined_inf: threashold at we are defining the infinity distance.
 */
static __inline__ CUDA_HOSTDEV void _BinaryToDistanceVerticalKernel(
    float* output_distance, float* output_Z, const float* input, const int height, const int width,
    const int x, const bool to_down, const float defined_inf) {
    // Code common to CPU and GPU kernel
    // first pass
    for (int y = 0; y < height; y++) {
        const int index = y * width + x;
        if (input[index] > 0) {
            output_distance[index] = 0;
        } else {
            output_distance[index] = defined_inf;
        }
    }
    // second pass
    if (to_down) {
        for (int y = 0; y < height; y++) {
            const int index = y * width + x;
            output_distance[index] =
                MIN(output_distance[index], output_distance[MAX(0, y - 1) * width + x] + 1);
        }

        // To get normalization Z.....we do backward scan....
        float Z = 0;
        float prev_Z = -1.0;
        int zero_index = -1;
        for (int y = 0; y < height; y++) {
            const int index = (height - 1 - y) * width + x;
            float dist = output_distance[index];
            if (dist > FLT_MIN) {  // we can keep update
                if (dist > Z) {
                    Z = dist;
                }
                if (zero_index > 0) {
                    output_Z[zero_index] = prev_Z;
                    zero_index = -1;
                }
            } else {
                if (dist == 0) {
                    if (zero_index > 0 && y < height - 1) {
                        output_Z[zero_index] = prev_Z;
                    }
                    zero_index = index;
                    prev_Z = Z;
                    if (y == height - 1) {
                        output_Z[index] = Z;
                        continue;
                    }
                }
                Z = 0;
            }
            output_Z[index] = Z;
        }
    } else {
        for (int y = 0; y < height; y++) {
            const int index = (height - 1 - y) * width + x;
            output_distance[index] =
                MIN(output_distance[index],
                    output_distance[MIN(height - 1, height - y) * width + x] + 1);
        }
        // To get normalization Z.....we do backward scan....
        float Z = 0;
        float prev_Z = -1.0;
        int zero_index = -1;
        for (int y = 0; y < height; y++) {
            const int index = y * width + x;
            float dist = output_distance[index];
            if (dist > FLT_MIN) {  // we can keep update
                if (dist > Z) {
                    Z = dist;
                }
                if (zero_index > 0) {
                    output_Z[zero_index] = prev_Z;
                    zero_index = -1;
                }
            } else {
                if (dist == 0) {
                    if (zero_index > 0) {
                        output_Z[zero_index] = prev_Z;
                    }
                    zero_index = index;
                    prev_Z = Z;
                    if (y == height - 1) {
                        output_Z[index] = Z;
                        continue;
                    }
                }
                Z = 0;
            }
            output_Z[index] = Z;
        }
    }
}

static __inline__ CUDA_HOSTDEV void _ComputeMaskKernel(float* output, const float* input1,
                                                       const float* input2, const int x,
                                                       const int y, const int width,
                                                       const int height,
                                                       const float distance_threshold) {
    // Code common to CPU and GPU kernel
    const int index = y * width + x;
    output[index] = MIN(input1[index], input2[index]) < distance_threshold ? 1 : 0;
}

class _BinaryToDistanceOp : public OpKernel {
 protected:
    static constexpr float ALPHA = 3;
    static constexpr int DEFAULT_OUTPUT_CHANNEL = 5;
    bool compute_vertical_;
    float distance_threshold_;
    bool verbose_;
    bool inverse_;
    int width_;
    int height_;
    int nbatch_;
    int n_channel_output_;
    int n_channel_input_;
    int scale_w_;
    int scale_h_;
    int target_height_;
    int target_width_;

 public:
    virtual void ComputeArch(OpKernelContext* context, float* output_images, const float* images,
                             const float distance_threshold) = 0;

    explicit _BinaryToDistanceOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("compute_vertical", &compute_vertical_));
        OP_REQUIRES_OK(context, context->GetAttr("distance_threshold", &distance_threshold_));
        OP_REQUIRES_OK(context, context->GetAttr("verbose", &verbose_));
        OP_REQUIRES_OK(context, context->GetAttr("inverse", &inverse_));
        OP_REQUIRES_OK(context, context->GetAttr("target_height", &target_height_));
        OP_REQUIRES_OK(context, context->GetAttr("target_width", &target_width_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& images_tensor = context->input(0);
        OP_REQUIRES(context, 4 == images_tensor.shape().dims(),
                    errors::InvalidArgument("Images tensor must have 4 dimensions, shape is: ",
                                            images_tensor.shape().DebugString(), "."));
        nbatch_ = images_tensor.shape().dim_size(0);
        n_channel_input_ = images_tensor.shape().dim_size(1);
        if (!inverse_) {
            OP_REQUIRES(context, 1 == n_channel_input_,
                        errors::InvalidArgument("Input images must have 1 channel, shape is: ",
                                                images_tensor.shape().DebugString(), "."));
        } else {
            OP_REQUIRES(context, 3 == n_channel_input_,
                        errors::InvalidArgument("Input images must have 3 channel, shape is: ",
                                                images_tensor.shape().DebugString(), "."));
        }

        height_ = images_tensor.shape().dim_size(2);
        width_ = images_tensor.shape().dim_size(3);

        // prepare output dimension
        if (compute_vertical_) {
            n_channel_output_ = DEFAULT_OUTPUT_CHANNEL * 2;
        } else {
            n_channel_output_ = DEFAULT_OUTPUT_CHANNEL;
        }

        if (inverse_) {
            scale_w_ = floor(target_width_ / width_);
            scale_h_ = floor(target_height_ / height_);
            if (target_height_ % height_ != 0 || target_width_ % width_ != 0) {
                scale_h_ = 1;
                scale_w_ = 4;
            }
            target_width_ = scale_w_ * width_;
            target_height_ = scale_h_ * height_;
            n_channel_output_ = 2;
        } else {
            target_height_ = height_;
            target_width_ = width_;
            scale_w_ = 1;
            scale_h_ = 1;
        }

        // Create an output tensor
        TensorShape output_shape({nbatch_, n_channel_output_, target_height_, target_width_});
        if (verbose_ && inverse_) {
            printf("\n input dim: nbatch = %d, nchannel = %d  width = %d, height = %d \n", nbatch_,
                   n_channel_output_, width_, height_);
            printf("   --- target_width=%d, target_height=%d \n", target_width_, target_height_);
            printf("   --- output shape: %s\n", output_shape.DebugString().c_str());
        }
        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

        // Call the Derived class's CPU/GPU specific implementation
        auto images = images_tensor.flat<float>();
        auto output_images = output_tensor->template flat<float>();
        ComputeArch(context, output_images.data(), images.data(), distance_threshold_);

        if (verbose_) {
            printf(" -----  Done  ----- \n");
        }
    }
};
#else
#endif  // PUBLIC_MAGLEV_SDK_LIB_SRC_BINARY_TO_DISTANCE_BINARY_TO_DISTANCE_H_
