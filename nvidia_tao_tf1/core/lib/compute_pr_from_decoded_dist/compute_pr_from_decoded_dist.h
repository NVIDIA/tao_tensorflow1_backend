// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

#ifndef _COMPUTE_PR_FROM_DECODED_DIST_H_
#define _COMPUTE_PR_FROM_DECODED_DIST_H_

#include <algorithm>
#include <climits>
#include <cmath>
#include <map>
#include <vector>

#include <cfloat>
#include <fstream>
#include <iostream>
#include <string>

#include "draw_characters.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/logging.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#ifdef __CUDACC__
#define CUDA_HOSTDEV __device__
#else
#define CUDA_HOSTDEV
#endif

constexpr int N_REGIONS = 3;
constexpr int DIM_FOR_PR = 2;

class _ComputePRFromDecodedDist : public OpKernel {
 protected:
    // Number of classes to evalute.
    int n_classes_;
    // Height of input blob.
    int height_;
    // Width of input blob.
    int width_;
    // If to draw metrics on image.
    bool draw_metrics_;
    // Radius used when matching is done.
    int search_radius_;
    // From bottom y coordinate to height_*bottom_ratio_ is considered as close area.
    float bottom_ratio_;
    // From height_*bottom_ratio_  to height_*top_ratio_ is considered as
    // middle area.
    float top_ratio_;
    bool verbose_;

 public:
    explicit _ComputePRFromDecodedDist(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("verbose", &verbose_));

        OP_REQUIRES_OK(context, context->GetAttr("draw_metrics", &draw_metrics_));

        OP_REQUIRES_OK(context, context->GetAttr("width", &width_));
        OP_REQUIRES(context, width_ > 0, errors::InvalidArgument("Need width > 0, got ", width_));
        OP_REQUIRES_OK(context, context->GetAttr("height", &height_));
        OP_REQUIRES(context, height_ > 0,
                    errors::InvalidArgument("Need height > 0, got ", height_));

        OP_REQUIRES_OK(context, context->GetAttr("n_classes", &n_classes_));
        OP_REQUIRES(context, n_classes_ > 0,
                    errors::InvalidArgument("Need n_classes > 0, got ", n_classes_));

        OP_REQUIRES_OK(context, context->GetAttr("search_radius", &search_radius_));
        OP_REQUIRES(context, search_radius_ > 0,
                    errors::InvalidArgument("Need search_radius > 0, got ", search_radius_));

        OP_REQUIRES_OK(context, context->GetAttr("bottom_ratio", &bottom_ratio_));
        OP_REQUIRES(context, bottom_ratio_ >= 0 && bottom_ratio_ <= 1,
                    errors::InvalidArgument("Need bottom_ratio_ >= 0 and bottom_ratio_<=1, got ",
                                            bottom_ratio_));

        OP_REQUIRES_OK(context, context->GetAttr("top_ratio", &top_ratio_));
        OP_REQUIRES(
            context, top_ratio_ >= 0 && top_ratio_ <= 1,
            errors::InvalidArgument("Need top_ratio_ >= 0 and top_ratio_<=1, got ", top_ratio_));

        OP_REQUIRES(context, bottom_ratio_ >= top_ratio_,
                    errors::InvalidArgument("Need bottom_ratio_ >= top_ratio_, got ", bottom_ratio_,
                                            " and ", top_ratio_));
    }

    void Compute(OpKernelContext* context) override { Preprocess(context); }

    virtual void evaluate(OpKernelContext* context, const int batch_size,
                          const int* tensor_GT_blobs, const int* tensor_pred_blobs,
                          const int* input_image, float* binary_pr, float* binary_pr_per_distance,
                          float* multiclass_pr, float* multiclass_pr_per_distance,
                          int* input_image_with_stat) = 0;

    // image > polygon > vertex
    void Preprocess(OpKernelContext* context) {
        //
        // Grab the input tensor.
        //
        const Tensor& tensor_gt = context->input(0);
        int batch_size = tensor_gt.shape().dim_size(0);

        const Tensor& tensor_pred = context->input(1);

        const Tensor& tensor_input_image = context->input(2);

        OP_REQUIRES(context, batch_size == tensor_pred.shape().dim_size(0),
                    errors::InvalidArgument("Need batch size equal but got ", batch_size,
                                            tensor_pred.shape().dim_size(0)));

        OP_REQUIRES(context, tensor_gt.shape().dim_size(2) == height_,
                    errors::InvalidArgument("tensor_gt.shape().dim_size(2) == height_, got ",
                                            tensor_gt.shape().dim_size(2), " and  ", height_));
        OP_REQUIRES(context, tensor_gt.shape().dim_size(3) == width_,
                    errors::InvalidArgument("tensor_gt.shape().dim_size(3) == width_, got ",
                                            tensor_gt.shape().dim_size(3), " and  ", width_));

        OP_REQUIRES(context, tensor_pred.shape().dim_size(2) == height_,
                    errors::InvalidArgument("tensor_pred.shape().dim_size(2) == height_, got ",
                                            tensor_pred.shape().dim_size(2), " and  ", height_));
        OP_REQUIRES(context, tensor_pred.shape().dim_size(3) == width_,
                    errors::InvalidArgument("tensor_pred.shape().dim_size(3) == width_, got ",
                                            tensor_pred.shape().dim_size(3), " and  ", width_));

        auto label = tensor_gt.flat<int>();
        auto pred = tensor_pred.flat<int>();
        auto input_image = tensor_input_image.flat<int>();

        Tensor* output_tensor = NULL;

        // Create an output tensor.
        TensorShape output_shape1({batch_size, 1, 1, DIM_FOR_PR});
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape1, &output_tensor));
        auto binary_pr = output_tensor->template flat<float>();

        TensorShape output_shape2({batch_size, N_REGIONS, 1, DIM_FOR_PR});
        OP_REQUIRES_OK(context, context->allocate_output(1, output_shape2, &output_tensor));
        auto binary_pr_per_distance = output_tensor->template flat<float>();

        TensorShape output_shape3({batch_size, 1, n_classes_, DIM_FOR_PR});
        OP_REQUIRES_OK(context, context->allocate_output(2, output_shape3, &output_tensor));
        auto multiclass_pr = output_tensor->template flat<float>();

        TensorShape output_shape4({batch_size, N_REGIONS, n_classes_, DIM_FOR_PR});
        OP_REQUIRES_OK(context, context->allocate_output(3, output_shape4, &output_tensor));
        auto multiclass_pr_per_distance = output_tensor->template flat<float>();

        TensorShape output_shape5({batch_size, height_, width_, 3});
        OP_REQUIRES_OK(context, context->allocate_output(4, output_shape5, &output_tensor));
        auto input_image_with_stat = output_tensor->template flat<int>();

        evaluate(context, batch_size, label.data(), pred.data(), input_image.data(),
                 binary_pr.data(), binary_pr_per_distance.data(), multiclass_pr.data(),
                 multiclass_pr_per_distance.data(), input_image_with_stat.data());
    }
};

#else
#endif  // _COMPUTE_PR_FROM_DECODED_DIST_H_
