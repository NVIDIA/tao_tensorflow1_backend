// Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.
// This file implements common code/class functionality for CPU/GPU versions of
// colortransform maglev op

#ifndef _COLORTRANSFORM_H_
#define _COLORTRANSFORM_H_

#include <algorithm>
#include <string>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

using namespace tensorflow;

#ifndef EIGEN_USE_GPU
float min(float a, float b) { return a < b ? a : b; }
float max(float a, float b) { return a > b ? a : b; }
#endif

#ifdef __CUDACC__
#define CUDA_HOSTDEV __device__
#else
#define CUDA_HOSTDEV
#endif

template <typename I, typename O>
static __inline__ CUDA_HOSTDEV void _ColorTransformKernel(int x, int y, const I* in,
                                                          const float* mat, O* out, float min_clip,
                                                          float max_clip, int height, int width,
                                                          bool input_channels_first,
                                                          bool output_channels_first) {
    // Code common to CPU and GPU kernel

    const int num_channels = 3;

    int input_x_stride, input_y_stride, input_c_stride;
    if (input_channels_first) {
        input_x_stride = 1;
        input_y_stride = width;
        input_c_stride = width * height;
    } else {
        input_x_stride = num_channels;
        input_y_stride = width * num_channels;
        input_c_stride = 1;
    }
    float c[3];
    for (int i = 0; i < num_channels; i++) {
        c[i] = static_cast<float>(in[y * input_y_stride + x * input_x_stride + i * input_c_stride]);
    }

    int output_x_stride, output_y_stride, output_c_stride;
    if (output_channels_first) {
        output_x_stride = 1;
        output_y_stride = width;
        output_c_stride = width * height;
    } else {
        output_x_stride = num_channels;
        output_y_stride = width * num_channels;
        output_c_stride = 1;
    }
    for (int i = 0; i < num_channels; i++) {
        float o = mat[i] * c[0] + mat[i + 4] * c[1] + mat[i + 8] * c[2] + mat[i + 12];
        o = min(max(o, min_clip), max_clip);
        out[y * output_y_stride + x * output_x_stride + i * output_c_stride] = static_cast<O>(o);
    }
}

class BaseColorTransformOp : public OpKernel {
 protected:
    bool verbose_;
    float min_clip_;
    float max_clip_;
    TensorFormat input_data_format_;
    TensorFormat output_data_format_;

 public:
    explicit BaseColorTransformOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("verbose", &verbose_));
        OP_REQUIRES_OK(context, context->GetAttr("min_clip", &min_clip_));
        OP_REQUIRES_OK(context, context->GetAttr("max_clip", &max_clip_));

        std::string input_data_format, output_data_format;
        OP_REQUIRES_OK(context, context->GetAttr("input_data_format", &input_data_format));
        OP_REQUIRES_OK(context, context->GetAttr("output_data_format", &output_data_format));
        OP_REQUIRES(context, FormatFromString(input_data_format, &input_data_format_),
                    errors::InvalidArgument("Invalid input data format"));
        OP_REQUIRES(context, FormatFromString(output_data_format, &output_data_format_),
                    errors::InvalidArgument("Invalid output data format"));
    }

    virtual void ComputeArch(OpKernelContext* context, Tensor* output_tensor,
                             const Tensor& input_images_tensor, const float* input_transf_mats,
                             int nbatch, int height, int width, bool input_channels_first,
                             bool output_channels_first) = 0;

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor.
        const Tensor& input_images_tensor = context->input(0);
        OP_REQUIRES(context, input_images_tensor.shape().dims() == 4,
                    errors::InvalidArgument("input_images_tensor shape should be 4D, got ",
                                            input_images_tensor.shape().dims(), " dims."));
        int batch_dim = 0;
        int height_dim = 1;
        int width_dim = 2;
        int channels_dim = 3;
        if (input_data_format_ == FORMAT_NCHW) {
            channels_dim = 1;
            height_dim = 2;
            width_dim = 3;
        }
        int batch_size = input_images_tensor.shape().dim_size(batch_dim);
        int height = input_images_tensor.shape().dim_size(height_dim);
        int width = input_images_tensor.shape().dim_size(width_dim);
        int num_channels = input_images_tensor.shape().dim_size(channels_dim);
        OP_REQUIRES(context, num_channels == 3,
                    errors::InvalidArgument("input images must have 3 channels, shape is ",
                                            input_images_tensor.shape().DebugString(), "."));

        const Tensor& input_transf_mats_tensor = context->input(1);
        OP_REQUIRES(context, input_transf_mats_tensor.shape().dims() == 3,
                    errors::InvalidArgument("input_transf_mats_tensor shape should be 3D, got ",
                                            input_transf_mats_tensor.shape().dims(), " dims."));
        int matrix_batch_size = input_transf_mats_tensor.shape().dim_size(0);
        int matrix_height = input_transf_mats_tensor.shape().dim_size(1);
        int matrix_width = input_transf_mats_tensor.shape().dim_size(2);
        OP_REQUIRES(context, batch_size == matrix_batch_size,
                    errors::InvalidArgument("number of images and matrices must match"));
        OP_REQUIRES(context, 4 == matrix_height, errors::InvalidArgument("matrix must be 4x4"));
        OP_REQUIRES(context, 4 == matrix_width, errors::InvalidArgument("matrix must be 4x4"));

        // Create an output tensor.
        TensorShape output_shape = output_data_format_ == FORMAT_NCHW
                                       ? TensorShape({batch_size, num_channels, height, width})
                                       : TensorShape({batch_size, height, width, num_channels});
        if (verbose_) {
            printf("batch_size = %d, width = %d, height = %d\n", batch_size, width, height);
            for (int i = 0; i < output_shape.dims(); i++) {
                printf("output dim %d size = %lld\n", i, output_shape.dim_size(i));
            }
        }
        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

        // Call derived class's compute here.
        auto input_transf_mats = input_transf_mats_tensor.flat<float>();
        ComputeArch(context, output_tensor, input_images_tensor, input_transf_mats.data(),
                    batch_size, height, width, input_data_format_ == FORMAT_NCHW,
                    output_data_format_ == FORMAT_NCHW);

        if (verbose_) {
            printf("done\n");
        }
    }
};

#endif  // _COLORTRANSFORM_H_
