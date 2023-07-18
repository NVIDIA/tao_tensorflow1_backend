// Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.
// This header implements common code for CPU/GPU versions of
// maglev's spatialtransform_op

#ifndef _SPATIALTRANSFORM_H_
#define _SPATIALTRANSFORM_H_

#include <string>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

using namespace tensorflow;

#ifdef __CUDACC__
#define CUDA_HOSTDEV __device__
#else
#define CUDA_HOSTDEV
#endif

enum FilterMode { FILTER_MODE_NEAREST = 0, FILTER_MODE_BILINEAR, FILTER_MODE_BICUBIC };

template <typename I, typename O>
static __inline__ CUDA_HOSTDEV void _SpatialTransformKernel(
    int x, int y, const I* in, const float* mat, O* out, int num_channels, int height, int width,
    int output_height, int output_width, FilterMode filter_mode, float background,
    bool input_channels_first, bool output_channels_first) {
    // Code common to CPU and GPU kernel.

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
    int output_x_stride, output_y_stride, output_c_stride;
    if (output_channels_first) {
        output_x_stride = 1;
        output_y_stride = output_width;
        output_c_stride = output_width * output_height;
    } else {
        output_x_stride = num_channels;
        output_y_stride = output_width * num_channels;
        output_c_stride = 1;
    }

    // Add 0.5 to output pixel coordinates to move the sampling point (=pixel center) to
    // half integers. This matches OpenGL rasterization rules (OpenGL 4.5 spec, Chapter 14).
    float fx = static_cast<float>(x) + 0.5f;
    float fy = static_cast<float>(y) + 0.5f;
    // Transform output pixel coordinate into input image space. Note that this allows for
    // perspective warps by transforming from 2D output image plane into 3D, followed by
    // projection back to 2D.
    // See: https://github.com/ssloy/tinyrenderer/wiki/Lesson-4:-Perspective-projection#..
    // ..wait-a-minute-may-i-touch-this-magical-bottom-row-of-the-3x3-matrix
    float ifx = mat[0] * fx + mat[3] * fy + mat[6];
    float ify = mat[1] * fx + mat[4] * fy + mat[7];
    float ifz = mat[2] * fx + mat[5] * fy + mat[8];
    // Project to 2D input image plane.
    ifx /= ifz;
    ify /= ifz;
    // Subtract pixel center. This is needed to avoid half a pixel shift so that identity
    // transform produces an identical image with bilinear sampling mode. This is not done
    // for nearest filter mode as we're flooring the coordinate. This matches OpenGL
    // texture filtering rules (OpenGL 4.5 spec, section 8.14 Texture Minification).
    if (filter_mode != FILTER_MODE_NEAREST) {
        ifx -= 0.5f;
        ify -= 0.5f;
    }

    // Compute integer input image coordinates by flooring.
    int ix = static_cast<int>(floor(ifx));
    int iy = static_cast<int>(floor(ify));
    // Compute distance from the exact sampling point to the integer coordinates. This is
    // used for weighting the adjacent samples to produce a filtered result.
    float bx = ifx - static_cast<float>(ix);
    float by = ify - static_cast<float>(iy);
    float ibx = 1.0f - bx;
    float iby = 1.0f - by;

    // Compute filter kernel weights based on sampling mode.
    float wx[4], wy[4];
    int kernel_size;
    switch (filter_mode) {
        case FILTER_MODE_NEAREST:
            // Read 1 pixel at the sampling point.
            wx[0] = 1.0f;
            wy[0] = 1.0f;
            kernel_size = 1;
            break;
        case FILTER_MODE_BILINEAR:
            // Read 2x2 pixels around the sampling point.
            wx[0] = ibx;
            wx[1] = bx;
            wy[0] = iby;
            wy[1] = by;
            kernel_size = 2;
            break;
        case FILTER_MODE_BICUBIC:
        default:
            // Read 4x4 pixels around the sampling point. Note that while in general bicubic
            // gives higher quality upsampling than bilinear, it introduces a slight blur to
            // the image, and thus identity mapping doesn't produce an identical image.
            // Bicubic weights reference http://vec3.ca/bicubic-filtering-in-fewer-taps
            wx[0] = 1.0f / 6.0f * ibx * ibx * ibx;
            wx[1] = 1.0f / 6.0f * (4.0f + 3.0f * bx * bx * bx - 6.0f * bx * bx);
            wx[2] = 1.0f / 6.0f * (4.0f + 3.0f * ibx * ibx * ibx - 6.0f * ibx * ibx);
            wx[3] = 1.0f / 6.0f * bx * bx * bx;

            wy[0] = 1.0f / 6.0f * iby * iby * iby;
            wy[1] = 1.0f / 6.0f * (4.0f + 3.0f * by * by * by - 6.0f * by * by);
            wy[2] = 1.0f / 6.0f * (4.0f + 3.0f * iby * iby * iby - 6.0f * iby * iby);
            wy[3] = 1.0f / 6.0f * by * by * by;

            ix--;
            iy--;
            kernel_size = 4;
            break;
    }

    // Sample and filter.
    for (int c = 0; c < num_channels; c++) {
        float o = 0.0f;
        for (int t = 0; t < kernel_size; t++) {
            int yy = iy + t;
            for (int s = 0; s < kernel_size; s++) {
                int xx = ix + s;
                // Pixels outside the input image are set to background value. Note that this
                // code correctly handles cases where the filter kernel is partially in and
                // partially out of the image.
                float sample = background;
                if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
                    int in_offset = yy * input_y_stride + xx * input_x_stride + c * input_c_stride;
                    sample = static_cast<float>(in[in_offset]);
                }
                o += sample * wx[s] * wy[t];
            }
        }
        int out_offset = y * output_y_stride + x * output_x_stride + c * output_c_stride;
        out[out_offset] = static_cast<O>(o);
    }
}

class BaseSpatialTransformOp : public OpKernel {
 protected:
    FilterMode filter_mode_;
    float background_;
    bool verbose_;
    TensorFormat input_data_format_;
    TensorFormat output_data_format_;

 public:
    explicit BaseSpatialTransformOp(OpKernelConstruction* context) : OpKernel(context) {
        std::string filter_mode;
        OP_REQUIRES_OK(context, context->GetAttr("filter_mode", &filter_mode));
        if (filter_mode == "nearest")
            filter_mode_ = FILTER_MODE_NEAREST;
        else if (filter_mode == "bilinear")
            filter_mode_ = FILTER_MODE_BILINEAR;
        else if (filter_mode == "bicubic")
            filter_mode_ = FILTER_MODE_BICUBIC;
        // else unknown filter mode (this should have been caught by TF already).

        OP_REQUIRES_OK(context, context->GetAttr("background_value", &background_));
        OP_REQUIRES_OK(context, context->GetAttr("verbose", &verbose_));

        std::string input_data_format, output_data_format;
        OP_REQUIRES_OK(context, context->GetAttr("input_data_format", &input_data_format));
        OP_REQUIRES_OK(context, context->GetAttr("output_data_format", &output_data_format));
        OP_REQUIRES(context, FormatFromString(input_data_format, &input_data_format_),
                    errors::InvalidArgument("Invalid input data format"));
        OP_REQUIRES(context, FormatFromString(output_data_format, &output_data_format_),
                    errors::InvalidArgument("Invalid output data format"));
    }

    // Derived class should implement this
    virtual void ComputeArch(OpKernelContext* context, Tensor* output_tensor,
                             const Tensor& input_images_tensor,
                             const float* transformation_matrices, int batch_size, int num_channels,
                             int height, int width, int output_height, int output_width,
                             bool input_channels_first, bool output_channels_first) = 0;

    void Compute(OpKernelContext* context) override {
        // Check input `images`.
        const Tensor& images_tensor = context->input(0);
        OP_REQUIRES(context, 4 == images_tensor.shape().dims(),
                    errors::InvalidArgument("images tensor must have 4 dimensions, shape is: ",
                                            images_tensor.shape().DebugString(), "."));
        int batch_dim = 0;
        int height_dim = 1;
        int width_dim = 2;
        int channels_dim = 3;
        if (input_data_format_ == FORMAT_NCHW) {
            channels_dim = 1;
            height_dim = 2;
            width_dim = 3;
        }
        int batch_size = images_tensor.shape().dim_size(batch_dim);
        int height = images_tensor.shape().dim_size(height_dim);
        int width = images_tensor.shape().dim_size(width_dim);
        int num_channels = images_tensor.shape().dim_size(channels_dim);

        // Check input `transformation_matrices`.
        const Tensor& transformation_matrices_tensor = context->input(1);
        OP_REQUIRES(context, 3 == transformation_matrices_tensor.shape().dims(),
                    errors::InvalidArgument(
                        "shape transformation_matrices must have 3 dimensions, ", "shape is: ",
                        transformation_matrices_tensor.shape().DebugString(), "."));
        int matrix_batch_size = transformation_matrices_tensor.shape().dim_size(0);
        int matrix_height = transformation_matrices_tensor.shape().dim_size(1);
        int matrix_width = transformation_matrices_tensor.shape().dim_size(2);
        OP_REQUIRES(context, batch_size == matrix_batch_size,
                    errors::InvalidArgument("number of images and matrices must match"));
        OP_REQUIRES(context, 3 == matrix_height, errors::InvalidArgument("matrix must be 3x3"));
        OP_REQUIRES(context, 3 == matrix_width, errors::InvalidArgument("matrix must be 3x3"));

        // Check input `shape`.
        const Tensor& shape_tensor = context->input(2);
        auto shape = shape_tensor.flat<int>();
        OP_REQUIRES(context, 1 == shape_tensor.shape().dims(),
                    errors::InvalidArgument("shape tensor must have 1 dimensions, shape is: ",
                                            shape_tensor.shape().DebugString(), "."));
        OP_REQUIRES(context, 2 == shape_tensor.shape().dim_size(0),
                    errors::InvalidArgument("shape tensor must have 2 elements, shape is: ",
                                            shape_tensor.shape().DebugString(), "."));
        int output_height = shape.data()[0];
        int output_width = shape.data()[1];

        // Create an output tensor.
        TensorShape output_shape =
            output_data_format_ == FORMAT_NCHW
                ? TensorShape({batch_size, num_channels, output_height, output_width})
                : TensorShape({batch_size, output_height, output_width, num_channels});
        if (verbose_) {
            printf("input dim: batch_size = %d, num_channels = %d, height = %d, width = %d \n",
                   batch_size, num_channels, height, width);
            printf("output shape: %s\n", output_shape.DebugString().c_str());
        }
        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

        // Call derived class's CPU/GPU specific implementation.
        auto transformation_matrices = transformation_matrices_tensor.flat<float>();
        ComputeArch(context, output_tensor, images_tensor, transformation_matrices.data(),
                    batch_size, num_channels, height, width, output_height, output_width,
                    input_data_format_ == FORMAT_NCHW, output_data_format_ == FORMAT_NCHW);

        if (verbose_) {
            printf("done\n");
        }
    }
};

#endif  // _SPATIALTRANSFORM_H_
