// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
// This header file implements common functionality of
// of generate distance mask from bezier curve op in modulus.

#ifndef _GENERATE_DIST_FROM_BEZIER_H_
#define _GENERATE_DIST_FROM_BEZIER_H_

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <vector>

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

typedef enum {
    BEZIER_MASK_POS = 0,
    CTRL_PTS1_X_POS = 1,
    CTRL_PTS1_Y_POS = 2,
    CTRL_PTS2_X_POS = 3,
    CTRL_PTS2_Y_POS = 4,
    CTRL_PTS3_X_POS = 5,
    CTRL_PTS3_Y_POS = 6,
    CTRL_PTS4_X_POS = 7,
    CTRL_PTS4_Y_POS = 8,
    BIT_START_POS = 9,
} BEZIER_ENCODING_POS;

typedef enum {
    CTRL_PTS1_X = 0,
    CTRL_PTS1_Y = 1,
    CTRL_PTS2_X = 2,
    CTRL_PTS2_Y = 3,
    CTRL_PTS3_X = 4,
    CTRL_PTS3_Y = 5,
    CTRL_PTS4_X = 6,
    CTRL_PTS4_Y = 7,
} BEZIER_CTRL_PTS;

typedef enum {
    PTS1_X = 0,
    PTS1_Y = 1,
    PTS2_X = 2,
    PTS2_Y = 3,
} BEZIER_SAMPLING_PTS;

static __inline__ CUDA_HOSTDEV float dot_product(const float x1, const float y1, const float x2,
                                                 const float y2) {
    return x1 * x2 + y1 * y2;
}

static __inline__ CUDA_HOSTDEV float norm2(const float dx, const float dy) {
    return sqrt(dx * dx + dy * dy);
}

static __inline__ CUDA_HOSTDEV float dist2(const float x1, const float y1, const float x2,
                                           const float y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    return norm2(dx, dy);
}

static __inline__ CUDA_HOSTDEV float get_distance_from_point_to_line_segment(
    const float x1, const float y1, const float x2, const float y2, const float src_x,
    const float src_y) {
    float dist;
    // If the line segment degenerates to a point, use the distance to this point.
    if (x1 == x2 && y1 == y2) {
        dist = dist2(x1, y1, src_x, src_y);
        return dist;
    }
    // Projection of the point onto the line segment.
    float pos = dot_product(x2 - x1, y2 - y1, src_x - x1, src_y - y1) /
                dot_product(x2 - x1, y2 - y1, x2 - x1, y2 - y1);
    if (pos <= 0) {
        // If the projection is out of the starting point, use the distance to the starting point.
        dist = dist2(x1, y1, src_x, src_y);
    } else if (pos >= 1) {
        // If the projection is out of the ending point, use the distance to the ending point.
        dist = dist2(x2, y2, src_x, src_y);
    } else {
        // If the projection is within the line segment, calculate the distance by Pythagorean
        // theorem.
        // Hypotenuse can be calculated by ||pt_src - pt_1||, expressed below as dot product.
        float hypotenuse_squared = dot_product(src_x - x1, src_y - y1, src_x - x1, src_y - y1);
        // Leg can be calculated by ||pt_2 - pt_1|| * pos, but in order to avoid precision issue, it
        // is expanded as below.
        float leg_squared = dot_product(x2 - x1, y2 - y1, src_x - x1, src_y - y1) *
                            dot_product(x2 - x1, y2 - y1, src_x - x1, src_y - y1) /
                            dot_product(x2 - x1, y2 - y1, x2 - x1, y2 - y1);
        dist = sqrt(hypotenuse_squared - leg_squared);
    }
    return dist;
}

static __inline__ CUDA_HOSTDEV void get_bezier_sample(const float* bezier_curve, const float t,
                                                      float* x, float* y) {
    // Bezier coefficients.
    float tp = 1 - t;
    float tp2 = tp * tp;
    float tp3 = tp2 * tp;
    float t2 = t * t;
    float t3 = t2 * t;
    float B0 = tp3;
    float B1 = 3 * tp2 * t;
    float B2 = 3 * t2 * tp;
    float B3 = t3;

    // Bezier control points.
    float x0 = bezier_curve[CTRL_PTS1_X];
    float y0 = bezier_curve[CTRL_PTS1_Y];
    float x1 = bezier_curve[CTRL_PTS2_X];
    float y1 = bezier_curve[CTRL_PTS2_Y];
    float x2 = bezier_curve[CTRL_PTS3_X];
    float y2 = bezier_curve[CTRL_PTS3_Y];
    float x3 = bezier_curve[CTRL_PTS4_X];
    float y3 = bezier_curve[CTRL_PTS4_Y];

    // Sample point on bezier curve.
    *x = x0 * B0 + x1 * B1 + x2 * B2 + x3 * B3;
    *y = y0 * B0 + y1 * B1 + y2 * B2 + y3 * B3;
    return;
}

static __inline__ CUDA_HOSTDEV void encode_bits(float* output, const int offset, const int class_id,
                                                const int num_bits) {
    uint32_t cid = static_cast<uint32_t>(class_id);
    uint32_t checker = 1;
    for (int i = 0; i < num_bits; i++) {
        uint32_t val = cid & checker;
        if (val > 0) {
            output[offset * i] = 1;
        }
        checker <<= 1;
    }
}

static __inline__ CUDA_HOSTDEV void encode_dist(float* output, const int offset,
                                                const float* bezier_curve, const float ox,
                                                const float oy, const int class_id,
                                                const int num_bits, const float scale_factor) {
    output[offset * BEZIER_MASK_POS] = 1;
    output[offset * CTRL_PTS1_X_POS] = (bezier_curve[CTRL_PTS1_X] - ox) / scale_factor;
    output[offset * CTRL_PTS1_Y_POS] = (bezier_curve[CTRL_PTS1_Y] - oy) / scale_factor;
    output[offset * CTRL_PTS2_X_POS] = (bezier_curve[CTRL_PTS2_X] - ox) / scale_factor;
    output[offset * CTRL_PTS2_Y_POS] = (bezier_curve[CTRL_PTS2_Y] - oy) / scale_factor;
    output[offset * CTRL_PTS3_X_POS] = (bezier_curve[CTRL_PTS3_X] - ox) / scale_factor;
    output[offset * CTRL_PTS3_Y_POS] = (bezier_curve[CTRL_PTS3_Y] - oy) / scale_factor;
    output[offset * CTRL_PTS4_X_POS] = (bezier_curve[CTRL_PTS4_X] - ox) / scale_factor;
    output[offset * CTRL_PTS4_Y_POS] = (bezier_curve[CTRL_PTS4_Y] - oy) / scale_factor;
    encode_bits(&output[offset * BIT_START_POS], offset, class_id, num_bits);
}

class _GenerateDistFromBezier : public OpKernel {
 protected:
    // Static constants.
    static constexpr int NUM_BEZIER_CTRL_PTS = 4;
    static constexpr int NUM_BEZIER_AXES = 2;
    static constexpr int NUM_MASK_CHANNELS = 1;
    static constexpr int NUM_BEZIER_CHANNELS = NUM_BEZIER_CTRL_PTS * NUM_BEZIER_AXES;
    static constexpr int MAX_NUM_BEZIER_CURVES = 200;
    static constexpr int NUM_SAMPLES_PER_CURVE = 16;

    // Class member for attributes.
    std::vector<int> n_classes_;
    int n_tasks_;
    int src_width_;
    int src_height_;
    int down_scale_factor_;
    float encode_scale_factor_;
    int radius_;
    int start_sample_id_;

    // Derived class member from attributes.
    int target_width_;
    int target_height_;
    std::vector<int> target_channels_;
    std::vector<int> target_bits_;
    int target_total_channels_;

 public:
    explicit _GenerateDistFromBezier(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("n_classes", &n_classes_));
        n_tasks_ = n_classes_.size();
        OP_REQUIRES(context, n_tasks_ > 0,
                    errors::InvalidArgument("Need n_classes > 0, got ", n_tasks_));
        for (int i = 0; i < n_tasks_; i++) {
            OP_REQUIRES(context, n_classes_[i] > 0,
                        errors::InvalidArgument("Need every element of n_classes > 0, got ",
                                                n_classes_[i], " at index ", i));
        }

        OP_REQUIRES_OK(context, context->GetAttr("src_width", &src_width_));
        OP_REQUIRES(context, src_width_ > 0,
                    errors::InvalidArgument("Need src_width > 0, got ", src_width_));
        OP_REQUIRES_OK(context, context->GetAttr("src_height", &src_height_));
        OP_REQUIRES(context, src_height_ > 0,
                    errors::InvalidArgument("Need src_height > 0, got ", src_height_));

        OP_REQUIRES_OK(context, context->GetAttr("down_scale_factor", &down_scale_factor_));
        OP_REQUIRES(
            context,
            down_scale_factor_ == 1 || down_scale_factor_ == 2 || down_scale_factor_ == 4 ||
                down_scale_factor_ == 8 || down_scale_factor_ == 16,
            errors::InvalidArgument("Need down_scale_factor need to be either 1, 2, 4, 8, 16, got ",
                                    down_scale_factor_));

        OP_REQUIRES_OK(context, context->GetAttr("encode_scale_factor", &encode_scale_factor_));
        OP_REQUIRES(context, encode_scale_factor_ > 0,
                    errors::InvalidArgument("encode_scale_factor:", encode_scale_factor_,
                                            " should be greater than zero."));

        OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
        OP_REQUIRES(
            context, radius_ >= 0,
            errors::InvalidArgument("radius:", radius_, " should be greater or equal to zero."));

        OP_REQUIRES_OK(context, context->GetAttr("start_sample_id", &start_sample_id_));
        OP_REQUIRES(context, start_sample_id_ >= 0 && 2 * start_sample_id_ < NUM_SAMPLES_PER_CURVE,
                    errors::InvalidArgument("start_sample_id:", start_sample_id_,
                                            " should be greater or equal to zero, and less than "
                                            "half of num_samples_per_curve."));

        target_width_ = src_width_ / down_scale_factor_;
        target_height_ = src_height_ / down_scale_factor_;
        target_total_channels_ = 0;
        for (int i = 0; i < n_tasks_; i++) {
            int num_bits =
                static_cast<int>(std::ceil(std::log2(static_cast<float>(n_classes_[i]))));
            target_channels_.push_back(NUM_MASK_CHANNELS + NUM_BEZIER_CHANNELS + num_bits);
            target_bits_.push_back(num_bits);
            target_total_channels_ += target_channels_[i];
        }
    }

    void Compute(OpKernelContext* context) override { Preprocess(context); }

    virtual void EncodeCore(OpKernelContext* context, const float* bezier_curves,
                            const int* vertices_count_per_bezier_curve,
                            const int* bezier_curves_count_per_image, const int* bezier_task_ids,
                            const int* bezier_class_ids, const int batch_size, const int rows,
                            const int cols, float* output, const int output_total_channels,
                            const int output_rows, const int output_cols, const int radius,
                            const int down_scale_factor, const float encode_scale_factor,
                            const int start_sample_id, const int num_input_tasks,
                            const int* num_classes, const int* num_bits,
                            const int* output_channels) = 0;

    void Preprocess(OpKernelContext* context) {
        // All the bezier curves in all images.
        const Tensor& bezier_curves_tensor = context->input(0);
        auto input_bezier_curves = bezier_curves_tensor.flat<float>();

        // Number of vertices per curve.
        const Tensor& vertices_count_per_bezier_curve_tensor = context->input(1);
        auto vertices_count_per_bezier_curve = vertices_count_per_bezier_curve_tensor.flat<int>();

        // Number of curves per image.
        const Tensor& bezier_curves_count_per_image_tensor = context->input(2);
        auto bezier_curves_count_per_image = bezier_curves_count_per_image_tensor.flat<int>();

        // Task id of each curve.
        const Tensor& bezier_task_ids_tensor = context->input(3);
        auto bezier_task_ids = bezier_task_ids_tensor.flat<int>();

        // Class id of each curve.
        const Tensor& bezier_class_ids_tensor = context->input(4);
        auto bezier_class_ids = bezier_class_ids_tensor.flat<int>();

        int total_vertices = bezier_curves_tensor.shape().dim_size(0);
        int dims = bezier_curves_tensor.shape().dim_size(1);
        int total_bezier_curves = vertices_count_per_bezier_curve_tensor.shape().dim_size(0);
        int batch_size = bezier_curves_count_per_image_tensor.shape().dim_size(0);
        int total_task_ids = bezier_task_ids_tensor.shape().dim_size(0);
        int total_class_ids = bezier_class_ids_tensor.shape().dim_size(0);

        OP_REQUIRES(context, dims == NUM_BEZIER_AXES,
                    errors::InvalidArgument("bezier_curves tensor row dim should be ",
                                            NUM_BEZIER_AXES, "."));

        int total_vertex_count = 0;
        for (int i = 0; i < total_bezier_curves; i++) {
            total_vertex_count += vertices_count_per_bezier_curve.data()[i];
        }

        OP_REQUIRES(context, total_vertex_count == total_vertices,
                    errors::InvalidArgument("total_vertex_count computed from "
                                            "vertices_count_per_bezier_curve_tensor ",
                                            total_vertex_count,
                                            " does not match to total_vertices "
                                            "returned by bezier_curves_tensor "
                                            "tensor shape ",
                                            total_vertices, "."));

        int total_curve_count = 0;
        for (int i = 0; i < batch_size; i++) {
            total_curve_count += bezier_curves_count_per_image.data()[i];
        }

        OP_REQUIRES(context, total_curve_count == total_bezier_curves,
                    errors::InvalidArgument("total_curve_count computed from "
                                            "bezier_curves_count_per_image_tensor ",
                                            total_curve_count,
                                            " does not match to total_bezier_curves "
                                            "returned by vertices_count_per_bezier_curve_tensor "
                                            "tensor shape ",
                                            total_bezier_curves, "."));

        OP_REQUIRES(context, total_bezier_curves == total_task_ids,
                    errors::InvalidArgument("vertices_count_per_bezier_curve_tensor ",
                                            "tensor shape ", total_bezier_curves,
                                            " does not match bezier_task_ids_tensor "
                                            "tensor shape ",
                                            total_task_ids, "."));

        OP_REQUIRES(context, total_bezier_curves == total_class_ids,
                    errors::InvalidArgument("vertices_count_per_bezier_curve_tensor ",
                                            "tensor shape ", total_bezier_curves,
                                            " does not match bezier_class_ids_tensor "
                                            "tensor shape ",
                                            total_class_ids, "."));

        Tensor* output_tensor = NULL;
        // Create an output tensor.
        TensorShape output_shape(
            {batch_size, target_total_channels_, target_height_, target_width_});

        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
        auto output = output_tensor->template flat<float>();

        EncodeCore(context, input_bezier_curves.data(), vertices_count_per_bezier_curve.data(),
                   bezier_curves_count_per_image.data(), bezier_task_ids.data(),
                   bezier_class_ids.data(), batch_size, total_bezier_curves, dims, output.data(),
                   target_total_channels_, target_height_, target_width_, radius_,
                   down_scale_factor_, encode_scale_factor_, start_sample_id_, n_tasks_,
                   n_classes_.data(), target_bits_.data(), target_channels_.data());
    }
};

#else
#endif  // _GENERATE_DIST_FROM_BEZIER_H_
