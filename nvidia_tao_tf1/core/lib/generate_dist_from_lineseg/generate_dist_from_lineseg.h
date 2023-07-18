// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
// This header file implements common functionality of
// of generate distance mask from line segment op in maglev

#ifndef _GENERATE_DIST_FROM_LINESEG_H_
#define _GENERATE_DIST_FROM_LINESEG_H_

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

constexpr float MARGIN_FOR_ON_THE_LINE = 0.01f;
constexpr float MARGIN_FOR_BEING_VERTEX = 0.15f;  // percentage
constexpr float VERY_SMALL_FLOAT_VAL = FLT_MIN;
constexpr float DONTCARE_ANGLE_THRESHOLD = -900.0f;

typedef enum {
    MASK_POS_OPTION_0 = 0,
    DX_POS_OPTION_0 = 1,
    DY_POS_OPTION_0 = 2,
    NDX_POS_OPTION_0 = 3,
    NDY_POS_OPTION_0 = 4,
    COS_POS_OPTION_0 = 5,
    SIN_POS_OPTION_0 = 6,
    WIDTH_LEFT_POS_OPTION_0 = 7,
    WIDTH_RIGHT_POS_OPTION_0 = 8,
    BITS_START_POS_OPTION_0 = 9
} ENCODING_OPTION_0;

typedef enum {
    MASK_POS_OPTION_1 = 0,
    D_POS_OPTION_1 = 1,
    D_COS_POS_OPTION_1 = 2,
    D_SIN_POS_OPTION_1 = 3,
    COS_POS_OPTION_1 = 4,
    SIN_POS_OPTION_1 = 5,
    WIDTH_LEFT_POS_OPTION_1 = 6,
    WIDTH_RIGHT_POS_OPTION_1 = 7,
    BITS_START_POS_OPTION_1 = 8
} ENCODING_OPTION_1;

static __inline__ CUDA_HOSTDEV bool cross_product(const float a_x, const float a_y, const float a_z,
                                                  const float b_x, const float b_y, const float b_z,
                                                  float* out_x, float* out_y, float* out_z,
                                                  bool check_z = true) {
    *out_x = a_y * b_z - a_z * b_y;
    *out_y = a_z * b_x - a_x * b_z;
    *out_z = a_x * b_y - a_y * b_x;
    bool result = true;
    if (check_z == true) {
        if (fabs(*out_z) > VERY_SMALL_FLOAT_VAL) {
            (*out_x) /= (*out_z);
            (*out_y) /= (*out_z);
            (*out_z) = 1;
        } else {
            result = false;
        }
    }
    return result;
}

static __inline__ CUDA_HOSTDEV void get_line(const float x1, const float y1, const float x2,
                                             const float y2, float* a, float* b, float* c) {
    cross_product(x1, y1, 1, x2, y2, 1, a, b, c, false);
}

static __inline__ CUDA_HOSTDEV void get_normal_line_passing(const float a, const float b,
                                                            const float c, const float x,
                                                            const float y, float* out_a,
                                                            float* out_b, float* out_c) {
    (*out_a) = -b;
    (*out_b) = a;
    (*out_c) = b * x - a * y;
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

static __inline__ CUDA_HOSTDEV float weight_scheme(const int y, const int y0) {
    float diff = y - y0;
    if (diff < 0) {
        diff = 0;
    }
    return y0 - diff;
}

static __inline__ CUDA_HOSTDEV bool get_closest_vector_and_angle_from_point_to_line(
    const float x1, const float y1, const float x2, const float y2, const float src_x,
    const float src_y, const float angle_top, const float angle_bottom, const float width_left_top,
    const float width_right_top, const float width_left_bottom, const float width_right_bottom,
    float* vx, float* vy, int* interection_with_bottom_middle_top /*0: bottom, 1: top, 2: middle*/,
    float* angle, float* width_left, float* width_right) {
    float a;
    float b;
    float c;
    get_line(x1, y1, x2, y2, &a, &b, &c);
    float n_a;
    float n_b;
    float n_c;
    get_normal_line_passing(a, b, c, src_x, src_y, &n_a, &n_b, &n_c);
    // now we compute intersection between a,b,c, and n_a, n_b, n_c
    float ix;
    float iy;
    float iz;
    bool result = false;
    if (cross_product(a, b, c, n_a, n_b, n_c, &ix, &iy, &iz)) {
        float magLine = norm2(x1 - x2, y1 - y2);
        // check, ix,iy,iz is on the line a,b,c.....
        float mag1 = norm2(ix - x1, iy - y1);
        float mag2 = norm2(ix - x2, iy - y2);

        float dist_sum = mag1 + mag2;
        // slightly over the line will still be selected...
        if (dist_sum <= magLine + MARGIN_FOR_ON_THE_LINE) {  // we have an intersection.....
            *vx = ix - src_x;
            *vy = iy - src_y;
            *interection_with_bottom_middle_top = 1;

            *width_left = (mag2 * width_left_bottom + mag1 * width_left_top) / (mag1 + mag2);
            *width_right = (mag2 * width_right_bottom + mag1 * width_right_top) / (mag1 + mag2);
            float sin_mixed = (mag2 * sin(angle_bottom) + mag1 * sin(angle_top)) / (mag1 + mag2);
            float cos_mixed = (mag2 * cos(angle_bottom) + mag1 * cos(angle_top)) / (mag1 + mag2);
            *angle = atan2(sin_mixed, cos_mixed);

            if (mag1 < mag2) {
                if (mag1 < MARGIN_FOR_BEING_VERTEX * dist_sum) {
                    // intersected with bottom point
                    *interection_with_bottom_middle_top = 0;
                    *angle = angle_bottom;
                    *width_left = width_left_bottom;
                    *width_right = width_right_bottom;
                }
            } else {
                if (mag2 < MARGIN_FOR_BEING_VERTEX * dist_sum) {
                    // intersected with top point
                    *interection_with_bottom_middle_top = 2;
                    *angle = angle_top;
                    *width_left = width_left_top;
                    *width_right = width_right_top;
                }
            }
        } else {
            float mag1 = norm2(src_x - x1, src_y - y1);
            float mag2 = norm2(src_x - x2, src_y - y2);
            if (mag1 < mag2) {
                *vx = x1 - src_x;
                *vy = y1 - src_y;
                *interection_with_bottom_middle_top = 0;
                *angle = angle_bottom;
                *width_left = width_left_bottom;
                *width_right = width_right_bottom;

            } else {
                *vx = x2 - src_x;
                *vy = y2 - src_y;
                *interection_with_bottom_middle_top = 2;
                *angle = angle_top;
                *width_left = width_left_top;
                *width_right = width_right_top;
            }
        }
        result = true;
    }
    return result;
}

static __inline__ CUDA_HOSTDEV void encode_mask(float* output, int x, int y, const int output_rows,
                                                const int output_cols, const int radius,
                                                const float distance, const int mask_pos) {
    if (distance < radius) {  // set valid mask on otherwise set to off
        output[y * output_cols + x + output_rows * output_cols * mask_pos] = 1;
    } else {
        output[y * output_cols + x + output_rows * output_cols * mask_pos] = 0;
    }
}

static __inline__ CUDA_HOSTDEV void encode_bits(float* output, int x, int y, const int output_rows,
                                                const int output_cols, const float class_id,
                                                const int bitcoding_starting_pos,
                                                const int channel_count_for_bit_coding) {
    // bit coding
    uint32_t cid = static_cast<uint32_t>(class_id);
    uint32_t checker = 1;
    for (int bit = 0; bit < channel_count_for_bit_coding; bit++) {
        uint32_t val = cid & checker;
        if (val > 0) {
            output[y * output_cols + x +
                   output_rows * output_cols * (bitcoding_starting_pos + bit)] = 1;
        } else {
            output[y * output_cols + x +
                   output_rows * output_cols * (bitcoding_starting_pos + bit)] = 0;
        }
        checker <<= 1;
    }
}

static __inline__ CUDA_HOSTDEV void encode_option_0(float* output, int x, int y,
                                                    const int output_rows, const int output_cols,
                                                    const int radius, const float vx,
                                                    const float vy, const float distance,
                                                    const float width_left, const float width_right,
                                                    const float class_id, const float angle,
                                                    const int channel_count_for_bit_coding) {
    encode_mask(output, x, y, output_rows, output_cols, radius, distance, MASK_POS_OPTION_0);

    if (vx >= 0) {
        output[y * output_cols + x + output_rows * output_cols * DX_POS_OPTION_0] = vx;
        output[y * output_cols + x + output_rows * output_cols * NDX_POS_OPTION_0] = 0;
    } else {
        output[y * output_cols + x + output_rows * output_cols * DX_POS_OPTION_0] = 0;
        output[y * output_cols + x + output_rows * output_cols * NDX_POS_OPTION_0] = -vx;
    }
    if (vy >= 0) {
        output[y * output_cols + x + output_rows * output_cols * DY_POS_OPTION_0] = vy;
        output[y * output_cols + x + output_rows * output_cols * NDY_POS_OPTION_0] = 0;
    } else {
        output[y * output_cols + x + output_rows * output_cols * DY_POS_OPTION_0] = 0;
        output[y * output_cols + x + output_rows * output_cols * NDY_POS_OPTION_0] = -vy;
    }

    output[y * output_cols + x + output_rows * output_cols * COS_POS_OPTION_0] =
        (cos(angle) + 1) * 0.5f;
    output[y * output_cols + x + output_rows * output_cols * SIN_POS_OPTION_0] =
        (sin(angle) + 1) * 0.5f;

    output[y * output_cols + x + output_rows * output_cols * WIDTH_LEFT_POS_OPTION_0] = width_left;
    output[y * output_cols + x + output_rows * output_cols * WIDTH_RIGHT_POS_OPTION_0] =
        width_right;

    encode_bits(output, x, y, output_rows, output_cols, class_id, BITS_START_POS_OPTION_0,
                channel_count_for_bit_coding);
}

static __inline__ CUDA_HOSTDEV void encode_option_1(float* output, int x, int y,
                                                    const int output_rows, const int output_cols,
                                                    const int radius, const float vx,
                                                    const float vy, const float distance,
                                                    const float width_left, const float width_right,
                                                    const float class_id, const float angle,
                                                    const int channel_count_for_bit_coding) {
    encode_mask(output, x, y, output_rows, output_cols, radius, distance, MASK_POS_OPTION_1);

    output[y * output_cols + x + output_rows * output_cols * D_POS_OPTION_1] =
        std::sqrt(vx * vx + vy * vy);

    float dist2d_angle = std::atan2(vy, vx);

    output[y * output_cols + x + output_rows * output_cols * D_COS_POS_OPTION_1] =
        (cos(dist2d_angle) + 1) * 0.5f;

    output[y * output_cols + x + output_rows * output_cols * D_SIN_POS_OPTION_1] =
        (sin(dist2d_angle) + 1) * 0.5f;

    output[y * output_cols + x + output_rows * output_cols * COS_POS_OPTION_1] =
        (cos(angle) + 1) * 0.5f;
    output[y * output_cols + x + output_rows * output_cols * SIN_POS_OPTION_1] =
        (sin(angle) + 1) * 0.5f;

    output[y * output_cols + x + output_rows * output_cols * WIDTH_LEFT_POS_OPTION_1] = width_left;
    output[y * output_cols + x + output_rows * output_cols * WIDTH_RIGHT_POS_OPTION_1] =
        width_right;

    encode_bits(output, x, y, output_rows, output_cols, class_id, BITS_START_POS_OPTION_1,
                channel_count_for_bit_coding);
}

class _GenerateDistFromLineseg : public OpKernel {
 protected:
    // static constants
    static constexpr int NUM_INPUT_CHANNELS = 13;
    static constexpr int BASE_ENCODING_DIM_FOR_OPTION0 = 9;
    static constexpr int BASE_ENCODING_DIM_FOR_OPTION1 = 8;

    // class member for attributes
    int n_classes_;
    int src_width_;
    int src_height_;
    int down_scale_factor_;
    int encoding_option_;
    int radius_;
    int cluster_radius_;
    int class_radius_;
    int defined_infinity_;
    bool normalize_;
    bool verbose_;

    // derived class member from attributes
    int target_width_;
    int target_height_;
    int starting_encoding_dim_for_bitcoding_;
    int target_channels_;
    int channel_count_for_bit_coding_;

    // encode option will support
    // different output
    // output order
    // default option 0
    // 0:MASK
    // 1:dx
    // 2:dy
    // 3:ndx
    // 4:ndy
    // 5:(cos+1)*0.5
    // 6:(sin+1)*0.5
    // 7: width_left
    // 8: width_right
    // 9: extras.....bit coding...etc

    // new option 1
    // 0:MASK
    // 1:dist
    // 2:(cos+1)*0.5 toward 0 dist
    // 3:(sin+1)*0.5 toward 0 dist
    // 4:(cos+1)*0.5 direction
    // 5:(sin+1)*0.5 direction
    // 6: width_left
    // 7: width_right
    // 8: extras.....bit coding...etc
 public:
    explicit _GenerateDistFromLineseg(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("n_classes", &n_classes_));
        OP_REQUIRES(context, n_classes_ > 0,
                    errors::InvalidArgument("Need n_classes > 0, got ", n_classes_));

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

        std::string encoded_option;
        OP_REQUIRES_OK(context, context->GetAttr("encoding_option", &encoded_option));
        if (!encoded_option.compare("dist")) {
            encoding_option_ = 0;
        } else if (!encoded_option.compare("angle")) {
            encoding_option_ = 1;
        } else {
            OP_REQUIRES(context,
                        !encoded_option.compare("angle") && !encoded_option.compare("dist"),
                        errors::InvalidArgument("encoded_option only supports `dist` now."
                                                "encoded_option `angle` not yet supported"));
            encoding_option_ = 0;
        }
        OP_REQUIRES_OK(context, context->GetAttr("defined_infinity", &defined_infinity_));

        OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
        OP_REQUIRES(
            context, radius_ >= 0,
            errors::InvalidArgument("radius:", radius_, " should be greater or equal to zero."));
        OP_REQUIRES_OK(context, context->GetAttr("cluster_radius", &cluster_radius_));
        OP_REQUIRES_OK(context, context->GetAttr("class_radius", &class_radius_));
        OP_REQUIRES(context, class_radius_ >= 0,
                    errors::InvalidArgument("class_radius_:", class_radius_,
                                            " should be greater or equal to zero."));
        OP_REQUIRES(
            context, radius_ < defined_infinity_,
            errors::InvalidArgument("radius:", radius_, " should be smaller than defined infinity:",
                                    defined_infinity_, ". "));
        OP_REQUIRES(context, cluster_radius_ <= radius_,
                    errors::InvalidArgument("cluster_radius:", cluster_radius_,
                                            " should be smaller than radius_:", radius_, "."));
        OP_REQUIRES(context, cluster_radius_ >= 0,
                    errors::InvalidArgument("cluster_radius_:", cluster_radius_,
                                            " should be greater or equal to zero."));

        OP_REQUIRES_OK(context, context->GetAttr("normalize", &normalize_));
        OP_REQUIRES_OK(context, context->GetAttr("verbose", &verbose_));

        target_width_ = src_width_ / down_scale_factor_;
        target_height_ = src_height_ / down_scale_factor_;
        if (n_classes_ == 1) {
            channel_count_for_bit_coding_ = 1;
        } else {
            channel_count_for_bit_coding_ =
                static_cast<int>(std::ceil(std::log2(static_cast<float>(n_classes_))));
        }

        if (encoding_option_ == 0) {
            target_channels_ = BASE_ENCODING_DIM_FOR_OPTION0 + channel_count_for_bit_coding_;
            starting_encoding_dim_for_bitcoding_ = BASE_ENCODING_DIM_FOR_OPTION0;
        } else {
            target_channels_ = BASE_ENCODING_DIM_FOR_OPTION1 + channel_count_for_bit_coding_;
            starting_encoding_dim_for_bitcoding_ = BASE_ENCODING_DIM_FOR_OPTION1;
        }
    }

    void Compute(OpKernelContext* context) override { Preprocess(context); }

    virtual void EncodeCore(OpKernelContext* context, const int* lineseg_count_per_image,
                            const int batch_size, const float* linesegments,
                            const int rows,  // total length
                            const int cols,  // fixed...dimension
                            const int encoding_option, float* output, float* output_dist2d,
                            float* cluster_id, float* class_id_int, float* weights,
                            float* dontcare_angles, const int output_channels,
                            const int output_rows, const int output_cols, const int radius,
                            const int defined_infinity, const int down_scale_factor,
                            const int starting_encoding_dim_for_bitcoding,
                            const int channel_count_for_bit_coding, const int num_input_classes,
                            const bool normalize) = 0;

    // image > polygon > vertex
    void Preprocess(OpKernelContext* context) {
        //
        // Grab the input tensor
        //
        const Tensor& line_segments_count_per_image_tensor = context->input(0);
        auto line_segments_count_per_image = line_segments_count_per_image_tensor.flat<int>();
        // all the vertices of polygons in all images
        const Tensor& line_segments_tensor = context->input(1);
        auto input_line_segments = line_segments_tensor.flat<float>();

        int batch_size = line_segments_count_per_image_tensor.shape().dim_size(0);
        /*
         *  From linesegments operator.
         *
         output1[i * cols1] = line_segments[i].top_.x;
         output1[i * cols1 + 1] = line_segments[i].top_.y;
         output1[i * cols1 + 2] = line_segments[i].bottom_.x;
         output1[i * cols1 + 3] = line_segments[i].bottom_.y;
         output1[i * cols1 + 4] = line_segments[i].angle_;
         output1[i * cols1 + 5] = line_segments[i].angle_top_;
         output1[i * cols1 + 6] = line_segments[i].angle_bottom_;
         output1[i * cols1 + 7] = line_segments[i].class_id_;
         output1[i * cols1 + 8] = line_segments[i].cluster_id_;
         output1[i * cols1 + 9] = line_segments[i].width_left_top_;
         output1[i * cols1 + 10] = line_segments[i].width_right_top_;
         output1[i * cols1 + 11] = line_segments[i].width_left_bottom_;
         output1[i * cols1 + 12] = line_segments[i].width_right_bottom_;
         */
        int dims = line_segments_tensor.shape().dim_size(1);                // cols
        int total_line_sements = line_segments_tensor.shape().dim_size(0);  // rows

        OP_REQUIRES(context, dims == NUM_INPUT_CHANNELS,
                    errors::InvalidArgument("linesegments tensor row dim should be ",
                                            NUM_INPUT_CHANNELS, "."));

        int total_seg_count = 0;
        for (int i = 0; i < batch_size; i++) {
            total_seg_count += line_segments_count_per_image.data()[i];
        }

        OP_REQUIRES(context, total_seg_count == total_line_sements,
                    errors::InvalidArgument("total_seg_count computed from "
                                            "line_segments_count_per_image_tensor",
                                            total_seg_count,
                                            " does not match to total_line_sements "
                                            "returned by line_segments_tensor "
                                            "tensor shape ",
                                            total_line_sements, "."));

        Tensor* output_tensor = NULL;

        // Create an output tensor
        TensorShape output_shape({batch_size, target_channels_, target_height_, target_width_});

        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
        auto output = output_tensor->template flat<float>();

        TensorShape output_shape1({batch_size, 1, target_height_, target_width_});

        OP_REQUIRES_OK(context, context->allocate_output(1, output_shape1, &output_tensor));
        auto dist_transform = output_tensor->template flat<float>();

        OP_REQUIRES_OK(context, context->allocate_output(2, output_shape1, &output_tensor));
        auto cluster_id = output_tensor->template flat<float>();

        OP_REQUIRES_OK(context, context->allocate_output(3, output_shape1, &output_tensor));
        auto class_id = output_tensor->template flat<float>();

        OP_REQUIRES_OK(context, context->allocate_output(4, output_shape1, &output_tensor));
        auto weights = output_tensor->template flat<float>();

        OP_REQUIRES_OK(context, context->allocate_output(5, output_shape1, &output_tensor));
        auto dontcare_angles = output_tensor->template flat<float>();

        EncodeCore(context, line_segments_count_per_image.data(), batch_size,
                   input_line_segments.data(), total_line_sements, dims, encoding_option_,
                   output.data(), dist_transform.data(), cluster_id.data(), class_id.data(),
                   weights.data(), dontcare_angles.data(), target_channels_, target_height_,
                   target_width_, radius_, defined_infinity_, down_scale_factor_,
                   starting_encoding_dim_for_bitcoding_, channel_count_for_bit_coding_, n_classes_,
                   normalize_);
    }
};

#else
#endif  // _GENERATE_DIST_FROM_LINESEG_H_
