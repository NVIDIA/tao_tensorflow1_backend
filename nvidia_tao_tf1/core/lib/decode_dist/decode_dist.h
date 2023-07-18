// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

#ifndef _DECODE_DIST_H_
#define _DECODE_DIST_H_

#include <algorithm>
#include <climits>
#include <cmath>
#include <map>
#include <vector>

#include <cfloat>
#include <fstream>
#include <iostream>
#include <string>

#include "lrn_params.h"
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

class _DecodeDist : public OpKernel {
 protected:
    // Define constants for channels and index.
    static constexpr int BASE_DECODING_DIM_FOR_OPTION0 = 7;
    static constexpr int BASE_DECODING_DIM_FOR_OPTION1 = 5;

    // enum for decoding option.
    enum DECODING_OPTION { DIST, ANGLE };
    // number of classes when encoding was done
    int n_classes_;
    // width of original image at a target resolution
    int target_width_;
    // height of original image at a target resolution
    int target_height_;
    // width of network output blob
    int src_width_;
    // height of network output blob
    int src_height_;
    // channels of network output blob
    int src_channels_;
    // down scale factor when encoding was done
    // so that is up scale factor when decoding
    int up_scale_factor_;

    // various encoding possible in the future but right now one option "0"
    DECODING_OPTION decoding_option_;
    lineregressordecoder::LRNEncodingParam encoding_param_;
    // radius used when encoding was done
    int radius_;
    // defined_infinity used when encoding was done
    int defined_infinity_;

    int starting_decoding_dim_for_bitcoding_;
    int channel_count_for_bit_coding_;

    // decoding specific parameters
    int minimum_votes_;
    float min_valid_mask_;
    int non_max_radius_;
    int background_class_id_;
    int max_possible_nodes_;
    int max_distance_for_nodes_;
    // arrow_length
    int arrow_length_;
    bool normalize_;
    bool verbose_;
    // computed quantity based on the parameters above for convenience.
    int total_blob_size_per_image_;
    // Input order for two decoding option:
    // default option 0
    // 0:MASK
    // 1:dx
    // 2:dy
    // 3:ndx
    // 4:ndy
    // 5:(cos+1)*0.5
    // 6:(sin+1)*0.5
    // 7: extras.....bit coding...etc

    // option 1
    // 0:MASK
    // 1:dist (magnitude of dx and dy for above option)
    // 2:(cos+1)*0.5 toward 0 dist
    // 3:(sin+1)*0.5 toward 0 dist
    // 4:(cos+1)*0.5 direction
    // 5:(sin+1)*0.5 direction
    // (6-N): bit coding channels, etc
 public:
    explicit _DecodeDist(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("n_classes", &n_classes_));
        OP_REQUIRES(context, n_classes_ > 0,
                    errors::InvalidArgument("Need n_classes > 0, got ", n_classes_));

        OP_REQUIRES_OK(context, context->GetAttr("target_width", &target_width_));
        OP_REQUIRES(context, target_width_ > 0,
                    errors::InvalidArgument("Need target_width > 0, got ", target_width_));

        OP_REQUIRES_OK(context, context->GetAttr("target_height", &target_height_));
        OP_REQUIRES(context, target_height_ > 0,
                    errors::InvalidArgument("Need target_height > 0, got ", target_height_));

        OP_REQUIRES_OK(context, context->GetAttr("src_width", &src_width_));
        OP_REQUIRES(context, src_width_ > 0,
                    errors::InvalidArgument("Need src_width > 0, got ", src_width_));
        OP_REQUIRES_OK(context, context->GetAttr("src_height", &src_height_));
        OP_REQUIRES(context, src_height_ > 0,
                    errors::InvalidArgument("Need src_height > 0, got ", src_height_));

        OP_REQUIRES_OK(context, context->GetAttr("up_scale_factor", &up_scale_factor_));
        OP_REQUIRES_OK(context, context->GetAttr("defined_infinity", &defined_infinity_));
        OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
        OP_REQUIRES_OK(context, context->GetAttr("non_max_radius", &non_max_radius_));
        OP_REQUIRES_OK(context, context->GetAttr("background_class_id", &background_class_id_));
        OP_REQUIRES_OK(context, context->GetAttr("max_possible_nodes", &max_possible_nodes_));
        OP_REQUIRES_OK(context, context->GetAttr("min_valid_mask", &min_valid_mask_));
        std::string decoding_option;
        OP_REQUIRES_OK(context, context->GetAttr("decoding_option", &decoding_option));
        if (decoding_option.compare("dist") == 0) {
            encoding_param_.set(lineregressordecoder::LRNEncodingType::LRN_LREDGE_ENCODING);
            decoding_option_ = DIST;
        } else if (decoding_option.compare("angle") == 0) {
            encoding_param_.set(lineregressordecoder::LRNEncodingType::LRN_LREDGE_DIST2D_ENCODING);
            decoding_option_ = ANGLE;
        } else {
            OP_REQUIRES(
                context,
                decoding_option.compare("angle") == 0 && decoding_option.compare("dist") == 0,
                errors::InvalidArgument("decoding_option only supports `dist` and 'angle'."));
            decoding_option_ = DIST;
        }
        if (n_classes_ == 1) {
            channel_count_for_bit_coding_ = 1;
        } else {
            channel_count_for_bit_coding_ =
                static_cast<int>(std::ceil(std::log2(static_cast<float>(n_classes_))));
        }

        starting_decoding_dim_for_bitcoding_ =
            static_cast<int>(encoding_param_.getBitStartPosition());
        src_channels_ = starting_decoding_dim_for_bitcoding_ + channel_count_for_bit_coding_;

        OP_REQUIRES(
            context, up_scale_factor_ == 1 || up_scale_factor_ == 2 || up_scale_factor_ == 4 ||
                         up_scale_factor_ == 8 || up_scale_factor_ == 16,
            errors::InvalidArgument("Need up_scale_factor need to be either 1, 2, 4, 8, 16, got ",
                                    up_scale_factor_));

        OP_REQUIRES(
            context, src_width_ * up_scale_factor_ >= target_width_,
            errors::InvalidArgument("`src_width_`(", src_width_,
                                    ") multiply by `up_scale_factor_`=", up_scale_factor_,
                                    ") should be greate equal to `target_width_`=", target_width_));

        OP_REQUIRES(context, src_height_ * up_scale_factor_ >= target_height_,
                    errors::InvalidArgument("`src_height_`(", src_height_,
                                            ") multiply by `up_scale_factor_`", up_scale_factor_,
                                            ") should be greate equal to `target_height_`=",
                                            target_height_));

        OP_REQUIRES_OK(context, context->GetAttr("verbose", &verbose_));
        OP_REQUIRES_OK(context, context->GetAttr("normalize", &normalize_));
        OP_REQUIRES_OK(context, context->GetAttr("arrow_length", &arrow_length_));
        OP_REQUIRES(context, non_max_radius_ >= 0,
                    errors::InvalidArgument("non_max_radius_:", non_max_radius_,
                                            " should be non-negative."));
        OP_REQUIRES(context, max_possible_nodes_ > 0,
                    errors::InvalidArgument("max_possible_nodes_:", max_possible_nodes_,
                                            " must be positive."));
        OP_REQUIRES_OK(context,
                       context->GetAttr("max_distance_for_nodes", &max_distance_for_nodes_));
        OP_REQUIRES(context, max_distance_for_nodes_ > 0,
                    errors::InvalidArgument("max_distance_for_nodes_:", max_distance_for_nodes_,
                                            " must be positive."));
        OP_REQUIRES_OK(context, context->GetAttr("minimum_votes", &minimum_votes_));
        OP_REQUIRES(
            context, minimum_votes_ > 0,
            errors::InvalidArgument("minimum_votes_:", minimum_votes_, " must be positive."));
        OP_REQUIRES(
            context, min_valid_mask_ >= 0,
            errors::InvalidArgument("min_valid_mask_:", min_valid_mask_, " must be non-negative."));
        OP_REQUIRES(context, radius_ > 0,
                    errors::InvalidArgument("`radius_`:", radius_, " must be positive."));
        OP_REQUIRES(context, defined_infinity_ > 0,
                    errors::InvalidArgument("`defined_infinity_`:", defined_infinity_,
                                            " must be positive."));
        OP_REQUIRES(
            context, radius_ < defined_infinity_,
            errors::InvalidArgument("radius:", radius_, " should be smaller than defined infinity:",
                                    defined_infinity_, ". "));
        total_blob_size_per_image_ = src_channels_ * src_height_ * src_width_;
    }

    void Compute(OpKernelContext* context) override { Preprocess(context); }

    virtual void decode_core(OpKernelContext* context, const float* tensor_encoded_blobs,
                             const float* tensor_input_nchw, const int batch_size, int* output0,
                             int* output1, int* output2, int* output3) = 0;

    // image > polygon > vertex
    void Preprocess(OpKernelContext* context) {
        //
        // Grab the input tensor
        //
        const Tensor& tensor_encoded_blobs = context->input(0);
        int batch_size = tensor_encoded_blobs.shape().dim_size(0);

        int channels = tensor_encoded_blobs.shape().dim_size(1);  // cols
        OP_REQUIRES(context, src_height_ == tensor_encoded_blobs.shape().dim_size(2),
                    errors::InvalidArgument("src_height_ should be ",
                                            tensor_encoded_blobs.shape().dim_size(2), " but ",
                                            src_height_, " is given."));
        OP_REQUIRES(context, src_width_ == tensor_encoded_blobs.shape().dim_size(3),
                    errors::InvalidArgument("src_width_  should be ",
                                            tensor_encoded_blobs.shape().dim_size(3), " but ",
                                            src_width_, " is given."));

        OP_REQUIRES(context, channels == src_channels_,
                    errors::InvalidArgument(" tensor num of channels should be ", src_channels_,
                                            ".", " It has ", channels, " channels."));

        const Tensor& tensor_input_nchw = context->input(1);
        OP_REQUIRES(
            context, batch_size == tensor_input_nchw.shape().dim_size(0),
            errors::InvalidArgument("tensor_input_nchw.shape().dim_size(0) should be ", batch_size,
                                    " but ", tensor_input_nchw.shape().dim_size(0), " is given."));
        OP_REQUIRES(
            context, 3 == tensor_input_nchw.shape().dim_size(1),
            errors::InvalidArgument("tensor_input_nchw.shape().dim_size(1) should be ", 3, " but ",
                                    tensor_input_nchw.shape().dim_size(1), " is given."));

        OP_REQUIRES(context, target_height_ == tensor_input_nchw.shape().dim_size(2),
                    errors::InvalidArgument("tensor_input_nchw.shape().dim_size(2) should be ",
                                            target_height_, " but ",
                                            tensor_input_nchw.shape().dim_size(2), " is given."));
        OP_REQUIRES(context, target_width_ == tensor_input_nchw.shape().dim_size(3),
                    errors::InvalidArgument("tensor_input_nchw.shape().dim_size(3) should be ",
                                            target_width_, " but ",
                                            tensor_input_nchw.shape().dim_size(3), " is given."));

        // output_tensor should be voting image with color scheme for each class.
        Tensor* output_tensor = NULL;

        // Create an output tensor
        TensorShape output_shape1({batch_size, target_height_, target_width_, 3});

        // binarized by voting with colors indicating class
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape1, &output_tensor));
        auto output0 = output_tensor->template flat<int>();

        // Intensity by voting with colors indicating class
        OP_REQUIRES_OK(context, context->allocate_output(1, output_shape1, &output_tensor));
        auto output1 = output_tensor->template flat<int>();

        // binarized by voting and direction vector drawing with colors indicating class
        OP_REQUIRES_OK(context, context->allocate_output(2, output_shape1, &output_tensor));
        auto output2 = output_tensor->template flat<int>();

        // to output blob for computing precision recall. The computation will be done in other ops.
        TensorShape output_shape2({batch_size, 1, target_height_, target_width_});
        OP_REQUIRES_OK(context, context->allocate_output(3, output_shape2, &output_tensor));

        auto output_for_metric = output_tensor->template flat<int>();
        auto encoded_blobs = tensor_encoded_blobs.flat<float>();

        auto input_nchw = tensor_input_nchw.flat<float>();
        decode_core(context, encoded_blobs.data(), input_nchw.data(), batch_size, output0.data(),
                    output1.data(), output2.data(), output_for_metric.data());
    }
};

#else
#endif  // _DECODE_DIST_H_
