// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#ifndef LRNDECODER_H_
#define LRNDECODER_H_

#include <cuda.h>
#include <stdio.h>
#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include "color.h"
#include "draw_basics.h"
#include "lrn_params.h"
#ifdef __CUDACC__
#define CUDA_HOSTDEV __device__
#else
#define CUDA_HOSTDEV
#endif

namespace lineregressordecoder {

typedef enum {
    DNN_LRN_MASK_POS = 0,
    DNN_LRN_DX_POS = 1,
    DNN_LRN_DY_POS = 2,
    DNN_LRN_NDX_POS = 3,
    DNN_LRN_NDY_POS = 4,
    DNN_LRN_COS_POS = 5,
    DNN_LRN_SIN_POS = 6,
    DNN_LRN_BITS_START_POS = 7
} LRNNetworksOuputChannels;

using VoteType = uint32_t;
using VoteLocalMaxType = uint8_t;
using ClassVoteType = uint32_t;
using BlurTypeForLRN = float;

constexpr int32_t MAX_CLASSES = 100;
constexpr int32_t MAX_SCALES = 8;
constexpr float LRN_RADIAN_TO_DEG = 180.0f / M_PI;
constexpr float LRN_DEG_TO_RADIAN = M_PI / 180.0f;
constexpr int8_t INVALID_VALUE = -1;
constexpr int16_t INVALID_VECTOR = std::numeric_limits<int16_t>::max();
constexpr uint8_t MAX_VOTE_THRESHOLD = 50;
constexpr VoteLocalMaxType MAX_GAUSSIAN_VOTE = std::numeric_limits<VoteLocalMaxType>::max();

struct LRNNode {
    int16_t x;
    int16_t y;
    int16_t angle;
    // Decoded_bit_code value.
    int8_t decoded_bit_code;
    uint8_t normalized_votes;
    // edge left
    uint8_t edge_left_width;
    // edge right
    uint8_t edge_right_width;
};  // 80 bits.

class LRNDecoderBase {
 public:
    static constexpr int32_t DIMX = 32;
    static constexpr int32_t DIMY = 8;
    static constexpr int32_t MAX_GAUSSIAN_FILTER_RADIUS = 10;

    //
    //  @brief Construct a new LRNDecoderBase object.
    //
    //  --------- spec ---------------
    //  @param dnn_width
    //  @param dnn_height
    //  @param dnn_channels
    //  @param input_image_width
    //  @param input_image_height
    //  @param scale_used_for_encoding
    //  @param defined_infinity
    //  @param normalize
    //  @param dnn_radius
    //  @param encoding_param
    //  --------algorithm parameters-----------------
    //  @param non_max_radius
    //  @param max_possible_nodes
    //  @param min_valid_mask_val
    //  @param min_votes_for_nodes
    //  @param max_dist_for_nodes
    //  @param ctx
    //
    LRNDecoderBase(
        // ---------spec----------------------------
        uint32_t dnn_width, uint32_t dnn_height, uint32_t dnn_channels, uint32_t input_image_width,
        uint32_t input_image_height, int16_t scale_used_for_encoding, int16_t defined_infinity,
        bool normalize, int8_t dnn_radius, LRNEncodingParam encoding_param,
        // --------algorithm parameters-----------------
        int8_t non_max_radius, uint16_t max_possible_nodes, float min_valid_mask_val,
        uint16_t min_votes_for_nodes, uint16_t max_dist_for_nodes)
        : LRNDecoderBase(dnn_width, dnn_height, dnn_channels, input_image_width, input_image_height,
                         scale_used_for_encoding, defined_infinity, normalize, dnn_radius,
                         encoding_param, non_max_radius, max_possible_nodes, min_valid_mask_val,
                         min_votes_for_nodes, max_dist_for_nodes, false, 0) {}

    //  @brief Construct a new LRNDecoderBase object.
    //
    //  ---------spec---------------
    //  @param dnn_width
    //  @param dnn_height
    //  @param dnn_channels
    //  @param input_image_width
    //  @param input_image_height
    //  @param scale_used_for_encoding
    //  @param defined_infinity
    //  @param normalize
    //  @param dnn_radius
    //  @param encoding_param
    //  --------algorithm parameters-----------------
    //  @param non_max_radius
    //  @param max_possible_nodes
    //  @param min_valid_mask_val
    //  @param min_votes_for_nodes
    //  @param max_dist_for_nodes
    //  --------algorithm options-----------------
    //  @param use_direction
    //  @param max_possible_classes
    //  @param ctx
    //
    LRNDecoderBase(
        // ---------spec----------------------------
        uint32_t dnn_width, uint32_t dnn_height, uint32_t dnn_channels, uint32_t input_image_width,
        uint32_t input_image_height, int16_t scale_used_for_encoding, int16_t defined_infinity,
        bool normalize, int8_t dnn_radius, LRNEncodingParam encoding_param,
        // --------algorithm parameters-----------------
        int8_t non_max_radius, uint16_t max_possible_nodes, float min_valid_mask_val,
        uint16_t min_votes_for_nodes, uint16_t max_dist_for_nodes,
        // --------algorithm options-----------------
        bool use_direction, int32_t max_possible_classes) {
        // The OpKernal should already check this condition before calling this function.
        // Adding here is just to show the assumptions.
        if (non_max_radius > MAX_GAUSSIAN_FILTER_RADIUS) {
            throw std::runtime_error("LRNDecoderBase: non_max_radius is too large.");
        }
        dnn_width_ = dnn_width;
        dnn_height_ = dnn_height;
        dnn_channels_ = dnn_channels;
        input_image_width_ = input_image_width;
        input_image_height_ = input_image_height;
        // Training param (parameters used during training).
        scale_used_for_encoding_ = scale_used_for_encoding;
        defined_infinity_ = defined_infinity;
        normalize_ = normalize;
        dnn_radius_ = dnn_radius;
        encoding_param_ = encoding_param;
        // Algorithm performance factor:
        // Radius of blurring (reduce noise).
        non_max_radius_ = non_max_radius;
        // Number of maximun nodes we allow.
        max_possible_nodes_ = max_possible_nodes;
        // Number of votes we declare is not noise.
        min_votes_for_nodes_ = min_votes_for_nodes;
        // Maximum distance we search.
        max_dist_for_nodes_ = max_dist_for_nodes;
        // Value of mask we take threshold in as valid.
        min_valid_mask_val_ = min_valid_mask_val;
        // Scaled threshold as above.
        min_scaled_valid_mask_val_ = min_valid_mask_val * 255.0f;
        // Algorithm options (type of encoding, number max classes).
        use_direction_ = use_direction;
        max_possible_classes_ = max_possible_classes;

        if (min_scaled_valid_mask_val_ > 255) {
            min_scaled_valid_mask_val_ = 255U;
        }
        // Number of bit channels to expect.
        bit_channels_ =
            (static_cast<int32_t>(dnn_channels) - encoding_param_.getBitStartPosition());
        bit_channels_ = bit_channels_ < 0 ? 0 : bit_channels_;
        if (max_possible_classes_ > 0) {
            bool channels_mismatch =
                bit_channels_ !=
                static_cast<int32_t>(std::ceil(log2(static_cast<float>(max_possible_classes_))));
            if (channels_mismatch) {
                throw std::runtime_error(
                    "LRNDecoderBase: max_possible_classes and bit_channels_ mismatch");
            }
        }
        computeGaussianCoefficient();
    }

    /**
     * @brief Computes gaussian coefficients.
     */
    void computeGaussianCoefficient() {
        float sigma = 0.3f * ((non_max_radius_ * 2.0f) * 0.5f - 1.0f) + 0.8f;
        float coeff_sum = 0;

        for (int32_t r = -non_max_radius_; r <= non_max_radius_; r++) {
            float dr = 0.5f * static_cast<float>(r) / sigma;
            float coeff = std::exp(-dr * dr);
            coeff_sum += coeff;
            gaussian_coeff_[r + non_max_radius_] = coeff;
        }
        for (int32_t r = -non_max_radius_; r <= non_max_radius_; r++) {
            gaussian_coeff_[r + non_max_radius_] /= coeff_sum;
        }
    }
    void reset() {}
    int32_t get_input_width() { return input_image_width_; }
    int32_t get_input_height() { return input_image_height_; }
    int32_t get_net_width() { return dnn_width_; }
    int32_t get_net_height() { return dnn_height_; }
    int32_t get_net_channels() { return dnn_channels_; }
    int32_t get_total_nodes() { return host_nodes_count_; }

 protected:
    float gaussian_coeff_[MAX_GAUSSIAN_FILTER_RADIUS * 2 + 1];
    int32_t dnn_width_;
    int32_t dnn_height_;
    int32_t dnn_channels_;
    int32_t input_image_width_;
    int32_t input_image_height_;

    // DNN training parameter that defines effective radius of distance transform.
    int8_t dnn_radius_;
    int8_t non_max_radius_;
    uint16_t max_possible_nodes_;
    uint16_t min_votes_for_nodes_;
    uint16_t max_dist_for_nodes_;
    // DNN training parameter that defines define infinity.
    int16_t defined_infinity_;
    bool normalize_;
    // DNN training parameter that defines down sampling scale must be integer.
    int16_t scale_used_for_encoding_;
    float min_valid_mask_val_;
    uint16_t min_scaled_valid_mask_val_;
    int32_t bit_channels_;
    bool use_direction_;
    int32_t max_possible_classes_;
    int32_t host_nodes_count_;
    ColorList color_list_;
    LRNEncodingParam encoding_param_;
    uint32_t lrn_bit_start_position_;
    uint32_t lrn_cos_start_position_;
};

template <typename T>
CUDA_HOSTDEV inline void clamp(const T minval, const T maxval, T* val) {
    if (*val < minval) {
        *val = minval;
    }
    if (*val > maxval) {
        *val = maxval;
    }
}

template <typename T, typename DST_TYPE>
CUDA_HOSTDEV inline DST_TYPE clampMax(const T val, const T maxval) {
    T ret = val;
    if (ret > maxval) {
        ret = maxval;
    }
    return static_cast<DST_TYPE>(ret);
}

template <typename DNN_BLOB_TYPE>
CUDA_HOSTDEV inline uint32_t bitsToIntegers(const DNN_BLOB_TYPE* dnn_output, const int32_t x,
                                            const int32_t y, const int32_t width,
                                            const int32_t height, const uint32_t bit_channels,
                                            const int8_t bit_start_pos) {
    uint32_t finalClass = 0U;
    for (uint32_t bit = 0; bit < bit_channels; bit++) {
        DNN_BLOB_TYPE v = dnn_output[height * width * (bit_start_pos + bit) + y * width + x];
        if (v >= 0.5f) {
            finalClass |= (1U << bit);
        }
    }
    return finalClass;
}

}  // namespace lineregressordecoder
#endif  // LRNDECODER_H_
