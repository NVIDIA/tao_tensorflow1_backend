// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#include "lrn_decoder_core.h"
#include "lrn_decoder_core_cpu.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace lineregressordecoder {

template <typename DNN_BLOB_TYPE>
LRNDecoderCPU<DNN_BLOB_TYPE>::LRNDecoderCPU(
    // ---------spec------------------------
    uint32_t dnn_width, uint32_t dnn_height, uint32_t dnn_channels, uint32_t input_image_width,
    uint32_t input_image_height, int16_t scale_used_for_encoding, int16_t defined_infinity,
    bool normalize, int8_t dnn_radius, LRNEncodingParam encoding_param,
    // --------algorithm parameters------------
    int8_t non_max_radius, uint16_t max_possible_nodes, float min_valid_mask_val,
    uint16_t min_votes_for_nodes, uint16_t max_dist_for_nodes,
    // --------algorithm options-----------
    bool use_direction, int32_t max_possible_classes)
    : LRNDecoderBase(dnn_width, dnn_height, dnn_channels, input_image_width, input_image_height,
                     scale_used_for_encoding, defined_infinity, normalize, dnn_radius,
                     encoding_param, non_max_radius, max_possible_nodes, min_valid_mask_val,
                     min_votes_for_nodes, max_dist_for_nodes, use_direction, max_possible_classes) {
    host_votes_.reset(new VoteType[input_image_width * input_image_height]);

    gaussian_votes_.reset(new VoteLocalMaxType[input_image_width * input_image_height]);
    gaussian_votes_tmp_.reset(new BlurTypeForLRN[input_image_width * input_image_height]);

    // This will be used to select prediction. It better be blurred with Gaussian.
    host_mask_.reset(new uint8_t[dnn_width * dnn_height]);

    host_dist_.reset(new uint8_t[dnn_width * dnn_height]);

    host_dx_.reset(new int16_t[dnn_width * dnn_height]);
    host_dy_.reset(new int16_t[dnn_width * dnn_height]);

    if (encoding_param_.getLeftWidthStartPosition() != LRNEncodingParam::INVALID) {
        host_left_width_.reset(new uint16_t[dnn_width * dnn_height]);
        host_right_width_.reset(new uint16_t[dnn_width * dnn_height]);
        host_left_width_sum_.reset(new uint32_t[input_image_width * input_image_height]);
        host_right_width_sum_.reset(new uint32_t[input_image_width * input_image_height]);
    }
    if (use_direction == true) {
        host_cos_.reset(new DNN_BLOB_TYPE[dnn_width * dnn_height]);
        host_sin_.reset(new DNN_BLOB_TYPE[dnn_width * dnn_height]);
        host_avg_direction_.reset(new int16_t[input_image_width * input_image_height]);
        host_cos_votes_.reset(new float[input_image_width * input_image_height]);
        host_sin_votes_.reset(new float[input_image_width * input_image_height]);
    }
    if (max_possible_classes > 0) {
        host_class_.reset(new int8_t[dnn_width * dnn_height]);
        host_votes_tmp_.reset(
            new VoteLocalMaxType[input_image_width * input_image_height * max_possible_classes]);
        for (int32_t i = 0; i < max_possible_classes; i++) {
            host_class_votes_[i].reset(new ClassVoteType[input_image_width * input_image_height]);
        }
    }
    host_nodes_.reset(new LRNNode[max_possible_nodes]);
}

template <typename DNN_BLOB_TYPE>
LRNDecoderCPU<DNN_BLOB_TYPE>::LRNDecoderCPU(
    // ---------spec------------------
    uint32_t dnn_width, uint32_t dnn_height, uint32_t dnn_channels, uint32_t input_image_width,
    uint32_t input_image_height, int16_t scale_used_for_encoding, int16_t defined_infinity,
    bool normalize, int8_t dnn_radius, LRNEncodingParam encoding_param,
    // --------algorithm parameters-------
    int8_t non_max_radius, uint16_t max_possible_nodes, float min_valid_mask_val,
    uint16_t min_votes_for_nodes, uint16_t max_dist_for_nodes)
    : LRNDecoderCPU(dnn_width, dnn_height, dnn_channels, input_image_width, input_image_height,
                    scale_used_for_encoding, defined_infinity, normalize, dnn_radius,
                    encoding_param, non_max_radius, max_possible_nodes, min_valid_mask_val,
                    min_votes_for_nodes, max_dist_for_nodes, false, 0) {}

template <typename DNN_BLOB_TYPE>
void LRNDecoderCPU<DNN_BLOB_TYPE>::initVote() {
    std::memset(host_votes_.get(), 0, sizeof(VoteType) * input_image_height_ * input_image_width_);

    std::memset(host_avg_direction_.get(), 0,
                sizeof(int16_t) * input_image_height_ * input_image_width_);
    for (int32_t j = 0; j < max_possible_classes_; j++) {
        std::memset(host_class_votes_[j].get(), 0,
                    sizeof(ClassVoteType) * input_image_height_ * input_image_width_);
    }
    std::memset(host_cos_votes_.get(), 0, sizeof(float) * input_image_height_ * input_image_width_);
    std::memset(host_sin_votes_.get(), 0, sizeof(float) * input_image_height_ * input_image_width_);

    std::memset(host_left_width_sum_.get(), 0,
                sizeof(uint32_t) * input_image_height_ * input_image_width_);
    std::memset(host_right_width_sum_.get(), 0,
                sizeof(uint32_t) * input_image_height_ * input_image_width_);
}

template <typename DNN_BLOB_TYPE>
void LRNDecoderCPU<DNN_BLOB_TYPE>::drawLine(int* input, const int x, const int y, const int cols,
                                            const int rows, const int16_t angle,
                                            const int arrow_length, const int line_thickness,
                                            const int r, const int g, const int b) {
    if (arrow_length > 0 && x > 0 && x < cols && y > 0 && y < rows) {
        float x2 = x + arrow_length * cos(angle * LRN_DEG_TO_RADIAN + M_PI);
        float y2 = y + arrow_length * sin(angle * LRN_DEG_TO_RADIAN + M_PI);
        draw_line(input, rows, cols, x, y, x2, y2, line_thickness, r, g, b);
    }
}

template <typename DNN_BLOB_TYPE>
void LRNDecoderCPU<DNN_BLOB_TYPE>::getOutput(int* output, int* output_with_angle,
                                             int* output_with_nonmax, int* output_for_metric,
                                             const float* input_img, const int rows, const int cols,
                                             const int arrow_length, const int line_thickness,
                                             const int background_class_id) {
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            output_for_metric[y * cols + x] = 0;
            int img_r = static_cast<int>(std::min(255.0f, input_img[y * cols + x] * 255));
            int img_g =
                static_cast<int>(std::min(255.0f, input_img[y * cols + x + rows * cols] * 255));
            int img_b =
                static_cast<int>(std::min(255.0f, input_img[y * cols + x + rows * cols * 2] * 255));
            output_with_nonmax[y * cols * 3 + 3 * x] = img_r;
            output_with_nonmax[y * cols * 3 + 3 * x + 1] = img_g;
            output_with_nonmax[y * cols * 3 + 3 * x + 2] = img_b;
            output[y * cols * 3 + 3 * x] = img_r;
            output[y * cols * 3 + 3 * x + 1] = img_g;
            output[y * cols * 3 + 3 * x + 2] = img_b;
            output_with_angle[y * cols * 3 + 3 * x] = img_r;
            output_with_angle[y * cols * 3 + 3 * x + 1] = img_g;
            output_with_angle[y * cols * 3 + 3 * x + 2] = img_b;
        }
    }

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            int votes = static_cast<int>(host_votes_[y * cols + x]);
            if (votes >= min_votes_for_nodes_) {
                uint32_t left_sum = host_left_width_sum_[y * cols + x];
                uint32_t right_sum = host_right_width_sum_[y * cols + x];
                int16_t angle = host_avg_direction_.get()[y * cols + x];

                uint8_t class_id = static_cast<uint8_t>(host_class_votes_[0][y * cols + x]);
                class_id = class_id < 20 ? class_id : 20;
                int r = color_list_.get(class_id).r_;
                int g = color_list_.get(class_id).g_;
                int b = color_list_.get(class_id).b_;
                output[y * cols * 3 + 3 * x] = r;
                output[y * cols * 3 + 3 * x + 1] = g;
                output[y * cols * 3 + 3 * x + 2] = b;
                output_with_angle[y * cols * 3 + 3 * x] = r;
                output_with_angle[y * cols * 3 + 3 * x + 1] = g;
                output_with_angle[y * cols * 3 + 3 * x + 2] = b;
                drawLine(output_with_angle, x, y, cols, rows, angle, arrow_length, line_thickness,
                         r, g, b);
                // output left edges.
                uint32_t left_edge_x = x - left_sum * cos(angle * LRN_DEG_TO_RADIAN - M_PI_2);
                uint32_t left_edge_y = y - left_sum * sin(angle * LRN_DEG_TO_RADIAN - M_PI_2);
                if (left_sum > 0 && left_edge_x > 0 && static_cast<int>(left_edge_x) < cols &&
                    left_edge_y > 0 && static_cast<int>(left_edge_y) < rows) {
                    output[left_edge_y * cols * 3 + 3 * left_edge_x] = std::max(0, r - 50);
                    output[left_edge_y * cols * 3 + 3 * left_edge_x + 1] = std::max(0, g - 50);
                    output[left_edge_y * cols * 3 + 3 * left_edge_x + 2] = std::max(0, b - 50);
                    drawLine(output_with_angle, left_edge_x, left_edge_y, cols, rows, angle,
                             arrow_length, line_thickness, std::max(0, r - 50), std::max(0, g - 50),
                             std::max(0, b - 50));
                    if (class_id >= background_class_id) {
                        output_for_metric[left_edge_y * cols + left_edge_x] =
                            class_id - background_class_id;
                    } else {
                        output_for_metric[left_edge_y * cols + left_edge_x] = class_id;
                    }
                }
                // output right edges.
                uint32_t right_edge_x = x + right_sum * cos(angle * LRN_DEG_TO_RADIAN - M_PI_2);
                uint32_t right_edge_y = y + right_sum * sin(angle * LRN_DEG_TO_RADIAN - M_PI_2);
                if (right_sum > 0 && right_edge_x > 0 && static_cast<int>(right_edge_x) < cols &&
                    right_edge_y > 0 && static_cast<int>(right_edge_y) < rows) {
                    output[right_edge_y * cols * 3 + 3 * right_edge_x] = std::max(0, r - 50);
                    output[right_edge_y * cols * 3 + 3 * right_edge_x + 1] = std::max(0, g - 50);
                    output[right_edge_y * cols * 3 + 3 * right_edge_x + 2] = std::max(0, b - 50);
                    drawLine(output_with_angle, right_edge_x, right_edge_y, cols, rows, angle,
                             arrow_length, line_thickness, std::max(0, r - 50), std::max(0, g - 50),
                             std::max(0, b - 50));

                    if (class_id >= background_class_id) {
                        output_for_metric[right_edge_y * cols + right_edge_x] =
                            class_id - background_class_id;
                    } else {
                        output_for_metric[right_edge_y * cols + right_edge_x] = class_id;
                    }
                }
                if (class_id >= background_class_id) {
                    output_for_metric[y * cols + x] = class_id - background_class_id;
                } else {
                    output_for_metric[y * cols + x] = class_id;
                }
            }
        }
    }
    if (arrow_length > 0) {
        for (int i = 0; i < host_nodes_count_; i++) {
            int x = host_nodes_[i].x;
            int y = host_nodes_[i].y;
            int angle = host_nodes_[i].angle;
            int class_id = host_nodes_[i].decoded_bit_code;
            float x2 = x + arrow_length * cos(angle * LRN_DEG_TO_RADIAN + M_PI);
            float y2 = y + arrow_length * sin(angle * LRN_DEG_TO_RADIAN + M_PI);
            draw_line(output_with_nonmax, rows, cols, x, y, x2, y2, line_thickness,
                      color_list_.get(class_id));
        }
    }
    // radius of output.
    const int radius = 1;
    for (int i = 0; i < host_nodes_count_; i++) {
        int x = host_nodes_[i].x;
        int y = host_nodes_[i].y;
        int class_id = host_nodes_[i].decoded_bit_code;
        int angle = host_nodes_[i].angle;
        uint8_t left_width = host_nodes_[i].edge_left_width;
        uint8_t right_width = host_nodes_[i].edge_right_width;
        int left_x = x - left_width * cos(angle * LRN_DEG_TO_RADIAN - M_PI_2);
        int left_y = y - left_width * sin(angle * LRN_DEG_TO_RADIAN - M_PI_2);
        int right_x = x + right_width * cos(angle * LRN_DEG_TO_RADIAN - M_PI_2);
        int right_y = y + right_width * sin(angle * LRN_DEG_TO_RADIAN - M_PI_2);
        int r = color_list_.get(class_id).r_;
        int g = color_list_.get(class_id).g_;
        int b = color_list_.get(class_id).b_;

        drawLine(output_with_nonmax, x, y, cols, rows, angle, arrow_length, line_thickness, r, g,
                 b);
        if (left_width > 0) {
            drawLine(output_with_nonmax, left_x, left_y, cols, rows, angle, arrow_length,
                     line_thickness, r, g, b);
        }
        if (right_width > 0) {
            drawLine(output_with_nonmax, right_x, right_y, cols, rows, angle, arrow_length,
                     line_thickness, r, g, b);
        }
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                float dist = dx * dx + dy * dy;
                if (dist <= radius * radius) {
                    int yp = y + dy;
                    int xp = x + dx;
                    if (xp >= 0 && xp < cols && yp >= 0 && yp < rows) {
                        output_with_nonmax[yp * cols * 3 + 3 * xp] =
                            255 - color_list_.get(class_id).r_;
                        output_with_nonmax[yp * cols * 3 + 3 * xp + 1] =
                            255 - color_list_.get(class_id).g_;
                        output_with_nonmax[yp * cols * 3 + 3 * xp + 2] =
                            255 - color_list_.get(class_id).b_;
                    }
                }
            }
        }
    }
}

template <typename DNN_BLOB_TYPE>
void LRNDecoderCPU<DNN_BLOB_TYPE>::process(const DNN_BLOB_TYPE* host_from_dnn) {
    decode(host_from_dnn);
    vote();
    extractNodes();
}

template <typename DNN_BLOB_TYPE>
void LRNDecoderCPU<DNN_BLOB_TYPE>::decode(const DNN_BLOB_TYPE* host_from_dnn) {
    decodeCore(host_from_dnn);
}

template <typename DNN_BLOB_TYPE>
void LRNDecoderCPU<DNN_BLOB_TYPE>::vote() {
    initVote();
    doVotes();
    computeDirectionAndArgMax();
}

template <typename DNN_BLOB_TYPE>
VoteLocalMaxType LRNDecoderCPU<DNN_BLOB_TYPE>::getMaxVal(const int32_t x, const int32_t y) {
    uint32_t current_class_id = host_class_votes_[0][y * input_image_width_ + x];
    VoteLocalMaxType max_val = 0;
    for (int32_t dy = -non_max_radius_; dy <= non_max_radius_; dy++) {
        int32_t newy = y + dy;
        if (newy >= 0 && newy < input_image_height_) {
            if (current_class_id < static_cast<uint32_t>(max_possible_classes_)) {
                VoteLocalMaxType votes =
                    host_votes_tmp_[current_class_id * input_image_height_ * input_image_width_ +
                                    newy * input_image_width_ + x];
                if (max_val < votes) {
                    max_val = votes;
                }
            }
        }
    }
    return max_val;
}

template <typename DNN_BLOB_TYPE>
void LRNDecoderCPU<DNN_BLOB_TYPE>::extractNodes() {
    host_nodes_count_ = 0;
    computeMaxHorizontally();
    for (int32_t y = 0; y < input_image_height_; y++) {
        for (int32_t x = 0; x < input_image_width_; x++) {
            VoteLocalMaxType max_val = getMaxVal(x, y);
            VoteType raw_votes = host_votes_[y * input_image_width_ + x];
            VoteLocalMaxType g_votes = gaussian_votes_[y * input_image_width_ + x];
            if (g_votes == max_val && raw_votes >= min_votes_for_nodes_ && max_val > 0) {
                if (host_nodes_count_ < max_possible_nodes_) {
                    setHostNodes(host_nodes_count_, x, y);
                    host_nodes_count_++;
                }
            }
        }
    }
}

template <typename DNN_BLOB_TYPE>
void LRNDecoderCPU<DNN_BLOB_TYPE>::doVotes() {
    int32_t effective_radius = scale_used_for_encoding_ * dnn_radius_;
    for (int32_t y = 0; y < dnn_height_; y++) {
        for (int32_t x = 0; x < dnn_width_; x++) {
            int32_t original_x = -1;
            int32_t original_y = -1;
            if (computeOriginalCoord(&original_x, &original_y, effective_radius, x, y)) {
                doVotesCore(original_x, original_y, x, y);
            }
        }
    }
    // blur horizontally
    for (int32_t y = 0; y < input_image_height_; y++) {
        for (int32_t x = 0; x < input_image_width_; x++) {
            float sum = 0.0f;
            for (int32_t dx = -non_max_radius_; dx <= non_max_radius_; dx++) {
                int32_t newx = std::max(0, std::min(input_image_width_ - 1, x + dx));
                sum += host_votes_[y * input_image_width_ + newx] *
                       gaussian_coeff_[dx + non_max_radius_];
            }
            gaussian_votes_tmp_[y * input_image_width_ + x] = sum;
        }
    }
    // blur vertically later
    for (int32_t y = 0; y < input_image_height_; y++) {
        for (int32_t x = 0; x < input_image_width_; x++) {
            float sum = 0.0f;
            for (int32_t dy = -non_max_radius_; dy <= non_max_radius_; dy++) {
                int32_t newy = std::max(0, std::min(input_image_height_ - 1, y + dy));
                sum += gaussian_votes_tmp_[newy * input_image_width_ + x] *
                       gaussian_coeff_[dy + non_max_radius_];
            }
            gaussian_votes_[y * input_image_width_ + x] =
                std::min(static_cast<VoteLocalMaxType>(MAX_GAUSSIAN_VOTE),
                         static_cast<VoteLocalMaxType>(0.5f + sum * 50.0f));
        }
    }
}

template <typename DNN_BLOB_TYPE>
bool LRNDecoderCPU<DNN_BLOB_TYPE>::computeOriginalCoord(int32_t* original_x, int32_t* original_y,
                                                        const int32_t effective_radius,
                                                        const int32_t x, const int32_t y) {
    bool result = true;
    float vecx = host_dx_[y * dnn_width_ + x];
    float vecy = host_dy_[y * dnn_width_ + x];
    if (vecx < -effective_radius || vecx > effective_radius || vecy < -effective_radius ||
        vecy > effective_radius) {
        return false;
    }
    *original_x = scale_used_for_encoding_ * x + vecx + 0.5f;
    *original_y = scale_used_for_encoding_ * y + vecy + 0.5f;
    if (*original_x < 0 || *original_x >= input_image_width_ || *original_y < 0 ||
        *original_y >= input_image_height_) {
        result = false;
    }
    return result;
}

template <typename DNN_BLOB_TYPE>
void LRNDecoderCPU<DNN_BLOB_TYPE>::doVotesCore(int32_t original_x, int32_t original_y,
                                               const int32_t x, const int32_t y, const int32_t R) {
    for (int32_t dx = -R; dx <= R; dx++) {
        int32_t nx = original_x + dx;
        for (int32_t dy = -R; dy <= R; dy++) {
            int32_t ny = original_y + dy;
            if (nx >= 0 && nx < input_image_width_ && ny >= 0 && ny < input_image_height_) {
                host_votes_[ny * input_image_width_ + nx]++;
            }
        }
    }
    host_left_width_sum_[original_y * input_image_width_ + original_x] +=
        host_left_width_[y * dnn_width_ + x];
    host_right_width_sum_[original_y * input_image_width_ + original_x] +=
        host_right_width_[y * dnn_width_ + x];

    host_cos_votes_[original_y * input_image_width_ + original_x] += host_cos_[y * dnn_width_ + x];
    host_sin_votes_[original_y * input_image_width_ + original_x] += host_sin_[y * dnn_width_ + x];
    int8_t class_id = host_class_[y * dnn_width_ + x];
    if (class_id >= 0 && class_id < max_possible_classes_) {
        host_class_votes_[class_id][original_y * input_image_width_ + original_x]++;
    }
}

template <typename DNN_BLOB_TYPE>
void LRNDecoderCPU<DNN_BLOB_TYPE>::computeDirectionAndArgMax() {
    for (int32_t y = 0; y < input_image_height_; y++) {
        for (int32_t x = 0; x < input_image_width_; x++) {
            ClassVoteType max_count = 0;
            int32_t max_class = -1;
            for (int32_t c = 0; c < max_possible_classes_; c++) {
                ClassVoteType val = host_class_votes_[c][y * input_image_width_ + x];
                if (val > max_count) {
                    max_count = val;
                    max_class = c;
                }
            }
            host_class_votes_[0][y * input_image_width_ + x] = max_class;
            float cos_val = host_cos_votes_[y * input_image_width_ + x];
            float sin_val = host_sin_votes_[y * input_image_width_ + x];
            int16_t angle =
                static_cast<int16_t>(0.5f + atan2f(sin_val, cos_val) * LRN_RADIAN_TO_DEG);
            host_avg_direction_[y * input_image_width_ + x] = angle;
            if (host_left_width_sum_ != nullptr && host_right_width_sum_ != nullptr) {
                uint32_t total_votes = host_votes_[y * input_image_width_ + x];
                float left_width = 0.0f;
                float right_width = 0.0f;
                if (total_votes > 0) {
                    left_width =
                        0.5f + host_left_width_sum_[y * input_image_width_ + x] / total_votes;
                    right_width =
                        0.5f + host_right_width_sum_[y * input_image_width_ + x] / total_votes;
                }
                host_left_width_sum_[y * input_image_width_ + x] =
                    static_cast<uint32_t>(left_width);
                host_right_width_sum_[y * input_image_width_ + x] =
                    static_cast<uint32_t>(right_width);
            }
        }
    }
}

//  We compute horinzontal max within windows per each class.
template <typename DNN_BLOB_TYPE>
void LRNDecoderCPU<DNN_BLOB_TYPE>::computeMaxValFromHorizontalScan(VoteLocalMaxType* max_vals,
                                                                   const int32_t x,
                                                                   const int32_t y) {
    for (int8_t c = 0; c < max_possible_classes_; c++) {
        max_vals[c] = 0;
    }
    for (int32_t dx = -non_max_radius_; dx <= non_max_radius_; dx++) {
        int32_t newx = x + dx;
        if (newx >= 0 && newx < input_image_width_) {
            int8_t cid = host_class_votes_[0][y * input_image_width_ + newx];

            if (cid >= 0 && cid < max_possible_classes_ &&
                gaussian_votes_[y * input_image_width_ + newx] > max_vals[cid]) {
                max_vals[cid] = gaussian_votes_[y * input_image_width_ + newx];
            }
        }
    }
}

// Perform non-max suppression horizontally.
template <typename DNN_BLOB_TYPE>
void LRNDecoderCPU<DNN_BLOB_TYPE>::computeMaxHorizontally() {
    VoteLocalMaxType max_vals[MAX_CLASSES];
    for (int32_t y = 0; y < input_image_height_; y++) {
        for (int32_t x = 0; x < input_image_width_; x++) {
            computeMaxValFromHorizontalScan(max_vals, x, y);
            for (int8_t c = 0; c < max_possible_classes_; c++) {
                host_votes_tmp_[c * input_image_height_ * input_image_width_ +
                                y * input_image_width_ + x] = max_vals[c];
            }
        }
    }
}

template <typename DNN_BLOB_TYPE>
void LRNDecoderCPU<DNN_BLOB_TYPE>::setHostNodes(int32_t node, int32_t x, int32_t y) {
    host_nodes_[node].x = x;
    host_nodes_[node].y = y;
    host_nodes_[node].normalized_votes =
        clampMax<uint16_t, VoteType>(gaussian_votes_[y * input_image_width_ + x], 255U);
    host_nodes_[node].edge_left_width = 0;
    host_nodes_[node].edge_right_width = 0;
    if (host_avg_direction_.get() != nullptr) {
        host_nodes_[node].angle = host_avg_direction_[y * input_image_width_ + x];
    } else {
        host_nodes_[node].angle = INVALID_VALUE;
    }
    if (host_class_votes_ != nullptr) {
        host_nodes_[node].decoded_bit_code = host_class_votes_[0][y * input_image_width_ + x];
    } else {
        host_nodes_[node].decoded_bit_code = INVALID_VALUE;
    }
    if (host_left_width_sum_ != nullptr) {
        host_nodes_[node].edge_left_width = host_left_width_sum_[y * input_image_width_ + x];
    }
    if (host_right_width_sum_ != nullptr) {
        host_nodes_[node].edge_right_width = host_right_width_sum_[y * input_image_width_ + x];
    }
}

template <typename DNN_BLOB_TYPE>
void LRNDecoderCPU<DNN_BLOB_TYPE>::decodeFromDNN(const DNN_BLOB_TYPE* host_from_dnn,
                                                 const int32_t x, const int32_t y) {
    float conf = host_from_dnn[y * dnn_width_ + x];
    float vx = defined_infinity_;
    float vy = defined_infinity_;
    float cos_val = -2.0f;
    float sine_val = -2.0f;
    uint8_t dist_val = defined_infinity_;
    uint8_t conf_v = 0U;
    int8_t class_id = MAX_CLASSES;
    float left_width = 0.0f;
    float right_width = 0.0f;
    if (conf > min_valid_mask_val_) {
        vx = host_from_dnn[y * dnn_width_ + x +
                           dnn_width_ * dnn_height_ * encoding_param_.getDXPosition()];
        vy = host_from_dnn[y * dnn_width_ + x +
                           dnn_width_ * dnn_height_ * encoding_param_.getDYPosition()];
        float nvx = host_from_dnn[y * dnn_width_ + x +
                                  dnn_width_ * dnn_height_ * encoding_param_.getNDXPosition()];
        float nvy = host_from_dnn[y * dnn_width_ + x +
                                  dnn_width_ * dnn_height_ * encoding_param_.getNDYPosition()];
        vx = vx < nvx ? -nvx : vx;
        vy = vy < nvy ? -nvy : vy;
        float dist = std::sqrt(vx * vx + vy * vy);
        uint16_t u_tmp = static_cast<uint16_t>(conf * 255.0f + 0.5f);
        conf_v = clampMax<uint16_t, uint8_t>(u_tmp, static_cast<uint16_t>(255U));
        u_tmp = static_cast<uint16_t>(dist + 0.5f);
        dist_val = clampMax<uint16_t, uint8_t>(u_tmp, static_cast<uint16_t>(255U));

        if (host_cos_.get() != nullptr) {
            float v = host_from_dnn[dnn_height_ * dnn_width_ * encoding_param_.getDirCosPosition() +
                                    y * dnn_width_ + x];
            v = v * 2.0f - 1.0f;
            clamp<float>(-1.0f, 1.0f, &v);
            cos_val = v;
        }
        if (host_sin_.get() != nullptr) {
            float v = host_from_dnn[dnn_height_ * dnn_width_ * encoding_param_.getDirSinPosition() +
                                    y * dnn_width_ + x];
            v = v * 2.0f - 1.0f;
            clamp<float>(-1.0f, 1.0f, &v);
            sine_val = v;
        }
        if (host_class_.get() != nullptr) {
            uint32_t int_class =
                bitsToIntegers(host_from_dnn, x, y, dnn_width_, dnn_height_, bit_channels_,
                               encoding_param_.getBitStartPosition());
            class_id = clampMax<uint32_t, int8_t>(int_class, static_cast<int8_t>(MAX_CLASSES));
        }
    }
    host_mask_[y * dnn_width_ + x] = conf_v;
    if (normalize_) {
        dist_val *= defined_infinity_;
        vx *= defined_infinity_;
        vy *= defined_infinity_;
    }
    host_dist_[y * dnn_width_ + x] = dist_val;
    host_dx_[y * dnn_width_ + x] = static_cast<int16_t>(0.5f + vx * scale_used_for_encoding_);
    host_dy_[y * dnn_width_ + x] = static_cast<int16_t>(0.5f + vy * scale_used_for_encoding_);

    if (host_cos_.get() != nullptr) {
        host_cos_[y * dnn_width_ + x] = cos_val;
    }
    if (host_sin_.get() != nullptr) {
        host_sin_[y * dnn_width_ + x] = sine_val;
    }
    if (host_class_.get() != nullptr) {
        host_class_[y * dnn_width_ + x] = class_id;
    }
    if (host_left_width_ != nullptr) {
        DNN_BLOB_TYPE tmp =
            host_from_dnn[dnn_height_ * dnn_width_ * encoding_param_.getLeftWidthStartPosition() +
                          y * dnn_width_ + x];
        if (normalize_) {
            left_width = tmp * defined_infinity_;
        }
        host_left_width_[y * dnn_width_ + x] =
            static_cast<uint16_t>(0.5f + left_width * scale_used_for_encoding_);
    }
    if (host_right_width_ != nullptr) {
        DNN_BLOB_TYPE tmp =
            host_from_dnn[dnn_height_ * dnn_width_ * encoding_param_.getRightWidthStartPosition() +
                          y * dnn_width_ + x];
        if (normalize_) {
            right_width = tmp * defined_infinity_;
        }
        host_right_width_[y * dnn_width_ + x] =
            static_cast<uint16_t>(0.5f + right_width * scale_used_for_encoding_);
    }
}

template <typename DNN_BLOB_TYPE>
void LRNDecoderCPU<DNN_BLOB_TYPE>::decodeFromDNNDist2D(const DNN_BLOB_TYPE* host_from_dnn,
                                                       const int32_t x, const int32_t y) {
    float conf = host_from_dnn[y * dnn_width_ + x];
    float vx = defined_infinity_;
    float vy = defined_infinity_;
    float dist = defined_infinity_;
    uint8_t dist_val = defined_infinity_;
    float cos_val = -2.0f;
    float sine_val = -2.0f;
    uint8_t conf_v = 0U;
    int8_t class_id = MAX_CLASSES;
    float left_width = 0.0f;
    float right_width = 0.0f;
    if (conf > min_valid_mask_val_) {
        dist = host_from_dnn[y * dnn_width_ + x +
                             dnn_width_ * dnn_height_ * encoding_param_.getMagnitudePosition()];
        float normal_cos =
            host_from_dnn[y * dnn_width_ + x +
                          dnn_width_ * dnn_height_ * encoding_param_.getNormalCosPosition()];
        normal_cos = normal_cos * 2.0f - 1.0f;
        float normal_sine =
            host_from_dnn[y * dnn_width_ + x +
                          dnn_width_ * dnn_height_ * encoding_param_.getNormalSinePosition()];
        normal_sine = normal_sine * 2.0f - 1.0f;
        clamp(-1.0f, 1.0f, &normal_cos);
        clamp(-1.0f, 1.0f, &normal_sine);
        vx = dist * normal_cos;
        vy = dist * normal_sine;

        uint16_t u_tmp = static_cast<uint16_t>(conf * 255.0f + 0.5f);
        conf_v = clampMax<uint16_t, uint8_t>(u_tmp, static_cast<uint16_t>(255U));
        u_tmp = static_cast<uint16_t>(dist + 0.5f);
        dist_val = clampMax<uint16_t, uint8_t>(u_tmp, static_cast<uint16_t>(255U));
        if (host_cos_.get() != nullptr) {
            float v = host_from_dnn[dnn_height_ * dnn_width_ * encoding_param_.getDirCosPosition() +
                                    y * dnn_width_ + x];
            v = v * 2.0f - 1.0f;
            clamp(-1.0f, 1.0f, &v);
            cos_val = v;
        }
        if (host_sin_.get() != nullptr) {
            float v = host_from_dnn[dnn_height_ * dnn_width_ * encoding_param_.getDirSinPosition() +
                                    y * dnn_width_ + x];
            v = v * 2.0f - 1.0f;
            clamp(-1.0f, 1.0f, &v);
            sine_val = v;
        }
        if (host_class_.get() != nullptr) {
            uint32_t int_class =
                bitsToIntegers(host_from_dnn, x, y, dnn_width_, dnn_height_, bit_channels_,
                               encoding_param_.getBitStartPosition());
            class_id = clampMax<uint32_t, int8_t>(int_class, static_cast<int8_t>(MAX_CLASSES));
        }
    }
    host_mask_[y * dnn_width_ + x] = conf_v;
    if (normalize_) {
        dist_val *= defined_infinity_;
        vx *= defined_infinity_;
        vy *= defined_infinity_;
    }
    host_dist_[y * dnn_width_ + x] = dist_val;
    host_dx_[y * dnn_width_ + x] = static_cast<int16_t>(0.5f + vx * scale_used_for_encoding_);
    host_dy_[y * dnn_width_ + x] = static_cast<int16_t>(0.5f + vy * scale_used_for_encoding_);
    if (host_cos_.get() != nullptr) {
        host_cos_[y * dnn_width_ + x] = cos_val;
    }
    if (host_sin_.get() != nullptr) {
        host_sin_[y * dnn_width_ + x] = sine_val;
    }
    if (host_class_.get() != nullptr) {
        host_class_[y * dnn_width_ + x] = class_id;
    }
    if (host_left_width_ != nullptr) {
        DNN_BLOB_TYPE tmp =
            host_from_dnn[dnn_height_ * dnn_width_ * encoding_param_.getLeftWidthStartPosition() +
                          y * dnn_width_ + x];
        if (normalize_) {
            left_width = tmp * defined_infinity_;
        }
        host_left_width_[y * dnn_width_ + x] =
            static_cast<uint16_t>(0.5f + left_width * scale_used_for_encoding_);
    }
    if (host_right_width_ != nullptr) {
        DNN_BLOB_TYPE tmp =
            host_from_dnn[dnn_height_ * dnn_width_ * encoding_param_.getRightWidthStartPosition() +
                          y * dnn_width_ + x];
        if (normalize_) {
            right_width = tmp * defined_infinity_;
        }
        host_right_width_[y * dnn_width_ + x] =
            static_cast<uint16_t>(0.5f + right_width * scale_used_for_encoding_);
    }
}

template <typename DNN_BLOB_TYPE>
void LRNDecoderCPU<DNN_BLOB_TYPE>::decodeCore(const DNN_BLOB_TYPE* host_from_dnn) {
    for (int32_t y = 0; y < dnn_height_; y++) {
        for (int32_t x = 0; x < dnn_width_; x++) {
            if (encoding_param_.isEncoderDist2D() == false) {
                decodeFromDNN(host_from_dnn, x, y);
            } else {
                decodeFromDNNDist2D(host_from_dnn, x, y);
            }
        }
    }
}

template class LRNDecoderCPU<float>;
}  // namespace lineregressordecoder
