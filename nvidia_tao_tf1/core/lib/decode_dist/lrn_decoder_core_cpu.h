// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

#ifndef LRN_DECODER_CORE_CPU_H_
#define LRN_DECODER_CORE_CPU_H_

#include "lrn_decoder_core.h"

namespace lineregressordecoder {

template <typename DNN_BLOB_TYPE>
class LRNDecoderCPU : public LRNDecoderBase {
 public:
    //
    // @brief Construct a new LRNDecoderCPU object
    //
    // ---------spec---------------
    // @param dnn_width
    // @param dnn_height
    // @param dnn_channels
    // @param input_image_width
    // @param input_image_height
    // @param scale_used_for_encoding
    // @param defined_infinity
    // @param normalize
    // @param dnn_radius
    // @param encoding_param
    // --------algorithm parameters-----------------
    // @param non_max_radius
    // @param max_possible_nodes
    // @param min_valid_mask_val
    // @param min_votes_for_nodes
    // @param max_dist_for_nodes
    // --------algorithm options-----------------
    // @param use_direction
    // @param max_possible_classes
    //
    LRNDecoderCPU(
        // ---------spec-----------------
        uint32_t dnn_width, uint32_t dnn_height, uint32_t dnn_channels, uint32_t input_image_width,
        uint32_t input_image_height, int16_t scale_used_for_encoding, int16_t defined_infinity,
        bool normalize, int8_t dnn_radius, LRNEncodingParam encoding_param,
        // --------algorithm parameters------
        int8_t non_max_radius, uint16_t max_possible_nodes, float min_valid_mask_val,
        uint16_t min_votes_for_nodes, uint16_t max_dist_for_nodes,
        // --------algorithm options-----
        bool use_direction, int32_t max_possible_classes);
    //
    // @brief Construct a new LRNDecoderCPU object
    //
    // ---------spec---------------
    // @param dnn_width
    // @param dnn_height
    // @param dnn_channels
    // @param input_image_width
    // @param input_image_height
    // @param scale_used_for_encoding
    // @param defined_infinity
    // @param normalize
    // @param dnn_radius
    // @param encoding_param
    // --------algorithm parameters-----------------
    // @param non_max_radius
    // @param max_possible_nodes
    // @param min_valid_mask_val
    // @param min_votes_for_nodes
    // @param max_dist_for_nodes
    //
    LRNDecoderCPU(
        // ---------spec----------------------------
        uint32_t dnn_width, uint32_t dnn_height, uint32_t dnn_channels, uint32_t input_image_width,
        uint32_t input_image_height, int16_t scale_used_for_encoding, int16_t defined_infinity,
        bool normalize, int8_t dnn_radius, LRNEncodingParam encoding_param,
        // --------algorithm parameters-----------------
        int8_t non_max_radius, uint16_t max_possible_nodes, float min_valid_mask_val,
        uint16_t min_votes_for_nodes, uint16_t max_dist_for_nodes);

    void process(const DNN_BLOB_TYPE* host_from_dnn);
    void getOutput(int* output, int* output_with_angle, int* output_with_nonmax,
                   int* output_for_metric, const float* input_img, const int rows, const int cols,
                   const int arrow_length, const int line_thickness, const int background_class_id);

 protected:
    void initVote();
    void vote();
    VoteLocalMaxType getMaxVal(const int32_t x, const int32_t y);
    void extractNodes(int class_id);
    void extractNodes();
    void decode(const DNN_BLOB_TYPE* host_from_dnn);
    void decodeCore(const DNN_BLOB_TYPE* host_from_dnn);
    void decodeFromDNN(const DNN_BLOB_TYPE* host_from_dnn, const int32_t x, const int32_t y);
    void decodeFromDNNDist2D(const DNN_BLOB_TYPE* host_from_dnn, const int32_t x, const int32_t y);
    void doVotes();
    bool computeOriginalCoord(int32_t* original_x, int32_t* original_y,
                              const int32_t effective_radius, const int32_t x, const int32_t y);
    void doVotesCore(int32_t original_x, int32_t original_y, const int32_t x, const int32_t y,
                     const int32_t R = 0);
    void computeDirectionAndArgMax();
    void computeMaxValFromHorizontalScan(VoteLocalMaxType* max_vals, const int32_t x,
                                         const int32_t y);
    void computeMaxHorizontally();
    void setHostNodes(int32_t node, int32_t x, int32_t y);
    void verticalFilter();
    void gaussianBlur(const DNN_BLOB_TYPE* host_from_dnn);
    void setNMSVotesParams(const uint16_t min_votes_for_nodes);
    void drawLine(int* input, const int x, const int y, const int cols, const int rows,
                  const int16_t angle, const int arrow_length, const int line_thickness,
                  const int r, const int g, const int b);

    // This will be used to select prediction. This will be blurred with Gaussian.
    std::unique_ptr<uint8_t[]> host_mask_;

    // This will be used to evaluate fit proposals. It better be blurred with Gaussian.
    std::unique_ptr<uint8_t[]> host_dist_;

    // These are in the original dimension.
    std::unique_ptr<VoteLocalMaxType[]> gaussian_votes_;
    std::unique_ptr<BlurTypeForLRN[]> gaussian_votes_tmp_;
    std::unique_ptr<VoteType[]> host_votes_;
    std::unique_ptr<VoteLocalMaxType[]> host_votes_tmp_;
    std::unique_ptr<int16_t[]> host_avg_direction_;
    std::unique_ptr<ClassVoteType[]> host_class_votes_[MAX_CLASSES];
    std::unique_ptr<float[]> host_cos_votes_;
    std::unique_ptr<float[]> host_sin_votes_;
    std::unique_ptr<int16_t[]> host_dx_;
    std::unique_ptr<int16_t[]> host_dy_;
    std::unique_ptr<DNN_BLOB_TYPE[]> host_cos_;
    std::unique_ptr<DNN_BLOB_TYPE[]> host_sin_;
    std::unique_ptr<int8_t[]> host_class_;
    std::unique_ptr<LRNNode[]> host_nodes_;

    std::unique_ptr<uint16_t[]> host_left_width_;
    std::unique_ptr<uint16_t[]> host_right_width_;
    std::unique_ptr<uint32_t[]> host_left_width_sum_;
    std::unique_ptr<uint32_t[]> host_right_width_sum_;
};
}  // namespace lineregressordecoder
#endif  // LRN_DECODER_CORE_CPU_H_
