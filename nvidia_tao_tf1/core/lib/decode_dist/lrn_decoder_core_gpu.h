// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

#ifndef LRN_DECODER_CORE_GPU_H_
#define LRN_DECODER_CORE_GPU_H_

#include "lrn_decoder_core.h"
#include "tmp_tensor.h"

#include "tensorflow/core/framework/op_kernel.h"

namespace lineregressordecoder {
template <typename DNN_BLOB_TYPE>
class LRNDecoder : public LRNDecoderBase {
 public:
    //
    // @brief Construct a new LRNDecoder object
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
    // @param use_direction
    // @param max_possible_classes
    // @param min_valid_mask_val
    // @param context
    //
    LRNDecoder(
        // ---------spec---------------
        uint32_t dnn_width, uint32_t dnn_height, uint32_t dnn_channels, uint32_t input_image_width,
        uint32_t input_image_height, int16_t scale_used_for_encoding, int16_t defined_infinity,
        bool normalize, int8_t dnn_radius, LRNEncodingParam encoding_param,
        // --------algorithm parameters-----------------
        int8_t non_max_radius, uint16_t max_possible_nodes, float min_valid_mask_val,
        uint16_t min_votes_for_nodes, uint16_t max_dist_for_nodes,
        // --------algorithm options-----------------
        bool use_direction, int32_t max_possible_classes,
        // --------ETC-----------------
        tensorflow::OpKernelContext* context);
    //
    // @brief Construct a new LRNDecoder object
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
    // @param context
    //
    LRNDecoder(
        // ---------spec---------------
        uint32_t dnn_width, uint32_t dnn_height, uint32_t dnn_channels, uint32_t input_image_width,
        uint32_t input_image_height, int16_t scale_used_for_encoding, int16_t defined_infinity,
        bool normalize, int8_t dnn_radius, LRNEncodingParam encoding_param,
        // --------algorithm parameters-----------------
        int8_t non_max_radius, uint16_t max_possible_nodes, float min_valid_mask_val,
        uint16_t min_votes_for_nodes, uint16_t max_dist_for_nodes,
        // --------ETC-----------------
        tensorflow::OpKernelContext* context);

    // @brief
    // @param device_from_dnn
    void decode(const DNN_BLOB_TYPE* device_from_dnn);
    void vote();
    void extractNodes();

    void extractNodes(int8_t classId);

    // @brief
    // @param setNMSVotesParams
    void setNMSVotesParams(const uint16_t min_votes_for_nodes);

    // @brief
    // @param device_from_dnn
    void interpretDeviceAsync(const DNN_BLOB_TYPE* device_from_dnn);

    // @brief
    // @param stream
    void setCUDAStream(cudaStream_t stream);

    void getOutputNonmax(int* output_with_nonmax, const int arrow_length, const int line_thickness);
    void getOutputWithColor(int* output, int* output_with_angle, int* output_for_metric,
                            const int arrow_length, const int line_thickness,
                            const int background_class_id);

    void getOutput(int* output, int* output_with_angle, int* output_with_nonmax,
                   int* output_for_metric, const int arrow_length, const int line_thickness,
                   const int background_class_id);

    void free();

 protected:
    // These are in the original dimension.
    TmpDeviceTensor<VoteType> device_votes_;
    TmpDeviceTensor<int16_t> device_avg_direction_;
    // Shape is MAX_CLASSES by input_image_width and input_image_height/
    TmpDeviceTensor<ClassVoteType> device_class_votes_;
    TmpDeviceTensor<VoteLocalMaxType> device_gaussian_votes_;
    TmpDeviceTensor<BlurTypeForLRN> device_gaussian_votes_tmp_;
    TmpDeviceTensor<float> device_cos_votes_;
    TmpDeviceTensor<float> device_sin_votes_;

    // Belows are in shape of dnn_width, dnn_height.
    // This will be used to select prediction. This will be blurred with Gaussian
    TmpDeviceTensor<uint8_t> device_mask_;

    // This will be used to evaluate fit proposals. It better be blurred with Gaussian.
    TmpDeviceTensor<uint8_t> device_dist_;

    TmpDeviceTensor<int16_t> device_dx_;
    TmpDeviceTensor<int16_t> device_dy_;
    TmpDeviceTensor<uint16_t> device_left_width_;
    TmpDeviceTensor<uint16_t> device_right_width_;
    TmpDeviceTensor<uint32_t> device_left_width_sum_;
    TmpDeviceTensor<uint32_t> device_right_width_sum_;

    TmpDeviceTensor<DNN_BLOB_TYPE> device_cos_;
    TmpDeviceTensor<DNN_BLOB_TYPE> device_sin_;
    TmpDeviceTensor<int8_t> device_class_;

    LRNNode* device_nodes_;
    // One value.
    TmpDeviceTensor<int32_t> device_nodes_count_;
    TmpDeviceTensor<VoteLocalMaxType> device_votes_local_max_tmp_;

    cudaStream_t stream_;

    dim3 blocks_;
    dim3 blocks_vertical_;
    dim3 threads_;

    dim3 blocks_for_original_dim_;
    dim3 blocks_for_original_dim_vertical_;

    Color* colors_;
};
}  // namespace lineregressordecoder
#endif  // LRN_DECODER_CORE_GPU_H_
