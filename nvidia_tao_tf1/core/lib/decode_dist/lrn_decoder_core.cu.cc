// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

#include "cuda_helper.h"
#include "draw_basics.h"
#include "lrn_decoder_core_gpu.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __device__
#else
#define CUDA_HOSTDEV
#endif

namespace lineregressordecoder {
__constant__ float gConstantGaussianCoeffs[LRNDecoderBase::MAX_GAUSSIAN_FILTER_RADIUS * 2 + 1];

template <typename TARGET_T, typename SRC_T, int32_t DIMX, int32_t DIMY>
__device__ void applyGaussianKernel(TARGET_T* __restrict__ device_gaussian_votes,
                                    const size_t device_gaussian_votes_pitch,
                                    const int32_t spatial_radius,
                                    const SRC_T ss_data[DIMY][DIMX * 3], const int32_t x,
                                    const int32_t y, const dim3& threadIdx,
                                    const float scale = 1.0f) {
    // No sync necessary because of SIMT for this case.
    float p = 0;
    for (int32_t dx = -spatial_radius; dx <= spatial_radius; dx++) {
        p += ss_data[threadIdx.y][DIMX + dx + threadIdx.x] *
             gConstantGaussianCoeffs[dx + spatial_radius];
    }
    device_gaussian_votes[y * device_gaussian_votes_pitch + x] =
        min(static_cast<TARGET_T>(MAX_GAUSSIAN_VOTE), static_cast<TARGET_T>(0.5f + p * scale));
}

// Try out to do Gaussian blurring on `device_votes_`.
template <int32_t DIMX, int32_t DIMY>
__global__ void blurVotesStep1(BlurTypeForLRN* __restrict__ device_gaussian_votes_tmp,
                               const size_t device_gaussian_votes_tmp_pitch,
                               const VoteType* __restrict__ device_votes,
                               const size_t device_votes_pitch, const int8_t spatial_radius,
                               const int32_t w, const int32_t h) {
    const int32_t x = blockIdx.x * DIMX + threadIdx.x;
    const int32_t y = blockIdx.y * DIMY + threadIdx.y;
    __shared__ BlurTypeForLRN ss_data[DIMY][DIMX * 3];

    // Limit number of accesses to the global memory to 3 by using the shared.
    int32_t safey = min(h - 1, y);
    ss_data[threadIdx.y][threadIdx.x] = static_cast<BlurTypeForLRN>(
        device_votes[safey * device_votes_pitch + min(w - 1, max(0, x - DIMX))]);
    ss_data[threadIdx.y][DIMX + threadIdx.x] =
        static_cast<BlurTypeForLRN>(device_votes[safey * device_votes_pitch + min(w - 1, x)]);
    ss_data[threadIdx.y][2 * DIMX + threadIdx.x] = static_cast<BlurTypeForLRN>(
        device_votes[safey * device_votes_pitch + min(w - 1, x + DIMX)]);

    if (x < w && y < h) {
        // No sync necessary because of SIMT for this case.
        applyGaussianKernel<BlurTypeForLRN, BlurTypeForLRN, DIMX, DIMY>(
            device_gaussian_votes_tmp, device_gaussian_votes_tmp_pitch, spatial_radius, ss_data, x,
            y, threadIdx);
    }
}

template <int32_t DIMX, int32_t DIMY>
__global__ void blurVotesStep2(VoteLocalMaxType* __restrict__ device_gaussian_votes,
                               const size_t device_gaussian_votes_pitch,
                               const BlurTypeForLRN* __restrict__ device_gaussian_votes_tmp,
                               const size_t device_gaussian_votes_tmp_pitch,
                               const int8_t spatial_radius, const int32_t w, const int32_t h) {
    const int32_t y = blockIdx.x * DIMX + threadIdx.x;
    const int32_t x = blockIdx.y * DIMY + threadIdx.y;
    __shared__ BlurTypeForLRN ss_data[DIMY][DIMX * 3];

    // Limit number of accesses to the global memory to 3 by using the shared.
    int32_t safex = min(w - 1, x);
    ss_data[threadIdx.y][threadIdx.x] = static_cast<BlurTypeForLRN>(
        device_gaussian_votes_tmp[min(h - 1, max(0, y - DIMX)) * device_gaussian_votes_tmp_pitch +
                                  safex]);
    ss_data[threadIdx.y][DIMX + threadIdx.x] = static_cast<BlurTypeForLRN>(
        device_gaussian_votes_tmp[min(h - 1, y) * device_gaussian_votes_tmp_pitch + safex]);
    ss_data[threadIdx.y][2 * DIMX + threadIdx.x] = static_cast<BlurTypeForLRN>(
        device_gaussian_votes_tmp[min(h - 1, y + DIMX) * device_gaussian_votes_tmp_pitch + safex]);

    if (x < w && y < h) {
        // No sync necessary because of SIMT for this case.
        applyGaussianKernel<VoteLocalMaxType, BlurTypeForLRN, DIMX, DIMY>(
            device_gaussian_votes, device_gaussian_votes_pitch, spatial_radius, ss_data, x, y,
            threadIdx, 50.0f);
    }
}

//
// Decode vx, vy, scaled confidence, and dist.
//  @param vx
//  @param vy
//  @param conf_val
//  @param dist_val
//  @param width
//  @param height
//  @param dnn_output
//  @param index
//
template <typename DNN_BLOB_TYPE, typename OP_TYPE>
__forceinline__ __device__ void decodeVxVy(DNN_BLOB_TYPE* vx, DNN_BLOB_TYPE* vy, uint8_t* conf_val,
                                           uint8_t* dist_val,
                                           const DNN_BLOB_TYPE* __restrict__ dnn_output,
                                           const uint32_t width, const uint32_t height,
                                           const uint32_t index, const DNN_BLOB_TYPE conf,
                                           const uint8_t dx_pos, const uint8_t dy_pos,
                                           const uint8_t ndx_pos, const uint8_t ndy_pos) {
    *vx = dnn_output[height * width * dx_pos + index];
    *vy = dnn_output[height * width * dy_pos + index];
    DNN_BLOB_TYPE nvx = dnn_output[height * width * ndx_pos + index];
    DNN_BLOB_TYPE nvy = dnn_output[height * width * ndy_pos + index];
    if (*vx < nvx) {
        *vx = -nvx;
    }
    if (*vy < nvy) {
        *vy = -nvy;
    }

    uint16_t u_tmp = static_cast<uint16_t>(conf * 255.0f + 0.5f);
    *conf_val = clampMax<uint16_t, uint8_t>(u_tmp, static_cast<uint16_t>(255U));
    OP_TYPE fdist = __fsqrt_rn((*vx) * (*vx) + (*vy) * (*vy));
    u_tmp = static_cast<uint16_t>(fdist + 0.5f);
    *dist_val = clampMax<uint16_t, uint8_t>(u_tmp, static_cast<uint16_t>(255U));
}

/**
 * @brief Decods Vx Vy from dist 2D.
 *
 * @param vx
 * @param vy
 * @param conf_value
 * @param dist_value
 * @param dnn_output
 * @param width
 * @param height
 * @param index
 * @param conf
 * @param mag_pos
 * @param normal_cos_pos
 * @param normal_sine_pos
 */
template <typename DNN_BLOB_TYPE, typename OP_TYPE>
__forceinline__ __device__ void decodeVxVyFromDist2D(
    DNN_BLOB_TYPE* vx, DNN_BLOB_TYPE* vy, uint8_t* conf_value, uint8_t* dist_value,
    const DNN_BLOB_TYPE* __restrict__ dnn_output, const uint32_t width, const uint32_t height,
    const uint32_t index, const DNN_BLOB_TYPE conf, const uint8_t mag_pos,
    const uint8_t normal_cos_pos, const uint8_t normal_sine_pos) {
    OP_TYPE fdist = dnn_output[width * height * mag_pos + index];
    OP_TYPE normal_cos = dnn_output[width * height * normal_cos_pos + index] * 2.0f - 1.0f;
    OP_TYPE normal_sine = dnn_output[width * height * normal_sine_pos + index] * 2.0f - 1.0f;
    /**
     * mag = dnn_output[height * width + index];
     * costnormal = 0.5f*(dnn_output[height * width * 2 + index]+1.0f);
     * clamp between -1 to 1
     * sintnormal = 0.5f*(dnn_output[height * width * 3 + index]+1.0f);
     * clamp between -1 to 1
     *
     * vx= mag*cosnormal
     * vy= mag*sinnormal
     */
    clamp(-1.0f, 1.0f, &normal_cos);
    clamp(-1.0f, 1.0f, &normal_sine);
    *vx = fdist * normal_cos;
    *vy = fdist * normal_sine;

    uint16_t u_tmp = static_cast<uint16_t>(conf * 255.0f + 0.5f);
    *conf_value = clampMax<uint16_t, uint8_t>(u_tmp, static_cast<uint16_t>(255U));

    u_tmp = static_cast<uint16_t>(fdist + 0.5f);
    *dist_value = clampMax<uint16_t, uint8_t>(u_tmp, static_cast<uint16_t>(255U));
}

template <typename DNN_BLOB_TYPE, typename OP_TYPE>
__forceinline__ __device__ void decodeOptionals(
    float* v, OP_TYPE* cos_value, OP_TYPE* sine_value, float* left_width, float* right_width,
    int8_t* class_id, const uint16_t* __restrict__ device_edge_left_width,
    const size_t device_edge_left_width_pitch, const uint16_t* __restrict__ device_edge_right_width,
    const size_t device_edge_right_width_pitch, const DNN_BLOB_TYPE* __restrict__ device_cos,
    const size_t device_cos_pitch, const DNN_BLOB_TYPE* __restrict__ device_sin,
    const size_t device_sin_pitch, const int8_t* __restrict__ device_classes,
    const size_t device_classes_pitch, const int32_t scale_used_for_encoding,
    const DNN_BLOB_TYPE* __restrict__ dnn_output, const int32_t x, const int32_t y,
    const uint32_t width, const uint32_t height, const uint32_t channels,
    const uint32_t bit_channels, const int8_t dir_cos_start_pos, const int8_t dir_sine_start_pos,
    const int8_t width_left_start_pos, const int8_t width_right_start_pos,
    const int8_t bit_start_pos) {
    if (device_edge_left_width != nullptr) {
        *v = dnn_output[height * width * width_left_start_pos + y * width + x];
        *left_width = *v;
    }
    if (device_edge_right_width != nullptr) {
        *v = dnn_output[height * width * width_right_start_pos + y * width + x];
        *right_width = *v;
    }
    if (device_cos != nullptr) {
        *v = dnn_output[height * width * dir_cos_start_pos + y * width + x];
        *v = (*v) * 2.0f - 1.0f;
        clamp(-1.0f, 1.0f, v);
        *cos_value = *v;
    }
    if (device_sin != nullptr) {
        *v = dnn_output[height * width * dir_sine_start_pos + y * width + x];
        *v = (*v) * 2.0f - 1.0f;
        clamp(-1.0f, 1.0f, v);
        *sine_value = *v;
    }
    if (device_classes != nullptr) {
        // we need to transform bits to integer.
        uint32_t intClass =
            bitsToIntegers(dnn_output, x, y, width, height, bit_channels, bit_start_pos);
        *class_id = clampMax<uint32_t, int8_t>(intClass, static_cast<int8_t>(MAX_CLASSES));
    }
}

template <typename DNN_BLOB_TYPE, typename OP_TYPE, int32_t DIMX, int32_t DIMY>
__global__ void decodeDNN(
    uint8_t* __restrict__ device_mask, const size_t device_mask_pitch,
    uint8_t* __restrict__ device_dist, const size_t device_dist_pitch,
    int16_t* __restrict__ device_dx, const size_t device_dx_pitch, int16_t* __restrict__ device_dy,
    const size_t device_dy_pitch, uint16_t* __restrict__ device_edge_left_width,
    const size_t device_edge_left_width_pitch, uint16_t* __restrict__ device_edge_right_width,
    const size_t device_edge_right_width_pitch, DNN_BLOB_TYPE* __restrict__ device_cos,
    const size_t device_cos_pitch, DNN_BLOB_TYPE* __restrict__ device_sin,
    const size_t device_sin_pitch, int8_t* __restrict__ device_classes,
    const size_t device_classes_pitch, const float float_min_valid_score,
    const int32_t scale_used_for_encoding, const DNN_BLOB_TYPE* __restrict__ dnn_output,
    const int32_t defined_infinite_dist, const bool normalize_flag, const uint32_t width,
    const uint32_t height, const uint32_t channels, const uint32_t bit_channels,
    const int8_t dx_pos, const int8_t dy_pos, const int8_t ndx_pos, const int8_t ndy_pos,
    const int8_t dir_cos_start_pos, const int8_t dir_sine_start_pos,
    const int8_t width_left_start_pos, const int8_t width_right_start_pos,
    const int8_t bit_start_pos) {
    static_assert(DIMX == 32, "DIMX should be 32");
    const int32_t x = DIMX * blockIdx.x + threadIdx.x;
    const int32_t y = DIMY * blockIdx.y + threadIdx.y;
    // We need to do horizontal blur first for mask and computed dist.
    if (x < width && y < height) {
        uint32_t index = y * width + x;
        DNN_BLOB_TYPE conf = dnn_output[index];
        DNN_BLOB_TYPE vx = defined_infinite_dist;
        DNN_BLOB_TYPE vy = defined_infinite_dist;
        OP_TYPE cos_value = -2.0f;
        OP_TYPE sine_value = -2.0f;
        uint8_t dist_value = defined_infinite_dist;
        uint8_t conf_value = 0U;
        int8_t class_id = MAX_CLASSES;
        float v = 0.0f;
        float left_width = 0.0f;
        float right_width = 0.0f;
        if (conf > float_min_valid_score) {
            decodeVxVy<DNN_BLOB_TYPE, OP_TYPE>(&vx, &vy, &conf_value, &dist_value, dnn_output,
                                               width, height, index, conf, dx_pos, dy_pos, ndx_pos,
                                               ndy_pos);
            decodeOptionals<DNN_BLOB_TYPE, OP_TYPE>(
                &v, &cos_value, &sine_value, &left_width, &right_width, &class_id,
                device_edge_left_width, device_edge_left_width_pitch, device_edge_right_width,
                device_edge_right_width_pitch, device_cos, device_cos_pitch, device_sin,
                device_sin_pitch, device_classes, device_classes_pitch, scale_used_for_encoding,
                dnn_output, x, y, width, height, channels, bit_channels, dir_cos_start_pos,
                dir_sine_start_pos, width_left_start_pos, width_right_start_pos, bit_start_pos);
        }
        device_mask[y * device_mask_pitch + x] = conf_value;
        if (normalize_flag) {
            dist_value *= defined_infinite_dist;
            dist_value = clampMax<uint8_t, uint8_t>(dist_value, static_cast<uint8_t>(255U));
            vx *= static_cast<int16_t>(defined_infinite_dist);
            vy *= static_cast<int16_t>(defined_infinite_dist);
            left_width *= static_cast<float>(defined_infinite_dist);
            right_width *= static_cast<float>(defined_infinite_dist);
        }
        device_dist[y * device_dist_pitch + x] = dist_value;
        device_dx[y * device_dx_pitch + x] =
            static_cast<int16_t>(0.5f + vx * scale_used_for_encoding);
        device_dy[y * device_dy_pitch + x] =
            static_cast<int16_t>(0.5f + vy * scale_used_for_encoding);
        if (device_cos != nullptr) {
            device_cos[y * device_cos_pitch + x] = cos_value;
        }
        if (device_sin != nullptr) {
            device_sin[y * device_sin_pitch + x] = sine_value;
        }
        if (device_classes != nullptr) {
            device_classes[y * device_classes_pitch + x] = class_id;
        }
        if (device_edge_left_width != nullptr) {
            device_edge_left_width[y * device_edge_left_width_pitch + x] =
                static_cast<uint16_t>(0.5f + left_width * scale_used_for_encoding);
        }
        if (device_edge_right_width != nullptr) {
            device_edge_right_width[y * device_edge_right_width_pitch + x] =
                static_cast<uint16_t>(0.5f + right_width * scale_used_for_encoding);
        }
    }
}

template <typename DNN_BLOB_TYPE, typename OP_TYPE, int32_t DIMX, int32_t DIMY>
__global__ void decodeDNNFromDist2D(
    uint8_t* __restrict__ device_mask, const size_t device_mask_pitch,
    uint8_t* __restrict__ device_dist, const size_t device_dist_pitch,
    int16_t* __restrict__ device_dx, const size_t device_dx_pitch, int16_t* __restrict__ device_dy,
    const size_t device_dy_pitch, uint16_t* __restrict__ device_edge_left_width,
    const size_t device_edge_left_width_pitch, uint16_t* __restrict__ device_edge_right_width,
    const size_t device_edge_right_width_pitch, DNN_BLOB_TYPE* __restrict__ device_cos,
    const size_t device_cos_pitch, DNN_BLOB_TYPE* __restrict__ device_sin,
    const size_t device_sin_pitch, int8_t* __restrict__ device_classes,
    const size_t device_classes_pitch, const float float_min_valid_score,
    const int32_t scale_used_for_encoding, const DNN_BLOB_TYPE* __restrict__ dnn_output,
    const int32_t defined_infinite_dist, const bool normalize_flag, const uint32_t width,
    const uint32_t height, const uint32_t channels, const uint32_t bit_channels,
    const uint8_t mag_pos, const uint8_t normal_cos_pos, const uint8_t normal_sine_pos,
    const int8_t dir_cos_start_pos, const int8_t dir_sine_start_pos,
    const int8_t width_left_start_pos, const int8_t width_right_start_pos,
    const int8_t bit_start_pos) {
    static_assert(DIMX == 32, "DIMX should be 32");
    const int32_t x = DIMX * blockIdx.x + threadIdx.x;
    const int32_t y = DIMY * blockIdx.y + threadIdx.y;
    // We need to do horizontal blur first for mask and computed dist.
    if (x < width && y < height) {
        uint32_t index = y * width + x;

        DNN_BLOB_TYPE conf = dnn_output[index];
        DNN_BLOB_TYPE vx = defined_infinite_dist;
        DNN_BLOB_TYPE vy = defined_infinite_dist;
        OP_TYPE cos_value = -2.0f;
        OP_TYPE sine_value = -2.0f;
        uint8_t dist_value = defined_infinite_dist;
        uint8_t conf_value = 0U;
        int8_t class_id = MAX_CLASSES;
        float v = 0.0f;
        float left_width = 0.0f;
        float right_width = 0.0f;
        if (conf > float_min_valid_score) {
            decodeVxVyFromDist2D<DNN_BLOB_TYPE, OP_TYPE>(&vx, &vy, &conf_value, &dist_value,
                                                         dnn_output, width, height, index, conf,
                                                         mag_pos, normal_cos_pos, normal_sine_pos);
            decodeOptionals<DNN_BLOB_TYPE, OP_TYPE>(
                &v, &cos_value, &sine_value, &left_width, &right_width, &class_id,
                device_edge_left_width, device_edge_left_width_pitch, device_edge_right_width,
                device_edge_right_width_pitch, device_cos, device_cos_pitch, device_sin,
                device_sin_pitch, device_classes, device_classes_pitch, scale_used_for_encoding,
                dnn_output, x, y, width, height, channels, bit_channels, dir_cos_start_pos,
                dir_sine_start_pos, width_left_start_pos, width_right_start_pos, bit_start_pos);
        }
        device_mask[y * device_mask_pitch + x] = conf_value;
        if (normalize_flag) {
            dist_value *= defined_infinite_dist;
            dist_value = clampMax<uint8_t, uint8_t>(dist_value, static_cast<uint8_t>(255U));
            vx *= static_cast<int16_t>(defined_infinite_dist);
            vy *= static_cast<int16_t>(defined_infinite_dist);
            left_width *= static_cast<float>(defined_infinite_dist);
            right_width *= static_cast<float>(defined_infinite_dist);
        }
        device_dist[y * device_dist_pitch + x] = dist_value;
        device_dx[y * device_dx_pitch + x] =
            static_cast<int16_t>(0.5f + vx * scale_used_for_encoding);
        device_dy[y * device_dy_pitch + x] =
            static_cast<int16_t>(0.5f + vy * scale_used_for_encoding);
        if (device_cos != nullptr) {
            device_cos[y * device_cos_pitch + x] = cos_value;
        }
        if (device_sin != nullptr) {
            device_sin[y * device_sin_pitch + x] = sine_value;
        }
        if (device_classes != nullptr) {
            device_classes[y * device_classes_pitch + x] = class_id;
        }
        if (device_edge_left_width != nullptr) {
            device_edge_left_width[y * device_edge_left_width_pitch + x] =
                static_cast<uint16_t>(0.5f + left_width * scale_used_for_encoding);
        }
        if (device_edge_right_width != nullptr) {
            device_edge_right_width[y * device_edge_right_width_pitch + x] =
                static_cast<uint16_t>(0.5f + right_width * scale_used_for_encoding);
        }
    }
}

__global__ void resetVotes(VoteType* device_votes, const size_t device_votes_pitch,
                           float* device_cos_votes, const size_t device_cos_votes_pitch,
                           float* device_sin_votes, const size_t device_sin_votes_pitch,
                           uint32_t* device_left_width_sum,
                           const size_t device_left_width_sum_pitch,
                           uint32_t* device_right_width_sum,
                           const size_t device_right_width_sum_pitch, int16_t* device_avg_direction,
                           const size_t device_direction_votes_pitch,
                           ClassVoteType* device_class_votes, const int32_t max_possible_classes,
                           const size_t device_class_votes_pitch,
                           const int32_t scale_used_for_encoding, const int32_t input_image_width,
                           const int32_t input_image_height) {
    const int32_t nx = blockDim.x * blockIdx.x + threadIdx.x;
    const int32_t ny = blockDim.y * blockIdx.y + threadIdx.y;

    if (nx < input_image_width && ny < input_image_height) {
        device_votes[ny * device_votes_pitch + nx] = 0;
        if (device_left_width_sum != nullptr) {
            device_left_width_sum[ny * device_left_width_sum_pitch + nx] = 0;
        }
        if (device_right_width_sum != nullptr) {
            device_right_width_sum[ny * device_right_width_sum_pitch + nx] = 0;
        }
        if (device_avg_direction != nullptr) {
            device_avg_direction[ny * device_direction_votes_pitch + nx] = 0;
        }
        if (device_cos_votes != nullptr) {
            device_cos_votes[ny * device_cos_votes_pitch + nx] = 0;
        }
        if (device_sin_votes != nullptr) {
            device_sin_votes[ny * device_sin_votes_pitch + nx] = 0;
        }
        if (device_class_votes != nullptr) {
            for (int32_t c = 0; c < max_possible_classes; c++) {
                device_class_votes[c * device_class_votes_pitch * input_image_height +
                                   ny * device_class_votes_pitch + nx] = 0;
            }
        }
    }
}

__global__ void computeDirectionAndArgMaxKernel(
    VoteType* __restrict__ device_votes, const size_t device_votes_pitch,
    uint32_t* __restrict__ device_left_width_sum, const size_t device_left_width_sum_pitch,
    uint32_t* __restrict__ device_right_width_sum, const size_t device_right_width_sum_pitch,
    float* __restrict__ device_cos_votes, const size_t device_cos_votes_pitch,
    float* __restrict__ device_sin_votes, const size_t device_sin_votes_pitch,
    int16_t* __restrict__ device_avg_direction, const size_t device_avg_direction_pitch,
    ClassVoteType* __restrict__ device_class_votes, const int32_t max_possible_classes,
    const size_t device_class_votes_pitch, const int32_t scale_used_for_encoding,
    const int32_t input_image_width, const int32_t input_image_height) {
    const int32_t x = blockDim.x * blockIdx.x + threadIdx.x;
    const int32_t y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < input_image_width && y < input_image_height) {
        ClassVoteType max_count = 0;
        int32_t max_class = -1;
        for (int32_t c = 0; c < max_possible_classes; c++) {
            ClassVoteType val =
                device_class_votes[c * input_image_height * device_class_votes_pitch +
                                   y * device_class_votes_pitch + x];
            if (val > max_count) {
                max_count = val;
                max_class = c;
            }
        }
        device_class_votes[y * device_class_votes_pitch + x] =
            static_cast<ClassVoteType>(max_class);
        float cos_val = device_cos_votes[y * device_cos_votes_pitch + x];
        float sin_val = device_sin_votes[y * device_sin_votes_pitch + x];
        int16_t angle = static_cast<int16_t>(0.5f + atan2f(sin_val, cos_val) * LRN_RADIAN_TO_DEG);
        device_avg_direction[y * device_avg_direction_pitch + x] = angle;

        if (device_left_width_sum != nullptr && device_right_width_sum != nullptr) {
            uint32_t totalVotes = device_votes[y * device_votes_pitch + x];
            float left_width = 0;
            float right_width = 0;
            if (totalVotes > 0) {
                left_width =
                    0.5f + device_left_width_sum[y * device_left_width_sum_pitch + x] / totalVotes;
                right_width =
                    0.5f +
                    device_right_width_sum[y * device_right_width_sum_pitch + x] / totalVotes;
            }
            device_left_width_sum[y * device_left_width_sum_pitch + x] =
                static_cast<uint32_t>(left_width);
            device_right_width_sum[y * device_right_width_sum_pitch + x] =
                static_cast<uint32_t>(right_width);
        }
    }
}

__forceinline__ __device__ bool computeOriginalCoord(
    int32_t* original_x, int32_t* original_y, const int32_t x, const int32_t y,
    const int32_t scale_used_for_encoding, const int32_t effective_radius, const int16_t* device_dx,
    const size_t device_dx_pitch, const int16_t* device_dy, const size_t device_dy_pitch,
    const int32_t width, const int32_t height, const int32_t input_image_width,
    const int32_t input_image_height) {
    bool result = true;
    float vecx = device_dx[y * device_dx_pitch + x];
    float vecy = device_dy[y * device_dx_pitch + x];
    if (vecx < -effective_radius || vecx > effective_radius || vecy < -effective_radius ||
        vecy > effective_radius) {
        return false;
    }
    if (result) {
        *original_x = scale_used_for_encoding * x + vecx + 0.5f;
        *original_y = scale_used_for_encoding * y + vecy + 0.5f;

        if (*original_x < 0 || *original_x >= input_image_width || *original_y < 0 ||
            *original_y >= input_image_height) {
            result = false;
        }
    }
    return result;
}

//
// Vote using atomic operation around coordinates x and y in original(output) dimension.
// @param device_votes
// @param device_votes_pitch
// @param original_x
// @param original_y
// @param input_image_width
// @param input_image_height
//
template <int32_t RADIUS = 0>
__forceinline__ __device__ void voteAroundPoint(VoteType* device_votes,
                                                const size_t device_votes_pitch,
                                                const int32_t original_x, const int32_t original_y,
                                                const int32_t input_image_width,
                                                const int32_t input_image_height) {
#pragma unroll
    for (int32_t dx = -RADIUS; dx <= RADIUS; dx++) {
        int32_t nx = original_x + dx;
#pragma unroll
        for (int32_t dy = -RADIUS; dy <= RADIUS; dy++) {
            int32_t ny = original_y + dy;
            if (nx >= 0 && nx < input_image_width && ny >= 0 && ny < input_image_height) {
                atomicAdd(&device_votes[ny * device_votes_pitch + nx], 1);
            }
        }
    }
}

template <typename DNN_BLOB_TYPE>
__forceinline__ __device__ void voteOptionals(
    uint32_t* device_left_width_sum, const size_t device_left_width_sum_pitch,
    uint32_t* device_right_width_sum, const size_t device_right_width_sum_pitch,
    float* device_cos_votes, const size_t device_cos_votes_pitch, float* device_sin_votes,
    const size_t device_sin_votes_pitch, ClassVoteType* device_class_votes,
    const size_t max_possible_classes, const size_t device_class_votes_pitch,
    const size_t device_class_vote_slice_pitch, const uint16_t* device_left_width,
    const size_t device_left_width_pitch, const uint16_t* device_right_width,
    const size_t device_right_width_pitch, const DNN_BLOB_TYPE* device_cos,
    const size_t device_cos_pitch, const DNN_BLOB_TYPE* device_sin, const size_t device_sin_pitch,
    const int8_t* device_classes, const size_t device_classes_pitch, const int32_t original_x,
    const int32_t original_y, const int32_t x, const int32_t y) {
    if (device_left_width_sum != nullptr && device_right_width_sum != nullptr) {
        atomicAdd(&device_left_width_sum[original_y * device_left_width_sum_pitch + original_x],
                  device_left_width[y * device_left_width_pitch + x]);
        atomicAdd(&device_right_width_sum[original_y * device_right_width_sum_pitch + original_x],
                  device_right_width[y * device_right_width_pitch + x]);
    }
    if (device_cos_votes != nullptr && device_sin_votes != nullptr) {
        atomicAdd(&device_cos_votes[original_y * device_cos_votes_pitch + original_x],
                  device_cos[y * device_cos_pitch + x]);
        atomicAdd(&device_sin_votes[original_y * device_sin_votes_pitch + original_x],
                  device_sin[y * device_sin_pitch + x]);
    }
    if (device_class_votes != nullptr) {
        int8_t class_id = device_classes[y * device_classes_pitch + x];
        if (class_id >= 0 && class_id < max_possible_classes) {
            atomicAdd(&device_class_votes[class_id * device_class_vote_slice_pitch +
                                          original_y * device_class_votes_pitch + original_x],
                      1);
        }
    }
}

template <typename DNN_BLOB_TYPE, int32_t DIMX, int32_t DIMY>
__global__ void computeVotesCosSinClasses(
    VoteType* device_votes, const size_t device_votes_pitch, uint32_t* device_left_width_sum,
    const size_t device_left_width_sum_pitch, uint32_t* device_right_width_sum,
    const size_t device_right_width_sum_pitch, float* device_cos_votes,
    const size_t device_cos_votes_pitch, float* device_sin_votes,
    const size_t device_sin_votes_pitch, int16_t* device_avg_direction,
    const size_t device_direction_votes_pitch, ClassVoteType* device_class_votes,
    const size_t max_possible_classes, const size_t device_class_votes_pitch,
    const uint8_t* device_mask, const size_t device_mask_pitch, const uint8_t* device_dist,
    const size_t device_dist_pitch, const int16_t* device_dx, const size_t device_dx_pitch,
    const int16_t* device_dy, const size_t device_dy_pitch, const uint16_t* device_left_width,
    const size_t device_left_width_pitch, const uint16_t* device_right_width,
    const size_t device_right_width_pitch, const DNN_BLOB_TYPE* device_cos,
    const size_t device_cos_pitch, const DNN_BLOB_TYPE* device_sin, const size_t device_sin_pitch,
    const int8_t* device_classes, const size_t device_classes_pitch,
    const int32_t scale_used_for_encoding, const uint8_t min_valid_value,
    const int32_t defined_infinite_dist, const int32_t effective_radius, const uint32_t width,
    const uint32_t height, const int32_t input_image_width, const int32_t input_image_height) {
    static_assert(DIMX <= 32, "DIMX should be less than or equal to 32.");
    int32_t x = DIMX * blockIdx.x + threadIdx.x;
    int32_t y = DIMY * blockIdx.y + threadIdx.y;
    int32_t original_x = 0;
    int32_t original_y = 0;
    if (x >= width || y >= height) {
        return;
    }
    if (computeOriginalCoord(&original_x, &original_y, x, y, scale_used_for_encoding,
                             effective_radius, device_dx, device_dx_pitch, device_dy,
                             device_dy_pitch, width, height, input_image_width,
                             input_image_height)) {
        voteAroundPoint(device_votes, device_votes_pitch, original_x, original_y, input_image_width,
                        input_image_height);
        voteOptionals<DNN_BLOB_TYPE>(
            device_left_width_sum, device_left_width_sum_pitch, device_right_width_sum,
            device_right_width_sum_pitch, device_cos_votes, device_cos_votes_pitch,
            device_sin_votes, device_sin_votes_pitch, device_class_votes, max_possible_classes,
            device_class_votes_pitch, device_class_votes_pitch * input_image_height,
            device_left_width, device_left_width_pitch, device_right_width,
            device_right_width_pitch, device_cos, device_cos_pitch, device_sin, device_sin_pitch,
            device_classes, device_classes_pitch, original_x, original_y, x, y);
    }
}

template <int32_t DIMX, int32_t DIMY, typename T = VoteType>
__forceinline__ __device__ void loadToSharedMemory(
    T ss[DIMY][DIMX * 3], int8_t ss_class[DIMY][DIMX * 3], const dim3& threadIdx, const int32_t x,
    const int32_t y, const int32_t width, const int32_t height, const T* device_votes,
    const size_t device_votes_pitch, ClassVoteType* device_class_votes,
    const size_t device_class_votes_pitch) {
    ss[threadIdx.y][threadIdx.x] = 0;
    ss[threadIdx.y][threadIdx.x + DIMX] = 0;
    ss[threadIdx.y][threadIdx.x + DIMX * 2] = 0;

    ss_class[threadIdx.y][threadIdx.x] = INVALID_VALUE;
    ss_class[threadIdx.y][threadIdx.x + DIMX] = INVALID_VALUE;
    ss_class[threadIdx.y][threadIdx.x + DIMX * 2] = INVALID_VALUE;
    if (y < height) {
#pragma unroll
        for (int32_t i = -1; i <= 1; i++) {
            int32_t new_x = x + i * DIMX;
            T val = 0;
            int8_t class_val = INVALID_VALUE;
            if (new_x >= 0 && new_x < width) {
                val = device_votes[y * device_votes_pitch + new_x];
                class_val = device_class_votes[y * device_class_votes_pitch + new_x];
            }
            ss[threadIdx.y][threadIdx.x + (i + 1) * DIMX] = val;
            ss_class[threadIdx.y][threadIdx.x + (i + 1) * DIMX] = class_val;
        }
    }
}

template <int32_t DIMX, int32_t DIMY, typename T = VoteLocalMaxType>
__forceinline__ __device__ void updateLocalMaxPerEachClass(
    T* device_local_max_temp, const size_t device_local_max_temp_pitch, const int32_t x,
    const int32_t y, const dim3& threadIdx, const uint32_t width, const uint32_t height,
    const int8_t max_possible_classes, const int8_t non_max_radius, const T ss[DIMY][DIMX * 3],
    const int8_t ss_class[DIMY][DIMX * 3]) {
    if (x < width && y < height) {
        T max_val[MAX_CLASSES];
        for (int8_t c = 0; c < max_possible_classes; c++) {
            max_val[c] = 0U;
        }
        for (int8_t d = -non_max_radius; d <= non_max_radius; d++) {
            int8_t cid = ss_class[threadIdx.y][threadIdx.x + DIMX + d];
            if (cid >= 0 && cid < max_possible_classes) {
                T t = ss[threadIdx.y][threadIdx.x + DIMX + d];
                if (max_val[cid] < t) {
                    max_val[cid] = t;
                }
            }
        }
        uint32_t slice_pitch = device_local_max_temp_pitch * height;
        for (int8_t c = 0; c < max_possible_classes; c++) {
            device_local_max_temp[static_cast<uint32_t>(c) * slice_pitch +
                                  y * device_local_max_temp_pitch + x] = max_val[c];
        }
    }
}

template <int32_t DIMX, int32_t DIMY>
__global__ void extractNodesStep1(VoteLocalMaxType* device_local_max_temp,
                                  const size_t device_local_max_temp_pitch,
                                  int32_t* device_nodes_count,  // just one value
                                  const VoteLocalMaxType* device_votes,
                                  const size_t device_votes_pitch,
                                  ClassVoteType* device_class_votes,
                                  const size_t device_class_votes_pitch,
                                  const int8_t non_max_radius, const uint32_t height,
                                  const uint32_t width, const int8_t max_possible_classes) {
    // First find max votes per each class within (x -non_max_radius) ~ (x + non_max_radius).
    static_assert(DIMX == 32, "DIMY should be 32");
    const int32_t x = DIMX * blockIdx.x + threadIdx.x;
    const int32_t y = DIMY * blockIdx.y + threadIdx.y;
    // We need to do horizontal blur first for mask and computed dist.
    // We will do voting in the next kernel. after vertical blur is done.
    __shared__ VoteLocalMaxType ss[DIMY][DIMX * 3];
    __shared__ int8_t ss_class[DIMY][DIMX * 3];

    loadToSharedMemory<DIMX, DIMY, VoteLocalMaxType>(ss, ss_class, threadIdx, x, y, width, height,
                                                     device_votes, device_votes_pitch,
                                                     device_class_votes, device_class_votes_pitch);
    // Take advantage of this kernel to reset `devicePeaksCount` to 0.
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        device_nodes_count[0] = 0;
    }

    // No sync necessary we will be doing horizontally only because it is linear separable.
    updateLocalMaxPerEachClass<DIMX, DIMY>(device_local_max_temp, device_local_max_temp_pitch, x, y,
                                           threadIdx, width, height, max_possible_classes,
                                           non_max_radius, ss, ss_class);
}

__forceinline__ __device__ VoteLocalMaxType getMaxValForClass(
    const int32_t x, const int32_t y, const int8_t current_class_id,
    const VoteLocalMaxType* device_local_max_temp, const size_t device_local_max_temp_pitch,
    const int8_t max_possible_classes, const int8_t non_max_radius, const uint32_t height) {
    VoteLocalMaxType max_val = VoteLocalMaxType(0);
    if (current_class_id >= 0 && current_class_id < max_possible_classes) {
        uint32_t slice_pitch = device_local_max_temp_pitch * height;
        for (int32_t d = -non_max_radius; d <= non_max_radius; d++) {
            int32_t ny = y + d;
            if (ny >= 0 && ny < height) {
                VoteLocalMaxType t =
                    device_local_max_temp[static_cast<uint32_t>(current_class_id) * slice_pitch +
                                          ny * device_local_max_temp_pitch + x];
                if (max_val < t) {
                    max_val = t;
                }
            }
        }
    }
    return max_val;
}

__forceinline__ __device__ LRNNode setLRNNode(
    const int32_t x, const int32_t y, const VoteLocalMaxType max_val, int8_t current_class_id,
    const int16_t* device_avg_direction, const size_t device_direction_votes_pitch,
    ClassVoteType* device_class_votes, const size_t device_class_votes_pitch,
    uint32_t* __restrict__ device_left_width_sum, const size_t device_left_width_sum_pitch,
    uint32_t* __restrict__ device_right_width_sum, const size_t device_right_width_sum_pitch) {
    // if deviceNodesCount is greater than maxPossibleNodes,
    // the number of found nodes is maxPossibleNodes.
    LRNNode tmp;
    tmp.x = x;
    tmp.y = y;
    tmp.normalized_votes = clampMax<uint16_t, uint8_t>(max_val, 255U);
    tmp.edge_left_width = 0;
    tmp.edge_right_width = 0;
    if (device_avg_direction != nullptr) {
        tmp.angle = device_avg_direction[y * device_direction_votes_pitch + x];
    } else {
        tmp.angle = INVALID_VALUE;
    }
    if (device_class_votes != nullptr) {
        tmp.decoded_bit_code = current_class_id;
    } else {
        tmp.decoded_bit_code = INVALID_VALUE;
    }
    if (device_left_width_sum != nullptr) {
        tmp.edge_left_width = device_left_width_sum[y * device_left_width_sum_pitch + x];
    }
    if (device_right_width_sum != nullptr) {
        tmp.edge_right_width = device_right_width_sum[y * device_right_width_sum_pitch + x];
    }
    return tmp;
}

template <int32_t DIMX, int32_t DIMY>
__global__ void extractNodesStep2(
    LRNNode* device_nodes, const size_t max_possible_nodes,
    int32_t* device_nodes_count,  // just one value
    const int16_t* device_avg_direction, const size_t device_direction_votes_pitch,
    ClassVoteType* device_class_votes, const size_t device_class_votes_pitch,
    const VoteType* device_votes, const size_t device_votes_pitch,
    uint32_t* __restrict__ device_left_width_sum, const size_t device_left_width_sum_pitch,
    uint32_t* __restrict__ device_right_width_sum, const size_t device_right_width_sum_pitch,
    const VoteLocalMaxType* device_gaussian_votes, const size_t device_gaussian_votes_pitch,
    const VoteLocalMaxType* device_local_max_temp, const size_t device_local_max_temp_pitch,
    const uint16_t min_votes_for_nodes, const uint16_t max_dist_for_nodes,
    const int8_t non_max_radius, const uint32_t height, const uint32_t width,
    const int32_t scale_used_for_encoding, const int8_t max_possible_classes) {
    static_assert(DIMX == 32, "DIMY should be 32");
    // The below is intentional change (swapping x and y).
    const int32_t y = DIMX * blockIdx.x + threadIdx.x;
    const int32_t x = DIMY * blockIdx.y + threadIdx.y;
    // We need to do horizontal blur first for mask and computed dist.
    // No sync necessary we will be doing horizontally only because it is linear separable.
    if (x < width && y < height) {
        int8_t current_class_id = device_class_votes[y * device_class_votes_pitch + x];
        VoteLocalMaxType max_val = getMaxValForClass(x, y, current_class_id, device_local_max_temp,
                                                     device_local_max_temp_pitch,
                                                     max_possible_classes, non_max_radius, height);

        VoteType raw_votes = device_votes[y * device_votes_pitch + x];
        VoteLocalMaxType g_votes = device_gaussian_votes[y * device_gaussian_votes_pitch + x];
        if (raw_votes >= min_votes_for_nodes && max_val == g_votes && max_val > 0) {
            // Add to the peak.
            uint32_t old = atomicAdd(device_nodes_count, 1);
            if (old < max_possible_nodes) {
                // If device_nodes_count is greater than max_possible_nodes, the # of found nodes is
                // max_possible_nodes.
                device_nodes[old] = setLRNNode(
                    x, y, max_val, current_class_id, device_avg_direction,
                    device_direction_votes_pitch, device_class_votes, device_class_votes_pitch,
                    device_left_width_sum, device_left_width_sum_pitch, device_right_width_sum,
                    device_right_width_sum_pitch);
            }
        }
    }
}

__forceinline__ __device__ void drawLine(int* input, const int x, const int y, const int cols,
                                         const int rows, const int16_t angle,
                                         const int arrow_length, const int line_thickness,
                                         const int r, const int g, const int b) {
    if (arrow_length > 0 && x > 0 && x < cols && y > 0 && y < rows) {
        float x2 = x + arrow_length * cosf(angle * LRN_DEG_TO_RADIAN + M_PI);
        float y2 = y + arrow_length * sinf(angle * LRN_DEG_TO_RADIAN + M_PI);
        draw_line(input, rows, cols, x, y, x2, y2, line_thickness, r, g, b);
    }
}

__forceinline__ __device__ void drawForEdges(int* output, int* output_with_angle,
                                             int* output_for_metric, const int edges_sum,
                                             const int class_id, const int edge_x, const int edge_y,
                                             const int cols, const int rows, const int16_t angle,
                                             const int arrow_length, const int line_thickness,
                                             const int background_class_id, const int r,
                                             const int g, const int b) {
    if (edges_sum > 0 && edge_x > 0 && edge_x < cols && edge_y > 0 && edge_y < rows) {
        output[edge_y * cols * 3 + 3 * edge_x] = max(0, r - 50);
        output[edge_y * cols * 3 + 3 * edge_x + 1] = max(0, g - 50);
        output[edge_y * cols * 3 + 3 * edge_x + 2] = max(0, b - 50);
        drawLine(output_with_angle, edge_x, edge_y, cols, rows, angle, arrow_length, line_thickness,
                 max(0, r - 50), max(0, g - 50), max(0, b - 50));
        if (class_id >= background_class_id) {
            output_for_metric[edge_y * cols + edge_x] = class_id - background_class_id;
        } else {
            output_for_metric[edge_y * cols + edge_x] = class_id;
        }
    }
}
__global__ void getVotesWithColor(int* output, int* output_with_angle, int* output_for_metric,
                                  const VoteType* device_votes, const uint32_t* defice_left_edge,
                                  const uint32_t* defice_right_edge,
                                  const ClassVoteType* device_class_votes,
                                  const int16_t* device_avg_direction, const int cols,
                                  const int rows, const int min_votes_for_nodes, Color* color_list,
                                  const int arrow_length, const int line_thickness,
                                  const int background_class_id) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;
    VoteType vote = device_votes[y * cols + x];
    if (vote >= min_votes_for_nodes) {
        uint32_t left_sum = defice_left_edge[y * cols + x];
        uint32_t right_sum = defice_right_edge[y * cols + x];
        int16_t angle = device_avg_direction[y * cols + x];
        int class_id = static_cast<int>(device_class_votes[y * cols + x]);
        class_id = class_id < 20 ? class_id : 20;
        int r = color_list[class_id].r_;
        int g = color_list[class_id].g_;
        int b = color_list[class_id].b_;
        output[y * cols * 3 + 3 * x] = r;
        output[y * cols * 3 + 3 * x + 1] = g;
        output[y * cols * 3 + 3 * x + 2] = b;

        output_with_angle[y * cols * 3 + 3 * x] = r;
        output_with_angle[y * cols * 3 + 3 * x + 1] = g;
        output_with_angle[y * cols * 3 + 3 * x + 2] = b;

        drawLine(output_with_angle, x, y, cols, rows, angle, arrow_length, line_thickness, r, g, b);
        // output left edges.
        uint32_t left_edge_x = x - left_sum * cosf(angle * LRN_DEG_TO_RADIAN - M_PI_2);
        uint32_t left_edge_y = y - left_sum * sinf(angle * LRN_DEG_TO_RADIAN - M_PI_2);
        drawForEdges(output, output_with_angle, output_for_metric, left_sum, class_id, left_edge_x,
                     left_edge_y, cols, rows, angle, arrow_length, line_thickness,
                     background_class_id, r, g, b);
        // output right edges.
        uint32_t right_edge_x = x + right_sum * cosf(angle * LRN_DEG_TO_RADIAN - M_PI_2);
        uint32_t right_edge_y = y + right_sum * sinf(angle * LRN_DEG_TO_RADIAN - M_PI_2);
        drawForEdges(output, output_with_angle, output_for_metric, right_sum, class_id,
                     right_edge_x, right_edge_y, cols, rows, angle, arrow_length, line_thickness,
                     background_class_id, r, g, b);

        if (class_id >= background_class_id) {
            output_for_metric[y * cols + x] = class_id - background_class_id;
        } else {
            output_for_metric[y * cols + x] = class_id;
        }
    }
}

__global__ void getNonmaxWithLineDrawn(int* output_with_nonmax, LRNNode* device_nodes,
                                       Color* color_list, const int device_nodes_count,
                                       const int cols, const int rows, const int arrow_length,
                                       const int line_thickness) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < device_nodes_count; i += stride) {
        int x = device_nodes[i].x;
        int y = device_nodes[i].y;
        int angle = device_nodes[i].angle;
        int class_id = device_nodes[i].decoded_bit_code;
        uint8_t left_width = device_nodes[i].edge_left_width;
        uint8_t right_width = device_nodes[i].edge_right_width;
        int left_x = x - left_width * cosf(angle * LRN_DEG_TO_RADIAN - M_PI_2);
        int left_y = y - left_width * sinf(angle * LRN_DEG_TO_RADIAN - M_PI_2);
        int right_x = x + right_width * cosf(angle * LRN_DEG_TO_RADIAN - M_PI_2);
        int right_y = y + right_width * sinf(angle * LRN_DEG_TO_RADIAN - M_PI_2);
        int r = color_list[class_id].r_;
        int g = color_list[class_id].g_;
        int b = color_list[class_id].b_;

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
    }
}

__global__ void getNonmaxWithColor(int* output_with_nonmax, LRNNode* device_nodes,
                                   Color* color_list, const int device_nodes_count, const int cols,
                                   const int rows) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < device_nodes_count; i += stride) {
        int x = device_nodes[i].x;
        int y = device_nodes[i].y;
        int angle = device_nodes[i].angle;
        uint8_t left_width = device_nodes[i].edge_left_width;
        uint8_t right_width = device_nodes[i].edge_right_width;
        int class_id = device_nodes[i].decoded_bit_code;
        int r = color_list[class_id].r_;
        int g = color_list[class_id].g_;
        int b = color_list[class_id].b_;

        int left_x = x - left_width * cos(angle * LRN_DEG_TO_RADIAN - M_PI_2);
        int left_y = y - left_width * sin(angle * LRN_DEG_TO_RADIAN - M_PI_2);
        int right_x = x + right_width * cos(angle * LRN_DEG_TO_RADIAN - M_PI_2);
        int right_y = y + right_width * sin(angle * LRN_DEG_TO_RADIAN - M_PI_2);
        const int radius = 1;
#pragma unroll
        for (int dy = -radius; dy <= radius; dy++) {
#pragma unroll
            for (int dx = -radius; dx <= radius; dx++) {
                float dist = dx * dx + dy * dy;
                if (dist <= radius * radius) {
                    int yp = y + dy;
                    int xp = x + dx;
                    if (xp >= 0 && xp < cols && yp >= 0 && yp < rows) {
                        output_with_nonmax[yp * cols * 3 + 3 * xp] = 255 - r;
                        output_with_nonmax[yp * cols * 3 + 3 * xp + 1] = 255 - g;
                        output_with_nonmax[yp * cols * 3 + 3 * xp + 2] = 255 - b;
                    }
                }
            }
        }
    }
}

template <typename DNN_BLOB_TYPE>
LRNDecoder<DNN_BLOB_TYPE>::LRNDecoder(
    // ---------spec---------------
    uint32_t dnn_width, uint32_t dnn_height, uint32_t dnn_channels, uint32_t input_image_width,
    uint32_t input_image_height, int16_t scale_used_for_encoding, int16_t defined_infinity,
    bool normalize, int8_t dnn_radius, LRNEncodingParam encoding_param,
    // --------algorithm parameters-----------------
    int8_t non_max_radius, uint16_t max_possible_nodes, float min_valid_mask_val,
    uint16_t min_votes_for_nodes, uint16_t max_dist_for_nodes,
    // --------algorithm options-----------------
    bool use_direction, int32_t max_possible_classes,
    // --------other context----------------
    tensorflow::OpKernelContext* context)
    : LRNDecoderBase(dnn_width, dnn_height, dnn_channels, input_image_width, input_image_height,
                     scale_used_for_encoding, defined_infinity, normalize, dnn_radius,
                     encoding_param, non_max_radius, max_possible_nodes, min_valid_mask_val,
                     min_votes_for_nodes, max_dist_for_nodes, use_direction, max_possible_classes) {
    CHECK_CUDA_ERROR(cudaMemcpyToSymbolAsync(gConstantGaussianCoeffs, gaussian_coeff_,
                                             (MAX_GAUSSIAN_FILTER_RADIUS * 2 + 1) * sizeof(float),
                                             0, cudaMemcpyHostToDevice));
    stream_ = 0;

    // These are in the original dimension.
    device_votes_.init(context, TensorShape({input_image_height, input_image_width}));

    device_gaussian_votes_.init(context, TensorShape({input_image_height, input_image_width}));
    device_gaussian_votes_tmp_.init(context, TensorShape({input_image_height, input_image_width}));

    CHECK_CUDA_ERROR(cudaMalloc(&device_nodes_, max_possible_nodes_ * sizeof(LRNNode)));
    device_nodes_count_.init(context, TensorShape({1U}));

    // This will be used to select prediction. It better be blurred with Gaussian.
    device_mask_.init(context, TensorShape({dnn_height, dnn_width}));
    device_dist_.init(context, TensorShape({dnn_height, dnn_width}));

    device_dx_.init(context, TensorShape({dnn_height, dnn_width}));
    device_dy_.init(context, TensorShape({dnn_height, dnn_width}));

    if (encoding_param_.getLeftWidthStartPosition() != LRNEncodingParam::INVALID) {
        device_left_width_.init(context, TensorShape({dnn_height, dnn_width}));
        device_right_width_.init(context, TensorShape({dnn_height, dnn_width}));
        device_left_width_sum_.init(context, TensorShape({input_image_height, input_image_width}));
        device_right_width_sum_.init(context, TensorShape({input_image_height, input_image_width}));
    }
    if (use_direction == true) {
        device_cos_.init(context, TensorShape({dnn_height, dnn_width}));
        device_sin_.init(context, TensorShape({dnn_height, dnn_width}));
        device_avg_direction_.init(context, TensorShape({input_image_height, input_image_width}));
        device_cos_votes_.init(context, TensorShape({input_image_height, input_image_width}));
        device_sin_votes_.init(context, TensorShape({input_image_height, input_image_width}));
    }
    if (max_possible_classes > 0) {
        device_votes_local_max_tmp_.init(
            context, TensorShape({max_possible_classes, input_image_height, input_image_width}));
        device_class_.init(context, TensorShape({dnn_height, dnn_width}));
        device_class_votes_.init(
            context, TensorShape({max_possible_classes, input_image_height, input_image_width}));
    }
    threads_ = dim3(DIMX, DIMY);
    blocks_.x = (dnn_width + DIMX - 1) / DIMX;
    blocks_.y = (dnn_height + DIMY - 1) / DIMY;
    blocks_vertical_ = dim3((dnn_height + DIMX - 1) / DIMX, (dnn_width + DIMY - 1) / DIMY);

    blocks_for_original_dim_ =
        dim3((input_image_width + DIMX - 1) / DIMX, (input_image_height + DIMY - 1) / DIMY);
    blocks_for_original_dim_vertical_ =
        dim3((input_image_height + DIMX - 1) / DIMX, (input_image_width + DIMY - 1) / DIMY);

    // Copy color to device memory from `ColorList`.
    int max_num_color = ColorList::get_max_size();
    CHECK_CUDA_ERROR(cudaMallocManaged(&colors_, max_num_color * sizeof(Color)));

    CHECK_CUDA_ERROR(cudaMemPrefetchAsync(colors_, max_num_color * sizeof(Color), cudaCpuDeviceId));
    for (int i = 0; i < max_num_color; ++i) {
        colors_[i] = color_list_.get(i);
    }
    // From below, we only need `colors` to live in device.
    // To avoid page-faulting, call move `colors` to device.
    int deviceId;
    cudaGetDevice(&deviceId);
    CHECK_CUDA_ERROR(cudaMemPrefetchAsync(colors_, max_num_color * sizeof(Color), deviceId));
}

template <typename DNN_BLOB_TYPE>
LRNDecoder<DNN_BLOB_TYPE>::LRNDecoder(
    // ---------spec---------------
    uint32_t dnn_width, uint32_t dnn_height, uint32_t dnn_channels, uint32_t input_image_width,
    uint32_t input_image_height, int16_t scale_used_for_encoding, int16_t defined_infinity,
    bool normalize, int8_t dnn_radius, LRNEncodingParam encoding_param,
    // --------algorithm parameters-----------------
    int8_t non_max_radius, uint16_t max_possible_nodes, float min_valid_mask_val,
    uint16_t min_votes_for_nodes, uint16_t max_dist_for_nodes, OpKernelContext* context)
    : LRNDecoder(dnn_width, dnn_height, dnn_channels, input_image_width, input_image_height,
                 scale_used_for_encoding, defined_infinity, normalize, dnn_radius, encoding_param,
                 non_max_radius, max_possible_nodes, min_valid_mask_val, min_votes_for_nodes,
                 max_dist_for_nodes, false, 0, context) {}

template <typename DNN_BLOB_TYPE>
void LRNDecoder<DNN_BLOB_TYPE>::decode(const DNN_BLOB_TYPE* device_from_dnn) {
    if (encoding_param_.isEncoderDist2D() == false) {
        decodeDNN<DNN_BLOB_TYPE, float, DIMX, DIMY><<<blocks_, threads_, 0, stream_>>>(
            device_mask_.get_dptr(), device_mask_.get_tensor().dim_size(1), device_dist_.get_dptr(),
            device_dist_.get_tensor().dim_size(1), device_dx_.get_dptr(),
            device_dx_.get_tensor().dim_size(1), device_dy_.get_dptr(),
            device_dy_.get_tensor().dim_size(1), device_left_width_.get_dptr(),
            device_left_width_.get_tensor().dim_size(1), device_right_width_.get_dptr(),
            device_right_width_.get_tensor().dim_size(1), device_cos_.get_dptr(),
            device_cos_.get_tensor().dim_size(1), device_sin_.get_dptr(),
            device_sin_.get_tensor().dim_size(1), device_class_.get_dptr(),
            device_class_.get_tensor().dim_size(1), min_valid_mask_val_, scale_used_for_encoding_,
            device_from_dnn, defined_infinity_, normalize_, dnn_width_, dnn_height_, dnn_channels_,
            bit_channels_, encoding_param_.getDXPosition(), encoding_param_.getDYPosition(),
            encoding_param_.getNDXPosition(), encoding_param_.getNDYPosition(),
            encoding_param_.getDirCosPosition(), encoding_param_.getDirSinPosition(),
            encoding_param_.getLeftWidthStartPosition(),
            encoding_param_.getRightWidthStartPosition(), encoding_param_.getBitStartPosition());
    } else {
        decodeDNNFromDist2D<DNN_BLOB_TYPE, float, DIMX, DIMY><<<blocks_, threads_, 0, stream_>>>(
            device_mask_.get_dptr(), device_mask_.get_tensor().dim_size(1), device_dist_.get_dptr(),
            device_dist_.get_tensor().dim_size(1), device_dx_.get_dptr(),
            device_dx_.get_tensor().dim_size(1), device_dy_.get_dptr(),
            device_dy_.get_tensor().dim_size(1), device_left_width_.get_dptr(),
            device_left_width_.get_tensor().dim_size(1), device_right_width_.get_dptr(),
            device_right_width_.get_tensor().dim_size(1), device_cos_.get_dptr(),
            device_cos_.get_tensor().dim_size(1), device_sin_.get_dptr(),
            device_sin_.get_tensor().dim_size(1), device_class_.get_dptr(),
            device_class_.get_tensor().dim_size(1), min_valid_mask_val_, scale_used_for_encoding_,
            device_from_dnn, defined_infinity_, normalize_, dnn_width_, dnn_height_, dnn_channels_,
            bit_channels_, encoding_param_.getMagnitudePosition(),
            encoding_param_.getNormalCosPosition(), encoding_param_.getNormalSinePosition(),
            encoding_param_.getDirCosPosition(), encoding_param_.getDirSinPosition(),
            encoding_param_.getLeftWidthStartPosition(),
            encoding_param_.getRightWidthStartPosition(), encoding_param_.getBitStartPosition());
    }
    CHECK_LAST_CUDA_ERROR();
}

template <typename DNN_BLOB_TYPE>
void LRNDecoder<DNN_BLOB_TYPE>::extractNodes() {
    // We do non-maxima suppression: This the first step of 2D non-max suppression using 2D linear
    // separable way. `m_deviceLocalMaxBuffer` will contain local max value using 1D
    // horizontal kernel window.  At the same time m_devicePeaksCount will be set to
    // zero in this kernel not to do any op in CPU side.
    // `m_deviceVotingMap` is bare voting as an outcome of the previous kernel.
    extractNodesStep1<DIMX, DIMY><<<blocks_for_original_dim_, threads_, 0, stream_>>>(
        device_votes_local_max_tmp_.get_dptr(),
        device_votes_local_max_tmp_.get_tensor().dim_size(2), device_nodes_count_.get_dptr(),
        device_gaussian_votes_.get_dptr(), device_gaussian_votes_.get_tensor().dim_size(1),

        device_class_votes_.get_dptr(), device_class_votes_.get_tensor().dim_size(2),
        non_max_radius_, input_image_height_, input_image_width_, max_possible_classes_);

    CHECK_LAST_CUDA_ERROR();
    // This complete 2D non-max suppression.
    // Then survivors are collected in compact manner using atomic operation.
    // m_devicePeaksCount will keep increasing unbounded but no more survivors are inserted to
    // m_devicePeaks once overflow detected.  So later we need to modify actual size of
    // m_devicePeaksCount.
    extractNodesStep2<DIMX, DIMY><<<blocks_for_original_dim_vertical_, threads_, 0, stream_>>>(
        device_nodes_, max_possible_nodes_, device_nodes_count_.get_dptr(),
        device_avg_direction_.get_dptr(), device_avg_direction_.get_tensor().dim_size(1),
        device_class_votes_.get_dptr(), device_class_votes_.get_tensor().dim_size(2),
        device_votes_.get_dptr(), device_votes_.get_tensor().dim_size(1),
        device_left_width_sum_.get_dptr(), device_left_width_sum_.get_tensor().dim_size(1),
        device_right_width_sum_.get_dptr(), device_right_width_sum_.get_tensor().dim_size(1),
        device_gaussian_votes_.get_dptr(), device_gaussian_votes_.get_tensor().dim_size(1),
        device_votes_local_max_tmp_.get_dptr(),
        device_votes_local_max_tmp_.get_tensor().dim_size(2), min_votes_for_nodes_,
        max_dist_for_nodes_, non_max_radius_, input_image_height_, input_image_width_,
        scale_used_for_encoding_, max_possible_classes_);

    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMemcpy(&host_nodes_count_, device_nodes_count_.get_dptr(), sizeof(int32_t),
                                cudaMemcpyDeviceToHost));
    if (host_nodes_count_ > max_possible_nodes_) {
        host_nodes_count_ = max_possible_nodes_;
    }
}

template <typename DNN_BLOB_TYPE>
void LRNDecoder<DNN_BLOB_TYPE>::vote() {
    dim3 blocks((input_image_width_ + DIMX - 1) / DIMX, (input_image_height_ + DIMY - 1) / DIMY);
    dim3 threads(DIMX, DIMY);
    // Reset all voting values to 0.
    resetVotes<<<blocks, threads, 0, stream_>>>(
        device_votes_.get_dptr(), device_votes_.get_tensor().dim_size(1),
        device_cos_votes_.get_dptr(), device_cos_votes_.get_tensor().dim_size(1),
        device_sin_votes_.get_dptr(), device_sin_votes_.get_tensor().dim_size(1),
        device_left_width_sum_.get_dptr(), device_left_width_sum_.get_tensor().dim_size(1),
        device_right_width_sum_.get_dptr(), device_right_width_sum_.get_tensor().dim_size(1),
        device_avg_direction_.get_dptr(), device_avg_direction_.get_tensor().dim_size(1),
        device_class_votes_.get_dptr(), max_possible_classes_,
        device_class_votes_.get_tensor().dim_size(2), scale_used_for_encoding_, input_image_width_,
        input_image_height_);

    CHECK_LAST_CUDA_ERROR();
    blocks = dim3((dnn_width_ + DIMX - 1) / DIMX, (dnn_height_ + DIMY - 1) / DIMY);
    computeVotesCosSinClasses<DNN_BLOB_TYPE, DIMX, DIMY><<<blocks, threads, 0, stream_>>>(
        device_votes_.get_dptr(), device_votes_.get_tensor().dim_size(1),
        device_left_width_sum_.get_dptr(), device_left_width_sum_.get_tensor().dim_size(1),
        device_right_width_sum_.get_dptr(), device_right_width_sum_.get_tensor().dim_size(1),
        device_cos_votes_.get_dptr(), device_cos_votes_.get_tensor().dim_size(1),
        device_sin_votes_.get_dptr(), device_sin_votes_.get_tensor().dim_size(1),
        device_avg_direction_.get_dptr(), device_avg_direction_.get_tensor().dim_size(1),
        device_class_votes_.get_dptr(),  // 3d tensor
        max_possible_classes_, device_class_votes_.get_tensor().dim_size(2),
        device_mask_.get_dptr(), device_mask_.get_tensor().dim_size(1), device_dist_.get_dptr(),
        device_dist_.get_tensor().dim_size(1), device_dx_.get_dptr(),
        device_dx_.get_tensor().dim_size(1), device_dy_.get_dptr(),
        device_dy_.get_tensor().dim_size(1), device_left_width_.get_dptr(),
        device_left_width_.get_tensor().dim_size(1), device_right_width_.get_dptr(),
        device_right_width_.get_tensor().dim_size(1), device_cos_.get_dptr(),
        device_cos_.get_tensor().dim_size(1), device_sin_.get_dptr(),
        device_sin_.get_tensor().dim_size(1), device_class_.get_dptr(),
        device_class_.get_tensor().dim_size(1), scale_used_for_encoding_,
        min_scaled_valid_mask_val_, defined_infinity_, dnn_radius_ * scale_used_for_encoding_,
        dnn_width_, dnn_height_, input_image_width_, input_image_height_);
    CHECK_LAST_CUDA_ERROR();

    blocks = dim3((input_image_width_ + DIMX - 1) / DIMX, (input_image_height_ + DIMY - 1) / DIMY);

    computeDirectionAndArgMaxKernel<<<blocks, threads, 0, stream_>>>(
        device_votes_.get_dptr(), device_votes_.get_tensor().dim_size(1),

        device_left_width_sum_.get_dptr(), device_left_width_sum_.get_tensor().dim_size(1),
        device_right_width_sum_.get_dptr(), device_right_width_sum_.get_tensor().dim_size(1),

        device_cos_votes_.get_dptr(), device_cos_votes_.get_tensor().dim_size(1),
        device_sin_votes_.get_dptr(), device_sin_votes_.get_tensor().dim_size(1),

        device_avg_direction_.get_dptr(), device_avg_direction_.get_tensor().dim_size(1),
        device_class_votes_.get_dptr(),  // optional classes
        max_possible_classes_, device_class_votes_.get_tensor().dim_size(2),
        scale_used_for_encoding_, input_image_width_, input_image_height_);
    CHECK_LAST_CUDA_ERROR();

    // Try out to do Gaussian blurring on device_votes_.
    blurVotesStep1<DIMX, DIMY><<<blocks_for_original_dim_, threads_, 0, stream_>>>(
        device_gaussian_votes_tmp_.get_dptr(), device_gaussian_votes_tmp_.get_tensor().dim_size(1),
        device_votes_.get_dptr(), device_votes_.get_tensor().dim_size(1), non_max_radius_,
        input_image_width_, input_image_height_);
    CHECK_LAST_CUDA_ERROR();

    blurVotesStep2<DIMX, DIMY><<<blocks_for_original_dim_vertical_, threads_, 0, stream_>>>(
        device_gaussian_votes_.get_dptr(), device_gaussian_votes_.get_tensor().dim_size(1),
        device_gaussian_votes_tmp_.get_dptr(), device_gaussian_votes_tmp_.get_tensor().dim_size(1),
        non_max_radius_, input_image_width_, input_image_height_);
    CHECK_LAST_CUDA_ERROR();
}

template <typename DNN_BLOB_TYPE>
void LRNDecoder<DNN_BLOB_TYPE>::setNMSVotesParams(const uint16_t min_votes_for_nodes) {
    // Specify the threshold of supportive votes on NMS filtering
    // 0 - NMS raw output
    // 1 - at least one vote
    // 2 - at least two votes
    // ...
    min_votes_for_nodes_ = min_votes_for_nodes;
}

template <typename DNN_BLOB_TYPE>
void LRNDecoder<DNN_BLOB_TYPE>::interpretDeviceAsync(const DNN_BLOB_TYPE* device_from_dnn) {
    decode(device_from_dnn);
    vote();
    extractNodes();
}

template <typename DNN_BLOB_TYPE>
void LRNDecoder<DNN_BLOB_TYPE>::free() {
    CHECK_CUDA_ERROR(cudaFree(device_nodes_));
    CHECK_CUDA_ERROR(cudaFree(colors_));
}

template <typename DNN_BLOB_TYPE>
void LRNDecoder<DNN_BLOB_TYPE>::setCUDAStream(cudaStream_t stream) {
    stream_ = stream;
}

template <typename DNN_BLOB_TYPE>
void LRNDecoder<DNN_BLOB_TYPE>::getOutputWithColor(int* output, int* output_with_angle,
                                                   int* output_for_metric, const int arrow_length,
                                                   const int line_thickness,
                                                   const int background_class_id) {
    int blockX = 16;
    int blockY = 16;
    dim3 gridSize((input_image_width_ + blockX - 1) / blockX,
                  (input_image_height_ + blockY - 1) / blockY);
    dim3 blockSize(blockX, blockY);
    getVotesWithColor<<<gridSize, blockSize, 0, stream_>>>(
        output, output_with_angle, output_for_metric, device_votes_.get_dptr(),
        device_left_width_sum_.get_dptr(), device_right_width_sum_.get_dptr(),
        device_class_votes_.get_dptr(), device_avg_direction_.get_dptr(), input_image_width_,
        input_image_height_, min_votes_for_nodes_, colors_, arrow_length, line_thickness,
        background_class_id);

    CHECK_LAST_CUDA_ERROR();
}

template <typename DNN_BLOB_TYPE>
void LRNDecoder<DNN_BLOB_TYPE>::getOutputNonmax(int* output_with_nonmax, const int arrow_length,
                                                const int line_thickness) {
    int threads_per_block = 32;
    int blocks_per_grid = 4;
    if (host_nodes_count_ > 0) {
        if (arrow_length > 0) {
            getNonmaxWithLineDrawn<<<blocks_per_grid, threads_per_block, 0, stream_>>>(
                output_with_nonmax, device_nodes_, colors_, host_nodes_count_, input_image_width_,
                input_image_height_, arrow_length, line_thickness);
            CHECK_LAST_CUDA_ERROR();
        }
        getNonmaxWithColor<<<blocks_per_grid, threads_per_block, 0, stream_>>>(
            output_with_nonmax, device_nodes_, colors_, host_nodes_count_, input_image_width_,
            input_image_height_);
        CHECK_LAST_CUDA_ERROR();
    }
}

template <typename DNN_BLOB_TYPE>
void LRNDecoder<DNN_BLOB_TYPE>::getOutput(int* output, int* output_with_angle,
                                          int* output_with_nonmax, int* output_for_metric,
                                          const int arrow_length, const int line_thickness,
                                          const int background_class_id) {
    int max_num_color = ColorList::get_max_size();
    int deviceId;
    cudaGetDevice(&deviceId);
    CHECK_CUDA_ERROR(cudaMemPrefetchAsync(colors_, max_num_color * sizeof(Color), deviceId));

    getOutputWithColor(output, output_with_angle, output_for_metric, arrow_length, line_thickness,
                       background_class_id);
    getOutputNonmax(output_with_nonmax, arrow_length, line_thickness);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

template class LRNDecoder<float>;
}  // namespace lineregressordecoder
