// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.

#ifndef POLYFIT_H_
#define POLYFIT_H_

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <vector>

#ifdef __CUDACC__
#define CUDA_HOSTDEV __device__
#else
#define CUDA_HOSTDEV
#endif

static constexpr float MIN_DELTA = 0.0000001f;
static constexpr float INVALID_RESIDUAL = 10000.0f;

template <typename T>
struct Vector3 {
    std::vector<T> coeffs_;
    Vector3() {
        coeffs_.reserve(3);
        coeffs_[0] = static_cast<T>(0);
        coeffs_[1] = static_cast<T>(0);
        coeffs_[2] = static_cast<T>(0);
    }
    T& operator[](size_t index) {
        assert((index >= 0) && (index < 3));
        return coeffs_[index];
    }
    const T& operator[](size_t index) const {
        assert((index >= 0) && (index < 3));
        return coeffs_[index];
    }
};

typedef Vector3<float> Vector3f;

/**
* @brief Inverts 3 by 3 matrix
*
* @param out
* @param in
* @return
*/
CUDA_HOSTDEV inline bool invert33(float* out, float* in) {
    bool ret = true;
    out[0] = (in[4] * in[8] - in[5] * in[7]);
    out[1] = (in[5] * in[6] - in[3] * in[8]);
    out[2] = (in[3] * in[7] - in[4] * in[6]);
    out[3] = (in[2] * in[7] - in[1] * in[8]);
    out[4] = (in[0] * in[8] - in[2] * in[6]);
    out[5] = (in[1] * in[6] - in[0] * in[7]);
    out[6] = (in[1] * in[5] - in[2] * in[4]);
    out[7] = (in[2] * in[3] - in[0] * in[5]);
    out[8] = (in[0] * in[4] - in[1] * in[3]);
    float det = in[0] * out[0] + in[1] * out[1] + in[2] * out[2];
    if (fabsf(det) < MIN_DELTA) {
        ret = false;
    }
    for (int32_t i = 0; i < 9; i++) {
        out[i] /= det;
    }
    return ret;
}

CUDA_HOSTDEV inline void get_quad_sum_input(const float x, const float y, float* sum_y1,
                                            float* sum_y2, float* sum_y3, float* sum_y4,
                                            float* sum_x1, float* sum_x1y1, float* sum_x1y2,
                                            float* sum_x2, float* count) {
    *sum_y1 += -y;
    *sum_y2 += y * y;
    *sum_y3 += y * y * y;
    *sum_y4 += y * y * y * y;
    *sum_x1 += x;
    *sum_x1y1 += x * y;
    *sum_x1y2 += x * y * y;
    *sum_x2 += x * x;
    *count += 1.0f;
}

/*
* Fits data to quadratic model. Compute the terms needed.
*
*/
CUDA_HOSTDEV inline float fit_quad(Vector3f* coeffs, const float num_sample,
                                   const float sum_input_1, const float sum_input_2,
                                   const float sum_input_3, const float sum_input_4,
                                   const float sum_output_1, const float sum_output1_input1,
                                   const float sum_output1_input2, const float sumO2) {
    float in[9];
    float out[9];
    float residual_1 = INVALID_RESIDUAL;
    float residual_2 = INVALID_RESIDUAL;
    float ret = residual_1;
    float lambda = 0.0f;
    in[0] = num_sample + lambda;
    in[1] = sum_input_1;
    in[2] = sum_input_2;
    in[3] = sum_input_1;
    in[4] = sum_input_2 + lambda;
    in[5] = sum_input_3;
    in[6] = sum_input_2;
    in[7] = sum_input_3;
    in[8] = sum_input_4 + lambda;
    Vector3f tmp1;
    Vector3f tmp2;
    if (invert33(out, in)) {
        // quadratic fitting
        tmp1[0] = out[0] * sum_output_1 + out[1] * sum_output1_input1 + out[2] * sum_output1_input2;
        tmp1[1] = out[3] * sum_output_1 + out[4] * sum_output1_input1 + out[5] * sum_output1_input2;
        tmp1[2] = out[6] * sum_output_1 + out[7] * sum_output1_input1 + out[8] * sum_output1_input2;
#ifdef __CUDACC__
        residual_1 = fabsf(sumO2 - (sum_output_1 * tmp1[0] + sum_output1_input1 * tmp1[1] +
                                    sum_output1_input2 * tmp1[2]));
#else
        residual_1 = std::fabs(sumO2 - (sum_output_1 * tmp1[0] + sum_output1_input1 * tmp1[1] +
                                        sum_output1_input2 * tmp1[2]));
#endif
    }
    float difference = num_sample * sum_input_2 - sum_input_1 * sum_input_1;
    if (difference > MIN_DELTA) {
        // line fitting
        float reciprocal = 1.0f / difference;
        tmp2[0] = reciprocal * (sum_input_2 * sum_output_1 - sum_input_1 * sum_output1_input1);
        tmp2[1] = reciprocal * (-sum_input_1 * sum_output_1 + num_sample * sum_output1_input1);
        tmp2[2] = 0.0f;
#ifdef __CUDACC__
        residual_2 = fabsf(sumO2 - (sum_output_1 * tmp2[0] + sum_output1_input1 * tmp2[1]));
#else
        residual_2 = std::fabs(sumO2 - (sum_output_1 * tmp2[0] + sum_output1_input1 * tmp2[1]));
#endif
    }
    // choose line based on residual values.
    if (residual_1 < residual_2) {
        *coeffs = tmp1;
        ret = residual_1;
    } else {
        *coeffs = tmp2;
        ret = residual_2;
    }
    return ret;
}
#endif /* POLYFIT_H_ */
