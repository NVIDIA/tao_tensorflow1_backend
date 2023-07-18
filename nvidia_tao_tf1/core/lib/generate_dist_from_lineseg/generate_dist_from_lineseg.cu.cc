// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#define EIGEN_USE_GPU

// Include common code
#include "generate_dist_from_lineseg.h"

// Put GPU code into a namespace to avoid name clashes when
// linking CPU and GPU versions to the same library
namespace GPUCode {

constexpr int GDL_DIMX = 32;
constexpr int GDL_DIMY = 4;

__global__ void reset_kernel(float* device_output, float* device_dist_2d, float* device_cluster_id,
                             float* device_class_id_in_integer, float* device_weights,
                             float* device_dontcare_angles, const int batch_size,
                             const int output_channels, const int output_rows,
                             const int output_cols, const int defined_infinity) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    int linear_image_size = output_cols * output_rows;
    if (index < linear_image_size * batch_size) {
        device_cluster_id[index] = 0;
        device_class_id_in_integer[index] = 0;
        device_dist_2d[index] = defined_infinity;
        device_weights[index] = 0;
        device_dontcare_angles[index] = 0;
        int batch_index = index / linear_image_size;
        int linear_index = index % linear_image_size;
        for (int i = 0; i < output_channels; i++) {
            device_output[batch_index * output_channels * linear_image_size +
                          i * linear_image_size + linear_index] = 0.0f;
        }
    }
}

__global__ void encode_vector_angles_bits_kernel(
    const float* device_first_linesement, int fixed_cols, int total_lineseg_count,
    float* device_output, float* device_dist_2d, float* device_cluster_id,
    float* device_class_id_in_integer, float* device_weights, float* device_dontcare_angles,
    const int output_channels, const int output_rows, const int output_cols, const int radius,
    const int cluster_radius, const int class_radius, const int defined_infinity,
    const int down_scale_factor, const int starting_encoding_dim_for_bitcoding,
    const int channel_count_for_bit_coding, const int num_input_classes, const int encoding_option,
    const bool normalize) {
    // for every pixel we will measure distance from it to the line segments....
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= output_cols || y >= output_rows) return;
    float oy = y * down_scale_factor;
    float ox = x * down_scale_factor;

    device_weights[y * output_cols + x] = weight_scheme(y, output_rows * 0.5f);
    // for every linesegment.
    float min_distance = static_cast<float>(defined_infinity);
    float min_vx;
    float min_vy;
    float min_angle;
    float min_class;
    float min_id;
    float min_width_left;
    float min_width_right;

    for (int i = 0; i < total_lineseg_count; i++) {
        float x2 = device_first_linesement[i * fixed_cols];
        float y2 = device_first_linesement[i * fixed_cols + 1];
        float x1 = device_first_linesement[i * fixed_cols + 2];
        float y1 = device_first_linesement[i * fixed_cols + 3];
        float angle = device_first_linesement[i * fixed_cols + 4];
        float angle_top = device_first_linesement[i * fixed_cols + 5];
        float angle_bottom = device_first_linesement[i * fixed_cols + 6];
        float class_id = device_first_linesement[i * fixed_cols + 7];
        float instance_id = device_first_linesement[i * fixed_cols + 8];
        float width_left_top = device_first_linesement[i * fixed_cols + 9];
        float width_right_top = device_first_linesement[i * fixed_cols + 10];
        float width_left_bottom = device_first_linesement[i * fixed_cols + 11];
        float width_right_bottom = device_first_linesement[i * fixed_cols + 12];

        // from ox and oy we need to draw normal vector
        // to the line (x1,y1) to (x2,y2)

        float vx;
        float vy;
        int interection_with_bottom_middle_top;
        float final_angle = angle;
        float width_left = width_left_top;
        float width_right = width_right_top;

        if (get_closest_vector_and_angle_from_point_to_line(
                x1, y1, x2, y2, ox, oy, angle_top, angle_bottom, width_left_top, width_right_top,
                width_left_bottom, width_right_bottom, &vx, &vy,
                &interection_with_bottom_middle_top, &final_angle, &width_left, &width_right)) {
            vx /= down_scale_factor;
            vy /= down_scale_factor;
            width_left /= down_scale_factor;
            width_right /= down_scale_factor;

            float dist = norm2(vx, vy);

            if (dist < min_distance) {
                min_distance = dist;
                min_angle = final_angle;
                min_vx = vx;
                min_vy = vy;
                min_class = class_id;
                min_id = instance_id;
                min_width_left = width_left;
                min_width_right = width_right;
            }
        }
    }

    if (min_distance < defined_infinity && device_dist_2d[y * output_cols + x] > min_distance) {
        // Update channel.
        device_dist_2d[y * output_cols + x] = min_distance;
        if (min_distance < cluster_radius) {
            device_cluster_id[y * output_cols + x] = min_id;
        }
        if (min_distance <= class_radius) {
            if (!(min_class < 0.f) && (min_class < (static_cast<float>(num_input_classes) + 1))) {
                device_class_id_in_integer[y * output_cols + x] = min_class;
            }
        }

        if (min_angle < DONTCARE_ANGLE_THRESHOLD) {
            device_dontcare_angles[y * output_cols + x] = 255;
        }
        if (normalize) {
            min_vx /= static_cast<float>(defined_infinity);
            min_vy /= static_cast<float>(defined_infinity);
            min_width_left /= static_cast<float>(defined_infinity);
            min_width_right /= static_cast<float>(defined_infinity);
        }
        if (encoding_option == 0) {
            encode_option_0(device_output, x, y, output_rows, output_cols, radius, min_vx, min_vy,
                            min_distance, min_width_left, min_width_right, min_class, min_angle,
                            channel_count_for_bit_coding);
        } else {
            encode_option_1(device_output, x, y, output_rows, output_cols, radius, min_vx, min_vy,
                            min_distance, min_width_left, min_width_right, min_class, min_angle,
                            channel_count_for_bit_coding);
        }
    }
}

class GenerateDistFromLineseg : public _GenerateDistFromLineseg {
 public:
    explicit GenerateDistFromLineseg(OpKernelConstruction* context)
        : _GenerateDistFromLineseg(context) {}

    void EncodeCore(OpKernelContext* context, const int* lineseg_count_per_image,
                    const int batch_size, const float* device_linesegments, const int rows,
                    const int cols, const int encoding_option, float* device_output,
                    float* device_output_dist2d, float* device_output_cluster_id,
                    float* device_output_class_id_int, float* device_weights,
                    float* device_dontcare_angles, const int output_channels, const int output_rows,
                    const int output_cols, const int radius, const int defined_infinity,
                    const int down_scale_factor, const int starting_encoding_dim_for_bitcoding,
                    const int channel_count_for_bit_coding, const int num_input_classes,
                    const bool normalize) {
        if (verbose_) {
            printf("running GPU version\n");
        }

        // set
        // device_output_dist2d to defined_infinity
        // device_output to 0
        // device_output_cluster_id to 0
        // device_output_class_id_int to 0
        dim3 threads(512);
        dim3 blocks((output_rows * output_cols * batch_size + threads.x - 1) / threads.x);
        const Eigen::GpuDevice& d = context->eigen_gpu_device();
        reset_kernel<<<blocks, threads, 0, d.stream()>>>(
            device_output, device_output_dist2d, device_output_cluster_id,
            device_output_class_id_int, device_weights, device_dontcare_angles, batch_size,
            output_channels, output_rows, output_cols, defined_infinity);

        // GPU encoding starts below.
        dim3 threadsMain(GDL_DIMX, GDL_DIMY);
        dim3 blocksMain((output_cols + GDL_DIMX - 1) / GDL_DIMX,
                        (output_rows + GDL_DIMY - 1) / GDL_DIMY);

        int offset = 0;
        for (int img_index = 0; img_index < batch_size; img_index++) {
            if (img_index > 0) {
                offset += lineseg_count_per_image[img_index - 1];
            }
            const float* first_linesegment = (&device_linesegments[cols * offset]);

            encode_vector_angles_bits_kernel<<<blocksMain, threadsMain, 0, d.stream()>>>(
                first_linesegment, cols, lineseg_count_per_image[img_index],
                &device_output[img_index * output_channels * output_rows * output_cols],
                &device_output_dist2d[img_index * output_rows * output_cols],
                &device_output_cluster_id[img_index * output_rows * output_cols],
                &device_output_class_id_int[img_index * output_rows * output_cols],
                &device_weights[img_index * output_rows * output_cols],
                &device_dontcare_angles[img_index * output_rows * output_cols], output_channels,
                output_rows, output_cols, radius, cluster_radius_, class_radius_, defined_infinity,
                down_scale_factor, starting_encoding_dim_for_bitcoding,
                channel_count_for_bit_coding, num_input_classes, encoding_option, normalize);
        }
    }
};

#pragma message("Registering GPU kernel")
REGISTER_KERNEL_BUILDER(Name("GenerateDistFromLineseg")
                            .Device(DEVICE_GPU)
                            .HostMemory("line_segments_count_per_image"),  // this input must be in
                                                                           // CPU accessible memory
                                                                           // since we need to touch
                                                                           // when we launch kernel
                        GenerateDistFromLineseg);

}  // namespace GPUCode
