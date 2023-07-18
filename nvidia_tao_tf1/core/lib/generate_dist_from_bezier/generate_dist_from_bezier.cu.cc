// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#define EIGEN_USE_GPU

// Include common code.
#include "generate_dist_from_bezier.h"

// Below includes are checked into ai-infra repo.
#include "tensorflow/core/kernels/gpu_device_array.h"
#include "tensorflow/core/kernels/gpu_device_array_gpu.h"

// Put GPU code into a namespace to avoid name clashes when
// linking CPU and GPU versions to the same library.
namespace GPUCode {

constexpr int GDL_DIMX = 32;
constexpr int GDL_DIMY = 4;

__global__ void reset_kernel(float* device_output, const int batch_size, const int output_channels,
                             const int output_rows, const int output_cols) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    int linear_image_size = output_cols * output_rows;
    if (index < linear_image_size * batch_size) {
        int batch_index = index / linear_image_size;
        int linear_index = index % linear_image_size;
        for (int i = 0; i < output_channels; i++) {
            device_output[batch_index * output_channels * linear_image_size +
                          i * linear_image_size + linear_index] = 0.0f;
        }
    }
}

__global__ void encode_bezier_dist_maps_kernel(
    const float* device_bezier_curves, const int* vertices_counts, const int* task_ids,
    const int* class_ids, int total_bezier_curves_count, int fixed_cols, float* device_output,
    const int output_total_channels, const int output_rows, const int output_cols, const int radius,
    const int down_scale_factor, const float encode_scale_factor, const int num_input_tasks,
    const int* num_classes, const int* num_bits, const int* output_channels,
    const int num_samples_per_curve, const int start_sample_id, const int num_bezier_ctrl_pts) {
    // For every pixel we will measure distance from it to the bezier curves.
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= output_cols || y >= output_rows) return;
    float oy = y * down_scale_factor;
    float ox = x * down_scale_factor;

    // For every task.
    int offset_channel = 0;
    for (int current_task_id = 0; current_task_id < num_input_tasks; current_task_id++) {
        if (current_task_id > 0) {
            offset_channel += output_channels[current_task_id - 1];
        }
        // For every bezier curve.
        float min_dist = static_cast<float>(radius * down_scale_factor);
        int min_dist_offset = -1;
        int min_dist_class_id = -1;
        int offset_vertices = 0;
        for (int i = 0; i < total_bezier_curves_count; i++) {
            // Ignore any curve that has invalid number of vertices or invalid task/class id.
            if (vertices_counts[i] == num_bezier_ctrl_pts && task_ids[i] == current_task_id &&
                class_ids[i] >= 0 && class_ids[i] < num_classes[current_task_id]) {
                float min_dist_to_segment = FLT_MAX;
                const float* bezier_points = &device_bezier_curves[offset_vertices * fixed_cols];
                float x_prev, y_prev, x_next, y_next;
                // Bezier sample points.
                float t = static_cast<float>(start_sample_id) / num_samples_per_curve;
                get_bezier_sample(bezier_points, t, &x_prev, &y_prev);
                for (int n = start_sample_id; n < num_samples_per_curve - start_sample_id; n++) {
                    t = static_cast<float>(n + 1) / num_samples_per_curve;
                    get_bezier_sample(bezier_points, t, &x_next, &y_next);
                    // Get the distance from the point to a sample line segment.
                    float dist_to_segment = get_distance_from_point_to_line_segment(
                        x_prev, y_prev, x_next, y_next, ox, oy);
                    if (dist_to_segment < min_dist_to_segment) {
                        min_dist_to_segment = dist_to_segment;
                    }
                    // Move to the next sample line segment.
                    x_prev = x_next;
                    y_prev = y_next;
                }
                // Update the closest curve and the corresponding distance.
                float dist = min_dist_to_segment;
                if (dist < min_dist) {
                    min_dist = dist;
                    min_dist_offset = offset_vertices;
                    min_dist_class_id = class_ids[i];
                }
            }
            offset_vertices += vertices_counts[i];
        }
        // Render the pixel to the closest curve if within the radius.
        if (min_dist_offset >= 0) {
            encode_dist(
                device_output + offset_channel * output_rows * output_cols + y * output_cols + x,
                output_rows * output_cols, device_bezier_curves + min_dist_offset * fixed_cols, ox,
                oy, min_dist_class_id, num_bits[current_task_id],
                encode_scale_factor * down_scale_factor);
        }
    }
}

class GenerateDistFromBezier : public _GenerateDistFromBezier {
 public:
    explicit GenerateDistFromBezier(OpKernelConstruction* context)
        : _GenerateDistFromBezier(context) {}

    void EncodeCore(OpKernelContext* context, const float* device_bezier_curves,
                    const int* vertices_count_per_bezier_curve,
                    const int* bezier_curves_count_per_image, const int* bezier_task_ids,
                    const int* bezier_class_ids, const int batch_size, const int rows,
                    const int cols, float* device_output, const int output_total_channels,
                    const int output_rows, const int output_cols, const int radius,
                    const int down_scale_factor, const float encode_scale_factor,
                    const int start_sample_id, const int num_input_tasks, const int* num_classes,
                    const int* num_bits, const int* output_channels) {
        // Copy vertices_count from CPU memory to GPU memory, as it is used both before the kernel
        // and inside the kernel.
        GpuDeviceArrayOnHost<int, 0> vertices_array(context, rows);
        if (rows) {  // Only allocate memory if nonzero.
            OP_REQUIRES_OK(context, vertices_array.Init());
            for (int i = 0; i < rows; i++)
                vertices_array.Set(i, vertices_count_per_bezier_curve[i]);
            OP_REQUIRES_OK(context, vertices_array.Finalize());
        }
        auto vertices_data = vertices_array.data();
        const int* device_vertices_count_per_bezier_curve =
            GetGpuDeviceArrayOnDevice(&vertices_data);

        // Copy num_classes from CPU memory to GPU memory, as it is used both before the kernel
        // and inside the kernel.
        GpuDeviceArrayOnHost<int, 0> num_classes_array(context, num_input_tasks);
        OP_REQUIRES_OK(context, num_classes_array.Init());
        for (int i = 0; i < num_input_tasks; i++) num_classes_array.Set(i, num_classes[i]);
        OP_REQUIRES_OK(context, num_classes_array.Finalize());
        auto num_classes_data = num_classes_array.data();
        const int* device_num_classes = GetGpuDeviceArrayOnDevice(&num_classes_data);

        // Copy num_bits from CPU memory to GPU memory, as it is used both before the kernel
        // and inside the kernel.
        GpuDeviceArrayOnHost<int, 0> num_bits_array(context, num_input_tasks);
        OP_REQUIRES_OK(context, num_bits_array.Init());
        for (int i = 0; i < num_input_tasks; i++) num_bits_array.Set(i, num_bits[i]);
        OP_REQUIRES_OK(context, num_bits_array.Finalize());
        auto num_bits_data = num_bits_array.data();
        const int* device_num_bits = GetGpuDeviceArrayOnDevice(&num_bits_data);

        // Copy output_channels from CPU memory to GPU memory, and it is used both before the kernel
        // and inside the kernel.
        GpuDeviceArrayOnHost<int, 0> output_channels_array(context, num_input_tasks);
        OP_REQUIRES_OK(context, output_channels_array.Init());
        for (int i = 0; i < num_input_tasks; i++) output_channels_array.Set(i, output_channels[i]);
        OP_REQUIRES_OK(context, output_channels_array.Finalize());
        auto output_channels_data = output_channels_array.data();
        const int* device_output_channels = GetGpuDeviceArrayOnDevice(&output_channels_data);

        // Set device_output to 0.
        dim3 threads(512);
        dim3 blocks((output_rows * output_cols * batch_size + threads.x - 1) / threads.x);
        const Eigen::GpuDevice& d = context->eigen_gpu_device();
        reset_kernel<<<blocks, threads, 0, d.stream()>>>(
            device_output, batch_size, output_total_channels, output_rows, output_cols);

        // GPU encoding starts below.
        dim3 threadsMain(GDL_DIMX, GDL_DIMY);
        dim3 blocksMain((output_cols + GDL_DIMX - 1) / GDL_DIMX,
                        (output_rows + GDL_DIMY - 1) / GDL_DIMY);

        int offset_vertex_count = 0;
        int offset_curve_count = 0;
        for (int img_index = 0; img_index < batch_size; img_index++) {
            if (img_index > 0) {
                for (int curve_index = 0;
                     curve_index < bezier_curves_count_per_image[img_index - 1]; curve_index++) {
                    // Update the offset for bezier curves based on vertices count.
                    offset_vertex_count +=
                        vertices_count_per_bezier_curve[offset_curve_count + curve_index];
                }
                // Update the offset for vertices count based on curve count.
                offset_curve_count += bezier_curves_count_per_image[img_index - 1];
            }
            const int* vertices_counts_per_image =
                (&device_vertices_count_per_bezier_curve[offset_curve_count]);
            const int* bezier_task_ids_per_image = (&bezier_task_ids[offset_curve_count]);
            const int* bezier_class_ids_per_image = (&bezier_class_ids[offset_curve_count]);
            const float* bezier_curves_per_image =
                (&device_bezier_curves[cols * offset_vertex_count]);

            encode_bezier_dist_maps_kernel<<<blocksMain, threadsMain, 0, d.stream()>>>(
                bezier_curves_per_image, vertices_counts_per_image, bezier_task_ids_per_image,
                bezier_class_ids_per_image, bezier_curves_count_per_image[img_index], cols,
                &device_output[img_index * output_total_channels * output_rows * output_cols],
                output_total_channels, output_rows, output_cols, radius, down_scale_factor,
                encode_scale_factor, num_input_tasks, device_num_classes, device_num_bits,
                device_output_channels, NUM_SAMPLES_PER_CURVE, start_sample_id,
                NUM_BEZIER_CTRL_PTS);
        }
    }
};

#pragma message("Registering GPU kernel")
REGISTER_KERNEL_BUILDER(Name("GenerateDistFromBezier")
                            .Device(DEVICE_GPU)
                            // These inputs must be in CPU accessible memory,
                            // since we need to touch when we launch kernel.
                            .HostMemory("vertices_count_per_bezier_curve")
                            .HostMemory("bezier_curves_count_per_image"),
                        GenerateDistFromBezier);

}  // namespace GPUCode
