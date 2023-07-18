// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#undef EIGEN_USE_GPU
#include <float.h>

// Include common code.
#include "generate_dist_from_bezier.h"

// We should register only once (CPU).
REGISTER_OP("GenerateDistFromBezier")
    .Input("bezier_curves: float")
    .Input("vertices_count_per_bezier_curve: int32")
    .Input("bezier_curves_count_per_image: int32")
    .Input("bezier_task_ids: int32")
    .Input("bezier_class_ids: int32")
    .Output("encoded: float")
    .Attr("n_classes: list(int) = [1]")
    .Attr("src_width: int")
    .Attr("src_height: int")
    .Attr("down_scale_factor: int = 4")
    .Attr("encode_scale_factor: float = 35")
    .Attr("radius: int = 5")
    .Attr("start_sample_id: int = 0")
    .SetShapeFn([](::shape_inference::InferenceContext* c) {
        // Inherit the output shape from the attributes.
        int src_height;
        TF_RETURN_IF_ERROR(c->GetAttr("src_height", &src_height));
        int src_width;
        TF_RETURN_IF_ERROR(c->GetAttr("src_width", &src_width));

        int down_scale_factor;
        TF_RETURN_IF_ERROR(c->GetAttr("down_scale_factor", &down_scale_factor));

        std::vector<::shape_inference::DimensionHandle> dims_out;
        dims_out.push_back(c->UnknownDim());  // Number of images, like batch size.
        dims_out.push_back(c->UnknownDim());  // Number of channels of output.

        if (down_scale_factor < 1) {
            dims_out.push_back(c->UnknownDim());
            dims_out.push_back(c->UnknownDim());
        } else {
            int target_width = src_width / down_scale_factor;
            int target_height = src_height / down_scale_factor;
            dims_out.push_back(c->MakeDim(target_height));
            dims_out.push_back(c->MakeDim(target_width));
        }

        c->set_output(0, c->MakeShape(dims_out));
        return Status::OK();
    })
    .Doc(R"doc(
        Generate dist from Bezier op.
        Summary:
            Provided a set of bezier curves (represented as control points [[x1,y1],...,[x4,y4]]),
            this operator will encode distance channels to the bezier control points.

        References:
            [1] https://confluence.nvidia.com/display/AV/Bezier+curve+experiment

        Arguments:
            bezier_curves: fp32 tensor of (number_of_vertices, 2)
                containing the bezier control points.
            vertices_count_per_bezier_curve: int32 tensor (number_of_curves)
                containing total number of vertices in each bezier curve.
                sum(vertices_count_per_bezier_curve) == number_of_vertices.
            bezier_curves_count_per_image: int32 tensor (batch_size)
                containing total number of bezier curves in one image.
                sum(bezier_curves_count_per_image) == number_of_curves
            bezier_task_ids: int32 tensor (number_of_curves)
                containing the task id of each bezier curve.
                Task id is 0 based, so as an example, all the bezier curves with task id 0 will be
                rendered into the first set of channels. Bezier curves with negative task ids or
                task ids equal to or greater than n_tasks are ignored.
            bezier_class_ids: int32 tensor (number_of_curves)
                containing the class id of each bezier curve.
                Class id is 0 based. Bezier curves with negative class ids or
                class ids equal to or greater than n_classes are ignored.

        Attributes:
            n_classes (list of int): Number of classes for each task.
                Its length is the number of tasks.
            src_width (int): Expected input label width.
            src_height (int): Expected input label height.
            down_scale_factor (int): Factor that indicates how much would the encoded label
                dimension down scale to, only takes values: 1, 2, 4, 8, 16.
                Default value is 4.
            encode_scale_factor (float): Factor to divide the encoded output.
                Default value is 35.
            radius (int): Radius of pixel along bezier curve to encode distance values.
                Along each bezier curve sample line segment, this operator encode
                only within a radius of pixels with distance values.
                Pixel outside the radius will be encoded with 0.
                Default value is 5.
            start_sample_id (int): The index of the starting sample line segment when
                rendering the mask. The index of the ending sample line segment is
                (num_samples_per_curve - start_sample_id - 1). The starting index has to be
                smaller than the ending index, so it should be greater or equal to zero,
                and less than half of num_samples_per_curve. Note that num_samples_per_curve
                is hardcoded as 16, so that the memory can be allocated with constant size.
                Default value is 0, which renders along the whole curve.

        Returns:
            encoded: fp32 tensor with shape 'NCHW'.
                N: batch size, C: number of channels, H: height, W: width.
                If there are multiple tasks, C becomes number of channels * number of classes.
                The order of encoded channels are:
                [0] Valid mask, with values only 1 or 0, where 1 means within radius of
                    bezier curve, and 0 means outside the radius.
                [1] Distance to the first control point along x axis.
                    If pixel is encoded with value dist_x, it means moving the given pixel along
                    positive x-axis direction to the first bezier control point is dist_x.
                    If this pixel is outside of radius, it can be 0.
                [2] Distance to the first control point along y axis, similar with above but in
                    positive y-axis direction.
                [3] Distance to the second control point along x axis, similar with above.
                [4] Distance to the second control point along y axis, similar with above.
                [5] Distance to the third control point along x axis, similar with above.
                [6] Distance to the third control point along y axis, similar with above.
                [7] Distance to the fourth control point along x axis, similar with above.
                [8] Distance to the fourth control point along y axis, similar with above.
                [9] Depending on n_classes, log2(n_classes) bit coding channels are
                    encoded provide by class id from the inputs.
        )doc");

class GenerateDistFromBezier : public _GenerateDistFromBezier {
 public:
    explicit GenerateDistFromBezier(OpKernelConstruction* context)
        : _GenerateDistFromBezier(context) {}

    void EncodeCore(OpKernelContext* context, const float* bezier_curves,
                    const int* vertices_count_per_bezier_curve,
                    const int* bezier_curves_count_per_image, const int* bezier_task_ids,
                    const int* bezier_class_ids, const int batch_size, const int rows,
                    const int cols, float* output, const int output_total_channels,
                    const int output_rows, const int output_cols, const int radius,
                    const int down_scale_factor, const float encode_scale_factor,
                    const int start_sample_id, const int num_input_tasks, const int* num_classes,
                    const int* num_bits, const int* output_channels) {
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
                (&vertices_count_per_bezier_curve[offset_curve_count]);
            const int* bezier_task_ids_per_image = (&bezier_task_ids[offset_curve_count]);
            const int* bezier_class_ids_per_image = (&bezier_class_ids[offset_curve_count]);
            const float* bezier_curves_per_image = (&bezier_curves[cols * offset_vertex_count]);

            encode_bezier_dist_maps(
                bezier_curves_per_image, vertices_counts_per_image, bezier_task_ids_per_image,
                bezier_class_ids_per_image, bezier_curves_count_per_image[img_index], cols,
                &output[img_index * output_total_channels * output_rows * output_cols],
                output_total_channels, output_rows, output_cols, radius, down_scale_factor,
                encode_scale_factor, start_sample_id, num_input_tasks, num_classes, num_bits,
                output_channels);
        }
    }

    void encode_bezier_dist_maps(const float* bezier_curves, const int* vertices_counts,
                                 const int* task_ids, const int* class_ids,
                                 int total_bezier_curves_count, int fixed_cols, float* output,
                                 const int output_total_channels, const int output_rows,
                                 const int output_cols, const int radius,
                                 const int down_scale_factor, const float encode_scale_factor,
                                 const int start_sample_id, const int num_input_tasks,
                                 const int* num_classes, const int* num_bits,
                                 const int* output_channels) {
        // For every task.
        int offset_channel = 0;
        for (int current_task_id = 0; current_task_id < num_input_tasks; current_task_id++) {
            if (current_task_id > 0) {
                offset_channel += output_channels[current_task_id - 1];
            }
            // Bezier sample points.
            float bezier_sampling_points[MAX_NUM_BEZIER_CURVES * (NUM_SAMPLES_PER_CURVE + 1) * 2];
            int offset_per_curve[MAX_NUM_BEZIER_CURVES];
            int offset_vertices = 0;
            int valid_curve_count = 0;
            for (int i = 0; i < total_bezier_curves_count; i++) {
                // Ignore any curve that has invalid number of vertices or invalid task/class id.
                if (valid_curve_count < MAX_NUM_BEZIER_CURVES &&
                    vertices_counts[i] == NUM_BEZIER_CTRL_PTS && task_ids[i] == current_task_id &&
                    class_ids[i] >= 0 && class_ids[i] < num_classes[current_task_id]) {
                    const float* bezier_points = &bezier_curves[offset_vertices * fixed_cols];
                    offset_per_curve[valid_curve_count] = offset_vertices;
                    float x, y;
                    for (int n = 0; n <= NUM_SAMPLES_PER_CURVE; n++) {
                        float t = static_cast<float>(n) / NUM_SAMPLES_PER_CURVE;
                        get_bezier_sample(bezier_points, t, &x, &y);

                        bezier_sampling_points
                            [(valid_curve_count * (NUM_SAMPLES_PER_CURVE + 1) + n) * 2 + PTS1_X] =
                                x;
                        bezier_sampling_points
                            [(valid_curve_count * (NUM_SAMPLES_PER_CURVE + 1) + n) * 2 + PTS1_Y] =
                                y;
                    }
                    valid_curve_count++;
                }
                offset_vertices += vertices_counts[i];
            }

            // For every pixel we will measure distance from it to the bezier curves.
            for (int y = 0; y < output_rows; y++) {
                float oy = y * down_scale_factor;
                for (int x = 0; x < output_cols; x++) {
                    float ox = x * down_scale_factor;
                    for (int c = 0; c < output_channels[current_task_id]; c++) {
                        output[y * output_cols + x + output_rows * output_cols * c +
                               offset_channel * output_rows * output_cols] = 0;
                    }
                    // For every bezier curve.
                    float min_dist = static_cast<float>(radius * down_scale_factor);
                    int min_dist_offset = -1;
                    int min_dist_class_id = -1;
                    for (int i = 0; i < valid_curve_count; i++) {
                        float min_dist_to_segment = FLT_MAX;
                        // For every sample line segment.
                        for (int n = start_sample_id; n < NUM_SAMPLES_PER_CURVE - start_sample_id;
                             n++) {
                            int start_idx = (i * (NUM_SAMPLES_PER_CURVE + 1) + n) * 2;
                            float x1 = bezier_sampling_points[start_idx + PTS1_X];
                            float y1 = bezier_sampling_points[start_idx + PTS1_Y];
                            float x2 = bezier_sampling_points[start_idx + PTS2_X];
                            float y2 = bezier_sampling_points[start_idx + PTS2_Y];
                            // Get the distance from the point to a sample line segment.
                            float dist_to_segment =
                                get_distance_from_point_to_line_segment(x1, y1, x2, y2, ox, oy);
                            if (dist_to_segment < min_dist_to_segment) {
                                min_dist_to_segment = dist_to_segment;
                            }
                        }
                        // Update the closest curve and the corresponding distance.
                        float dist = min_dist_to_segment;
                        if (dist < min_dist) {
                            min_dist = dist;
                            min_dist_offset = offset_per_curve[i];
                            min_dist_class_id = class_ids[i];
                        }
                    }
                    // Render the pixel to the closest curve if within the radius.
                    if (min_dist_offset >= 0) {
                        encode_dist(output + offset_channel * output_rows * output_cols +
                                        y * output_cols + x,
                                    output_rows * output_cols,
                                    bezier_curves + min_dist_offset * fixed_cols, ox, oy,
                                    min_dist_class_id, num_bits[current_task_id],
                                    encode_scale_factor * down_scale_factor);
                    }
                }
            }
        }
    }
};

#pragma message("Registering CPU kernel")
REGISTER_KERNEL_BUILDER(Name("GenerateDistFromBezier")
                            .Device(DEVICE_CPU)
                            .HostMemory("bezier_curves")
                            .HostMemory("vertices_count_per_bezier_curve")
                            .HostMemory("bezier_curves_count_per_image")
                            .HostMemory("bezier_task_ids")
                            .HostMemory("bezier_class_ids"),
                        GenerateDistFromBezier);
