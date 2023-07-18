// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#undef EIGEN_USE_GPU
#include <float.h>

// Include common code
#include "generate_dist_from_lineseg.h"

// We should register only once (CPU)
REGISTER_OP("GenerateDistFromLineseg")
    .Input("line_segments_count_per_image: int32")
    .Input("line_segments: float")
    .Output("encoded: float")
    .Output("dist2d: float")
    .Output("cluster_id: float")
    .Output("class_id: float")
    .Output("weights: float")
    .Output("dontcare_angles: float")
    .Attr("n_classes: int = 1")
    .Attr("src_width: int")
    .Attr("src_height: int")
    .Attr("down_scale_factor: int = 4")
    .Attr("encoding_option: {'dist', 'angle'}")
    .Attr("radius: int = 20")
    .Attr("cluster_radius: int = 5")
    .Attr("class_radius: int = 1")
    .Attr("defined_infinity: int = 30")
    .Attr("normalize: bool = false")
    .Attr("verbose: bool = false")
    .SetShapeFn([](::shape_inference::InferenceContext* c) {
        // Inherit the output shape from the attributes
        int src_height;
        TF_RETURN_IF_ERROR(c->GetAttr("src_height", &src_height));
        int src_width;
        TF_RETURN_IF_ERROR(c->GetAttr("src_width", &src_width));

        int down_scale_factor;
        TF_RETURN_IF_ERROR(c->GetAttr("down_scale_factor", &down_scale_factor));

        std::vector<::shape_inference::DimensionHandle> dims_out;
        dims_out.push_back(c->UnknownDim());  // Number of images. like batch size
        dims_out.push_back(c->UnknownDim());  // Number of channels of output

        std::vector<::shape_inference::DimensionHandle> dims_out2;
        dims_out2.push_back(c->UnknownDim());  // Number of images. like batch size
        dims_out2.push_back(c->MakeDim(1));    // Number of channels of output

        if (down_scale_factor < 1) {
            dims_out.push_back(c->UnknownDim());
            dims_out.push_back(c->UnknownDim());
            dims_out2.push_back(c->UnknownDim());
            dims_out2.push_back(c->UnknownDim());
        } else {
            int target_width = src_width / down_scale_factor;
            int target_height = src_height / down_scale_factor;

            dims_out.push_back(c->MakeDim(target_height));
            dims_out.push_back(c->MakeDim(target_width));
            dims_out2.push_back(c->MakeDim(target_height));
            dims_out2.push_back(c->MakeDim(target_width));
        }

        c->set_output(0, c->MakeShape(dims_out));

        c->set_output(1, c->MakeShape(dims_out2));

        c->set_output(2, c->MakeShape(dims_out2));

        c->set_output(3, c->MakeShape(dims_out2));

        c->set_output(4, c->MakeShape(dims_out2));

        c->set_output(5, c->MakeShape(dims_out2));
        return Status::OK();
    })
    .Doc(R"doc(
        Generate dist from lineseg op.
        Summary:
            Provided a set of points (represent as [x,y]) that from a set line segments,
            number of line segments in each label object, such as a set of line segments
            can be from one lane polygon, number of line segments in each image, related
            angle values, class values, etc., this operator will encode distance channels,
            angle channels, bit coding channels.

        References:
            [1] https://confluence.nvidia.com/display/AV/Line+Regressor+Encoding

        Arguments:
            line_segments_count_per_image: a int32 tensor with (batch_size, 1)
                total number of line segments in one image.
            line_segments: a fp32 tensor of (Number_of_line_segments, 13)
                13 channels contain details information for each line segments:
                    -0- line segments top point x coordinate
                    -1- line segments top point y coordinate
                    -2- line segments bottom point x coordinate
                    -3- line segments bottom point y coordinate
                    -4- angle (in radians) for this line segment
                    -5- angle (in radians) at top point of this line segment, averaged from
                        two adjacent line segments.
                    -6- angle (in radians) at bottom point of this line segment, averaged from
                        two adjacent line segments.
                    -7- class id of this line segment
                    -8- cluster id of this line segment
                        since one lane class can have multiple lane polygon
                    -9- width to the left boundary from top point
                    -10- width to the right boundary from top point
                    -11- width to the left boundary from bottom point
                    -12- width to the left boundary from bottom point

        Attributes:
            n_classes: number of class type to identify the labels.
            src_width: expected input label width.
            src_height: expected input label height.
            down_scale_factor: factor of down scale for output label dimension.
                This operator requires down_scale_factor to be in one of following
                values: 1, 2, 4, 8, 16.
            encoding_option: encoding option. Currently only supports option 'dist'.
                (1) default option (distance oriented output channels) provides `encoded`
                output tensor as follows:
                -0:MASK
                -1:dx
                -2:dy
                -3:ndx
                -4:ndy
                -5:(cos+1)*0.5
                -6:(sin+1)*0.5
                -7:width_left
                -8:width_right
                -(9-N): bit coding channels,
                depends on how total number of classes we have (n_classes)
                we get ceil(log2(n_classes)) number of bit coding channels.
                (2) another option (angle oriented output channels) provides `encoded`
                output tensor as follows:
                option 1
                -0:MASK
                -1:dist
                -2:(cos+1)*0.5 toward 0 dist
                -3:(sin+1)*0.5 toward 0 dist
                -4:(cos+1)*0.5 direction
                -5:(sin+1)*0.5 direction
                -6:width_left
                -7:width_right
                -(8-N): bit coding
                same bit-codding channels as default option
            radius: radius of pixel along line segment to encode distance values.
                along pixel for each line segment, this operator encode
                only within a radius of pixels with distance values. Pixel
                outside the radius will be encoded with defined infinity value.
            cluster_radius: similar with radius above, instead encoded with cluster
                values that provided by line_segments input.
            class_radius: similar with radius above, instead encoded with class id
                values that provided by line_segments input.
            defined_infinity: distance threshold that defines infinity distance, see definition
                for radius. Pixel is not on the line segment and its shortest distance to closest
                line segment will be encoded this value. This operator requires
                defined_infinity > radius.
            normalize: boolean type to set if output distance channels should be normalized to have
                range between 0 and 1. (divide by defined_infinity).
            verbose: if print out some extra information

        Returns:
            encoded: a fp32 tensor with shape 'NCHW'.
                N: batch size, C: number of channels, H: height, W:width.
                The order of encoded channels (for encoding_option='dist') are:
                [0] valid mask, with values only 1 or 0, pixel has value
                    where 1 means within radius of line segment, 0 means
                    outside the radius.
                [1] positive x-axis direction vector.
                    if pixel encode with value dist_x, meaning moving along
                    positive x-axis direction with dist_x units, to closest
                    line segment pixel. if this pixel is along line segment,
                    dist_x = 0. if this pixel is outside of radius, dist_x
                    can be defined_infinity, etc.
                [2] positive y-axis direction vector, similar with above but in
                    in positive y-axis direction.
                [3] negative x-axis direction vector, similar with above but in
                    negative x-axis direction.
                [4] negative y-axis direction vector, similar with above but in
                    negative y-axis direction.
                [5] cosine channel. encode cosine channels, provided by
                    input line segment has angle values. Given angle value for
                    line segment theta, encoded as (cos(theta)+1)*0.5.
                [6] sine channel. encode sine channels, provided by
                    input line segment has angle values. Given angle value for
                    line segment theta, encoded as (sin(theta)+1)*0.5.
                [7] width length for the center line to left edge.
                [8] width length for the center line to right edge.
                [9] and above, depending nclasses, log2(nclasses) bit coding channels
                    encoded provide by class id from line segment inputs.
                The order of encoded channels (for encoding_option='angle') are:
                [0] valid mask, with values only 1 or 0, pixel has value
                    where 1 means within radius of line segment, 0 means
                    outside the radius.
                [1] distance magnitude, sqrt(x**2+y**2) using the the distance channels from 1-4
                    for the above encoding option ('dist').
                [2] cosine channel, angle=atan2(y, x) same x, y used above for getting the distance
                    magnitude, encoded as (cos(angle)+1)*0.5.
                [3] sine channel, angle=atan2(y, x) same x, y used above for getting the distance
                    magnitude, encoded as (cos(angle)+1)*0.5. please notice, the cosine and sine
                    are different as below channels.
                [4] cosine channel. encode cosine channels, provided by
                    input line segment has angle values. Given angle value for
                    line segment theta, encoded as (cos(theta)+1)*0.5.
                [5] sine channel. encode sine channels, provided by
                    input line segment has angle values. Given angle value for
                    line segment theta, encoded as (sin(theta)+1)*0.5.
                [6] width length for the center line to left edge.
                [7] width length for the center line to right edge.
                [8] and above, depending nclasses, log2(nclasses) bit coding channels
                    encoded provide by class id from line segment inputs.
            dist2d: a fp32 tensor with shape 'NCHW', where C=1, distance tensor indicating the
                2d distance from each pixel to closest line segment pixels.
            cluster_id: a fp32 tensor with shape 'NCHW', where C=1, distance tensor indicating
                the original cluster id if provided by line segment inputs.
            class_id_int: a fp32 tensor with shape 'NCHW', where C=1, distance tensor indicating
                the original class id if provided by line segment inputs.
            weights: a fp32 tensor with shape 'NCHW', where C=1, distance tensor indicating
                weights by y-axis position (higher y, lower weights) starting from the
                highest y-axis pixel position from line segments inputs.
            dont_care_angles: a fp32 tensor with shape 'NCHW', where C=1, incidating some label
                type which wish not to encode angle related values,
                based on class id provided by line segment inputs.
        )doc");

class GenerateDistFromLineseg : public _GenerateDistFromLineseg {
 public:
    explicit GenerateDistFromLineseg(OpKernelConstruction* context)
        : _GenerateDistFromLineseg(context) {}

    void EncodeCore(OpKernelContext* context, const int* lineseg_count_per_image,
                    const int batch_size, const float* linesegments,
                    const int rows,  // total length
                    const int cols,  // fixed  dimension
                    const int encoding_option, float* output, float* output_dist2d,
                    float* output_cluster_id, float* output_class_id_int, float* output_weights,
                    float* output_dontcare_angles, const int output_channels, const int output_rows,
                    const int output_cols, const int radius, const int defined_infinity,
                    const int down_scale_factor, const int starting_encoding_dim_for_bitcoding,
                    const int channel_count_for_bit_coding, const int num_input_classes,
                    const bool normalize) {
        int offset = 0;
        for (int img_index = 0; img_index < batch_size; img_index++) {
            for (int y = 0; y < output_rows; y++) {
                for (int x = 0; x < output_cols; x++) {
                    output_dist2d[img_index * output_rows * output_cols + y * output_cols + x] =
                        defined_infinity;
                }
            }
            if (img_index > 0) {
                offset += lineseg_count_per_image[img_index - 1];
            }
            const float* first_linesegment = (&linesegments[cols * offset]);

            encode_vector_angles_bits(
                first_linesegment, cols, lineseg_count_per_image[img_index],
                &output[img_index * output_channels * output_rows * output_cols],
                &output_dist2d[img_index * output_rows * output_cols],
                &output_cluster_id[img_index * output_rows * output_cols],
                &output_class_id_int[img_index * output_rows * output_cols],
                &output_weights[img_index * output_rows * output_cols],
                &output_dontcare_angles[img_index * output_rows * output_cols], output_channels,
                output_rows, output_cols, radius, defined_infinity, down_scale_factor,
                starting_encoding_dim_for_bitcoding, channel_count_for_bit_coding,
                num_input_classes, encoding_option, normalize);
        }
    }

    void encode_vector_angles_bits(
        const float* first_linesegment, int fixed_cols, int total_lineseg_count, float* output,
        float* dist_2d, float* cluster_id, float* class_id_in_integer, float* output_weights,
        float* output_dontcare_angles, const int output_channels, const int output_rows,
        const int output_cols, const int radius, const int defined_infinity,
        const int down_scale_factor, const int starting_encoding_dim_for_bitcoding,
        const int channel_count_for_bit_coding, const int num_input_classes,
        const int encoding_option, const bool normalize) {
        // for every pixel we will measure distance from it to the line segments.
        for (int y = 0; y < output_rows; y++) {
            float oy = y * down_scale_factor;
            for (int x = 0; x < output_cols; x++) {
                float ox = x * down_scale_factor;
                for (int c = 0; c < output_channels; c++) {
                    output[y * output_cols + x + output_rows * output_cols * c] = 0;
                }
                cluster_id[y * output_cols + x] = 0;
                class_id_in_integer[y * output_cols + x] = 0;
                output_weights[y * output_cols + x] = weight_scheme(y, output_rows * 0.5f);
                output_dontcare_angles[y * output_cols + x] = 0;
                // for every linesegment.
                float min_distance = defined_infinity_;
                float min_vx = std::numeric_limits<float>::max();
                float min_vy = std::numeric_limits<float>::max();
                float min_angle = std::numeric_limits<float>::max();
                float min_class = std::numeric_limits<float>::max();
                float min_id = std::numeric_limits<float>::max();
                float min_width_left = std::numeric_limits<float>::max();
                float min_width_right = std::numeric_limits<float>::max();

                for (int i = 0; i < total_lineseg_count; i++) {
                    float x2 = first_linesegment[i * fixed_cols];
                    float y2 = first_linesegment[i * fixed_cols + 1];
                    float x1 = first_linesegment[i * fixed_cols + 2];
                    float y1 = first_linesegment[i * fixed_cols + 3];
                    float angle = first_linesegment[i * fixed_cols + 4];
                    float angle_top = first_linesegment[i * fixed_cols + 5];
                    float angle_bottom = first_linesegment[i * fixed_cols + 6];
                    float class_id = first_linesegment[i * fixed_cols + 7];
                    float instance_id = first_linesegment[i * fixed_cols + 8];
                    float width_left_top = first_linesegment[i * fixed_cols + 9];
                    float width_right_top = first_linesegment[i * fixed_cols + 10];
                    float width_left_bottom = first_linesegment[i * fixed_cols + 11];
                    float width_right_bottom = first_linesegment[i * fixed_cols + 12];

                    // from ox and oy we need to draw normal vector
                    // to the line (x1,y1) to (x2,y2)
                    float vx;
                    float vy;
                    int interection_with_bottom_middle_top;
                    float final_angle = angle;
                    float width_left = width_left_top;
                    float width_right = width_right_top;

                    if (get_closest_vector_and_angle_from_point_to_line(
                            x1, y1, x2, y2, ox, oy, angle_top, angle_bottom, width_left_top,
                            width_right_top, width_left_bottom, width_right_bottom, &vx, &vy,
                            &interection_with_bottom_middle_top, &final_angle, &width_left,
                            &width_right)) {
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

                if (min_distance < defined_infinity &&
                    dist_2d[y * output_cols + x] > min_distance) {
                    // update
                    dist_2d[y * output_cols + x] = min_distance;
                    if (min_distance < cluster_radius_) {
                        cluster_id[y * output_cols + x] = min_id;
                    }
                    if (min_distance <= class_radius_) {
                        if (!(min_class < 0.f) &&
                            (min_class < (static_cast<float>(num_input_classes) + 1))) {
                            class_id_in_integer[y * output_cols + x] = min_class;
                        }
                    }
                    if (min_angle < DONTCARE_ANGLE_THRESHOLD) {
                        output_dontcare_angles[y * output_cols + x] = 255;
                    }
                    // Normalize distance values if `normalize` is true
                    if (normalize) {
                        min_vx /= static_cast<float>(defined_infinity);
                        min_vy /= static_cast<float>(defined_infinity);
                        min_width_left /= static_cast<float>(defined_infinity);
                        min_width_right /= static_cast<float>(defined_infinity);
                    }
                    if (encoding_option == 0) {
                        encode_option_0(output, x, y, output_rows, output_cols, radius, min_vx,
                                        min_vy, min_distance, min_width_left, min_width_right,
                                        min_class, min_angle, channel_count_for_bit_coding);
                    } else {
                        encode_option_1(output, x, y, output_rows, output_cols, radius, min_vx,
                                        min_vy, min_distance, min_width_left, min_width_right,
                                        min_class, min_angle, channel_count_for_bit_coding);
                    }
                }
            }
        }
    }
};

#pragma message("Registering CPU kernel")
REGISTER_KERNEL_BUILDER(Name("GenerateDistFromLineseg")
                            .Device(DEVICE_CPU)
                            .HostMemory("line_segments_count_per_image")
                            .HostMemory("line_segments"),
                        GenerateDistFromLineseg);
