/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
   ==============================================================================*/
// Modifications: Copyright (c) 2017-2018, NVIDIA CORPORATION.  All rights reserved.

#define EIGEN_USE_THREADS
#define DASHED_LINE_PIXEL_LENGTH 5
#define LINE_MIN_BRIGHTNESS 0.2
#define THICK_LINE_WIDTH 2

#include <float.h>
#include <stdexcept>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;

// We should register only once (CPU)
REGISTER_OP("DrawPolygonOutlines")
    .Input("images: float")
    .Input("polygons: float")
    .Input("points_per_polygon: int32")
    .Input("confidences: float")
    .Input("color_value: float")
    .Attr("line_type: {'solid', 'solid-thick', 'dashed', 'dashed-thick'}")
    .Output("output: float")
    .SetShapeFn([](::shape_inference::InferenceContext* c) {
        return ::shape_inference::UnchangedShapeWithRankAtLeast(c, 3);
    })
    .Doc(R"doc(
      Draw arbitrary polygon outlines on a batch of images.

      Outputs a copy of `images` but draws on top of the pixels zero or more polygons
      specified by the locations in `polygons`. The coordinates of each
      polygon in `polygons` are encoded as `[y_1, x_1, y_2, x_2, ..., y_n, x_n]`.
      n is specified as the input points_per_polygon.

      The polygon coordinates are floats in `[0.0, 1.0]` relative to the width and
      height of the underlying image.

      For example, if an image is 100 x 200 pixels (height x width) and the polygon
      is `[0.1, 0.2, 0.3, 0.4, 0.5, 0.9]`, the first coordinate of
      the bounding box will be `(40, 10)`, the second will be (80, 30) and the third
      will be `(180, 50)` (in (x,y) coordinates).

      Parts of the polygon may fall outside the image.

      Confidences is tensor containing confidence for each polygon between [0, 1]. The rank
      of confidences is one less than rank of polygons

      Line_type is the line_accent. Choose between 'solid', 'solid-thick', 'dashed' and
      'dashed-thick' accents.

      images: 4-D with shape `[batch, height, width, depth]`. A batch of images.
      polygons: 3-D with shape `[batch, num_polygons, points_per_polygon]` containing polygons.
      points_per_polygon: integer indicating the number of vertices in the polygon.
      output: 4-D with the same shape as `images`. The batch of input images with
      polygons drawn on the images.
      )doc");

// Define to make moving x,y locations into and out of functions easier.
struct int2 {
    int x;
    int y;
};

class DrawPolygonOutlinesOp : public OpKernel {
 private:
    static const int64 color_table_length_ = 13;
    string line_type_;

    // TODO(vijayc): Move this table to proto and make this table an input to this Op.
    // 0: red
    // 1: aqua
    // 2: lime
    // 3: navy blue
    // 4: purple
    // 5: fuchsia
    // 6: olive
    // 7: blue
    // 8: maroon
    // 9: yellow
    // 10:orange
    // 11:black
    // 12:white
    float color_table_[color_table_length_][4] = {
        {1, 0, 0, 1},   {0, 1, 1, 1},     {0, 1, 0, 1}, {0, 0, 0.5, 1}, {0.5, 0, 0.5, 1},
        {1, 0, 1, 1},   {0.5, 0.5, 0, 1}, {0, 0, 1, 1}, {0.5, 0, 0, 1}, {1, 1, 0, 1},
        {1, 0.5, 0, 1}, {0, 0, 0, 1},     {1, 1, 1, 1},
    };

 public:
    explicit DrawPolygonOutlinesOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("line_type", &line_type_));
        OP_REQUIRES(context, sizeof(color_table_) / sizeof(*color_table_) == color_table_length_,
                    errors::InvalidArgument("Invalid color table"));
    }

    // Extend a point by THICK_LINE_WIDTH towards top and right directions.
    void _thicken_point(std::vector<int2>* _line_pixels) {
        if (_line_pixels->empty()) {
            return;
        }
        int2 this_point = _line_pixels->back();
        _line_pixels->pop_back();

        for (int i = 0; i < THICK_LINE_WIDTH; i++) {
            for (int j = 0; j < THICK_LINE_WIDTH; j++) {
                _line_pixels->push_back(int2{this_point.x + i, this_point.y + j});
            }
        }
    }

    /*
    // _is_within_image Determines whether point is within the image or not..
    // point:      The point to check.
    // width:      Image width (int64).
    // height:     Image height (int64).
    */
    bool _is_within_image(int2 point, int64 width, int64 height) {
        return (((point.y <= static_cast<int>(height - 1)) &&
                 (point.x <= static_cast<int>(width - 1))) &&
                ((point.y >= 0) && (point.x >= 0)));
    }

    /*
    // _get_line_pixels Finds all the pixels along a line and returns them.
    //                  An implementation of Bresenham’s Line Drawing Algorithm.
    // x0, y0:       First point on the line (int, int).
    // x1, y1:       Second point on the line (int, int).
    */
    std::vector<int2> _get_line_pixels(const int x0, const int y0, const int x1, const int y1,
                                       const string line_type_) {
        // Initialize.
        std::vector<int2> line_pixels;
        int x, y;
        int dx = x1 - x0;
        int dy = y1 - y0;
        int dx0 = fabs(dx);
        int dy0 = fabs(dy);
        int delta;
        int midpoint;
        int midpoint_increment;
        int midpoint_factor;
        int end_coordinate;
        int independent_coordinate;
        int dependent_coordinate;
        bool store_flag = true;

        // If the extent of the line in y is smaller than in x.
        if (dy0 <= dx0) {
            delta = dx;
        } else {
            delta = dy;
        }

        // Assign the first point to the line.
        if (delta >= 0) {
            x = x0;
            y = y0;
        } else {
            x = x1;
            y = y1;
        }
        int2 this_point;
        this_point.x = x;
        this_point.y = y;
        line_pixels.push_back(this_point);

        // Setup the variables based on the line direction. When the line is steeper in
        // x than y, you want to increment along x and find new ys and vice versa.
        if (dy0 <= dx0) {
            // Get the end coordinate.
            if (dx >= 0) {
                end_coordinate = x1;
            } else {
                end_coordinate = x0;
            }

            // Set up the midpoint terms.
            midpoint_increment = 2 * (dy0 - dx0);
            midpoint_factor = 2 * dy0;
            midpoint = 2 * dy0 - dx0;

            // Assign independent and dependent coordinates.
            independent_coordinate = x;
            dependent_coordinate = y;
        } else {
            // Get the end coordinate.
            if (dy >= 0) {
                end_coordinate = y1;
            } else {
                end_coordinate = y0;
            }

            // Set up the midpoint terms.
            midpoint_increment = 2 * (dx0 - dy0);
            midpoint_factor = 2 * dx0;
            midpoint = 2 * dx0 - dy0;

            // Assign independent and dependent coordinates.
            independent_coordinate = y;
            dependent_coordinate = x;
        }

        // Loop over extent of the line, adjusting the dependent coordinate when
        // midpoint is above 0.
        for (int i = 0; independent_coordinate < end_coordinate; i++) {
            independent_coordinate += 1;
            if (midpoint < 0) {
                midpoint += midpoint_factor;
            } else {
                if ((dx < 0 && dy < 0) || (dx > 0 && dy > 0)) {
                    dependent_coordinate = dependent_coordinate + 1;
                } else {
                    dependent_coordinate = dependent_coordinate - 1;
                }
                midpoint += midpoint_increment;
            }
            // Assign the point to the line.
            int2 this_point;
            if (dy0 <= dx0) {
                this_point.x = independent_coordinate;
                this_point.y = dependent_coordinate;
            } else {
                this_point.x = dependent_coordinate;
                this_point.y = independent_coordinate;
            }

            // Store the points based on the visualization type
            if (line_type_ == "solid") {
                line_pixels.push_back(this_point);
            } else if (line_type_ == "solid-thick") {
                line_pixels.push_back(this_point);
                _thicken_point(&line_pixels);
            } else if (line_type_ == "dashed") {
                if (i % DASHED_LINE_PIXEL_LENGTH == 0) {
                    store_flag = !store_flag;
                }
                if (store_flag) {
                    line_pixels.push_back(this_point);
                }
            } else if (line_type_ == "dashed-thick") {
                if (i % DASHED_LINE_PIXEL_LENGTH == 0) {
                    store_flag = !store_flag;
                }
                if (store_flag) {
                    line_pixels.push_back(this_point);
                    _thicken_point(&line_pixels);
                }
            }
        }

        return line_pixels;
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& images = context->input(0);
        const Tensor& polygons = context->input(1);
        const Tensor& points_per_polygon_tensor = context->input(2);
        const Tensor& confidences = context->input(3);
        const Tensor& color_value = context->input(4);
        int points_per_polygon = *(points_per_polygon_tensor.flat<int>().data());
        const int64 depth = images.dim_size(3);

        //  OP_REQUIRES_OK(context, context->GetAttr("line_type", &line_type_));

        // Check assumptions.

        OP_REQUIRES(context, images.dims() == 4,
                    errors::InvalidArgument("The rank of the images should be 4"));
        OP_REQUIRES(context, polygons.dims() == 3,
                    errors::InvalidArgument("The rank of the polygons tensor should be 3"));

        OP_REQUIRES(context, confidences.dims() == 2,
                    errors::InvalidArgument("The rank of the confidences tensor should be 2"));

        OP_REQUIRES(context, images.dim_size(0) == polygons.dim_size(0) &&
                                 confidences.dim_size(0) == polygons.dim_size(0),
                    errors::InvalidArgument("The batch sizes should be the same"));

        OP_REQUIRES(context, confidences.dim_size(1) == polygons.dim_size(1),
                    errors::InvalidArgument("The no. of polygons and confidences should be same"));

        OP_REQUIRES(context, depth == 4 || depth == 1 || depth == 3,
                    errors::InvalidArgument("Channel depth should be either 1 (GRY), "
                                            "3 (RGB), or 4 (RGBA)"));

        OP_REQUIRES(context, points_per_polygon > 0,
                    errors::InvalidArgument("The points per prior should be greater than 0."));

        OP_REQUIRES(context, color_value.dims() == 1,
                    errors::InvalidArgument("Invalid color type"));

        if (color_value.dim_size(0) != 0) {
            OP_REQUIRES(
                context, color_value.dim_size(0) == 3,
                errors::InvalidArgument("Color_value should have only 3 values (RGB). Got: ",
                                        color_value.dim_size(0)));
        }

        const int64 batch_size = images.dim_size(0);
        const int64 height = images.dim_size(1);
        const int64 width = images.dim_size(2);

        // Reset first color channel to 1 if image is GRY.
        // For GRY images, this means all polygons will be white.
        if (depth == 1) {
            for (int64 i = 0; i < color_table_length_; i++) {
                color_table_[i][0] = 1;
            }
        }

        // Set up output tensor.
        Tensor* output;
        OP_REQUIRES_OK(context, context->allocate_output(
                                    0, TensorShape({batch_size, height, width, depth}), &output));

        output->tensor<float, 4>() = images.tensor<float, 4>();
        auto canvas = output->tensor<float, 4>();

        // For each image.
        for (int64 b = 0; b < batch_size; ++b) {
            const int64 num_polygons = polygons.dim_size(1);
            const auto tpolygons = polygons.tensor<float, 3>();
            const auto tconfidences = confidences.tensor<float, 2>();
            const auto tcolor_value = color_value.tensor<float, 1>();

            // Draw each polygon.
            for (int64 p = 0; p < num_polygons; ++p) {
                int64 color_index = p % color_table_length_;
                const int64 min_polygon_row = static_cast<float>(tpolygons(b, p, 0)) * (height - 1);
                const int64 max_polygon_row =
                    static_cast<float>(tpolygons(b, p, (points_per_polygon * 4) - 2)) *
                    (height - 1);
                const int64 min_polygon_col = static_cast<float>(tpolygons(b, p, 1)) * (width - 1);
                const int64 max_polygon_col =
                    static_cast<float>(tpolygons(b, p, (points_per_polygon * 4) - 1)) * (width - 1);

                if (min_polygon_row >= height || max_polygon_row < 0 || min_polygon_col >= width ||
                    max_polygon_col < 0) {
                    LOG(WARNING) << "Polygon (" << min_polygon_row << "," << min_polygon_col << ","
                                 << max_polygon_row << "," << max_polygon_col
                                 << ") is completely outside the image"
                                 << " and will not be drawn.";
                    continue;
                }

                // For each point pair on the polygon, draw the line between them.
                int64 x0, y0, x1, y1;
                for (int64 pt = 0; pt < points_per_polygon; ++pt) {
                    // Extract the point pair.
                    x0 = static_cast<float>(tpolygons(b, p, (pt * 2) + 1)) * (width - 1);
                    y0 = static_cast<float>(tpolygons(b, p, (pt * 2))) * (height - 1);

                    if (pt < points_per_polygon - 1) {
                        x1 = static_cast<float>(tpolygons(b, p, (pt * 2) + 3)) * (width - 1);
                        y1 = static_cast<float>(tpolygons(b, p, (pt * 2) + 2)) * (height - 1);
                    } else {  // If the last point, then connect it to the first point.
                        x1 = static_cast<float>(tpolygons(b, p, 1)) * (width - 1);
                        y1 = static_cast<float>(tpolygons(b, p, 0)) * (height - 1);
                    }

                    // Gather the line pixels.
                    std::vector<int2> line_pixels;
                    line_pixels = _get_line_pixels(x0, y0, x1, y1, line_type_);

                    // Draw line, ensuring that each point is within the image.
                    for (const auto point : line_pixels) {
                        if (_is_within_image(point, width, height)) {
                            // Set the brightness based on confidence.
                            // This will set gray scale values too for monochrome images.
                            for (int64 c = 0; c < depth; c++) {
                                float background_color;
                                float foreground_color;
                                float alpha;  // brightness value
                                background_color = canvas(b, point.y, point.x, c);
                                alpha = LINE_MIN_BRIGHTNESS +
                                        ((1 - LINE_MIN_BRIGHTNESS) * tconfidences(b, p));

                                if (color_value.dim_size(0) == 0) {
                                    foreground_color =
                                        static_cast<float>(color_table_[color_index][c]);
                                } else {
                                    foreground_color = static_cast<float>(tcolor_value(c));
                                }

                                // If image is RGBA, do not alter RGB value but set alpha to
                                // A component.
                                if (depth == 4) {
                                    canvas(b, point.y, point.x, c) =
                                        (c == 4) ? alpha : foreground_color;
                                    // If image is GRY or RGB, scale RGB with alpha
                                } else {
                                    float alpha_adjusted_value = std::max<float>(
                                        0, std::min<float>(((1 - alpha) * background_color) +
                                                               (alpha * foreground_color),
                                                           1));
                                    canvas(b, point.y, point.x, c) = alpha_adjusted_value;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};

#pragma message("Registering CPU kernel")
REGISTER_KERNEL_BUILDER(Name("DrawPolygonOutlines")
                            .Device(DEVICE_CPU)
                            .HostMemory("images")
                            .HostMemory("polygons")
                            .HostMemory("points_per_polygon")
                            .HostMemory("confidences")
                            .HostMemory("color_value"),
                        DrawPolygonOutlinesOp);
