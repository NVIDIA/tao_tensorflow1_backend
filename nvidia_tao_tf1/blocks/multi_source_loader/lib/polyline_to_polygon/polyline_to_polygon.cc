// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.

/*
Polyline to Polygon conversion algorithm

*/

#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_types.h"

#include "../utils/point2.h"
#include "../utils/tensor_utils.h"

using namespace std;
using namespace tensorflow;

const float SEGMENT_IS_POINT_THRESHOLD = 0.001f;

REGISTER_OP("PolylineToPolygon")
    .Input("polygon_indices: int64")
    .Input("polygon_values: float32")
    .Input("polygon_dense_shape: int64")
    .Input("class_ids_indices: int64")
    .Input("class_ids_values: int32")
    .Input("class_ids_shape: int64")
    .Output("output_polygon_indices: int64")
    .Output("output_polygon_values: float32")
    .Output("output_polygon_dense_shape: int64")
    .Output("output_class_ids_indices: int64")
    .Output("output_class_ids_values: int32")
    .Output("output_class_ids_shape: int64")
    .Attr("target_class_id: int = -1")
    .Attr("line_width: float = 1")
    .Attr("debug: int = 0")
    .Doc(R"doc(
        Polyline to Polygon(s) conversion op.
        Summary:
            Takes in two SparseTensor[s] describing a set of polylines to convert.

            polygon_dense_shape must be >2D ([NT]PVC), where N is
            batch dimension, T is temporal dimension, P is polygons, V vertices, and C coordinate index (0 or 1).

            polygon_values is a flat fp32 list of interleaved vertex (x, y) coordinates.

            polygon_indices is a 2d tensor with dimension 0 the size of the polygons.values tensor,
            and dimension 1 is size 5.

            NOTE: Currently only one class per polygon is supported

        Tensor Arguments:
            polygon_indices: indices field of a SparseTensor describing the input polygons
            polygon_values: values field of a SparseTensor describing the input polygons.
            polygon_dense_shape: dense_shape field of a SparseTensor describing the input polygons.
            class_ids_indices: indices field of a SparseTensor describing the classes of each polygon/polyline
            class_ids_values: values field of a SparseTensor describing the classes of each polygon/polyline
            class_ids_shape: dense_shape field of a SparseTensor describing the classes of each polygon/polyline

        Scalar Arguments:
            target_class_id: the class id that represents the polyline to convert to polygon(s)
            line_width: the width of the resulting line

        Returns:
            output_polygon_indices: same format as polygon_indices
            output_polygon_values: same format as polygon_values
            output_polygon_dense_shape: same format as polygon_dense_shape
            output_class_ids_indices: same format as class_ids_indices
            output_class_ids_values: same format as class_ids_values
            output_class_ids_shape: same format as class_ids_shape
    )doc");

class PolylineToPolygonOp : public OpKernel {
 public:
    explicit PolylineToPolygonOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("target_class_id", &target_class_id_));
        OP_REQUIRES_OK(context, context->GetAttr("line_width", &line_width_));
        int dbg;
        OP_REQUIRES_OK(context, context->GetAttr("debug", &dbg));
        debug_ = dbg != 0;
    }

    void Compute(OpKernelContext* context) override;

 private:
    int32_t target_class_id_;
    float line_width_;
    bool debug_;
};

#pragma message("Registering CPU kernel")
REGISTER_KERNEL_BUILDER(Name("PolylineToPolygon").Device(DEVICE_CPU), PolylineToPolygonOp);

/*
Finds the ranges within the polygon_indices tensor that represent the
individual polygons. If you want to get the slice of indices for a given polygon "i",
then the start index is out_ranges(i) and the non-inclusive end index is out_ranges(i + 1).

NOTE: polygon_indices MUST be lexicographically sorted from the left column to the right
*/
TTypes<int64>::Vec GetPolygonRanges(OpKernelContext* context,
                                    const TTypes<int64>::ConstMatrix& polygon_indices,
                                    int64 num_polygons, Tensor* out_ranges_tensor) {
    // Create a tensor that can hold the row ranges for each polygon.
    // These ranges can be used to slice both the indices and values
    // tensors to a specific polygon
    Status alloc_status =
        context->allocate_temp(DT_INT64,
                               TensorShape{// Note that if num_classes == 0,
                                           // we still need at least 2 elements to define
                                           // the initial range, and both values will be 0
                                           // to signify the empty range
                                           max(num_polygons + 1, 2ll)},
                               out_ranges_tensor);
    if (!alloc_status.ok()) throw runtime_error(alloc_status.ToString());

    TTypes<int64>::Vec ranges = out_ranges_tensor->vec<int64>();

    const int64 num_indices = polygon_indices.dimension(0);

    // The final two columns of the index tensor are what actually
    // define the coordinates of the tensor. All preceding dimensions are specifying
    // which polygon we're dealing with. Essentially, all but the last 2 columns
    // form an "id" tuple that uniquely identifies a polygon
    const int64 num_id_columns = polygon_indices.dimension(1) - 2;

    // First polygon starts at index 0
    ranges(0) = 0;

    // The index of the start of the last polygon
    int64 prev_index = 0;
    // The index of the range we're currently dealing with
    int64 curr_range = 1;
    // Coordinates always come in pairs, so we can skip every other row
    for (int64 i = 2; i < num_indices; i += 2) {
        // Check to see if the id tuple of the current indice matches that of the previous one.
        // If they don't match, then this is the start of a new polygon
        if (!IsSubDimEqual(num_id_columns, polygon_indices, prev_index, polygon_indices, i)) {
            if (curr_range >= ranges.dimension(0))
                throw runtime_error(
                    "Something went wrong! The number of ranges doesn't "
                    "match the number of classes.");

            ranges(curr_range) = i;
            prev_index = i;
            ++curr_range;
        }
    }

    if (curr_range != ranges.dimension(0) - 1)
        throw runtime_error("Error computing the polygon ranges!");

    ranges(curr_range) = num_indices;

    return ranges;
}

void PolylineToPolygonOp::Compute(OpKernelContext* context) {
    const Tensor& tf_indices = context->input(0);
    OP_REQUIRES(context, tf_indices.dims() == 2 && tf_indices.dim_size(1) > 2,
                errors::InvalidArgument("polygon_indices must be a 2D tensor"
                                        " with >2 columns. Shape is ",
                                        tf_indices.shape().DebugString()));
    TTypes<int64>::ConstMatrix indices = tf_indices.matrix<int64>();

    const Tensor& tf_values = context->input(1);
    OP_REQUIRES(context, tf_values.dims() == 1,
                errors::InvalidArgument("polygon_values must be 1D. Shape is ",
                                        tf_values.shape().DebugString()));
    TTypes<float>::ConstVec values = tf_values.vec<float>();

    const Tensor& tf_dense_shape = context->input(2);
    OP_REQUIRES(context, tf_dense_shape.dims() == 1,
                errors::InvalidArgument("polygon_dense_shape must be 1D. Shape is ",
                                        tf_dense_shape.shape().DebugString()));
    TTypes<int64>::ConstVec dense_shape = tf_dense_shape.vec<int64>();

    const Tensor& tf_class_ids_indices = context->input(3);
    OP_REQUIRES(context, tf_class_ids_indices.dims() == 2 &&
                             tf_class_ids_indices.dim_size(1) == tf_indices.dim_size(1) - 1,
                errors::InvalidArgument("class_ids_indices must be a 2D tensor "
                                        " with 1 less column than polygon_indices. Shape is ",
                                        tf_class_ids_indices.shape().DebugString(), " vs. ",
                                        tf_indices.shape().DebugString()));
    TTypes<int64>::ConstMatrix class_ids_indices = tf_class_ids_indices.matrix<int64>();

    const Tensor& tf_class_ids_values = context->input(4);
    OP_REQUIRES(context, tf_class_ids_values.dims() == 1,
                errors::InvalidArgument("class_ids_values must be 1D. Shape is ",
                                        tf_class_ids_values.shape().DebugString()));
    TTypes<int32>::ConstVec class_ids_values = tf_class_ids_values.vec<int32>();

    const Tensor& tf_class_ids_shape = context->input(5);
    OP_REQUIRES(context, tf_class_ids_shape.dims() == 1,
                errors::InvalidArgument("class_ids_shape must be 1D. Shape is ",
                                        tf_class_ids_shape.shape().DebugString()));
    TTypes<int64>::ConstVec class_ids_shape = tf_class_ids_shape.vec<int64>();

    if (debug_) {
        cout << "Target Class: " << target_class_id_ << endl;
        cout << "Polygon Width: " << line_width_ << endl;
        cout << "### Input:" << endl << "# Coordinates" << endl;
        PrintSparseTensor(cout, indices, values, dense_shape);
        cout << endl << "# Classes" << endl;
        PrintSparseTensor(cout, class_ids_indices, class_ids_values, class_ids_shape);
        cout << endl;
    }

    int64 max_class_index = -1;
    for (int64 row = 0; row < class_ids_indices.dimension(0); ++row) {
        max_class_index =
            max(max_class_index, class_ids_indices(row, class_ids_indices.dimension(1) - 1));
    }
    OP_REQUIRES(context, max_class_index <= 0, errors::InvalidArgument("Only one class per polygon "
                                                                       "currently supported."));

    // Note: num_classes is equivalent to the number of polygons + polylines
    const int64 num_classes = class_ids_values.dimension(0);

    try {
        Tensor tf_ranges;
        TTypes<int64>::Vec ranges = GetPolygonRanges(context, indices, num_classes, &tf_ranges);

        if (debug_) {
            cout << "Ranges: " << DebugString(ranges) << endl;
        }

        // Figure out how many classes, and how many examples, there
        // will be in the output. Doing this allows this function to minimize
        // temporary allocations to just calculating the polygon ranges.
        int64 out_num_classes = 0;
        int64 out_num_indices = 0;
        for (int64 i = 0; i < num_classes; ++i) {
            int32 class_id = class_ids_values(i);
            int64 range_start = ranges(i);
            int64 range_end = ranges(i + 1);

            if (class_id != target_class_id_) {
                // Basically, a no-op
                ++out_num_classes;
                out_num_indices += range_end - range_start;
            } else {
                int64 num_points = (range_end - range_start) / 2;
                int64 num_polygons = num_points - 1;
                // There are 6 points per polygon in the output
                int64 out_num_points = 6 * num_polygons;
                out_num_indices += 2 * out_num_points;
                out_num_classes += num_polygons;
            }
        }

        if (debug_) {
            cout << "Out Num Classes: " << out_num_classes << endl;
            cout << "Out Num Indices: " << out_num_indices << endl;
        }

        Tensor* tf_output_indices = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
                                    0, TensorShape{out_num_indices, tf_indices.dim_size(1)},
                                    &tf_output_indices));
        TTypes<int64>::Matrix output_indices = tf_output_indices->matrix<int64>();

        Tensor* tf_output_values = nullptr;
        OP_REQUIRES_OK(
            context, context->allocate_output(1, TensorShape{out_num_indices}, &tf_output_values));
        TTypes<float>::Vec output_values = tf_output_values->vec<float>();

        Tensor* tf_output_class_ids_indices = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(
                           3, TensorShape{out_num_classes, tf_class_ids_indices.dim_size(1)},
                           &tf_output_class_ids_indices));
        TTypes<int64>::Matrix output_class_ids_indices =
            tf_output_class_ids_indices->matrix<int64>();

        Tensor* tf_output_class_ids_values = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(4, TensorShape{out_num_classes},
                                                         &tf_output_class_ids_values));
        TTypes<int32>::Vec output_class_ids_values = tf_output_class_ids_values->vec<int32>();

        const float half_width = line_width_ / 2.0f;

        const int64 num_example_dims = indices.dimension(1) - 3;

        if (debug_) {
            cout << "Num Example Dims: " << num_example_dims << endl;
        }

        int64 output_offset = 0;
        int64 output_cls_offset = 0;
        int64 next_polygon_index = 0;
        int64 max_num_polygons = 0;
        int64 max_num_coordinates = 0;
        for (int64 i = 0; i < num_classes; ++i) {
            const int32 class_id = class_ids_values(i);
            const int64 range_start = ranges(i);
            const int64 range_end = ranges(i + 1);

            auto curr_poly_indices = Slice(indices, range_start, range_end);
            auto curr_poly_coords = Slice(values, range_start, range_end);

            // Figure out whether this is a different example
            if (i == 0 ||
                !IsSubDimEqual(num_example_dims, indices, ranges(i - 1), indices, ranges(i))) {
                // New example, reset the polygon index
                next_polygon_index = 0;
            }

            if (class_id != target_class_id_) {
                const int64 output_range_end = output_offset + range_end - range_start;
                auto output_poly_indices = Slice(output_indices, output_offset, output_range_end);
                auto output_poly_values = Slice(output_values, output_offset, output_range_end);

                output_poly_indices = curr_poly_indices;
                output_poly_indices.chip(num_example_dims, 1).setConstant(next_polygon_index);
                output_poly_values = curr_poly_coords;
                output_class_ids_values(output_cls_offset) = class_id;
                InnerAssign(output_class_ids_indices, output_cls_offset,
                            make_pair(curr_poly_indices.data(), num_example_dims),
                            next_polygon_index, 0);
                output_offset = output_range_end;
                ++output_cls_offset;
                ++next_polygon_index;
                max_num_coordinates = max(max_num_coordinates, (range_end - range_start) / 2);
            } else {
                max_num_coordinates = max(max_num_coordinates, 6ll);

                const auto curr_poly_points =
                    reinterpret_cast<const Point2<float>*>(curr_poly_coords.data());

                const int64 num_coords = curr_poly_coords.dimension(0) / 2;
                for (int64 k = 1; k < num_coords; ++k) {
                    const int64 output_range_end = output_offset + 12;
                    auto output_poly_indices =
                        Slice(output_indices, output_offset, output_range_end);
                    auto output_poly_coords = Slice(output_values, output_offset, output_range_end);

                    auto output_curr_poly_points =
                        reinterpret_cast<Point2<float>*>(output_poly_coords.data());

                    const Point2<float> start_coord = curr_poly_points[k - 1];
                    Point2<float> end_coord = curr_poly_points[k];

                    Point2<float> line_dir = end_coord - start_coord;
                    float line_length = norm(line_dir);

                    // Check to see if the start and end points are approximately the same.
                    // If so, then we want to introduce some very small length so that we
                    // don't perform any divisions by zero. At least the maps-a-japan dataset
                    // has such line segments.
                    if (line_length < SEGMENT_IS_POINT_THRESHOLD) {
                        line_dir.x = 1;
                        line_dir.y = 0;
                        line_length = 1;
                        end_coord = start_coord + SEGMENT_IS_POINT_THRESHOLD * line_dir;
                    } else {
                        line_dir = line_dir / line_length;
                    }

                    const Point2<float> line_tangent = perpendicular(line_dir);

                    // Compute the coordinates for the polygon. This creates a diamond-like
                    // shape
                    output_curr_poly_points[0] = start_coord + half_width * line_tangent;
                    output_curr_poly_points[1] = end_coord + half_width * line_tangent;
                    output_curr_poly_points[2] = end_coord + half_width * line_dir;
                    output_curr_poly_points[3] = end_coord - half_width * line_tangent;
                    output_curr_poly_points[4] = start_coord - half_width * line_tangent;
                    output_curr_poly_points[5] = start_coord - half_width * line_dir;

                    // Now assign the indices
                    for (int64 coord = 0; coord < 6; ++coord) {
                        for (int64 dim = 0; dim < 2; ++dim) {
                            const int64 row = coord * 2 + dim;
                            InnerAssign(output_poly_indices, row,
                                        make_pair(curr_poly_indices.data(), num_example_dims),
                                        next_polygon_index, coord, dim);
                        }
                    }

                    output_class_ids_values(output_cls_offset) = class_id;
                    InnerAssign(output_class_ids_indices, output_cls_offset,
                                make_pair(curr_poly_indices.data(), num_example_dims),
                                next_polygon_index, 0);

                    ++next_polygon_index;
                    ++output_cls_offset;
                    output_offset = output_range_end;
                }
            }

            max_num_polygons = max(max_num_polygons, next_polygon_index);
        }

        Tensor* tf_output_dense_shape = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape{dense_shape.dimension(0)},
                                                         &tf_output_dense_shape));
        auto output_dense_shape = tf_output_dense_shape->vec<int64>();
        InnerAssign(output_dense_shape, 0, make_pair(dense_shape.data(), num_example_dims),
                    max_num_polygons, max_num_coordinates, max_num_polygons > 0 ? 2 : 0);

        Tensor* tf_output_class_ids_shape = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(5, TensorShape{class_ids_shape.dimension(0)},
                                                &tf_output_class_ids_shape));
        auto output_class_ids_shape = tf_output_class_ids_shape->vec<int64>();
        InnerAssign(output_class_ids_shape, 0, make_pair(class_ids_shape.data(), num_example_dims),
                    max_num_polygons, max_class_index + 1);

        if (debug_) {
            cout << "### Output:" << endl << "# Coordinates" << endl;
            PrintSparseTensor(cout, output_indices, output_values, output_dense_shape);
            cout << endl << "# Classes" << endl;
            PrintSparseTensor(cout, output_class_ids_indices, output_class_ids_values,
                              output_class_ids_shape);
            cout << endl;
        }
    } catch (exception& ex) {
        OP_REQUIRES_OK(context, errors::Internal(ex.what()));
    }
}
