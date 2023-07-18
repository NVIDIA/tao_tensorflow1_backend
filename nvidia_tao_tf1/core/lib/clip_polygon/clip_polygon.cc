// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

#include "clipper.hpp"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;

// We should register only once (CPU)
REGISTER_OP("ClipPolygon")
    .Input("polygons: float")
    .Input("points_per_polygon: int32")
    .Input("polygon_mask: float")
    .Attr("closed: bool")
    .Attr("precision_factor: float = 1e9")
    .Output("output_polygons: float")
    .Output("output_points_per_polygon: int32")
    .Output("output_polygon_index_mapping: int32")
    .SetShapeFn([](::shape_inference::InferenceContext* c) {
        std::vector<::shape_inference::DimensionHandle> dims_polygons;
        dims_polygons.push_back(c->UnknownDim());
        dims_polygons.push_back(c->MakeDim(2));
        c->set_output(0, c->MakeShape(dims_polygons));
        std::vector<::shape_inference::DimensionHandle> dims_polygon_list;
        dims_polygon_list.push_back(c->UnknownDim());
        c->set_output(1, c->MakeShape(dims_polygon_list));
        c->set_output(2, c->MakeShape(dims_polygon_list));
        return Status::OK();
    })
    .Doc(R"doc(
    Op to clip polygons or polylines with an input polygon mask.

    Clipped polygons do not give any intra-polygon coordinate ordering guarantees. This is
    typically not a problem as lines or polygons are agnostic to direction.
    Polygons are assumed to be cyclical, and can therefore 'shift' indices in the array, and
    can even be inverted in direction. Polylines (`closed` is False) are not cyclical and can
    therefore only revert in direction, but can never be shifted.

    Self-intersecting polygons will be split into multiple non-intersecting polygons. This
    means that the amount of output polygons can increase or decrease. This does not apply to
    polylines (`closed` is False). Similarly, the amount of output polygons and polylines can
    decrease if they are clipped entirely.

    Arguments:
        polygons: a tensor in the form of a list of lists. The top-level list contains
            sub-lists with 2 elements each; each sub-list contains absolute x/y coordinates
            (in that order) of a single vertex of a single polygon for a single image
            (= raster map). The length of the top-level list is therefore equal to the total
            number of vertices over all polygons that we are drawing over all raster maps.
        points_per_polygon: a tensor in the form of a flat list. The elements of the list
            are the vertex counts for each polygon that we will draw during rasterization. Thus,
            the length of this list is equal to the number of polygons we will draw, and if we
            were to sum all the values in this list, the sum should equal the length of the
            ``polygon_vertices`` list above.
        polygon_mask: a (n, 2) fp32 tensor containing one polygon, to be used as the clipping mask.
        closed (bool): if the polygon is closed, or open (polyline).
        precision_factor (float): this factor is used to correct for an float-int-float roundtrip
            done in the backend clipper library[1]. For good reasons[2], the backend is entirely based
            on integer computations. Because we want to clip floating point values, we use this
            factor to account for any lost precision in the conversion.

    Returns:
        polygons: same as input `polygon` but clipped.
        points_per_polygon: same as input `points_per_polygon` but clipped.
        polygon_index_mapping: mapping of each output polygon to the index of the input polygon
            it originated from (because polygons can be clipped entirely or split into multiple
            parts).

    [1] http://www.angusj.com/delphi/clipper/documentation/Docs/Units/ClipperLib/Types/IntPoint.htm
    [2] http://www.angusj.com/delphi/clipper/documentation/Docs/Overview/Rounding.htm
    )doc");

static const ClipperLib::ClipType clip_type = ClipperLib::ctIntersection;
static const ClipperLib::PolyFillType poly_fill_type = ClipperLib::pftNonZero;

class ClipPolygonOp : public OpKernel {
 private:
    bool closed_;
    float precision_factor_;

 public:
    explicit ClipPolygonOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("closed", &closed_));
        OP_REQUIRES_OK(context, context->GetAttr("precision_factor", &precision_factor_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& polygons_tensor = context->input(0);
        const Tensor& points_per_polygon_tensor = context->input(1);
        const Tensor& polygon_mask_tensor = context->input(2);

        OP_REQUIRES(context, 2 == polygons_tensor.shape().dims(),
                    errors::InvalidArgument("polygons tensor must have 2 dimensions, shape is: ",
                                            polygons_tensor.shape().DebugString(), "."));

        OP_REQUIRES(
            context, 2 == polygons_tensor.shape().dim_size(1),
            errors::InvalidArgument("polygons tensor dimension index 1 must be exactly 2,",
                                    " shape is: ", polygons_tensor.shape().DebugString(), "."));

        OP_REQUIRES(context, 1 == points_per_polygon_tensor.shape().dims(),
                    errors::InvalidArgument("points_per_polygon must be a 1 dimensional vector,",
                                            " shape is: ",
                                            points_per_polygon_tensor.shape().DebugString(), "."));

        OP_REQUIRES(
            context, 2 == polygon_mask_tensor.shape().dims(),
            errors::InvalidArgument("polygon_mask tensor must have 2 dimensions, shape is: ",
                                    polygon_mask_tensor.shape().DebugString(), "."));

        OP_REQUIRES(
            context, 2 == polygon_mask_tensor.shape().dim_size(1),
            errors::InvalidArgument("polygon_mask tensor dimension index 1 must be exactly 2,",
                                    " shape is: ", polygon_mask_tensor.shape().DebugString(), "."));

        // Get some constants from the tensors
        const int64 nvertices = polygons_tensor.dim_size(0);
        const int64 npolygons = points_per_polygon_tensor.dim_size(0);
        const int64 nvertices_mask = polygon_mask_tensor.dim_size(0);

        // Disabling due to non-polygon labels with 0 vertices being sent through this op.
        // This should be re-enabled when only polygons are sent through this op.
        // OP_REQUIRES(
        //     context, nvertices >= npolygons,
        //     errors::InvalidArgument("Number of polygons ", npolygons, " is larger than the
        //     number",
        //                             " of vertices ", nvertices, "."));

        OP_REQUIRES(
            context, nvertices_mask >= 3,
            errors::InvalidArgument("Number of vertices of the polygon mask needs to be greater"
                                    " than or equal to 3, but is: ",
                                    nvertices_mask));

        const auto input_polygon = polygons_tensor.flat<float>();
        const auto input_points_per_polygon = points_per_polygon_tensor.flat<int>();
        const auto input_polygon_mask = polygon_mask_tensor.flat<float>();

        // Test that the sum of the `points_per_polygon_tensor` vector adds up to `nvertices`.
        int nvertices_from_vertex_counts = 0;
        for (int i = 0; i < npolygons; i++) {
            nvertices_from_vertex_counts += input_points_per_polygon.data()[i];
        }
        OP_REQUIRES(context, nvertices_from_vertex_counts == nvertices,
                    errors::InvalidArgument(
                        "Sum of points_per_polygon_tensor", nvertices_from_vertex_counts,
                        " over all polygons does not add up to nvertices ", nvertices, "."));

        // Create a Clipper path from the polygonal mask.
        ClipperLib::Path path_polygon_mask;
        int current_mask_vertex = 0;
        for (int v = 0; v < nvertices_mask; v++, current_mask_vertex += 2) {
            float x = input_polygon_mask.data()[current_mask_vertex];
            float y = input_polygon_mask.data()[current_mask_vertex + 1];
            x = x * precision_factor_;
            y = y * precision_factor_;
            path_polygon_mask << ClipperLib::IntPoint(static_cast<ClipperLib::cInt>(x),
                                                      static_cast<ClipperLib::cInt>(y));
        }

        ClipperLib::Paths polygon_intersections;
        std::vector<int> polygon_index_mapping;

        int current_input_vertex = 0;
        int num_output_vertices = 0;
        for (int polygon_id = 0; polygon_id < npolygons; polygon_id++) {
            // Number of vertices for this polygon.
            int polygon_vertices = input_points_per_polygon.data()[polygon_id];

            ClipperLib::Path path_polygon;

            // Loop through all vertices in the current polygon.
            for (int v = 0; v < polygon_vertices; v++, current_input_vertex += 2) {
                float x = input_polygon.data()[current_input_vertex];
                float y = input_polygon.data()[current_input_vertex + 1];
                x = x * precision_factor_;
                y = y * precision_factor_;
                path_polygon << ClipperLib::IntPoint(static_cast<ClipperLib::cInt>(x),
                                                     static_cast<ClipperLib::cInt>(y));
            }

            // Perform polygon clipping.
            ClipperLib::Paths paths_intersected;
            try {
                // Catch errors to make sure third-party problems bubble up to TensorFlow.
                ClipperLib::Clipper c;
                c.AddPath(path_polygon, ClipperLib::ptSubject, closed_);
                c.AddPath(path_polygon_mask, ClipperLib::ptClip, true);
                if (closed_) {
                    c.Execute(clip_type, paths_intersected, poly_fill_type, poly_fill_type);
                } else {
                    // If the polygon is not closed (alias a 'line', or 'path'), we obtain the
                    // result through a PolyTree, which is then converted to a path.
                    ClipperLib::PolyTree polytree_intersected;
                    c.Execute(clip_type, polytree_intersected, poly_fill_type, poly_fill_type);
                    ClipperLib::OpenPathsFromPolyTree(polytree_intersected, paths_intersected);
                }
            } catch (const std::exception& e) {  // Gotta catch 'em all!
                OP_REQUIRES(context, false, errors::InvalidArgument(e.what()));
            }

            // Accumulate the amount of output vertices.
            for (const auto& path_intersected : paths_intersected) {
                num_output_vertices += path_intersected.size();
                polygon_index_mapping.push_back(polygon_id);
            }

            // Add to the output polygon vector.
            std::move(paths_intersected.begin(), paths_intersected.end(),
                      std::back_inserter(polygon_intersections));
        }

        const int num_output_polygons = polygon_intersections.size();

        // Create polygons output.
        Tensor* output_tensor_polygons = nullptr;
        OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                    {0, 2}, 0, TensorShape({num_output_vertices, 2}),
                                    &output_tensor_polygons));
        auto output_polygons = output_tensor_polygons->template flat<float>();

        // Create polygon lengths output.
        Tensor* output_tensor_points_per_polygon = nullptr;
        OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                    {0}, 1, TensorShape({num_output_polygons}),
                                    &output_tensor_points_per_polygon));
        auto output_points_per_polygon = output_tensor_points_per_polygon->template flat<int>();

        // Create output polygon index mapping.
        Tensor* output_tensor_polygon_index_mapping = nullptr;
        OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                    {0}, 2, TensorShape({num_output_polygons}),
                                    &output_tensor_polygon_index_mapping));
        auto output_polygon_index_mapping =
            output_tensor_polygon_index_mapping->template flat<int>();

        int current_output_vertex = 0;
        for (int pid = 0; pid < num_output_polygons; pid++) {
            ClipperLib::Path polygon = polygon_intersections[pid];
            output_points_per_polygon(pid) = polygon.size();
            output_polygon_index_mapping(pid) = polygon_index_mapping[pid];
            for (const auto& vertex : polygon) {
                output_polygons(current_output_vertex) =
                    static_cast<float>(vertex.X) / precision_factor_;
                output_polygons(current_output_vertex + 1) =
                    static_cast<float>(vertex.Y) / precision_factor_;
                current_output_vertex += 2;
            }
        }
    }
};

#pragma message("Registering CPU kernel")
REGISTER_KERNEL_BUILDER(Name("ClipPolygon").Device(DEVICE_CPU), ClipPolygonOp);
