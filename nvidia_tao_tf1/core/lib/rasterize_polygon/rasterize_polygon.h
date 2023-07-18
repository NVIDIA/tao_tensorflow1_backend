// Copyright (c) 2017-2018, NVIDIA CORPORATION.  All rights reserved.
// This header file implements common functionality of rasterize_polygon maglev op

#ifndef _RASTERIZE_POLYGON_H_
#define _RASTERIZE_POLYGON_H_

#include <algorithm>
#include <tuple>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#ifndef EIGEN_USE_GPU

using namespace std;

struct float2 {
    float x;
    float y;
};
// Population count (count the number of set bits in a 32b uint).
int __popc(unsigned int a) {
    int c = 0;
    for (unsigned int i = 0; i < 32; i++, a >>= 1) {
        c += (a & 1);
    }
    return c;
}
#endif

struct Polygon {
    float2 min = {FLT_MAX, FLT_MAX};
    float2 max = {-FLT_MAX, -FLT_MAX};
    int start_vertex = 0;
    int nvertices = 0;
    int image_id = 0;
    int class_id = -1;  // Class index below 0 means the polygon will be skipped.
};

struct Image {
    void combineBbox(const Polygon& polygon) {
        min.x = std::min(min.x, polygon.min.x);
        min.y = std::min(min.y, polygon.min.y);
        max.x = std::max(max.x, polygon.max.x);
        max.y = std::max(max.y, polygon.max.y);
    }
    float2 min = {FLT_MAX, FLT_MAX};
    float2 max = {-FLT_MAX, -FLT_MAX};
    int start_polygon = -1;
    int npolygons = 0;
    // Align (pad) struct size to 8:
    int padding1;
    int padding2;
};

#ifdef __CUDACC__
#define CUDA_HOSTDEV __device__
#else
#define CUDA_HOSTDEV
#endif

static __inline__ CUDA_HOSTDEV void _RasterizePolygonKernel(
    int x, int y, float* out, int height, int width, int num_samples, const Image* image,
    const Polygon* polys, const float2* vertices, const bool binarize, const bool one_hot) {
    // Code common to CPU and GPU kernel.

    // If pixel_filter_size value is more than 0.5, it will sample outside the pixel's box.
    const float pixel_filter_size = 0.5f;

    // Pixel coordinates. Note that sampling point is at half integers.
    float fx = static_cast<float>(x) + 0.5f;
    float fy = static_cast<float>(y) + 0.5f;

    // Pixel filter bounds.
    float fxmax = fx + pixel_filter_size;
    float fxmin = fx - pixel_filter_size;
    float fymax = fy + pixel_filter_size;
    float fymin = fy - pixel_filter_size;

    const float background = 0.0f;
    float fillvalue = 1.0f;

    // Check if the pixel is inside image bbox. Left and top are inclusive.
    float out_value = background;
    int npolygons = image->npolygons;

    if (npolygons && fxmax >= image->min.x && fxmin < image->max.x && fymax >= image->min.y &&
        fymin < image->max.y) {
        // Loop over polygons in an image from front to back.
        int start_polygon = image->start_polygon;
        int end_polygon = start_polygon + npolygons - 1;

        unsigned int coverage_mask = 0;  // Only used for one_hot.

        const int num_samples_squared = num_samples * num_samples;
        // Max samples is 5 since num_samples_squared must be <= 32 in order for coverage_mask to
        //   fit into 32b.

        const unsigned int full_coverage = (1 << (num_samples_squared)) - 1;
        const float oo_num_samples = 1.0f / static_cast<float>(num_samples);
        const float sample_coverage = 1.0f / static_cast<float>(num_samples_squared);

        for (int p = end_polygon; p >= start_polygon; p--) {
            int poly_class = polys[p].class_id;

            int numv = polys[p].nvertices;

            // If there are too few vertices, or the pixel is outside polygon bbox, continue to
            //   next polygon.
            if (numv <= 2 || fxmax < polys[p].min.x || fxmin >= polys[p].max.x ||
                fymax < polys[p].min.y || fymin >= polys[p].max.y) {
                continue;
            }

            // out_value = 0.5; // Enable this to visualize polygon bounding boxes.

            int winding_number[32];
            for (int s = 0; s < num_samples_squared; s++) {
                winding_number[s] = 0;
            }

            // Loop over polygon vertices.
            int startv = polys[p].start_vertex;
            int i0 = startv + numv - 1;  // Start from the last vertex.
            float2 p0 = vertices[i0];
            int i1;
            float2 p1;
            for (int v = 0; v < numv; v++, i0 = i1, p0 = p1) {
                // Construct an edge from the current and next vertices. The last vertex is
                //   connected to the first.
                i1 = startv + v;
                p1 = vertices[i1];

                // Discard edges that are horizontal. These can't contribute to the image.
                if (p0.y == p1.y) {
                    continue;
                }

                // Discard edges that are completely above, below, or to the right of the pixel.
                if ((fymax < min(p0.y, p1.y)) || (fymin >= max(p0.y, p1.y)) ||
                    (fxmax < min(p0.x, p1.x))) {
                    continue;
                }

                // Compute whether the edge is going up or down.
                int dir = 1;
                if (p0.y < p1.y) {
                    dir = -1;
                }

                // Evaluate winding at each sample location.
                const float fadd = 2.0f * pixel_filter_size * oo_num_samples;
                const float fstart = pixel_filter_size * oo_num_samples - pixel_filter_size;
                float fys = fy + fstart;
                for (int psy = 0, s = 0; psy < num_samples; psy++, fys += fadd) {
                    /*
                    //map sample number to half integers, eg. [1/2, 3/2, 5/2, 7/2]
                    float psy1 = float(psy)+0.5f;
                    //divide by number of sample, eg. [1/8, 3/8, 5/8, 7/8]
                    float psy2 = psy1 * oo_num_samples;
                    //map to [-1,1] range, eg. [-6/8, -2/8, 2/8, 6/8]
                    float psy3 = 2.0f*psy2 - 1.0f;
                    //multiply by pixel filter radius, eg. [-3/8, -1/8, 1/8, 3/8]
                    float psy4 = pixel_filter_size * psy3;
                    //add pixel coordinate, eg. 10.5 -> [10+1/8, 10+3/8, 10+5/8, 10+7/8]
                    float fys = fy + psy4;
                    */
                    float fxs = fx + fstart;
                    for (int psx = 0; psx < num_samples; psx++, fxs += fadd, s++) {
                        // Sample is already covered, spare the effort.
                        if (coverage_mask & (1 << s)) {
                            continue;
                        }

                        // Discard edges that are completely above, below, or to the right of the
                        //   sample.
                        if ((fys < min(p0.y, p1.y)) || (fys >= max(p0.y, p1.y)) ||
                            (fxs < min(p0.x, p1.x))) {
                            continue;
                        }
                        // If the edge is completely to the left from the pixel, update
                        //   winding_number.
                        if (fxs >= max(p0.x, p1.x)) {
                            winding_number[s] += dir;
                        } else {
                            // Compute edge x coordinate at fy by linear interpolation.
                            float r = (fys - p0.y) / (p1.y - p0.y);
                            // Clamp r between [0,1] to avoid numerical problems due to the
                            //   division.
                            r = min(max(r, 0.0f), 1.0f);
                            float ex = p0.x + r * (p1.x - p0.x);
                            if (fxs >= ex) {
                                // The edge is to the left from the pixel. If the edge is going
                                //   upward, increment winding_number, otherwise decrement.
                                winding_number[s] += dir;
                            }
                        }
                    }
                }
            }
            // If winding_number != 0, a sample is covered by the polygon.
            for (int s = 0; s < num_samples_squared; s++) {
                if (winding_number[s]) {
                    if (one_hot) {
                        coverage_mask |= 1 << s;
                    } else {
                        *out = poly_class + 1;
                        return;
                    }
                }
            }
            if (one_hot) {
                // If binarize is true, bail out if any sample is covered.
                // If binarize is false, bail out if all samples are covered.
                if ((binarize && coverage_mask) ||
                    (!binarize && (coverage_mask == full_coverage))) {
                    *out = fillvalue;
                    return;
                }
            }
        }
        // Blend fillvalue over background according to the fraction of samples covered.
        if (one_hot && !binarize) {
            float alpha = static_cast<float>(__popc(coverage_mask)) * sample_coverage;
            out_value = alpha * fillvalue + (1.0f - alpha) * background;
        }
    }
    // Write result.
    *out = out_value;
}

class _RasterizePolygonOp : public OpKernel {
 protected:
    int nclasses_;
    bool binarize_;
    bool one_hot_;
    bool verbose_;

 public:
    virtual void ComputeArch(OpKernelContext* context, TensorShape output_shape, const int width,
                             const int height, const int num_samples,
                             const std::vector<Image>& images, const std::vector<Polygon>& polygons,
                             const std::vector<float2>& vertices) = 0;

    explicit _RasterizePolygonOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("nclasses", &nclasses_));
        OP_REQUIRES_OK(context, context->GetAttr("binarize", &binarize_));
        OP_REQUIRES_OK(context, context->GetAttr("one_hot", &one_hot_));
        OP_REQUIRES(context, nclasses_ > 0,
                    errors::InvalidArgument("Need nclasses > 0, got ", nclasses_));
        OP_REQUIRES_OK(context, context->GetAttr("verbose", &verbose_));
    }

    // We assume that the coordinates per polygon are in order.
    // We assume that the polygons per image are in order.
    // However, we cannot guarantee that the classes per image are in order.
    // Therefore, we reorder all polygons (1) per image and then (2) per class id (3) per
    // polygon.
    struct {
        bool operator()(Polygon a, Polygon b) {
            if (a.image_id < b.image_id) {
                return true;
            } else if (a.image_id == b.image_id) {
                // Sort per class_id
                if (a.class_id > b.class_id) {
                    return false;
                } else if (a.class_id == b.class_id && a.start_vertex > b.start_vertex) {
                    return false;
                } else {
                    return true;
                }
            }
            return false;
        }
    } sortPolygons;

    void SparseInput(OpKernelContext* context) {
        // Grab the input tensor
        const Tensor& polygon_indices_tensor = context->input(0);
        auto input_polygon_indices = polygon_indices_tensor.flat<int>();

        const Tensor& dense_shape_tensor = context->input(1);
        auto dense_shape = dense_shape_tensor.flat<int>();

        const Tensor& polygon_values_tensor = context->input(2);
        auto input_polygon_values = polygon_values_tensor.flat<float>();

        const Tensor& class_ids_indices_tensor = context->input(3);
        const Tensor& class_ids_dense_shape = context->input(4);
        const Tensor& class_ids_values_tensor = context->input(5);

        const Tensor& width_tensor = context->input(6);
        int width = *(width_tensor.flat<int>().data());

        const Tensor& height_tensor = context->input(7);
        int height = *(height_tensor.flat<int>().data());

        const Tensor& num_samples_tensor = context->input(8);
        int num_samples = *(num_samples_tensor.flat<int>().data());

        OP_REQUIRES(context, num_samples >= 1 && num_samples <= 5,
                    errors::InvalidArgument("num_samples must be between 1 and 5, got: ",
                                            num_samples, "."));

        // Unpack and check polygon_dense_shape.
        OP_REQUIRES(context, dense_shape_tensor.shape().dims() == 1,
                    errors::InvalidArgument("polygon_dense_shape must be 1D, shape is ",
                                            dense_shape_tensor.shape().DebugString(), "."));
        int dense_shape_dims = dense_shape_tensor.shape().dim_size(0);
        OP_REQUIRES(context, dense_shape_dims == 3 || dense_shape_dims == 4,
                    errors::InvalidArgument("polygon_dense_shape must have 3 or 4 values, has ",
                                            dense_shape_dims, "."));
        bool single_image = dense_shape_dims == 3;
        int dim = 0;
        int batch_size = single_image ? 1 : dense_shape.data()[dim++];
        int max_polygons_per_image = dense_shape.data()[dim++];
        int max_vertices_per_polygon = dense_shape.data()[dim++];
        int num_coordinates_per_vertex = dense_shape.data()[dim];

        OP_REQUIRES(context, batch_size > 0,
                    errors::InvalidArgument(
                        "dense_shape batch dimension size must be > 0, dense_shape is [",
                        dense_shape_tensor.SummarizeValue(4), "]."));
        OP_REQUIRES(context, max_polygons_per_image >= 0,
                    errors::InvalidArgument(
                        "dense_shape polygons dimension size must be >= 0, dense_shape is [",
                        dense_shape_tensor.SummarizeValue(4), "]."));
        OP_REQUIRES(context, max_vertices_per_polygon >= 0,
                    errors::InvalidArgument(
                        "dense_shape vertices dimension size must be >= 0, dense_shape is [",
                        dense_shape_tensor.SummarizeValue(4), "]."));
        OP_REQUIRES(context, num_coordinates_per_vertex == 2,
                    errors::InvalidArgument(
                        "dense_shape coordinates dimension size must be 2, dense_shape is [",
                        dense_shape_tensor.SummarizeValue(4), "]."));

        // Polygon values tensor is a float vector. Each pair of floats makes
        // up a 2D vertex coordinate.
        OP_REQUIRES(context, polygon_values_tensor.shape().dims() == 1,
                    errors::InvalidArgument("polygon_values must be 1D, shape is ",
                                            polygon_values_tensor.shape().DebugString(), "."));
        OP_REQUIRES(context, (polygon_values_tensor.shape().dim_size(0) & 1) == 0,
                    errors::InvalidArgument("polygon_values size must be divisible by 2, shape is ",
                                            polygon_values_tensor.shape().DebugString(), "."));

        int nvertices = polygon_values_tensor.shape().dim_size(0) / 2;

        // TODO(jrasanen) this seems unnecessary.
        std::vector<float2> vertices(nvertices);
        for (int v = 0; v < nvertices; v++) {
            vertices[v].x = input_polygon_values.data()[v * 2 + 0];
            vertices[v].y = input_polygon_values.data()[v * 2 + 1];
        }

        // Polygon indices tensor is 2D. The first dimension is the same size as
        // the number of vertices, and the second dimension is either 3 (for PVC),
        // or 4 (for BPVC), where B is batch dimension, P is polygon dimension,
        // V is vertex dimension, and C is coordinate axis dimension.
        OP_REQUIRES(context, polygon_indices_tensor.shape().dims() == 2,
                    errors::InvalidArgument("polygon_indices must be 2D, shape is ",
                                            polygon_indices_tensor.shape().DebugString(), "."));
        OP_REQUIRES(context, polygon_indices_tensor.shape().dim_size(0) == nvertices * 2,
                    errors::InvalidArgument("polygon_indices dimension 0 size must match "
                                            "the number of polygon_values (",
                                            nvertices * 2, ") shape is ",
                                            polygon_indices_tensor.shape().DebugString(), "."));

        int dims = polygon_indices_tensor.shape().dim_size(1);
        OP_REQUIRES(context, dense_shape_dims == dims,
                    errors::InvalidArgument("polygon_indices dimension 1 size must match "
                                            "polygon_dense_shape size(",
                                            dims, " != ", dense_shape_dims, ")."));

        OP_REQUIRES(context, class_ids_dense_shape.shape().dims() == 1,
                    errors::InvalidArgument("class_ids_per_polygon_dense_shape must be 1D, "
                                            "shape is ",
                                            dense_shape_tensor.shape().DebugString(), "."));
        int class_id_dense_shape_dims = class_ids_dense_shape.shape().dim_size(0);
        OP_REQUIRES(context, class_id_dense_shape_dims == 3 || class_id_dense_shape_dims == 2,
                    errors::InvalidArgument("class_ids_per_polygon_dense_shape must have 2 or 3 "
                                            "values, has ",
                                            class_id_dense_shape_dims, "."));

        OP_REQUIRES(context, class_ids_indices_tensor.shape().dims() == 2,
                    errors::InvalidArgument("class_ids_indices_tensor must be 2D, shape is ",
                                            class_ids_indices_tensor.shape().DebugString(), "."));

        const auto num_classes = class_ids_values_tensor.NumElements();
        OP_REQUIRES(context, class_ids_indices_tensor.shape().dim_size(0) == num_classes,
                    errors::InvalidArgument("class_ids_indices_tensor dimension 0 size must match "
                                            "the number of class values (",
                                            num_classes, ") shape is ",
                                            class_ids_indices_tensor.shape().DebugString(), "."));

        auto class_ids_dims = class_ids_indices_tensor.shape().dim_size(1);
        int class_ids_dense_shape_dims = class_ids_dense_shape.shape().dim_size(0);
        OP_REQUIRES(
            context, dense_shape_dims == dims,
            errors::InvalidArgument("class_ids_indices dimension 1 size must match "
                                    "class_ids_dense_shape size(",
                                    class_ids_dims, " != ", class_ids_dense_shape_dims, ")."));

        const int* indices = input_polygon_indices.data();
        std::vector<Polygon> polygons;
        std::vector<int> polygons_per_image(batch_size, 0);  // Initialize to zeroes.

        int prev_image = 0, prev_polygon = 0, prev_vertex = 0, prev_coordinate = 0;
        if (nvertices) {
            Polygon polygon;
            int vertex_index = 0;
            // This loop essentially processes stuff read during the previous iteration,
            // so in order to avoid replicating the polygon output code, we instead add
            // one extra iteration.
            for (int i = 0; i <= input_polygon_indices.size(); i += dims) {
                int curr_image, curr_polygon, curr_vertex, curr_coordinate;
                if (i >= input_polygon_indices.size()) {
                    curr_image = prev_image + 1;
                    curr_polygon = prev_polygon;
                    curr_vertex = prev_vertex;
                    curr_coordinate = prev_coordinate;
                } else {
                    if (dims == 3) {  // Single image case.
                        curr_image = 0;
                        curr_polygon = indices[i + 0];
                        curr_vertex = indices[i + 1];
                        curr_coordinate = indices[i + 2];
                    } else {  // Batch case.
                        curr_image = indices[i + 0];
                        curr_polygon = indices[i + 1];
                        curr_vertex = indices[i + 2];
                        curr_coordinate = indices[i + 3];
                        if (i == 0) prev_image = curr_image;
                    }
                    OP_REQUIRES(
                        context, curr_image >= 0 && curr_image < batch_size,
                        errors::InvalidArgument("Image index out of range error: ", curr_image,
                                                " not in [0, ", batch_size - 1, "]."));
                    OP_REQUIRES(
                        context, curr_polygon >= 0 && curr_polygon < max_polygons_per_image,
                        errors::InvalidArgument("Polygon index out of range error: ", curr_polygon,
                                                " not in [0, ", max_polygons_per_image - 1, "]."));
                    OP_REQUIRES(context, curr_vertex >= 0 && curr_vertex < max_vertices_per_polygon,
                                errors::InvalidArgument("Vertex index out of range error: ",
                                                        curr_vertex, " not in [0, ",
                                                        max_vertices_per_polygon - 1, "]."));
                    OP_REQUIRES(context, curr_coordinate >= 0 && curr_coordinate <= 1,
                                errors::InvalidArgument("Coordinate index out of range error: ",
                                                        curr_coordinate, " not in [0, 1]."));
                }

                OP_REQUIRES(
                    context,
                    (curr_image > prev_image) || ((curr_image == prev_image) &&
                                                  ((curr_polygon > prev_polygon) ||
                                                   ((curr_polygon == prev_polygon) &&
                                                    ((curr_vertex > prev_vertex) ||
                                                     ((curr_vertex == prev_vertex) &&
                                                      (curr_coordinate >= prev_coordinate)))))),
                    errors::InvalidArgument("polygon_indices are not in lexicographical order."));

                if (prev_coordinate == 0) {
                    polygon.min.x = min(polygon.min.x, vertices[vertex_index].x);
                    polygon.max.x = max(polygon.max.x, vertices[vertex_index].x);
                } else {
                    polygon.min.y = min(polygon.min.y, vertices[vertex_index].y);
                    polygon.max.y = max(polygon.max.y, vertices[vertex_index].y);
                    vertex_index++;
                }

                if (curr_image != prev_image || curr_polygon != prev_polygon) {
                    polygon.image_id = prev_image;
                    // Storing the polygon id in the class id field so that it can be used to
                    // resolve the class id in AssignSparseClassIds. Need to re-use this field to
                    // preserve alignment in the polygon struct.
                    polygon.class_id = prev_polygon;
                    polygon.nvertices = vertex_index - polygon.start_vertex;

                    polygons.push_back(polygon);
                    polygons_per_image[prev_image]++;

                    polygon.min.x = FLT_MAX;
                    polygon.min.y = FLT_MAX;
                    polygon.max.x = -FLT_MAX;
                    polygon.max.y = -FLT_MAX;
                    polygon.start_vertex = vertex_index;
                }

                prev_image = curr_image;
                prev_polygon = curr_polygon;
                prev_vertex = curr_vertex;
                prev_coordinate = curr_coordinate;
            }
            OP_REQUIRES(context, vertex_index == nvertices,
                        errors::InvalidArgument("polygon_indices don't have one to one mapping to "
                                                "vertices: number of vertex indices (",
                                                vertex_index, ") != number of vertices (",
                                                nvertices, ")."));
            AssignSparseClassIds(context, &polygons, class_ids_values_tensor,
                                 class_ids_indices_tensor);
        }

        std::sort(polygons.begin(), polygons.end(), sortPolygons);

        DrawSorted(context, polygons, vertices, polygons_per_image, batch_size, height, width,
                   num_samples, single_image);
    }

    void DenseInput(OpKernelContext* context) {
        // Grab the input tensor
        const Tensor& polygon_vertices_tensor = context->input(0);
        auto input_polygon_vertices = polygon_vertices_tensor.flat<float>();

        const Tensor& vertex_counts_per_polygon_tensor = context->input(1);
        auto vertex_counts_per_polygon = vertex_counts_per_polygon_tensor.flat<int>();

        const Tensor& class_ids_per_polygon_tensor = context->input(2);
        auto class_ids_per_polygon = class_ids_per_polygon_tensor.flat<int>();

        const Tensor& polygons_per_image_tensor = context->input(3);
        auto polygons_per_image = polygons_per_image_tensor.flat<int>();

        const Tensor& width_tensor = context->input(4);
        int width = *(width_tensor.flat<int>().data());

        const Tensor& height_tensor = context->input(5);
        int height = *(height_tensor.flat<int>().data());

        const Tensor& num_samples_tensor = context->input(6);
        int num_samples = *(num_samples_tensor.flat<int>().data());

        OP_REQUIRES(context, num_samples >= 1 && num_samples <= 5,
                    errors::InvalidArgument("num_samples must be between 1 and 5, got: ",
                                            num_samples, "."));

        OP_REQUIRES(
            context, 2 == polygon_vertices_tensor.shape().dims(),
            errors::InvalidArgument("polygon_vertices tensor must have 2 dimensions, shape is: ",
                                    polygon_vertices_tensor.shape().DebugString(), "."));
        OP_REQUIRES(context, 2 == polygon_vertices_tensor.shape().dim_size(1),
                    errors::InvalidArgument(
                        "polygon_vertices tensor dimension index 1 must be exactly 2,",
                        " shape is: ", polygon_vertices_tensor.shape().DebugString(), "."));
        int nvertices = polygon_vertices_tensor.shape().dim_size(0);

        OP_REQUIRES(context, 1 == vertex_counts_per_polygon_tensor.shape().dims(),
                    errors::InvalidArgument(
                        "vertex_counts_per_polygon must be a 1 dimensional vector,", " shape is: ",
                        vertex_counts_per_polygon_tensor.shape().DebugString(), "."));
        OP_REQUIRES(context, 1 == class_ids_per_polygon_tensor.shape().dims(),
                    errors::InvalidArgument(
                        "class_ids_per_polygon must be a 1 dimensional vector,", " shape is: ",
                        class_ids_per_polygon_tensor.shape().DebugString(), "."));
        OP_REQUIRES(context, 1 == polygons_per_image_tensor.shape().dims(),
                    errors::InvalidArgument("polygons_per_image must be a 1 dimensional vector,",
                                            " shape is: ",
                                            polygons_per_image_tensor.shape().DebugString(), "."));

        int npolygons = vertex_counts_per_polygon_tensor.shape().dim_size(0);
        int len_class_ids_per_polygon = class_ids_per_polygon_tensor.shape().dim_size(0);
        int batch_size = polygons_per_image_tensor.shape().dim_size(0);
        bool single_image = false;
        if (batch_size == 0) {
            single_image = true;
            batch_size = 1;
        }

        OP_REQUIRES(
            context, npolygons == len_class_ids_per_polygon,
            errors::InvalidArgument("vertex_counts_per_polygon vector and class_ids_per_polygon ",
                                    "vector shapes are not equal; ",
                                    vertex_counts_per_polygon_tensor.shape().DebugString(), " and ",
                                    class_ids_per_polygon_tensor.shape().DebugString(), "."));

        OP_REQUIRES(
            context, nvertices >= npolygons,
            errors::InvalidArgument("Number of polygons ", npolygons, " is larger than the number",
                                    " of vertices ", nvertices, "."));

        // Test that the sum of the `vertex_counts_per_polygon` vector adds up to `nvertices`.
        int nvertices_from_vertex_counts = 0;
        for (int i = 0; i < npolygons; i++) {
            nvertices_from_vertex_counts += vertex_counts_per_polygon.data()[i];
        }
        OP_REQUIRES(context, nvertices_from_vertex_counts == nvertices,
                    errors::InvalidArgument(
                        "Sum of vertex_counts_per_polygon", nvertices_from_vertex_counts,
                        " over all polygons does not add up to nvertices ", nvertices, "."));

        if (!single_image) {
            // Test that the sum of the `polygons_per_image` vector adds up to `npolygons`.
            int npolygons_from_images = 0;
            for (int i = 0; i < batch_size; i++) {
                npolygons_from_images += polygons_per_image.data()[i];
            }

            OP_REQUIRES(
                context, npolygons_from_images == npolygons,
                errors::InvalidArgument("Sum of `polygons_from_images` over all images does not ",
                                        "add up to `npolygons`"));
        }

        // Initialize vertices, images and polygons struct.
        std::vector<float2> vertices(nvertices);

        std::vector<int> polygons_per_image_vector(batch_size);
        if (single_image) {
            polygons_per_image_vector[0] = vertex_counts_per_polygon.size();
        } else {
            for (int i = 0; i < polygons_per_image.size(); i++)
                polygons_per_image_vector[i] = polygons_per_image.data()[i];
        }

        std::vector<Polygon> polygons(npolygons);

        if (nvertices > 0) {
            int v = 0;
            int p = 0;
            for (int image_id = 0; image_id < batch_size; image_id++) {
                int npoly_image = polygons_per_image_vector[image_id];

                int last_image = p + npoly_image;

                for (; p < last_image; p++) {
                    int nvertices_poly = vertex_counts_per_polygon.data()[p];
                    int class_id = class_ids_per_polygon.data()[p];

                    if (one_hot_) {
                        OP_REQUIRES(context, class_id < nclasses_,
                                    errors::InvalidArgument(
                                        "class_id ", class_id, " of polygon #", p,
                                        " exceeds the given amount of classes ", nclasses_, "."));
                    }

                    int last_polygon = v + nvertices_poly;

                    float min_x = FLT_MAX;
                    float min_y = FLT_MAX;
                    float max_x = -FLT_MAX;
                    float max_y = -FLT_MAX;

                    polygons[p].start_vertex = v;
                    for (; v < last_polygon; v++) {
                        int d = v * 2;
                        float x = input_polygon_vertices.data()[d];
                        float y = input_polygon_vertices.data()[d + 1];

                        vertices[v].x = x;
                        vertices[v].y = y;

                        min_x = min(min_x, x);
                        min_y = min(min_y, y);
                        max_x = max(max_x, x);
                        max_y = max(max_y, y);
                    }
                    polygons[p].min.x = min_x;
                    polygons[p].min.y = min_y;
                    polygons[p].max.x = max_x;
                    polygons[p].max.y = max_y;
                    polygons[p].nvertices = nvertices_poly;
                    polygons[p].image_id = image_id;
                    polygons[p].class_id = class_id;
                }
            }
        }

        std::sort(polygons.begin(), polygons.end(), sortPolygons);

        DrawSorted(context, polygons, vertices, polygons_per_image_vector, batch_size, height,
                   width, num_samples, single_image);
    }

    void DrawSorted(OpKernelContext* context, const std::vector<Polygon>& polygons,
                    const std::vector<float2>& vertices, const std::vector<int>& polygons_per_image,
                    int batch_size, int height, int width, int num_samples, bool single_image) {
        int noutput_maps = batch_size;
        if (one_hot_) noutput_maps *= nclasses_;

        std::vector<Image> images(noutput_maps);

        int start_polygon = 0;
        // Loop over the images (batch dim).
        for (int image_id = 0; image_id < batch_size; image_id++) {
            int npoly_image = polygons_per_image[image_id];

            // Fill in the polygons in the vector of images (batch_dim * nclasses).
            for (int p = start_polygon; p < start_polygon + npoly_image; p++) {
                // Skip (don't draw) polygons with negative class indices.
                if (polygons[p].class_id < 0) {
                    continue;
                }
                int map_id = one_hot_ ? image_id * nclasses_ + polygons[p].class_id : image_id;

                images[map_id].combineBbox(polygons[p]);
                images[map_id].npolygons += 1;
                // Only save the start_polygon on first occurance.
                if (images[map_id].start_polygon == -1) {
                    images[map_id].start_polygon = p;
                }
            }
            start_polygon += npoly_image;
        }

        int noutputmaps_class = one_hot_ ? nclasses_ : 1;

        // Create an output tensor.
        TensorShape output_shape({batch_size, noutputmaps_class, height, width});
        TensorShape output_shape_single({noutputmaps_class, height, width});
        if (single_image) output_shape = output_shape_single;

        // Verbose summary.
        if (verbose_) {
            LOG(INFO) << "batch_size = " << batch_size
                      << ", noutputmaps_class = " << noutputmaps_class
                      << ", npolygons = " << polygons.size() << ", nvertices = " << vertices.size()
                      << ", width = " << width << ", height = " << height
                      << ", num_samples = " << num_samples;
            for (int i = 0; i < static_cast<int>(output_shape.dims()); i++) {
                LOG(INFO) << "output dim " << i << " size = " << output_shape.dim_size(i);
            }
            for (int b = 0; b < batch_size; b++) {
                for (int c = 0; c < noutputmaps_class; c++) {
                    int i = b * noutputmaps_class + c;
                    int sp = images[i].start_polygon;
                    int nump = images[i].npolygons;
                    LOG(INFO) << "image " << b << " class " << c << ": start_polygon = " << sp
                              << " npolygons = " << nump << " bmin = (" << images[i].min.x << ", "
                              << images[i].min.y << ") bmax = (" << images[i].max.x << ", "
                              << images[i].max.y << ")";
                    for (int p = sp; p < sp + nump; p++) {
                        int sv = polygons[p].start_vertex;
                        int numv = polygons[p].nvertices;
                        LOG(INFO) << "  polygon " << p << ": start_vertex = " << sv
                                  << " nvertices = " << numv << " bmin = (" << polygons[p].min.x
                                  << ", " << polygons[p].min.y << ") bmax = (" << polygons[p].max.x
                                  << ", " << polygons[p].max.y << ")";
                        for (int v = sv; v < sv + numv; v++) {
                            LOG(INFO) << "    vertex " << v << ": " << vertices[v].x << ", "
                                      << vertices[v].y;
                        }
                    }
                }
            }
        }

        // Call derived class's compute here.
        ComputeArch(context, output_shape, width, height, num_samples, images, polygons, vertices);

        if (verbose_) {
            LOG(INFO) << "done";
        }
    }

    void Compute(OpKernelContext* context) override { DenseInput(context); }

 protected:
    std::tuple<int, int> GetClassImageIdPolygonId(const int* class_ids_indices, int index,
                                                  int step) {
        auto offset = index * step;
        if (step == 2) {
            return std::make_tuple(0, class_ids_indices[offset]);
        } else {
            return std::make_tuple(class_ids_indices[offset], class_ids_indices[offset + 1]);
        }
    }

    /**
     * Helper function which assigns class ids to polygons from sparse tensors. The class ids
     * are matched based on the image index and the polygon index values being the same for the
     * polygon as they are for the class id.
     *
     * Skips class ids that do not match to a polygon.
     *
     * Here's an example of polygon indices being used to match class indices to determine class
     * values.
     *
     * The dimensions of the polygon indices (B, S, V, C) represent the following
     *   B: Batch index, which frame this is describing in the batch.
     *   S: Shape index, which shape this is in the frame.
     *   V: Vertex index, which vertex this is for the shape
     *   C: Coordinate index, which coordinate this is in the vertex, 0 for x, 1 for y
     *
     *
     * polygon_indices = \
     *     ([2, 0, 0, 0], # First polygon
     *      [2, 0, 0, 1],
     *      [2, 0, 1, 0],
     *      [2, 0, 1, 1],
     *      [2, 0, 2, 0],
     *      [2, 0, 2, 1],
     *      [2, 1, 0, 0], # Second polygon
     *      [2, 1, 0, 1],
     *      [2, 1, 1, 0],
     *      [2, 1, 1, 1],
     *      [2, 1, 2, 0],
     *      [2, 1, 2, 1],
     *      [3, 1, 0, 0], # Third polygon
     *      [3, 1, 0, 1],
     *      [3, 1, 1, 0],
     *      [3, 1, 1, 1],
     *      [3, 1, 2, 0],
     *      [3, 1, 2, 1])
     *
     *  Note: These polygons are describing three triangles.
     *
     *  The dimensions of the class_indices (B, S, C) represent the following
     *    B: Batch index, which frame this is describing in the batch.
     *    S: Shape index, which shape this is in the frame.
     *    C: Class index, which class this is for the shape, currently always 0, since always a
     *        single class
     *
     * class_indices =  \
     *     [[0, 0, 0], # Doesn't match
     *      [1, 0, 0], # Doesn't match
     *      [1, 1, 0], # Doesn't match
     *      [2, 0, 0], # Matches first polygon
     *      [2, 1, 0], # Matches second polygon
     *      [3, 0, 0], # Doesn't match
     *      [3, 1, 0]] # Matches third polygon
     *
     * class_values = \
     *     [0, # Ignored
     *      1, # Ignored
     *      2, # Ignored
     *      3, # Class id for first polygon
     *      4, # Class id for second polygon
     *      5, # Ignored.
     *      6  # Class id for third polygon
     *      ]
     *
     * @param context Context of the kernel op. Used for OP_REQUIRE checks.
     * @param polygons Pointer to vector of polygon structs to be populated with class ids
     * @param class_ids_values_tensor Tensor of class id values from class id SparseTensor.
     * @param class_ids_indices_tensor Tensor of class id indices from class id SparseTensor.
     */
    void AssignSparseClassIds(OpKernelContext* context, std::vector<Polygon>* polygons,
                              const Tensor& class_ids_values_tensor,
                              const Tensor& class_ids_indices_tensor) {
        const auto class_ids_values_flat_tensor = class_ids_values_tensor.flat<int>();
        const auto num_classes = class_ids_values_flat_tensor.size();
        const auto class_ids_values = class_ids_values_flat_tensor.data();

        const auto class_ids_indices = class_ids_indices_tensor.flat<int>().data();
        const auto class_id_index_step =
            static_cast<int>(class_ids_indices_tensor.shape().dim_size(1));

        OP_REQUIRES(context, num_classes >= static_cast<int>(polygons->size()),
                    errors::InvalidArgument("Number of classes(", num_classes,
                                            ") is less than "
                                            "number of polygons(",
                                            polygons->size(), ")."));

        int class_id_index = 0;
        int class_image_id = 0;
        int class_polygon_id = 0;
        std::tie(class_image_id, class_polygon_id) =
            GetClassImageIdPolygonId(class_ids_indices, class_id_index, class_id_index_step);

        for (auto& polygon : *polygons) {
            const auto image_id = polygon.image_id;
            const auto polygon_id = polygon.class_id;
            polygon.class_id = -1;
            // Some class sparse tensors may have class ids for polygons that don't exist.
            // in this case we skip over them.
            while ((image_id != class_image_id || polygon_id != class_polygon_id) &&
                   (class_id_index < num_classes)) {
                ++class_id_index;
                std::tie(class_image_id, class_polygon_id) = GetClassImageIdPolygonId(
                    class_ids_indices, class_id_index, class_id_index_step);
            }

            OP_REQUIRES(context, class_id_index < num_classes,
                        errors::InvalidArgument("No corresponding class id for image ", image_id,
                                                " polygon ", polygon_id));
            polygon.class_id = class_ids_values[class_id_index];
            if (one_hot_) {
                OP_REQUIRES(context, polygon.class_id < nclasses_,
                            errors::InvalidArgument(
                                "class_id ", polygon.class_id, " for polygon ", polygons->size(),
                                " exceeds the number of classes (", nclasses_, ")."));
            }
        }
    }
};

#else
#endif  // _RASTERIZE_POLYGON_H_
