// Copyright (c) 2017-2018, NVIDIA CORPORATION.  All rights reserved.

/*
Polygon rasterization algorithm
  Reference: https://en.wikipedia.org/wiki/Nonzero-rule
    Here we check whether a pixel is inside a polygon by computing winding number
    along a ray from the pixel to the left. The edges we encounter either increment
    or decrement winding number depending on whether they go up or down. If the
    resulting winding number is non-zero, we know we're inside the polygon.
  Input gathering
    N images, each contains P(N) polygons, each of which contains V(P(N)) vertices.
    TODO Snap coordinates to fixed point subpixel grid. Rasterizers usually work in fixed point to
    provide uniform accuracy across the whole image, and to avoid numerical problems. Downside of
    fixed point is that coordinates outside the number range require edges to be clipped to
    supported range. The current implementation works in floating point.
    Compute bbox for each polygon, bbox for each image.
  Rasterization kernel
    @TODO(jrasanen) One thread per image per pixel? Or launch one kernel per image? Or launch one
    kernel per image per polygon? Currently using the first option.
    For each pixel:
    If the pixel is outside image bbox, exit thread.
    Loop over polygons in an image from front to back.
      If the pixel is outside polygon bbox, continue to next polygon.
      Set WindingNumberCounter = 0.
      Loop over polygon vertices.
        Construct an edge from the current and next vertices. The last vertex is connected to the
        first.
        Discard edges that are completely above, below, or to the right of the pixel.
        Compute edge x coordinate given pixel y, discard the edge if it's to the right from the
        pixel.
        Now we know the edge is to the left from the pixel. If the edge is going upward,
        increment WindingNumberCounter, otherwise decrement.
      If WindingNumberCounter != 0, write class_id to pixel, exit thread (since we're drawing front
        to back).
*/

#undef EIGEN_USE_GPU

#include <float.h>

// Include common code
#include "rasterize_polygon.h"

// The code will be compiled for GPU and CPU, but we should register only once (CPU)
REGISTER_OP("RasterizePolygon")
    .Input("polygon_vertices: float")
    .Input("vertex_counts_per_polygon: int32")
    .Input("class_ids_per_polygon: int32")
    .Input("polygons_per_image: int32")
    .Input("width: int32")
    .Input("height: int32")
    .Input("num_samples: int32")
    .Output("output_images: float")
    .Attr("nclasses: int = 1")
    .Attr("binarize: bool = false")
    .Attr("one_hot: bool = true")
    .Attr("verbose: bool = false")
    .SetShapeFn([](::shape_inference::InferenceContext* c) {
        // Inherit the output shape from the attributes.
        int nclasses;
        TF_RETURN_IF_ERROR(c->GetAttr("nclasses", &nclasses));
        bool one_hot;
        TF_RETURN_IF_ERROR(c->GetAttr("one_hot", &one_hot));
        std::vector<::shape_inference::DimensionHandle> dims_out;
        // Batch dimension (N).
        // Check if we know the size of polygons_per_image vector.
        ::shape_inference::DimensionHandle dim = c->Dim(c->input(3), 0);
        if (c->ValueKnown(dim)) {
            int polygons_per_image_elements = c->Value(dim);
            if (polygons_per_image_elements > 0) {
                dims_out.push_back(c->MakeDim(polygons_per_image_elements));
            }
            // Else skip batch dimension.
        } else {
            // Size unknown, assume unknown batch dimension.
            dims_out.push_back(c->UnknownDim());
        }
        // Channels dimension (C).
        dims_out.push_back(c->MakeDim(one_hot ? nclasses : 1));
        // Height dimension (H).
        const Tensor* height_tensor = c->input_tensor(5);
        dims_out.push_back(height_tensor ? c->MakeDim(height_tensor->flat<int>().data()[0])
                                         : c->UnknownDim());
        // Width dimension (W).
        const Tensor* width_tensor = c->input_tensor(4);
        dims_out.push_back(width_tensor ? c->MakeDim(width_tensor->flat<int>().data()[0])
                                        : c->UnknownDim());
        c->set_output(0, c->MakeShape(dims_out));
        return Status::OK();
    })
    .Doc(R"doc(
         Polygon rasterizer op.
         Summary:
            * Takes in a set of ordered vertices and labels all the pixels that are within the
              vertices.
            * The vertices are assumed to belong to a polygon.

         Arguments:
            polygon_vertices: a tensor in the form of a list of lists. The top-level list contains
                sub-lists with 2 elements each; each sub-list contains absolute x/y coordinates
                (in that order) of a single vertex of a single polygon for a single image
                (= raster map). The length of the top-level list is therefore equal to the total
                number of vertices over all polygons that we are drawing over all raster maps.
            vertex_counts_per_polygon: a tensor in the form of a flat list. The elements of the list
                are the vertex counts for each polygon that we will draw during rasterization. Thus,
                the length of this list is equal to the number of polygons we will draw, and if we
                were to sum all the values in this list, the sum should equal the length of the
                ``polygon_vertices`` list above.
            class_ids_per_polygon: a tensor in the form of a flat list having the same shape as the
                ``vertex_counts_per_polygon`` list above. Each list element is an ID representing
                the class to which each polygon belongs.
            polygons_per_image: if an empty tensor, we assume only one single image (i.e. this
                call will output only a single raster map). Otherwise, this should be a tensor in
                the form of a flat list, where each list element is the number of polygons to be
                drawn for that image (raster). In this case, the sum of the list values should equal
                the length of the ``vertex_counts_per_polygon`` list above.

         Returns:
            cov: a fp32 tensor (`NCHW`) containing the output map. When ``one_hot`` is used, the
                number of channels `C` is equal to ``nclasses``, and when it is not used, it is
                equal to 1.
          )doc");

void RasterizePolygonKernel(int x, int y, float* out, int height, int width, int num_samples,
                            const Image* image, const Polygon* polys, const float2* vertices,
                            const bool binarize, const bool one_hot) {
    // Code common to CPU and GPU kernel
    _RasterizePolygonKernel(x, y, out, height, width, num_samples, image, polys, vertices, binarize,
                            one_hot);
}

class RasterizePolygonOp : public _RasterizePolygonOp {
 public:
    explicit RasterizePolygonOp(OpKernelConstruction* context) : _RasterizePolygonOp(context) {}

    void ComputeArch(OpKernelContext* context, TensorShape output_shape, const int width,
                     const int height, const int num_samples, const std::vector<Image>& images,
                     const std::vector<Polygon>& polygons, const std::vector<float2>& vertices) {
        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
        auto output_images = output_tensor->template flat<float>();

        if (verbose_) {
            LOG(INFO) << "running CPU kernel";
        }
        for (int b = 0; b < static_cast<int>(images.size()); b++) {
            float* out = output_images.data() + b * height * width;
            const Image* image = &images[b];
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    RasterizePolygonKernel(x, y, out + y * width + x, height, width, num_samples,
                                           image, &polygons[0], &vertices[0], binarize_, one_hot_);
                }
            }
        }
    }
};

#pragma message("Registering CPU kernel")
REGISTER_KERNEL_BUILDER(Name("RasterizePolygon").Device(DEVICE_CPU), RasterizePolygonOp);

REGISTER_OP("RasterizeSparsePolygon")
    .Input("polygon_indices: int32")      // TODO(jrasanen): int64 support?
    .Input("polygon_dense_shape: int32")  // TODO(jrasanen): int64 support?
    .Input("polygon_values: float")
    .Input("class_ids_per_polygon_indices: int32")      // TODO(jrasanen): int64 support?
    .Input("class_ids_per_polygon_dense_shape: int32")  // TODO(jrasanen): int64 support?
    .Input("class_ids_per_polygon_values: int32")       // TODO(jrasanen): int64 support?
    .Input("width: int32")
    .Input("height: int32")
    .Input("num_samples: int32")
    .Output("output_images: float")
    .Attr("nclasses: int = 1")
    .Attr("binarize: bool = false")
    .Attr("one_hot: bool = true")
    .Attr("verbose: bool = false")
    .SetShapeFn([](::shape_inference::InferenceContext* c) {
        // Inherit the output shape from the attributes.
        int nclasses;
        TF_RETURN_IF_ERROR(c->GetAttr("nclasses", &nclasses));
        bool one_hot;
        TF_RETURN_IF_ERROR(c->GetAttr("one_hot", &one_hot));
        std::vector<::shape_inference::DimensionHandle> dims_out;
        // Batch dimension (N).
        // Try figuring out the number of polygon_dense_shape elements. If it can't be infered,
        // assume 3D (= no batch dimension).
        ::shape_inference::DimensionHandle dim = c->Dim(c->input(1), 0);
        if (c->ValueKnown(dim)) {
            int dense_shape_elements = c->Value(dim);
            if (dense_shape_elements == 4) {
                // We have batch dimension. Try figuring out its exact size.
                const Tensor* dense_shape_tensor = c->input_tensor(1);
                if (dense_shape_tensor) {
                    int batch_size = dense_shape_tensor->flat<int>().data()[0];
                    dims_out.push_back(c->MakeDim(batch_size));
                } else {
                    dims_out.push_back(c->UnknownDim());
                }
            }
        }
        // Channels dimension (C).
        dims_out.push_back(c->MakeDim(one_hot ? nclasses : 1));
        // Height dimension (H).
        const Tensor* height_tensor = c->input_tensor(7);
        dims_out.push_back(height_tensor ? c->MakeDim(height_tensor->flat<int>().data()[0])
                                         : c->UnknownDim());
        // Width dimension (W).
        const Tensor* width_tensor = c->input_tensor(6);
        dims_out.push_back(width_tensor ? c->MakeDim(width_tensor->flat<int>().data()[0])
                                        : c->UnknownDim());
        c->set_output(0, c->MakeShape(dims_out));
        return Status::OK();
    })
    .Doc(R"doc(
         Sparse polygon rasterizer op.
         Summary:
            Takes in a SparseTensor describing a set of polygons to draw.

            polygon_dense_shape must be either 3D (PVC) or 4D (NPVC), where N is batch
            dimension, P is polygons, V vertices, and C coordinate index (0 or 1). In the
            3D case the op returns a 3D tensor (CHW or HWC). In the 4D case the first
            dimension of dense_shape specifies batch size, and the op returns a 4D tensor
            (NCHW or NHWC). Polygon_values is a flat fp32 list of interleaved vertex x
            and y coordinates. Polygon_indices is a 2D tensor with dimension 0 the size of
            the polygons.values tensor, and dimension 1 either 3D (PVC) or 4D (NPVC).

            class_ids_per_polygon_dense_shape must be either 2D(PC) or 3D(NPC) where N is batch
            dimension, P is polygon, C class index (currently always 0).

         Arguments:
            polygon_indices: indices field of a SparseTensor describing the input polygons.
            polygon_dense_shape: dense_shape field of a SparseTensor describing the input polygons.
            polygon_values: values field of a SparseTensor describing the input polygons.
            class_ids_per_polygon_values: values field of a SparseTensor describing the classe ids
                for each polygon.
            class_ids_per_polygon_indices: indices field of a SparseTensor describing the classe ids
                for each polygon.
            class_ids_per_polygon_dense_shape: shape field of a SparseTensor describing the classe ids
                for each polygon.
            width: width of the output map.
            height: height of the output map.
            num_samples:  number of samples per box filter dimension.

         Returns:
            cov: a fp32 tensor (`CHW` or `NCHW`) containing the output map. When ``one_hot`` is
                used, the number of channels `C` is equal to ``nclasses``, and when it is not used,
                it is equal to 1.
          )doc");

class RasterizeSparsePolygonOp : public RasterizePolygonOp {
 public:
    explicit RasterizeSparsePolygonOp(OpKernelConstruction* context)
        : RasterizePolygonOp(context) {}

    void Compute(OpKernelContext* context) override { SparseInput(context); }
};

#pragma message("Registering CPU kernel")
REGISTER_KERNEL_BUILDER(Name("RasterizeSparsePolygon").Device(DEVICE_CPU),
                        RasterizeSparsePolygonOp);
