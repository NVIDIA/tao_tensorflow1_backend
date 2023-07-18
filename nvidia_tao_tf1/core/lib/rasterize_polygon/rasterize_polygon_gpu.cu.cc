// Copyright (c) 2017-2018, NVIDIA CORPORATION.  All rights reserved.

#define EIGEN_USE_GPU

#include <float.h>

#include "tensorflow/core/util/gpu_launch_config.h"

// Below includes are checked into ai-infra repo.
#include "tensorflow/core/kernels/gpu_device_array.h"
#include "tensorflow/core/kernels/gpu_device_array_gpu.h"

// Include common code
#include "rasterize_polygon.h"

// Put GPU code into a namespace to avoid name clashes when linking CPU and GPU versions to the
// same library
namespace GPUCode {
__global__ void RasterizePolygonKernel(float* out, int height, int width, int num_samples,
                                       GpuDeviceArrayStruct<Image> images_data, int nimages,
                                       GpuDeviceArrayStruct<Polygon> polys_data,
                                       GpuDeviceArrayStruct<float2> vertices_data,
                                       const bool binarize, const bool one_hot) {
    const Image* images = GetGpuDeviceArrayOnDevice(&images_data);
    const Polygon* polys = GetGpuDeviceArrayOnDevice(&polys_data);
    const float2* vertices = GetGpuDeviceArrayOnDevice(&vertices_data);
    // GPU
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= height) {
        return;
    }
    int i = x / width;
    if (i >= nimages) {
        return;
    }
    x %= width;

    const Image* image = images + i;
    out += (i * height + y) * width + x;

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
            LOG(INFO) << "running GPU kernel";
        }

        GpuDeviceArrayOnHost<Image> images_array(context, images.size());
        OP_REQUIRES_OK(context, images_array.Init());
        for (int i = 0; i < images.size(); i++) images_array.Set(i, images[i]);
        OP_REQUIRES_OK(context, images_array.Finalize());
        auto device_images = images_array.data();

        GpuDeviceArrayOnHost<Polygon> polygons_array(context, polygons.size());
        if (polygons.size()) {  // Only allocate memory if nonzero.
            OP_REQUIRES_OK(context, polygons_array.Init());
            for (int i = 0; i < polygons.size(); i++) polygons_array.Set(i, polygons[i]);
            OP_REQUIRES_OK(context, polygons_array.Finalize());
        }
        auto device_polygons = polygons_array.data();

        GpuDeviceArrayOnHost<float2> vertices_array(context, vertices.size());
        if (vertices.size()) {  // Only allocate memory if nonzero.
            OP_REQUIRES_OK(context, vertices_array.Init());
            for (int i = 0; i < vertices.size(); i++) vertices_array.Set(i, vertices[i]);
            OP_REQUIRES_OK(context, vertices_array.Finalize());
        }
        auto device_vertices = vertices_array.data();

        // Launch the kernel.
        dim3 dim_block(8, 8);  // @TODO(jrasanen) optimize block size
        // Outputs are laid out horizontally since CUDA allows a large number of blocks only in
        // horizontal direction.
        dim3 dim_grid(((width * images.size()) + dim_block.x - 1) / dim_block.x,
                      (height + dim_block.y - 1) / dim_block.y);
        const Eigen::GpuDevice& d = context->eigen_gpu_device();
        RasterizePolygonKernel<<<dim_grid, dim_block, 0, d.stream()>>>(
            output_images.data(), height, width, num_samples, device_images, images.size(),
            device_polygons, device_vertices, binarize_, one_hot_);
    }
};

#pragma message("Registering GPU kernel")
REGISTER_KERNEL_BUILDER(Name("RasterizePolygon")
                            .Device(DEVICE_GPU)
                            .HostMemory("polygon_vertices")
                            .HostMemory("vertex_counts_per_polygon")
                            .HostMemory("class_ids_per_polygon")
                            .HostMemory("polygons_per_image")
                            .HostMemory("width")
                            .HostMemory("height")
                            .HostMemory("num_samples"),
                        RasterizePolygonOp);

class RasterizeSparsePolygonOp : public RasterizePolygonOp {
 public:
    explicit RasterizeSparsePolygonOp(OpKernelConstruction* context)
        : RasterizePolygonOp(context) {}
    void Compute(OpKernelContext* context) override { SparseInput(context); }
};

#pragma message("Registering GPU kernel")
REGISTER_KERNEL_BUILDER(Name("RasterizeSparsePolygon")
                            .Device(DEVICE_GPU)
                            .HostMemory("polygon_indices")
                            .HostMemory("polygon_dense_shape")
                            .HostMemory("polygon_values")
                            .HostMemory("class_ids_per_polygon_indices")
                            .HostMemory("class_ids_per_polygon_dense_shape")
                            .HostMemory("class_ids_per_polygon_values")
                            .HostMemory("width")
                            .HostMemory("height")
                            .HostMemory("num_samples"),
                        RasterizeSparsePolygonOp);

}  // namespace GPUCode
