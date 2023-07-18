// Copyright (c) 2016-2019, NVIDIA CORPORATION.  All rights reserved.

#define EIGEN_USE_GPU

// Include common code in GPUCode namespace
#include "rasterize_bbox.h"

// Below includes are checked into ai-infra repo.
#include "tensorflow/core/kernels/gpu_device_array.h"
#include "tensorflow/core/kernels/gpu_device_array_gpu.h"

// Put GPU code into a namespace to avoid name clashes when linking CPU and GPU versions to the
// same library
namespace GPUCode {

__global__ void RasterizeBboxKernel(GpuDeviceArrayStruct<Bbox> bboxes_data,
                                    const uint32_t gradient_flags,
                                    GpuDeviceArrayStruct<int32_t> class_indices_data, int width,
                                    int height, int num_classes, int num_gradients, int num_outputs,
                                    float* out) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= height) {
        return;
    }
    int c = x / width;
    if (c >= num_outputs) {
        return;
    }
    x %= width;

    const Bbox* bboxes = GetGpuDeviceArrayOnDevice(&bboxes_data);
    const int32_t* class_indices = GetGpuDeviceArrayOnDevice(&class_indices_data);

    int im = c / num_classes;
    int cl = c % num_classes;
    int bbox_base = class_indices[c * 2 + 0];
    int num_bboxes = class_indices[c * 2 + 1];

    bboxes += bbox_base;

    int o = im * num_classes + cl;
    out += o * num_gradients * width * height;

    // Call the common API for rest of the handling
    _RasterizeBboxKernel(x, y, bboxes, gradient_flags, num_bboxes, width, height, num_classes,
                         num_gradients, num_outputs, out);
}

class RasterizeBboxOp : public _RasterizeBboxOp {
 public:
    explicit RasterizeBboxOp(OpKernelConstruction* context) : _RasterizeBboxOp(context) {}

    // GPU specific computation code
    void ComputeArch(OpKernelContext* context, float* output_images,
                     const std::vector<Bbox>& bboxes, uint32_t combined_gradient_flags,
                     const std::vector<int>& class_indices, int num_images, int output_height,
                     int output_width, int num_gradients, int num_outputs) override {
        if (verbose_) {
            LOG(INFO) << "running GPU kernel";
        }

        GpuDeviceArrayOnHost<Bbox> bboxes_array(context, bboxes.size());
        if (bboxes.size()) {
            OP_REQUIRES_OK(context, bboxes_array.Init());
            for (int i = 0; i < bboxes.size(); i++) bboxes_array.Set(i, bboxes[i]);
            OP_REQUIRES_OK(context, bboxes_array.Finalize());
        }
        auto device_bboxes = bboxes_array.data();

        GpuDeviceArrayOnHost<int32_t> class_indices_array(context, class_indices.size());
        OP_REQUIRES_OK(context, class_indices_array.Init());
        for (int i = 0; i < class_indices.size(); i++) class_indices_array.Set(i, class_indices[i]);
        OP_REQUIRES_OK(context, class_indices_array.Finalize());
        auto device_indices = class_indices_array.data();

        // launch the kernel
        dim3 dim_block(8, 8);  // @TODO(jrasanen): optimize block size
        // outputs are laid out horizontally since CUDA allows a large number of blocks only in
        //   horizontal direction
        dim3 dim_grid(((output_width * num_outputs) + dim_block.x - 1) / dim_block.x,
                      (output_height + dim_block.y - 1) / dim_block.y);

        const Eigen::GpuDevice& d = context->eigen_gpu_device();
        RasterizeBboxKernel<<<dim_grid, dim_block, 0, d.stream()>>>(
            device_bboxes, combined_gradient_flags, device_indices, output_width, output_height,
            num_classes, num_gradients, num_outputs, output_images);
    }
};

#pragma message("Registering GPU kernel")
REGISTER_KERNEL_BUILDER(
    Name("RasterizeBbox")
        .Device(DEVICE_GPU)
        .HostMemory("num_images")     // these inputs must be in CPU accessible memory since we
        .HostMemory("num_gradients")  // need to touch the data before kernel launch
        .HostMemory("image_height")
        .HostMemory("image_width")
        .HostMemory("bboxes_per_image")
        .HostMemory("bbox_class_ids")
        .HostMemory("bbox_matrices")
        .HostMemory("bbox_gradients")
        .HostMemory("bbox_coverage_radii")
        .HostMemory("bbox_flags")
        .HostMemory("bbox_sort_values")
        .HostMemory("gradient_flags"),
    RasterizeBboxOp);

}  // namespace GPUCode
