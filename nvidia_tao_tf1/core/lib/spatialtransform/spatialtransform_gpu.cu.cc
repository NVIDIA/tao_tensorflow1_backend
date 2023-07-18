// Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.
#define EIGEN_USE_GPU

// Include common code
#include "spatialtransform.h"

template <typename I, typename O>
__global__ void SpatialTransformKernel(const I* input_images, const float* transformation_matrices,
                                       O* output_images, int nbatch, int num_channels, int height,
                                       int width, int output_height, int output_width,
                                       FilterMode filter_mode, float background,
                                       bool input_channels_first, bool output_channels_first) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= output_height) {
        return;
    }
    int b = x / output_width;
    if (b >= nbatch) {
        return;
    }
    x %= output_width;

    if (x >= output_width) {
        return;
    }

    const I* in = input_images + b * height * width * num_channels;
    const float* mat = transformation_matrices + b * 3 * 3;
    O* out = output_images + b * output_height * output_width * num_channels;

    // Code common to CPU and GPU kernel
    _SpatialTransformKernel<I, O>(x, y, in, mat, out, num_channels, height, width, output_height,
                                  output_width, filter_mode, background, input_channels_first,
                                  output_channels_first);
}

template <typename Device, typename I, typename O>
class SpatialTransformOp : public BaseSpatialTransformOp {
 public:
    explicit SpatialTransformOp(OpKernelConstruction* context) : BaseSpatialTransformOp(context) {}

    void ComputeArch(OpKernelContext* context, Tensor* output_tensor,
                     const Tensor& input_images_tensor, const float* transformation_matrices,
                     int nbatch, int num_channels, int height, int width, int output_height,
                     int output_width, bool input_channels_first,
                     bool output_channels_first) override {
        if (verbose_) printf("running GPU version\n");

        auto output_images = output_tensor->flat<O>().data();
        auto input_images = input_images_tensor.flat<I>().data();

        // Launch the kernel
        dim3 dimBlock(8, 8);  // @TODO(jrasanen) optimize block size
        // Outputs are laid out horizontally since CUDA allows a large number of blocks only in
        //   horizontal direction
        dim3 dimGrid(((output_width * nbatch) + dimBlock.x - 1) / dimBlock.x,
                     (output_height + dimBlock.y - 1) / dimBlock.y);
        const Eigen::GpuDevice& d = context->eigen_gpu_device();
        SpatialTransformKernel<I, O><<<dimGrid, dimBlock, 0, d.stream()>>>(
            input_images, transformation_matrices, output_images, nbatch, num_channels, height,
            width, output_height, output_width, filter_mode_, background_, input_channels_first,
            output_channels_first);
    }
};

#pragma message("Registering GPU kernel")
#define REGISTER_KERNEL(I, O)                                      \
    REGISTER_KERNEL_BUILDER(Name("SpatialTransform")               \
                                .Device(DEVICE_GPU)                \
                                .TypeConstraint<I>("input_dtype")  \
                                .TypeConstraint<O>("output_dtype") \
                                .HostMemory("shape"),              \
                            SpatialTransformOp<GPUDevice, I, O>);

REGISTER_KERNEL(uint8, uint8)
REGISTER_KERNEL(uint8, Eigen::half)
REGISTER_KERNEL(uint8, float)
REGISTER_KERNEL(Eigen::half, uint8)
REGISTER_KERNEL(Eigen::half, Eigen::half)
REGISTER_KERNEL(Eigen::half, float)
REGISTER_KERNEL(float, uint8)
REGISTER_KERNEL(float, Eigen::half)
REGISTER_KERNEL(float, float)

#undef REGISTER_KERNEL
