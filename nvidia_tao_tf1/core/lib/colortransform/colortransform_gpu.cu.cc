// Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.

#define EIGEN_USE_GPU

// Include common code
#include "colortransform.h"

template <typename I, typename O>
__global__ void ColorTransformKernel(const I* input_images, const float* input_transf_mats,
                                     O* output_images, int nbatch, float min_clip, float max_clip,
                                     int height, int width, bool input_channels_first,
                                     bool output_channels_first) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= height) {
        return;
    }
    int b = x / width;
    if (b >= nbatch) {
        return;
    }
    x %= width;

    const int num_channels = 3;
    const I* in = input_images + b * height * width * num_channels;
    O* out = output_images + b * height * width * num_channels;
    const float* mat = input_transf_mats + b * 4 * 4;

    // Call the common code, nothing else to do for CPU version of this file
    _ColorTransformKernel<I, O>(x, y, in, mat, out, min_clip, max_clip, height, width,
                                input_channels_first, output_channels_first);
}

template <typename Device, typename I, typename O>
class ColorTransformOp : public BaseColorTransformOp {
 public:
    explicit ColorTransformOp(OpKernelConstruction* context) : BaseColorTransformOp(context) {}

    void ComputeArch(OpKernelContext* context, Tensor* output_tensor,
                     const Tensor& input_images_tensor, const float* input_transf_mats, int nbatch,
                     int height, int width, bool input_channels_first,
                     bool output_channels_first) override {
        if (verbose_) printf("running GPU version\n");

        auto output_images = output_tensor->flat<O>().data();
        auto input_images = input_images_tensor.flat<I>().data();

        // Launch the kernel
        dim3 dimBlock(8, 8);  // @TODO(jrasanen) optimize block size
        // Outputs are laid out horizontally since CUDA allows a large number of blocks only in
        //   horizontal direction
        dim3 dimGrid(((width * nbatch) + dimBlock.x - 1) / dimBlock.x,
                     (height + dimBlock.y - 1) / dimBlock.y);
        const Eigen::GpuDevice& d = context->eigen_gpu_device();
        ColorTransformKernel<I, O><<<dimGrid, dimBlock, 0, d.stream()>>>(
            input_images, input_transf_mats, output_images, nbatch, min_clip_, max_clip_, height,
            width, input_channels_first, output_channels_first);
    }
};

#pragma message("Registering GPU kernel")
#define REGISTER_KERNEL(I, O)                                       \
    REGISTER_KERNEL_BUILDER(Name("Colortransform")                  \
                                .Device(DEVICE_GPU)                 \
                                .TypeConstraint<I>("input_dtype")   \
                                .TypeConstraint<O>("output_dtype"), \
                            ColorTransformOp<GPUDevice, I, O>);

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
