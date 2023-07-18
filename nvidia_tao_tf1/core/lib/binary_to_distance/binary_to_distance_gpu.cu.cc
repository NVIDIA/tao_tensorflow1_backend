// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#define EIGEN_USE_GPU

// Include common code
#include "binary_to_distance.h"

// Put GPU code into a namespace to avoid name clashes when linking CPU and GPU versions to the
//   same library
namespace GPUCode {

__global__ void BinaryToDistanceHorizontalKernel(float* output_images, const float* images,
                                                 const float defined_inf, int nbatch, int height,
                                                 int width, int n_channel_output) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int b = index / height;
    unsigned int y = index % height;
    if (y >= height) {
        return;
    }
    if (b >= nbatch) {
        return;
    }

    float* output = output_images + b * n_channel_output * height * width;
    const float* input = images + b * height * width;

    _BinaryToDistanceHorizontalKernel(output,  // first channel will be distance to the left
                                      output + 3 * height * width,  // Z_left
                                      input, width, y, true, defined_inf);

    // second channel will be distance to the right
    _BinaryToDistanceHorizontalKernel(output + height * width,
                                      output + 4 * height * width,  // Z_right
                                      input, width, y, false, defined_inf);

    for (int x = 0; x < width; x++) {
        _ComputeMaskKernel(output + 2 * height * width, output, output + height * width, x, y,
                           width, height, defined_inf);
    }
}

__global__ void DistanceToBinaryToHorizontalKernel(float* output_images, const float* images,
                                                   const float defined_inf, const int scale,
                                                   int nbatch, int height, int width,
                                                   int n_channel_output, int n_channel_input) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int b = index / height;
    unsigned int y = index % height;
    if (y >= height) {
        return;
    }
    if (b >= nbatch) {
        return;
    }

    float* output_l =
        output_images + b * n_channel_output * (height) * (scale * width) + y * scale * width;

    float* output_r = output_images + b * n_channel_output * (height) * (scale * width) +
                      (height) * (scale * width) + y * scale * width;

    const float* input_l = images + b * n_channel_input * height * width + y * width;
    const float* input_r =
        images + b * n_channel_input * height * width + height * width + y * width;
    const float* input_mask =
        images + b * n_channel_input * height * width + 2 * height * width + y * width;

    _DistanceToBinaryHorizontalKernel(output_l, input_l, input_mask, width, y, true, defined_inf,
                                      scale);
    _DistanceToBinaryHorizontalKernel(output_r, input_r, input_mask, width, y, false, defined_inf,
                                      scale);
}

__global__ void BinaryToDistanceVerticalKernel(float* output_images, const float* images,
                                               const float defined_inf, int nbatch, int height,
                                               int width, int n_channel_output,
                                               int defulat_output_channels) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int b = index / width;
    unsigned int x = index % width;
    if (x >= width) {
        return;
    }
    if (b >= nbatch) {
        return;
    }

    float* output = output_images + b * n_channel_output * height * width +
                    defulat_output_channels * height * width;
    const float* input = images + b * height * width;

    // first channel will be distance to the left
    _BinaryToDistanceVerticalKernel(output,
                                    output + 3 * height * width,  // Z_down
                                    input, height, width, x, true, defined_inf);

    // second channel will be distance to the right
    _BinaryToDistanceVerticalKernel(output + height * width,
                                    output + 4 * height * width,  // Z_up
                                    input, height, width, x, false, defined_inf);

    for (int y = 0; y < height; x++) {
        _ComputeMaskKernel(output + 2 * height * width, output, output + height * width, x, y,
                           width, height, defined_inf);
    }
}

class BinaryToDistanceOp : public _BinaryToDistanceOp {
 public:
    explicit BinaryToDistanceOp(OpKernelConstruction* context) : _BinaryToDistanceOp(context) {}

    void ComputeArch(OpKernelContext* context, float* output_images, const float* images,
                     const float distance_threshold) {
        if (verbose_ && inverse_) {
            printf("\n running GPU version\n");
            printf(" ---- target height=%d  and target width=%d  \n", target_height_,
                   target_width_);
            printf(" ---- input height=%d  and input width=%d  \n", height_, width_);
            printf(" ---- scale of height=%d and scale of width=%d,\n", scale_h_, scale_w_);
        }

        const Eigen::GpuDevice& d = context->eigen_gpu_device();

        if (!inverse_) {
            // Launch the kernel
            dim3 dimBlock(512);  // @TODO(jrasanen) optimize block size
            // Outputs are laid out horizontally since CUDA allows
            //  a large number of blocks only in
            //  horizontal direction
            dim3 dimGrid(((nbatch_ * height_) + dimBlock.x - 1) / dimBlock.x);

            BinaryToDistanceHorizontalKernel<<<dimGrid, dimBlock, 0, d.stream()>>>(
                output_images, images, distance_threshold * ALPHA, nbatch_, height_, width_,
                n_channel_output_);
            cudaDeviceSynchronize();
            if (compute_vertical_) {
                dimGrid = dim3(((nbatch_ * width_) + dimBlock.x - 1) / dimBlock.x);
                BinaryToDistanceVerticalKernel<<<dimGrid, dimBlock, 0, d.stream()>>>(
                    output_images, images, distance_threshold * ALPHA, nbatch_, height_, width_,
                    n_channel_output_, DEFAULT_OUTPUT_CHANNEL);
            }
        } else {
            // Launch the kernelimagesnbatch
            dim3 dimBlock(512);  // @TODO(jrasanen) optimize block size
            // Outputs are laid out horizontally since CUDA allows
            // large number of blocks only in
            // horizontal direction
            dim3 dimGrid(((nbatch_ * height_) + dimBlock.x - 1) / dimBlock.x);

            cudaMemset(output_images, 0.0, sizeof(float) * target_width_ * target_height_ *
                                               nbatch_ * n_channel_output_);

            DistanceToBinaryToHorizontalKernel<<<dimGrid, dimBlock, 0, d.stream()>>>(
                output_images, images, distance_threshold * ALPHA, scale_w_, nbatch_, height_,
                width_, n_channel_output_, n_channel_input_);
            cudaDeviceSynchronize();
        }
    }
};

#pragma message("Registering GPU kernel")
REGISTER_KERNEL_BUILDER(Name("BinaryToDistance").Device(DEVICE_GPU).HostMemory("images"),
                        BinaryToDistanceOp);

}  // namespace GPUCode
