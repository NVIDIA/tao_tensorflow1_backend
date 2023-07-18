// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

#define EIGEN_USE_GPU
#include "decode_dist.h"
#include "cuda_helper.h"
#include "lrn_decoder_core_gpu.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __device__
#else
#define CUDA_HOSTDEV
#endif

using namespace tensorflow;

// Put GPU code into a namespace to avoid name clashes when
// linking CPU and GPU versions to the same library
namespace GPUCode {

template <typename T>
__global__ void copy2ImageOutput(int* outbuf, const T* inbuf, const int ncols, const int nrows,
                                 const int channels, const int pixelPitch,
                                 const float multiply = 255.0f) {
    uint32_t outputX = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t outputY = blockIdx.y * blockDim.y + threadIdx.y;
    if (outputX >= ncols || outputY >= nrows) return;

    if (channels == 3) {
        float val1 = static_cast<float>(inbuf[outputY * pixelPitch + outputX]) * multiply;
        outbuf[outputY * ncols * 3 + outputX * 3 + 0] = static_cast<int>(val1);

        float val2 =
            static_cast<float>(inbuf[1 * nrows * pixelPitch + outputY * pixelPitch + outputX]) *
            multiply;
        outbuf[outputY * ncols * 3 + outputX * 3 + 1] = static_cast<int>(val2);

        float val3 =
            static_cast<float>(inbuf[2 * nrows * pixelPitch + outputY * pixelPitch + outputX]) *
            multiply;
        outbuf[outputY * ncols * 3 + outputX * 3 + 2] = static_cast<int>(val3);
    } else {
        float val = static_cast<float>(inbuf[outputY * pixelPitch + outputX]) * multiply;
        outbuf[outputY * ncols + outputX] = static_cast<int>(val);
    }
}

template <typename T>
void initOutputWithImage(OpKernelContext* context, int* output, int* output_with_angle,
                         int* output_with_nonmax, const T* input_img, const int width,
                         const int height, const float multiply = 255.0f) {
    int blockX = 16;
    int blockY = 16;
    dim3 gridSize((width + blockX - 1) / blockX, (height + blockY - 1) / blockY);
    dim3 blockSize(blockX, blockY);
    // TODO(xiaolinl): optimize using multi-cuda stream.
    // below 3 kernels can happen concurrently.
    const Eigen::GpuDevice& d = context->eigen_gpu_device();
    copy2ImageOutput<T><<<gridSize, blockSize, 0, d.stream()>>>(output, input_img, width, height, 3,
                                                                width, multiply);
    CHECK_LAST_CUDA_ERROR();

    copy2ImageOutput<T><<<gridSize, blockSize, 0, d.stream()>>>(output_with_angle, input_img, width,
                                                                height, 3, width, multiply);

    CHECK_LAST_CUDA_ERROR();

    copy2ImageOutput<T><<<gridSize, blockSize, 0, d.stream()>>>(output_with_nonmax, input_img,
                                                                width, height, 3, width, multiply);

    CHECK_LAST_CUDA_ERROR();
    cudaDeviceSynchronize();
}

class DecodeDist : public _DecodeDist {
 public:
    explicit DecodeDist(OpKernelConstruction* context) : _DecodeDist(context) {}

    void decode_core(OpKernelContext* context, const float* encoded_blobs,
                     const float* tensor_input_nchw, const int batch_size, int* output0,
                     int* output1, int* output2, int* output_for_metric) {
        std::unique_ptr<lineregressordecoder::LRNDecoder<float>> decoder_gpu;
        decoder_gpu.reset(new lineregressordecoder::LRNDecoder<float>(
            src_width_, src_height_, src_channels_, target_width_, target_height_, up_scale_factor_,
            defined_infinity_, normalize_, radius_, encoding_param_, non_max_radius_,
            max_possible_nodes_, min_valid_mask_, minimum_votes_, max_distance_for_nodes_, true,
            1 << channel_count_for_bit_coding_, context));
        // TODO(xiaolinl): see if below can be parallelized.
        for (int i = 0; i < batch_size; i++) {
            const float* encoded_blobs_this_batch = encoded_blobs + i * total_blob_size_per_image_;
            const float* im_start = tensor_input_nchw + i * target_height_ * target_width_ * 3;
            int* output_start0 = output0 + i * target_height_ * target_width_ * 3;  // votes
            int* output_start1 =
                output1 + i * target_height_ * target_width_ * 3;  // votes with angle
            int* output_start2 = output2 + i * target_height_ * target_width_ * 3;  // nonmax
            int* output_for_metric_start =
                output_for_metric + i * target_height_ * target_width_;  // class map

            initOutputWithImage<float>(context, output_start0, output_start1, output_start2,
                                       im_start, target_width_, target_height_);

            cudaMemset(output_for_metric_start, 0, target_width_ * target_height_ * sizeof(int));
            cudaDeviceSynchronize();
            CHECK_LAST_CUDA_ERROR();

            decoder_gpu->interpretDeviceAsync(encoded_blobs_this_batch);
            cudaDeviceSynchronize();
            CHECK_LAST_CUDA_ERROR();
            decoder_gpu->getOutput(output_start0, output_start1, output_start2,
                                   output_for_metric_start, arrow_length_, 0, background_class_id_);
        }
        decoder_gpu->free();
    }
};

#pragma message("Registering GPU kernel")
REGISTER_KERNEL_BUILDER(
    Name("DecodeDist").Device(DEVICE_GPU).HostMemory("encoded_blobs").HostMemory("images"),
    DecodeDist);
}  // namespace GPUCode
