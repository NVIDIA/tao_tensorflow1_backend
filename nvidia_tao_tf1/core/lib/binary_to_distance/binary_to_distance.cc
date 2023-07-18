// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
/*
Binary to distance
   Provided a input of binary image, compute the distance to left/right/top/down for
for a binary mask.
   For example, in one row of a binary image:
   where 1 means the pixel is for some classification value
   0 0 0 1 0 0 0 0 0 1 0 0 0

   Then output:
   distance to left (defined infinity: inf):
   inf inf inf 0 1 2 3 4 5 0 1 2 3

   distance to right:
   3 2 1 0 5 4 3 2 1 0 inf inf inf
*/

#undef EIGEN_USE_GPU

#include <float.h>

// Include common code
#include "binary_to_distance.h"

// The code will be compiled for GPU and CPU, but we should register only once (CPU)
REGISTER_OP("BinaryToDistance")
    .Input("images: float")
    .Output("output_images: float")
    .Attr("distance_threshold: float = 40")
    .Attr("target_height: int")
    .Attr("target_width: int")
    .Attr("compute_vertical: bool = false")
    .Attr("inverse: bool = false")
    .Attr("verbose: bool = false")
    .SetShapeFn([](::shape_inference::InferenceContext* c) {
        // Inherit the output shape from the attributes
        int target_height;
        TF_RETURN_IF_ERROR(c->GetAttr("target_height", &target_height));
        int target_width;
        TF_RETURN_IF_ERROR(c->GetAttr("target_width", &target_width));
        std::vector<::shape_inference::DimensionHandle> dims_out;
        dims_out.push_back(c->UnknownDim());
        dims_out.push_back(c->UnknownDim());
        dims_out.push_back(c->MakeDim(target_height));
        dims_out.push_back(c->MakeDim(target_width));
        c->set_output(0, c->MakeShape(dims_out));
        return Status::OK();
    })
    .Doc(R"doc(
        BinaryToDistance op.
        Summary:
            * Takes binary image (inverse=false) then convert it to generate multichannel distance map along with valid masks and normalization masks.
            * Down sampling is enabled by attribute and it is simple take every other scan-line vertically and horizontally.
            * If include_vertical is on number of distance maps becomes 4, # of valid masks 2, and # of normalization masks 2.
            * If inverse operation. take input as distance maps, recover to classified pixel with voting recovered from distance


        Arguments:
            (input) images: a tensor of binary images.
            (attribute) dist_threshold: number that greater than this we define as infinity
            (attribute) target_height: output height, if inverse=false, it is same with input height
            (attribute) target_width: output width, if inverse=false, it is same with input width
            (attribute) compute_vertical: it including distance to vertical direction
            (attribute) inverse: should this compute distance to votes for binary
            (attribute) verbose: print more information


        Returns:
            cov: a fp32 tensor (`NCHW`) containing the output map.
                 The order of C becomes
                 (if inverse = false)
                 1) distance to left
                 2) distance_to_right
                 3) horizontal valid mask
                 4) horizontal normalization

                 if include_vertical = true and inverse = true
                 5) distance_to_bottom
                 6) distance_to_top
                 7) vertical valid mask
                 8) vertical normalization

                (if inverse = true)
                1) voting for distance to left
                2) voting for distance to right

                if include_vertical = true and inverse = false (This is not yet implemented!)
                3) voting for distance to top
                4) voting for distance to bottom

          )doc");

class BinaryToDistanceOp : public _BinaryToDistanceOp {
 public:
    explicit BinaryToDistanceOp(OpKernelConstruction* context) : _BinaryToDistanceOp(context) {}

    void computeBinaryToDistance(float* output_images, const float* images,
                                 const float dist_threshold) {
        for (int b = 0; b < nbatch_; b++) {
            // order is batch, channels, height, and width. We can change this if necessary.
            // We just need 1 channel as input....
            // output channel is given as.....
            float* output = output_images + b * n_channel_output_ * height_ * width_;
            const float* input = images + b * height_ * width_;
            for (int y = 0; y < height_; y++) {
                _BinaryToDistanceHorizontalKernel(
                    // first channel will be distance to the left
                    output,
                    // Z_left
                    output + 3 * height_ * width_, input, width_, y, true, dist_threshold * ALPHA);

                // second channel will be distance to the right
                _BinaryToDistanceHorizontalKernel(output + height_ * width_,
                                                  output + 4 * height_ * width_, input, width_, y,
                                                  false, dist_threshold * ALPHA);
            }

            for (int y = 0; y < height_; y++) {
                for (int x = 0; x < width_; x++) {
                    // Basically min operation between the above....!!
                    _ComputeMaskKernel(output + 2 * height_ * width_, output,
                                       output + height_ * width_, x, y, width_, height_,
                                       dist_threshold * ALPHA);
                }
            }
            // When we need to do vertical...
            if (compute_vertical_) {
                output = output_images + b * n_channel_output_ * height_ * width_ +
                         DEFAULT_OUTPUT_CHANNEL * height_ * width_;
                for (int x = 0; x < width_; x++) {
                    _BinaryToDistanceVerticalKernel(output,
                                                    // Z_down
                                                    output + 3 * height_ * width_, input, height_,
                                                    width_, x, true, dist_threshold * ALPHA);

                    // second channel will be distance to the top
                    _BinaryToDistanceVerticalKernel(output + height_ * width_,
                                                    output + 4 * height_ * width_, input, height_,
                                                    width_, x, false, dist_threshold * ALPHA);
                }
                for (int y = 0; y < height_; y++) {
                    for (int x = 0; x < width_; x++) {
                        // Basically min operation between the above....!!
                        _ComputeMaskKernel(output + 2 * height_ * width_, output,
                                           output + height_ * width_, x, y, width_, height_,
                                           dist_threshold * ALPHA);
                    }
                }
            }
        }
    }

    void computeDistanceToBinary(float* output_images, const float* images,
                                 const float dist_threshold) {
        for (int b = 0; b < nbatch_; b++) {
            for (int y = 0; y < height_; y++) {
                float* output_l = output_images +
                                  b * n_channel_output_ * (height_) * (scale_w_ * width_) +
                                  y * scale_w_ * width_;

                float* output_r = output_images +
                                  b * n_channel_output_ * (height_) * (scale_w_ * width_) +
                                  (height_) * (scale_w_ * width_) + y * scale_w_ * width_;

                const float* input_l =
                    images + b * n_channel_input_ * height_ * width_ + y * width_;
                const float* input_r = images + b * n_channel_input_ * height_ * width_ +
                                       height_ * width_ + y * width_;
                const float* input_mask = images + b * n_channel_input_ * height_ * width_ +
                                          2 * height_ * width_ + y * width_;

                // to left
                _DistanceToBinaryHorizontalKernel(output_l, input_l, input_mask, width_, y, true,
                                                  dist_threshold * ALPHA, scale_w_);
                // to right
                _DistanceToBinaryHorizontalKernel(output_r, input_r, input_mask, width_, y, false,
                                                  dist_threshold * ALPHA, scale_w_);
            }
        }
    }

    void ComputeArch(OpKernelContext* context, float* output_images, const float* images,
                     const float dist_threshold) {
        if (verbose_) {
            printf("\n running CPU kernel\n");
        }

        if (!inverse_) {
            computeBinaryToDistance(output_images, images, dist_threshold);
        } else {
            computeDistanceToBinary(output_images, images, dist_threshold);
        }
    }
};

#pragma message("Registering BinaryToDistance Op CPU kernel")
REGISTER_KERNEL_BUILDER(Name("BinaryToDistance").Device(DEVICE_CPU).HostMemory("images"),
                        BinaryToDistanceOp);
