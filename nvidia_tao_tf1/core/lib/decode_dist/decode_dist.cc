// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

#undef EIGEN_USE_GPU
#include <float.h>

// Include common code
#include "decode_dist.h"
#include "lrn_decoder_core_cpu.h"

// We should register only once (CPU)
REGISTER_OP("DecodeDist")
    .Input("encoded_blobs: float")
    .Input("images: float")
    .Output("votes: int32")
    .Output("votes_with_direction: int32")
    .Output("non_max: int32")
    .Output("class_map: int32")
    .Attr("n_classes: int = 1")
    .Attr("target_width: int")
    .Attr("target_height: int")
    .Attr("src_width: int")
    .Attr("src_height: int")
    .Attr("up_scale_factor: int")
    .Attr("decoding_option: {'dist', 'angle'}")
    .Attr("radius: int = 20")
    .Attr("defined_infinity: int = 30")
    .Attr("minimum_votes: int = 1")
    .Attr("min_valid_mask: float = 0.5")
    .Attr("non_max_radius: int = 2")
    .Attr("background_class_id: int")
    .Attr("max_possible_nodes: int = 4096")
    .Attr("max_distance_for_nodes: int = 5")
    .Attr("arrow_length: int = 20")
    .Attr("normalize: bool = false")
    .Attr("verbose: bool = false")
    .SetShapeFn([](::shape_inference::InferenceContext* c) {
        // Inherit the output shape from the attributes
        int target_height;
        TF_RETURN_IF_ERROR(c->GetAttr("target_height", &target_height));
        int target_width;
        TF_RETURN_IF_ERROR(c->GetAttr("target_width", &target_width));

        std::vector<::shape_inference::DimensionHandle> dims_out1;
        dims_out1.push_back(c->UnknownDim());            // Number of images. like batch size
        dims_out1.push_back(c->MakeDim(target_height));  // target height
        dims_out1.push_back(c->MakeDim(target_width));   // target width
        dims_out1.push_back(c->MakeDim(3));              // color images with RGB
        c->set_output(0, c->MakeShape(dims_out1));
        c->set_output(1, c->MakeShape(dims_out1));
        c->set_output(2, c->MakeShape(dims_out1));

        std::vector<::shape_inference::DimensionHandle> dims_out2;
        dims_out2.push_back(c->UnknownDim());            // Number of images. like batch size
        dims_out2.push_back(c->MakeDim(1));              // single channel class map.
        dims_out2.push_back(c->MakeDim(target_height));  // target height
        dims_out2.push_back(c->MakeDim(target_width));   // target width
        c->set_output(3, c->MakeShape(dims_out2));
        return Status::OK();
    })
    .Doc(R"doc(
        Decode Dist Op.
        Summary:
            Provided set of distance, angle and bit-coding encoded channels, this operator
            decode the signals(See Generate_dist_from_lineseg Op Docs for Encode details.).

            For distance channels, one pixel has information such as where its closest label pixel,
            with label pixel type can be, defined by bit-coding channels. For example, in
            binary class case, number of bit-coding channels would be one with 1 and 0 as class id.
            Provided location information from distance channels and angle values, this operator also
            decode the angle features that original comes with encoding.

            For the pixel that is label-pixel itself (pixel with valid class type), it would get
            a vote from itself. Other pixels around this pixel within radius of choice would also
            vote for this pixel provided this label pixel is their closest label-pixel.
            More pixels are voting for this label-pixel would yield higher number of votes. Similarly,
            based on valid mask, voting, this operator also performs Gaussian bluring for reducing noise,
            non-max suppression, etc.

        References:
            [1] https://confluence.nvidia.com/display/AV/Line+Regressor+Encoding

        Arguments:
        encoded_blobs: a fp32 tensor with shape 'NCHW'.
            N: batch size, C: number of channels, H: height, W:width.
            Based on the original encoded option, encoded blobs as follows:
            (1) 'dist' default option (distance oriented output channels) provides
                encoded tensor as follows:
                -0:MASK
                -1:dx
                -2:dy
                -3:ndx
                -4:ndy
                -5:(cos+1)*0.5
                -6:(sin+1)*0.5
                -(7-N): bit coding channels,
                depends on how total number of classes we have (n_classes)
                we get ceil(log2(n_classes)) number of bit coding channels.
            (2) 'angle' another option (angle oriented output channels) provides
                encoded tensor as follows:
                -0:MASK
                -1:dist (magnitude of dx and dy for above option)
                -2:(cos+1)*0.5 toward 0 dist
                -3:(sin+1)*0.5 toward 0 dist
                -4:(cos+1)*0.5 direction
                -5:(sin+1)*0.5 direction
                -(6-N): bit coding channels, etc
                same bit-codding channels as default option
        images: a fp32 tensor of `NCHW`.
            N: batch size, C: number of channels, H: height, W:width.
            Tensor that this decode operator to visualize on top of. It can be the
            original color images, in this case C=3. The height and width should
            be matched with the expected output height and width from this operator.

        Attributes:
        n_classes: number of class we identify the labels (excluding background class).
            Must be positive.
        target_width: expected width of the output image.
        target_height: expected height of the output image.
        src_width: input label width.
        src_height: input label height.
        up_scale_factor: factor for up scale from encoded dimension to decoded dimension.
            This operator requires down_scale_factor to be in one of following
            values: 1, 2, 4, 8, 16.
            Assume src_width*up_scale_factor >= target_width. (same with height).
        decoding_option: 'dist' or 'angle' matched the original encoding option. See above.
        radius: original encoded radius of choice. (radius of encoded pixel).
            See Generate_dist_from_lineseg Op Docs for more details.
        defined_infinity: original encoded defined_infinity of choice.
        minimum_votes: minimum of votes one pixel to be consider label pixel.
        min_valid_mask: minimum threshold for differentiate pixel to be valid mask. Default
            to be 0.5, meaning pixel from mask channel greater than 0.5 would be considered valid.
        non_max_radius: radius of choice to perform Gaussian blurring and non-max suppression.
        background_class_id: integer id that identifies as background as starting class id.
            this parameter supports for practices that set background = -1.
            For visualization and consistency purposes, all class id decoded from background
            will be mapped to start from background = 0. For example, for the list of original
            class id = [-1, 0, 1, 2, ...], the output `class_map` would have correspoding
            class id = [0, 1, 2, 3, ...]. Extra step will be needed if one wishes to map back
            exactly the original class id.
        max_possible_nodes: predefined buffer size for node.
        max_distance_for_nodes: maximum distance values for consider node extraction.
        arrow_length: length for drawing directional vector from decoded angle values.
        verbose: if print out some extra information
        normalize: if the expected distance channels from the input is normalized.
            (multipied by defined_infinity) see doc string for GenerateDistFromLineseg for details.

        Returns:
        votes: a int32 tensor (`NHWC`) with voting from distance channels.
            With color-coded, visualization on class type, C=3.
        votes_with_direction: a int32 tensor (`NHWC`) with voting cooperated with
            angle decoded directions. With color-coded, visualization on class type, C=3.
        non_max: a int32 tensor (`NHWC`)  with non-max suppression.
            With color-coded, C=3.
        class_map: a int32 tensor (`NHWC`) decoded class map from original bit-coding channels.
            pixel values are related to class id, so C=1.
        )doc");

class DecodeDist : public _DecodeDist {
 public:
    explicit DecodeDist(OpKernelConstruction* context) : _DecodeDist(context) {}

    void decode_core(OpKernelContext* context, const float* encoded_blobs,
                     const float* tensor_input_nchw, const int batch_size, int* output0,
                     int* output1, int* output2, int* output_for_metric) {
        std::unique_ptr<lineregressordecoder::LRNDecoderCPU<float>> decoder_;
        decoder_.reset(new lineregressordecoder::LRNDecoderCPU<float>(
            src_width_, src_height_, src_channels_, target_width_, target_height_, up_scale_factor_,
            defined_infinity_, normalize_, radius_, encoding_param_, non_max_radius_,
            max_possible_nodes_, min_valid_mask_, minimum_votes_, max_distance_for_nodes_, true,
            1 << channel_count_for_bit_coding_));

        for (int i = 0; i < batch_size; i++) {
            const float* start = encoded_blobs + i * total_blob_size_per_image_;
            const float* im_start = tensor_input_nchw + i * target_height_ * target_width_ * 3;
            int* output_start0 = output0 + i * target_height_ * target_width_ * 3;
            int* output_start1 = output1 + i * target_height_ * target_width_ * 3;
            int* output_start2 = output2 + i * target_height_ * target_width_ * 3;
            int* output_for_metric_start = output_for_metric + i * target_height_ * target_width_;
            decoder_->process(start);
            decoder_->getOutput(output_start0, output_start1, output_start2,
                                output_for_metric_start, im_start, target_height_, target_width_,
                                arrow_length_, 0, background_class_id_);
        }
    }
};

#pragma message("Registering CPU kernel")
REGISTER_KERNEL_BUILDER(
    Name("DecodeDist").Device(DEVICE_CPU).HostMemory("encoded_blobs").HostMemory("images"),
    DecodeDist);
