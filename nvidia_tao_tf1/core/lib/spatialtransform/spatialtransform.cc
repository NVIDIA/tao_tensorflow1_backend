// Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.
#undef EIGEN_USE_GPU

#include <vector>

// Include common code
#include "spatialtransform.h"

// The code will be compiled for GPU and CPU, but we should register only once (CPU)
REGISTER_OP("SpatialTransform")
    .Input("images: input_dtype")
    .Input("transformation_matrices: float")
    .Input("shape: int32")
    .Output("output_images: output_dtype")
    .Attr("use_input_image_shape: bool")
    .Attr("input_dtype: {uint8, half, float}")
    .Attr("output_dtype: {uint8, half, float}")
    .Attr("filter_mode: {'nearest', 'bilinear', 'bicubic'} = 'bilinear'")
    .Attr("background_value: float = 0.0")
    .Attr("input_data_format: {'NHWC', 'NCHW'} = 'NCHW'")
    .Attr("output_data_format: {'NHWC', 'NCHW'} = 'NCHW'")
    .Attr("verbose: bool = false")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        std::string input_data_format_str;
        std::string output_data_format_str;
        TF_RETURN_IF_ERROR(c->GetAttr("input_data_format", &input_data_format_str));
        TF_RETURN_IF_ERROR(c->GetAttr("output_data_format", &output_data_format_str));
        TensorFormat input_data_format;
        TensorFormat output_data_format;
        FormatFromString(input_data_format_str, &input_data_format);
        FormatFromString(output_data_format_str, &output_data_format);

        std::vector<::shape_inference::DimensionHandle> dims_out;

        const int dims = 4;
        tensorflow::shape_inference::ShapeHandle cur;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), dims, &cur));
        if (!c->RankKnown(cur)) {
            for (int i = 0; i < dims; i++) dims_out.push_back(c->UnknownDim());
            c->set_output(0, c->MakeShape(dims_out));
            return Status::OK();
        }

        auto batch_size = c->Dim(cur, 0);
        dims_out.push_back(batch_size);

        int channels_dim = 3;
        int height_dim = 1;
        if (input_data_format == FORMAT_NCHW) {
            channels_dim = 1;
            height_dim = 2;
        }
        auto num_channels = c->Dim(cur, channels_dim);
        auto height = c->Dim(cur, height_dim);
        auto width = c->Dim(cur, height_dim + 1);

        if (output_data_format == FORMAT_NCHW) dims_out.push_back(num_channels);

        // Infer output width and height.
        bool use_input_image_shape;
        TF_RETURN_IF_ERROR(c->GetAttr("use_input_image_shape", &use_input_image_shape));
        if (use_input_image_shape) {
            // Infer size from the input tensor.
            dims_out.push_back(height);
            dims_out.push_back(width);
        } else {
            // Infer size from the shape tensor.
            const Tensor* shape_tensor = c->input_tensor(2);
            if (shape_tensor != nullptr) {
                // Shape tensor is known.
                if (shape_tensor->dims() != 1) {
                    return errors::InvalidArgument(
                        strings::StrCat("shape tensor must have 1 dimensions, shape is: ",
                                        shape_tensor->shape().DebugString(), "."));
                }
                if (shape_tensor->dim_size(0) != 2) {
                    return errors::InvalidArgument(
                        strings::StrCat("shape tensor must have 2 elements, shape is: ",
                                        shape_tensor->shape().DebugString(), "."));
                }
                auto height_and_width = shape_tensor->flat<int32>();
                dims_out.push_back(c->MakeDim(height_and_width(0)));
                dims_out.push_back(c->MakeDim(height_and_width(1)));
            } else {
                // Shape tensor is unknown.
                dims_out.push_back(c->UnknownDim());
                dims_out.push_back(c->UnknownDim());
            }
        }

        if (output_data_format == FORMAT_NHWC) dims_out.push_back(num_channels);

        c->set_output(0, c->MakeShape(dims_out));
        return Status::OK();
    })
    .Doc(R"doc(
           Spatial transformation op.

           Transforms a batch of input images with per image 3x3 image warp matrix. This
           operation supports specifying input and output data formats separately, thus it can be
           used for NCHW<->NHWC conversion. It also supports type casting by allowing output dtype
           to be specified explicitly.

           Args:
               input_images: 4D Tensor (NHWC or NCHW). Supported dtypes are float32,
                    float16, and uint8.
               transformation_matrices: 3D tensor (N, 3, 3).
               shape: 1D int32 Tensor with 2 elements: (H, W) containing the desired output canvas
                   shape.
               use_input_image_shape (bool): If True, infer output width and height from input
                   image. If False, infer from shape tensor.
               filter_mode (string): 'nearest', 'bilinear' (default), or 'bicubic'.
               background_value (float): The value to use when output pixel is not covered by
                   an input image. Defaults to 0.
               input_data_format (string): Either 'NCHW' (default) or 'NHWC'.
               output_data_format (string): Either 'NCHW' (default) or 'NHWC'.
               output_dtype (dtype): Output image dtype (float32, float16, or uint8).

           Returns:
               output_images: 4D Tensor (NHWC or NCHW).
           )doc");

template <typename Device, typename I, typename O>
class SpatialTransformOp : public BaseSpatialTransformOp {
 public:
    explicit SpatialTransformOp(OpKernelConstruction* context) : BaseSpatialTransformOp(context) {}

    void ComputeArch(OpKernelContext* context, Tensor* output_tensor,
                     const Tensor& input_images_tensor, const float* transformation_matrices,
                     int nbatch, int num_channels, int height, int width, int output_height,
                     int output_width, bool input_channels_first,
                     bool output_channels_first) override {
        if (verbose_) printf("running CPU version\n");

        auto output_images = output_tensor->flat<O>().data();
        auto input_images = input_images_tensor.flat<I>().data();

        for (int b = 0; b < nbatch; b++) {
            const I* in = input_images + b * height * width * num_channels;
            const float* mat = transformation_matrices + b * 3 * 3;
            O* out = output_images + b * output_height * output_width * num_channels;
            for (int y = 0; y < output_height; y++) {
                for (int x = 0; x < output_width; x++) {
                    _SpatialTransformKernel<I, O>(x, y, in, mat, out, num_channels, height, width,
                                                  output_height, output_width, filter_mode_,
                                                  background_, input_channels_first,
                                                  output_channels_first);
                }
            }
        }
    }
};

#pragma message("Registering CPU kernel")
#define REGISTER_KERNEL(I, O)                                          \
    REGISTER_KERNEL_BUILDER(Name("SpatialTransform")                   \
                                .Device(DEVICE_CPU)                    \
                                .TypeConstraint<I>("input_dtype")      \
                                .TypeConstraint<O>("output_dtype")     \
                                .HostMemory("images")                  \
                                .HostMemory("transformation_matrices") \
                                .HostMemory("shape"),                  \
                            SpatialTransformOp<CPUDevice, I, O>);

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
