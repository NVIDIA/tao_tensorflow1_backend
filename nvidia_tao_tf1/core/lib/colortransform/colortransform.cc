// Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.

#undef EIGEN_USE_GPU

#include <vector>

// Include common code
#include "colortransform.h"

// Register CPU op
REGISTER_OP("Colortransform")
    .Input("input_images: input_dtype")
    .Input("input_transf_mats: float")
    .Output("output_images: output_dtype")
    .Attr("input_dtype: {uint8, half, float}")
    .Attr("output_dtype: {uint8, half, float}")
    .Attr("min_clip: float = 0")
    .Attr("max_clip: float = 1")
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

        dims_out.push_back(height);
        dims_out.push_back(width);

        if (output_data_format == FORMAT_NHWC) dims_out.push_back(num_channels);

        c->set_output(0, c->MakeShape(dims_out));
        return Status::OK();
    })
    .Doc(R"doc(
           Color transformation op.

           Transforms a batch of input images with per image 4x4 color matrix. This operation
           supports specifying input and output data formats separately, thus it can be used
           for NCHW<->NHWC conversion. It also supports type casting by allowing output dtype
           to be specified explicitly.

           Args:
               input_images: 4D Tensor (NHWC or NCHW). C = [RGB]. Supported dtypes are float32,
                    float16, and uint8.
               input_transf_mats: 3D tensor (N, 4, 4).
               min_clip (float): Minimum color value after transformation. 
               max_clip (float): Maximum color value after transformation.
               input_data_format (string): Either 'NCHW' (default) or 'NHWC'.
               output_data_format (string): Either 'NCHW' (default) or 'NHWC'.
               output_dtype (dtype): Output image dtype (float32, float16, or uint8).

           Returns:
               output_images: 4D Tensor (NHWC or NCHW).
           )doc");

template <typename Device, typename I, typename O>
class ColorTransformOp : public BaseColorTransformOp {
 public:
    explicit ColorTransformOp(OpKernelConstruction* context) : BaseColorTransformOp(context) {}

    void ComputeArch(OpKernelContext* context, Tensor* output_tensor,
                     const Tensor& input_images_tensor, const float* input_transf_mats, int nbatch,
                     int height, int width, bool input_channels_first,
                     bool output_channels_first) override {
        if (verbose_) printf("running CPU version\n");

        auto output_images = output_tensor->flat<O>().data();
        auto input_images = input_images_tensor.flat<I>().data();

        for (int b = 0; b < nbatch; b++) {
            const int num_channels = 3;
            const I* in = input_images + b * height * width * num_channels;
            const float* mat = input_transf_mats + b * 4 * 4;
            O* out = output_images + b * height * width * num_channels;

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    _ColorTransformKernel<I, O>(x, y, in, mat, out, min_clip_, max_clip_, height,
                                                width, input_channels_first, output_channels_first);
                }
            }
        }
    }
};

#pragma message("Registering CPU kernel")
#define REGISTER_KERNEL(I, O)                                      \
    REGISTER_KERNEL_BUILDER(Name("Colortransform")                 \
                                .Device(DEVICE_CPU)                \
                                .TypeConstraint<I>("input_dtype")  \
                                .TypeConstraint<O>("output_dtype") \
                                .HostMemory("input_transf_mats"),  \
                            ColorTransformOp<CPUDevice, I, O>);

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
