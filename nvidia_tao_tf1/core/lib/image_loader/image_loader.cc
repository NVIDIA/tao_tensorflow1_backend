// Copyright (c) 2018~2019, NVIDIA CORPORATION.  All rights reserved.

#include <stdio.h>
#include <stdlib.h>

#include "image_loader_function.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;

/*
 * The image loader_function_callback defines a template that loads a single image from
 * a given path for an image format.
 *
 * To support a new image type, The image loader callback function must have the following
 * signature:
 *
 * @param path: the image path.
 * @param buffer: the buffer to keep the loaded image, usually the returned tensor
 * (tensor->flat<>().data()).
 * @param length: the length of the image.
 * @param verbose: verbose logging mode.
 *
 * void(const std::string& path, char* buffer, const size_t length, bool verbose)
 */

typedef std::function<void(const std::string&, char*, const size_t, bool)>
    image_loader_function_callback;

REGISTER_OP("ImageLoader")
    .Input("path: string")
    .Input("image_shape: int32")
    .Input("image_format: string")
    .Attr("dtype: {float, float16, uint8}")
    .Attr("padded_shape: shape")
    .Attr("verbose: bool")
    .Output("output_image: dtype")
    .SetShapeFn([](::shape_inference::InferenceContext* c) {
        ::shape_inference::ShapeHandle out;
        PartialTensorShape partial_padded_shape;
        TF_RETURN_IF_ERROR(c->GetAttr("padded_shape", &partial_padded_shape));
        TensorShape padded_shape;
        bool ok = partial_padded_shape.AsTensorShape(&padded_shape);
        if (!ok) {
            return Status(
                ::tensorflow::error::INVALID_ARGUMENT,
                "padded_shape not fully defined, was:" + partial_padded_shape.DebugString());
        }
        TF_RETURN_IF_ERROR(c->MakeShapeFromTensorShape(padded_shape, &out));
        c->set_output(0, out);
        return Status::OK();
    })
    .Doc(R"doc(
    Loads fp16/jpeg images and pads them on right and bottom, if padded_shape is larger than
    the image_shape.

    path: Full pathname to the image. 
    image_format: `fp16` or `jpeg`.
    image_shape: Tensor of image shape as loaded from disk (C, H, W).
    padded_shape: Tensor of output image, with optional padding (C, H, W).
    dtype: Output type, either float16 or float32.
    verbose: Whether to print the debug info.

    output_image: Image (C, H, W) with dimensions as defined in padded_shape dimensions.
    )doc");

class ImageLoaderOp : public OpKernel {
 private:
    bool verbose_;
    DataType dtype_;
    TensorShape output_shape_;

 public:
    explicit ImageLoaderOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
        OP_REQUIRES_OK(context, context->GetAttr("verbose", &verbose_));

        PartialTensorShape partial_padded_shape;
        OP_REQUIRES_OK(context, context->GetAttr("padded_shape", &partial_padded_shape));
        OP_REQUIRES(context, partial_padded_shape.dims() == 3,
                    errors::InvalidArgument("padded_shape must have 3 dimensions (C, H, W)"));
        OP_REQUIRES(context, partial_padded_shape.AsTensorShape(&output_shape_),
                    errors::InvalidArgument("padded_shape must be fully defined"));
    }

    // copy data line by line and convert data to desired format, half -> float.
    void _assign_row(float* row_dest, Eigen::half* input_ptr, int64 width) {
        auto data_cast = [](Eigen::half& value) -> float {
            return Eigen::half_impl::half_to_float(value);
        };
        std::transform(input_ptr, input_ptr + width, row_dest, data_cast);
    }

    // copy data line by line and convert data to desired format, half -> half
    void _assign_row(Eigen::half* row_dest, Eigen::half* input_ptr, int64 width) {
        std::memcpy(row_dest, input_ptr, sizeof(Eigen::half) * width);
    }

    // copy data line by line and convert data to desired format, uint8 -> uint8
    void _assign_row(uint8* row_dest, uint8* input_ptr, int64 width) {
        std::memcpy(row_dest, input_ptr, sizeof(uint8) * width);
    }

    // copy data line by line and convert data to desired format, uint8 -> float
    void _assign_row(float* row_dest, uint8* input_ptr, int64 width) {
        auto data_cast = [](uint8& value) -> float { return static_cast<float>(value); };
        std::transform(input_ptr, input_ptr + width, row_dest, data_cast);
    }

    // copy data line by line and convert data to desired format, uint8 -> half
    void _assign_row(Eigen::half* row_dest, uint8* input_ptr, int64 width) {
        auto data_cast = [](uint8& value) -> Eigen::half {
            // uint8 -> float -> half
            return Eigen::half(value);
        };
        std::transform(input_ptr, input_ptr + width, row_dest, data_cast);
    }
    /**
     * @brief Load image on a given path.
     * @param context: tensorflow OpKernelContext.
     * @param image_loader_function: callback function to load image into a given buffer.
     *
     * IN_T is the data type of image (e.g., Eigen::half for fp16, uint8_t for jpeg).
     * OUT_T is the data type of the returned tensor you want to convert to.
     *
     */
    template <typename IN_T, typename OUT_T>
    void _Compute(OpKernelContext* context, image_loader_function_callback image_loader_function) {
        const Tensor* path_tensor;
        OP_REQUIRES_OK(context, context->input("path", &path_tensor));
        OP_REQUIRES(context, TensorShapeUtils::IsScalar(path_tensor->shape()),
                    errors::InvalidArgument("Input 'path' tensor must be scalar, but had shape: ",
                                            path_tensor->shape().DebugString()));
        const std::string& path = path_tensor->flat<std::string>().data()[0];

        const Tensor* image_shape;
        OP_REQUIRES_OK(context, context->input("image_shape", &image_shape));
        OP_REQUIRES(context, image_shape->dim_size(0) == output_shape_.dims(),
                    errors::InvalidArgument(
                        "image shape and padded shape must have same cardinality: ",
                        image_shape->DebugString(), " vs. ", output_shape_.DebugString()));

        const int64 ndims = image_shape->dim_size(0);

        TensorShape input_shape;

        for (int i = 0; i < ndims; ++i) {
            int32 idim = image_shape->flat<int32>().data()[i];
            input_shape.AddDim(idim);
            OP_REQUIRES(context, idim > 0,
                        errors::InvalidArgument("Input shape dimension ", std::to_string(i),
                                                " was invalid:", idim));
        }

        size_t total_input_size = input_shape.num_elements();
        size_t expected_size = sizeof(IN_T) * total_input_size;
        // Figure out padding dimensions, do sanity checks.
        int64 output_width = output_shape_.dim_size(2);
        int64 input_width = input_shape.dim_size(2);
        int64 output_height = output_shape_.dim_size(1);
        int64 input_height = input_shape.dim_size(1);
        int64 padding_bottom = output_height - input_height;
        int64 padding_right = output_width - input_width;
        int64 num_channels = output_shape_.dim_size(0);

        OP_REQUIRES(context, input_width <= output_width,
                    errors::InvalidArgument("Input width must be <= output width"));
        OP_REQUIRES(context, input_height <= output_height,
                    errors::InvalidArgument("Input height must be <= output height"));
        OP_REQUIRES(
            context, output_shape_.dim_size(0) == input_shape.dim_size(0),
            errors::InvalidArgument("Input and output shape channels (dim 0) must be equal"));
        OP_REQUIRES(context, output_shape_.num_elements() * sizeof(OUT_T) >= expected_size,
                    errors::InvalidArgument("Output size must be >= input size"));

        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape_, &output_tensor));
        OUT_T* output_data = output_tensor->flat<OUT_T>().data();

        // read file into the end of the buffer, so we can move values around
        char* dst = reinterpret_cast<char*>(output_data) +
                    output_shape_.num_elements() * sizeof(OUT_T) - expected_size;

        /* Load file */
        image_loader_function(path, dst, expected_size, verbose_);

        if (padding_right == 0 && padding_bottom == 0 && sizeof(IN_T) == sizeof(OUT_T)) {
            // nothing to do: already have right size and type.
            return;
        }

        // Move and convert data into the padded image if needed.
        IN_T* input_ptr = reinterpret_cast<IN_T*>(dst);
        for (int64 c = 0; c < num_channels; ++c) {
            OUT_T* channel_dest = output_data + c * output_height * output_width;
            for (int64 y = 0; y < input_height; ++y) {
                OUT_T* row_dest = channel_dest + y * output_width;
                _assign_row(row_dest, input_ptr, input_width);
                input_ptr += input_width;
            }
        }

        // Zero out the padded areas. Must be done after all data is copied to avoid corrupting.
        for (int64 c = 0; c < num_channels; ++c) {
            OUT_T* channel_dest = output_data + c * output_height * output_width;

            // right pad
            for (int64 y = 0; y < input_height; ++y) {
                OUT_T* pad_dest = channel_dest + y * output_width + input_width;
                memset(pad_dest, 0, padding_right * sizeof(OUT_T));
            }

            // pad bottom
            for (int64 y = input_height; y < output_height; ++y) {
                OUT_T* pad_dest = channel_dest + y * output_width;
                memset(pad_dest, 0, output_width * sizeof(OUT_T));
            }
        }
    }

    enum class ImageFormat { FP16, JPEG, UNKNOWN_IMAGE_FORMAT };

    ImageFormat parse_image_format(const std::string& image_format) {
        // The image format is either the name of the image format or the extension of the image.
        if (image_format == "jpeg" || image_format == "jpg" || image_format == ".jpeg" ||
            image_format == ".jpg") {
            return ImageFormat::JPEG;
        } else if (image_format == "fp16" || image_format == ".fp16") {
            return ImageFormat::FP16;
        } else {
            return ImageFormat::UNKNOWN_IMAGE_FORMAT;
        }
    }

    void Compute(OpKernelContext* context) override {
        const Tensor* image_format_tensor;
        OP_REQUIRES_OK(context, context->input("image_format", &image_format_tensor));

        ImageFormat image_format =
            parse_image_format(image_format_tensor->flat<std::string>().data()[0]);

        OP_REQUIRES(context, image_format != ImageFormat::UNKNOWN_IMAGE_FORMAT,
                    errors::InvalidArgument("Unsupported image format"));

        if (image_format == ImageFormat::JPEG) {
            OP_REQUIRES(context, dtype_ == DT_UINT8 || dtype_ == DT_HALF || dtype_ == DT_FLOAT,
                        errors::InvalidArgument(
                            "Supported output format for jpeg are uint8, fp16 and fp32"));

            if (dtype_ == DT_UINT8) {
                _Compute<uint8, uint8>(context, jpeg_image_loader);

            } else if (dtype_ == DT_HALF) {
                _Compute<uint8, Eigen::half>(context, jpeg_image_loader);

            } else if (dtype_ == DT_FLOAT) {
                _Compute<uint8, float>(context, jpeg_image_loader);
            }

        } else if (image_format == ImageFormat::FP16) {
            OP_REQUIRES(
                context, dtype_ == DT_FLOAT || dtype_ == DT_HALF,
                errors::InvalidArgument("Supported output format for raw image are fp16 and fp32"));

            if (dtype_ == DT_FLOAT) {
                _Compute<Eigen::half, float>(context, fp16_image_loader);

            } else if (dtype_ == DT_HALF) {
                _Compute<Eigen::half, Eigen::half>(context, fp16_image_loader);
            }
        }
    }
};

#pragma message("Registering CPU kernel")
REGISTER_KERNEL_BUILDER(Name("ImageLoader").Device(DEVICE_CPU).TypeConstraint<float>("dtype"),
                        ImageLoaderOp);
REGISTER_KERNEL_BUILDER(Name("ImageLoader").Device(DEVICE_CPU).TypeConstraint<Eigen::half>("dtype"),
                        ImageLoaderOp);
REGISTER_KERNEL_BUILDER(Name("ImageLoader").Device(DEVICE_CPU).TypeConstraint<uint8>("dtype"),
                        ImageLoaderOp);
