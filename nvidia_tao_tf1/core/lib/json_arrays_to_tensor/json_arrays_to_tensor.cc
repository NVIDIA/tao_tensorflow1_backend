// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

#include "json.hpp"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"

using json = nlohmann::json;

namespace tensorflow {

REGISTER_OP("JsonArraysToTensor")
    .Input("value: string")
    .Attr("dtype: {int32, int64, float, string}")
    .Output("indices: int64")
    .Output("values: dtype")
    .Output("dense_shape: int64")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
        std::vector<tensorflow::shape_inference::DimensionHandle> vector_1d(1, c->UnknownDim());
        std::vector<tensorflow::shape_inference::DimensionHandle> vector_2d(2, c->UnknownDim());
        c->set_output(0, c->MakeShape(vector_2d));
        c->set_output(1, c->MakeShape(vector_1d));
        c->set_output(2, c->MakeShape(vector_1d));
        return Status::OK();
    })
    .Doc(R"doc(
    Convert a json-encoded values (that may be nested in lists) to a sparse tensor.

    Arguments:
        vertices (tf.string): A valid json-encoded string that contains values of strictly one
            datatype (corresponding to ``dtype`` argument). These values may be nested inside
            lists.
        dtype (tf.dtype): Supported datatype (tf.int32, tf.int64, tf.float32, tf.string), the
            output values will be in this ``dtype``.

    Returns:
        A ``tf.SparseTensor`` containing (by definition) the ``indices`` (``tf.int64``),
            ``values`` (``dtype``) and ``dense_shape`` (``tf.int64``) of the decoded json. 

    )doc");

template <typename T>
class JsonArraysToTensorOp : public OpKernel {
 private:
    int dims_;

    void recurse_array_depth(const json& js, const int depth, std::vector<int64>* dense_shape,
                             int* nvalues) {
        if (js.is_array()) {
            if (depth >= static_cast<int64>(dense_shape->size())) {
                dense_shape->push_back(0);
            }
            int64 len = static_cast<int64>(js.size());
            (*dense_shape)[depth] = std::max((*dense_shape)[depth], len);
            for (int64 i = 0; i < len; i++) {
                recurse_array_depth(js[i], depth + 1, dense_shape, nvalues);
            }
        } else {
            (*nvalues)++;
        }
    }

    void recurse_to_values(const json& js, int64 depth, std::vector<int64>* current_nd_index,
                           auto indices, auto values, int* pindex) {
        if (js.is_array()) {
            for (int64 i = 0; i < static_cast<int64>(js.size()); i++) {
                (*current_nd_index)[depth] = i;
                recurse_to_values(js[i], depth + 1, current_nd_index, indices, values, pindex);
            }
        } else {
            for (size_t d = 0; d < (*current_nd_index).size(); d++) {
                indices(*pindex, d) = (*current_nd_index)[d];
            }
            values(*pindex) = js;
            (*pindex)++;
        }
    }

 public:
    explicit JsonArraysToTensorOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
        const Tensor* input_tensor;
        OP_REQUIRES_OK(ctx, ctx->input("value", &input_tensor));
        OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(input_tensor->shape()),
                    errors::InvalidArgument("input string must be a scalar, got shape: ",
                                            input_tensor->shape().DebugString()));
        const auto input_str = input_tensor->flat<string>();
        std::string encoded_json = input_str.data()[0];

        // Make sure the json parsing is succesful, and bubble up a TensorFlow error otherwise.
        json js;
        try {
            js = json::parse(encoded_json);
        } catch (const std::exception& e) {  // Gotta catch 'em all!
            OP_REQUIRES(ctx, false, errors::InvalidArgument(e.what()));
        }

        std::vector<int64> dense_shape;

        int nvalues = 0;
        recurse_array_depth(js, 0, &dense_shape, &nvalues);
        const int ndims = static_cast<int>(dense_shape.size());

        // Now the shapes are known, we can allocate the output tensors.
        Tensor* output_tensor_indices = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("indices", TensorShape({nvalues, ndims}),
                                                 &output_tensor_indices));
        auto output_indices = output_tensor_indices->matrix<int64>();

        Tensor* output_tensor_values = nullptr;
        OP_REQUIRES_OK(
            ctx, ctx->allocate_output("values", TensorShape({nvalues}), &output_tensor_values));
        auto output_values = output_tensor_values->flat<T>();

        Tensor* output_tensor_dense_shape = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("dense_shape", TensorShape({ndims}),
                                                 &output_tensor_dense_shape));
        auto output_dense_shape = output_tensor_dense_shape->flat<int64>();

        // The current index vector keeps track in which dimension and element we are.
        std::vector<int64> current_nd_index(ndims);
        // Recurse through each array and copy data into the output tensors.
        int index = 0;
        try {
            recurse_to_values(js, 0, &current_nd_index, output_indices, output_values, &index);
        } catch (const nlohmann::detail::type_error& e) {
            OP_REQUIRES(ctx, false, errors::InvalidArgument(e.what()));
        }

        for (int d = 0; d < ndims; d++) {
            output_dense_shape(d) = dense_shape[d];
        }
    }
};

#define REGISTER_KERNEL(type)                                                        \
    REGISTER_KERNEL_BUILDER(                                                         \
        Name("JsonArraysToTensor").Device(DEVICE_CPU).TypeConstraint<type>("dtype"), \
        JsonArraysToTensorOp<type>)

REGISTER_KERNEL(int32);
REGISTER_KERNEL(int64);
REGISTER_KERNEL(float);
REGISTER_KERNEL(string);

}  // namespace tensorflow
