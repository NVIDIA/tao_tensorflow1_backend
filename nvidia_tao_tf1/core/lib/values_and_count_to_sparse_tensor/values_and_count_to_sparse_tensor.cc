// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

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

REGISTER_OP("ValuesAndCountToSparseTensor")
    .Input("values: dtype")
    .Input("counts: int32")
    .Input("counts_of_counts: int32")
    .Attr("dtype: {int32, int64, float, double, string}")
    .Output("indices: int64")
    .Output("output_values: dtype")
    .Output("dense_shape: int64")
    .SetShapeFn([](::shape_inference::InferenceContext* c) {
        // TODO(xiangbok): if any of the ranks are unknown, set unknown dims everywhere.
        const int rank = c->Rank(c->input(0)) + c->Rank(c->input(1)) + c->Rank(c->input(2));

        // Indices.
        std::vector<::shape_inference::DimensionHandle> dim_indices;
        dim_indices.push_back(c->UnknownDim());
        dim_indices.push_back(c->MakeDim(rank));
        c->set_output(0, c->MakeShape(dim_indices));

        // Dense_shape.
        c->set_output(2, c->Vector(rank));

        // Values is always a 1D array.
        // TODO(xiangbok): The length is the sum of the input values shape, but we can only
        // infer this if it's fully defined.
        c->set_output(1, c->Vector(c->UnknownDim()));
        return Status::OK();
    })
    .Doc(R"doc(
    Converts values and its counts into a sparse tensor.

    Args:
        values: A (dtype) tensor containing a long list of values of all the counts. The length of
            this list is therefore equal to the total number of counts that will be put into the
            sparse tensor.
        counts: A (tf.int32) 1D tensor. The elements of the list are the value counts that will be
            put into the sparse tensor. The sum of all the values in this list should equal the
            length of the ``values`` list above.
        counts_of_counts: an optional (tf.int32) 1D tensor. The elements of the list are the counts
            for the above counts that that will be put into the sparse tensor. The sum of all the
            values in this list should equal the length of the ``counts`` list above. To not
            specify this parameter, pass in an empty tensor of rank 0. If this parameter is not
            specified, then the output will be of a rank that is lower by 1

    Returns:
        A ``tf.SparseTensor`` containing (by definition) the ``indices`` (``tf.int64``),
            ``output_values`` (``dtype``) and ``dense_shape`` (``tf.int64``) of the values.
    )doc");

template <typename T>
class ValuesAndCountToSparseTensorOp : public OpKernel {
 public:
    explicit ValuesAndCountToSparseTensorOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& values_tensor = context->input(0);
        const Tensor& counts_tensor = context->input(1);
        const Tensor& counts_of_counts_tensor = context->input(2);

        OP_REQUIRES(context, 1 == counts_tensor.shape().dims(),
                    errors::InvalidArgument("counts must be a 1 dimensional vector,", " shape is: ",
                                            counts_tensor.shape().DebugString(), "."));

        OP_REQUIRES(
            context, 1 >= counts_of_counts_tensor.shape().dims(),
            errors::InvalidArgument("counts_of_counts_tensor must have rank <= 1,", " shape is: ",
                                    counts_of_counts_tensor.shape().DebugString(), "."));

        // Get the dimensions and sets properties that are necessary for computing the sparse
        const int n_values = values_tensor.dim_size(0);
        const int n_values_total = values_tensor.NumElements();
        const int n_counts = counts_tensor.dim_size(0);
        const bool has_counts_of_counts = counts_of_counts_tensor.shape().dims() != 0;
        const int n_counts_of_counts =
            has_counts_of_counts ? counts_of_counts_tensor.dim_size(0) : 0;
        const int rank = values_tensor.shape().dims() + counts_tensor.shape().dims() +
                         counts_of_counts_tensor.shape().dims();
        const auto input_values_per_count = counts_tensor.flat<int>();
        const auto input_counts_of_counts = counts_of_counts_tensor.flat<int>();

        // the dimensional product is neccessary to know how many elements are in a single
        // value, use 0 if there are no values
        int value_dim_product = n_values > 0 ? (n_values_total / n_values) : 0;

        // Validate that the amount of counts adds up to n_values
        int count_sum = 0;
        for (int i = 0; i < n_counts; i++) {
            count_sum += input_values_per_count.data()[i];
        }

        OP_REQUIRES(context, count_sum == n_values,
                    errors::InvalidArgument("Sum of counts_tensor ", count_sum,
                                            " over all values does not add up to n_values ",
                                            n_values, "."));

        if (has_counts_of_counts) {
            // Validate that the amount of counts of counts adds up to n_counts only if there
            // are counts_of_counts.
            int count_of_counts_sum = 0;
            for (int i = 0; i < n_counts_of_counts; i++) {
                count_of_counts_sum += input_counts_of_counts.data()[i];
            }

            OP_REQUIRES(
                context, count_of_counts_sum == n_counts,
                errors::InvalidArgument("Sum of counts_of_counts_tensor ", count_of_counts_sum,
                                        " over all counts_of_counts does not add up to n_counts ",
                                        n_counts, "."));
        }

        // Create the indices output
        Tensor* output_indices_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({n_values_total, rank}),
                                                         &output_indices_tensor));
        auto output_indices = output_indices_tensor->template flat<int64>();

        // The strategy here is to write this in a strip way. This writes out all the indices for
        // the sparse tensor one by one starting with the count of counts (if necessary), then
        // counts, then every index for the values.

        // used to keep track where the first element of the strip is for strip write
        int first_write_in_output = 0;

        // write out the counts of counts
        if (has_counts_of_counts) {
            int coc_to_write_index = first_write_in_output;
            int count_index_tracker = 0;

            for (int coc_index = 0; coc_index < n_counts_of_counts; coc_index++) {
                // first gather how many values belongs to this counts of counts index
                // which is affected by the counts tensor
                int coc_value = input_counts_of_counts.data()[coc_index];
                int numbers_to_write = 0;

                for (int i = 0; i < coc_value; i++) {
                    numbers_to_write += input_values_per_count.data()[count_index_tracker];
                    count_index_tracker++;
                }
                numbers_to_write *= value_dim_product;

                // next, write it to the output tensor
                for (int i = 0; i < numbers_to_write; i++) {
                    output_indices(coc_to_write_index) = coc_index;
                    coc_to_write_index += rank;
                }
            }
            first_write_in_output++;
        }

        // write out the counts
        int counts_to_write_index = first_write_in_output;
        if (has_counts_of_counts) {
            int count_index_tracker = 0;

            // find how many counts are in a count of count
            for (int coc_index = 0; coc_index < n_counts_of_counts; coc_index++) {
                int coc_value = input_counts_of_counts.data()[coc_index];

                // iterate through that many counts in a counts of counts
                for (int count_index_to_coc = 0; count_index_to_coc < coc_value;
                     count_index_to_coc++) {
                    int count_value = input_values_per_count.data()[count_index_tracker];

                    // write it out to the output tensor for that specific count
                    for (int i = 0; i < count_value * value_dim_product; i++) {
                        output_indices(counts_to_write_index) = count_index_to_coc;
                        counts_to_write_index += rank;
                    }
                    count_index_tracker++;
                }
            }
        } else {
            // if there are no counts of counts, then the logic is influenced differently
            // This means that only counts are necessary
            for (int counts_index = 0; counts_index < n_counts; counts_index++) {
                int counts_value = input_values_per_count.data()[counts_index];
                for (int i = 0; i < counts_value * value_dim_product; i++) {
                    output_indices(counts_to_write_index) = counts_index;
                    counts_to_write_index += rank;
                }
            }
        }
        first_write_in_output++;

        // write out the values[0] indices for sparse
        int values_0_to_write_index = first_write_in_output;

        // First gather how many values are in a count
        for (int counts_index = 0; counts_index < n_counts; counts_index++) {
            int counts_value = input_values_per_count.data()[counts_index];

            // next iterate through that many values and write it to output
            for (int values_0_to_counts = 0; values_0_to_counts < counts_value;
                 values_0_to_counts++) {
                for (int i = 0; i < value_dim_product; i++) {
                    output_indices(values_0_to_write_index) = values_0_to_counts;
                    values_0_to_write_index += rank;
                }
            }
        }
        first_write_in_output++;

        // write out the values[1:] indices for sparse
        int value_dim_curr_to_last = value_dim_product;
        for (int values_rank = 1; values_rank < values_tensor.shape().dims(); values_rank++) {
            int value_curr_to_write_index = first_write_in_output;
            int values_dim = values_tensor.dim_size(values_rank);
            value_dim_curr_to_last = value_dim_curr_to_last / values_dim;

            // repeat until end of output tensor
            while (value_curr_to_write_index < n_values_total * rank) {
                // write out the rest of the indices in a dense manner
                for (int i = 0; i < values_dim; i++) {
                    for (int j = 0; j < value_dim_curr_to_last; j++) {
                        output_indices(value_curr_to_write_index) = i;
                        value_curr_to_write_index += rank;
                    }
                }
            }
            first_write_in_output++;
        }

        // Creates the values output tensor
        Tensor* output_values_tensor = nullptr;
        OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                    {0}, 1, TensorShape({n_values_total}), &output_values_tensor));
        auto output_values = output_values_tensor->template flat<T>();
        const auto values_tensor_flatten = values_tensor.flat<T>();

        for (int i = 0; i < n_values_total; i++) {
            output_values(i) = values_tensor_flatten.data()[i];
        }

        // Creates the dense shape
        Tensor* output_dense_shape_tensor = nullptr;
        OP_REQUIRES_OK(
            context, context->allocate_output(2, TensorShape({rank}), &output_dense_shape_tensor));
        auto output_dense_shape = output_dense_shape_tensor->template flat<int64>();
        int max_counts_of_counts = 0;
        int max_values_in_count = 0;

        if (has_counts_of_counts) {
            for (int i = 0; i < n_counts_of_counts; i++) {
                if (max_counts_of_counts < input_counts_of_counts.data()[i]) {
                    max_counts_of_counts = input_counts_of_counts.data()[i];
                }
            }
        } else {
            max_counts_of_counts = n_counts;
        }

        for (int i = 0; i < n_counts; i++) {
            if (max_values_in_count < input_values_per_count.data()[i]) {
                max_values_in_count = input_values_per_count.data()[i];
            }
        }

        int output_dense_shape_tensor_idx = 0;
        if (has_counts_of_counts) {
            output_dense_shape(output_dense_shape_tensor_idx) = n_counts_of_counts;
            output_dense_shape_tensor_idx++;
        }
        output_dense_shape(output_dense_shape_tensor_idx) = max_counts_of_counts;
        output_dense_shape_tensor_idx++;
        output_dense_shape(output_dense_shape_tensor_idx) = max_values_in_count;
        output_dense_shape_tensor_idx++;
        int value_dims = values_tensor.shape().dims();
        for (int i = 1; i < value_dims; i++) {
            output_dense_shape(output_dense_shape_tensor_idx) = values_tensor.dim_size(i);
            output_dense_shape_tensor_idx++;
        }
    }
};

#define REGISTER_KERNEL(type)                                                                  \
    REGISTER_KERNEL_BUILDER(                                                                   \
        Name("ValuesAndCountToSparseTensor").Device(DEVICE_CPU).TypeConstraint<type>("dtype"), \
        ValuesAndCountToSparseTensorOp<type>)

REGISTER_KERNEL(int32);
REGISTER_KERNEL(int64);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
REGISTER_KERNEL(string);
