// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

#include <algorithm>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/lib/core/errors.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "insert_op_helpers.h"

using CPUDevice = Eigen::ThreadPoolDevice;

namespace tensorflow {

namespace {

// Computes the output vector when input tensor has rank 1.
template <typename T>
void ScatterCopy1(const Tensor& Tx,
                  const std::array<const std::vector<std::size_t>*, 5>& output_table,
                  Tensor* Toutput) {
    auto output = Toutput->vec<T>();
    auto x = Tx.vec<T>();

    for (std::size_t i = 0; i < static_cast<std::size_t>(x.size()); ++i) {
        output((*output_table[0])[i]) = x(i);
    }
}

// Computes the output vector when input tensor has rank 2.
template <typename T>
void ScatterCopy2(const Tensor& Tx,
                  const std::array<const std::vector<std::size_t>*, 5>& output_table,
                  Tensor* Toutput) {
    auto output = Toutput->tensor<T, 2>();
    auto x = Tx.tensor<T, 2>();

    auto sizes_raw = Tx.shape().dim_sizes();
    std::array<std::size_t, 2> sizes = {static_cast<std::size_t>(sizes_raw[0]),
                                        static_cast<std::size_t>(sizes_raw[1])};

    for (std::size_t i = 0; i < sizes[0]; ++i) {
        for (std::size_t j = 0; j < sizes[1]; ++j) {
            output((*output_table[0])[i], (*output_table[1])[j]) = x(i, j);
        }
    }
}

// Computes the output vector when input tensor has rank 3.
template <typename T>
void ScatterCopy3(const Tensor& Tx,
                  const std::array<const std::vector<std::size_t>*, 5>& output_table,
                  Tensor* Toutput) {
    auto output = Toutput->tensor<T, 3>();
    auto x = Tx.tensor<T, 3>();

    auto sizes_raw = Tx.shape().dim_sizes();
    std::array<std::size_t, 3> sizes = {static_cast<std::size_t>(sizes_raw[0]),
                                        static_cast<std::size_t>(sizes_raw[1]),
                                        static_cast<std::size_t>(sizes_raw[2])};

    for (std::size_t i = 0; i < sizes[0]; ++i) {
        for (std::size_t j = 0; j < sizes[1]; ++j) {
            for (std::size_t k = 0; k < sizes[2]; ++k) {
                output((*output_table[0])[i], (*output_table[1])[j], (*output_table[2])[k]) =
                    x(i, j, k);
            }
        }
    }
}

// Computes the output vector when input tensor has rank 4.
template <typename T>
void ScatterCopy4(const Tensor& Tx,
                  const std::array<const std::vector<std::size_t>*, 5>& output_table,
                  Tensor* Toutput) {
    auto output = Toutput->tensor<T, 4>();
    auto x = Tx.tensor<T, 4>();

    auto sizes_raw = Tx.shape().dim_sizes();
    std::array<std::size_t, 4> sizes = {
        static_cast<std::size_t>(sizes_raw[0]), static_cast<std::size_t>(sizes_raw[1]),
        static_cast<std::size_t>(sizes_raw[2]), static_cast<std::size_t>(sizes_raw[3])};

    for (std::size_t i = 0; i < sizes[0]; ++i) {
        for (std::size_t j = 0; j < sizes[1]; ++j) {
            for (std::size_t k = 0; k < sizes[2]; ++k) {
                for (std::size_t l = 0; l < sizes[3]; ++l) {
                    output((*output_table[0])[i], (*output_table[1])[j], (*output_table[2])[k],
                           (*output_table[3])[l]) = x(i, j, k, l);
                }
            }
        }
    }
}

// Computes the output vector when input tensor has rank 5.
template <typename T>
void ScatterCopy5(const Tensor& Tx,
                  const std::array<const std::vector<std::size_t>*, 5>& output_table,
                  Tensor* Toutput) {
    auto output = Toutput->tensor<T, 5>();
    auto x = Tx.tensor<T, 5>();

    auto sizes_raw = Tx.shape().dim_sizes();
    std::array<std::size_t, 5> sizes = {
        static_cast<std::size_t>(sizes_raw[0]), static_cast<std::size_t>(sizes_raw[1]),
        static_cast<std::size_t>(sizes_raw[2]), static_cast<std::size_t>(sizes_raw[3]),
        static_cast<std::size_t>(sizes_raw[4])};

    for (std::size_t i = 0; i < sizes[0]; ++i) {
        for (std::size_t j = 0; j < sizes[1]; ++j) {
            for (std::size_t k = 0; k < sizes[2]; ++k) {
                for (std::size_t l = 0; l < sizes[3]; ++l) {
                    for (std::size_t m = 0; m < sizes[4]; ++m) {
                        output((*output_table[0])[i], (*output_table[1])[j], (*output_table[2])[k],
                               (*output_table[3])[l], (*output_table[4])[m]) = x(i, j, k, l, m);
                    }
                }
            }
        }
    }
}

}  // anonymous namespace

// TODO(efagerholm): Add shape function.
REGISTER_OP("Insert")
    .Input("x: T")
    .Input("indices: Tindices")
    .Input("value: T")
    .Attr("T: type")
    .Attr("Tindices: {int32, int64} = DT_INT32")
    .Attr("axis: int >= 0 = 0")
    .Output("out: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        int32 axis;
        TF_RETURN_IF_ERROR(c->GetAttr("axis", &axis));
        shape_inference::ShapeHandle indices_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &indices_shape));

        shape_inference::ShapeHandle x_shape = c->input(0);
        if (!c->RankKnown(x_shape)) {
            c->set_output(0, c->UnknownShape());
            return Status::OK();
        }

        int32 rank = c->Rank(x_shape);
        std::vector<shape_inference::DimensionHandle> output_shape(rank, c->UnknownDim());

        for (int i = 0; i < rank; ++i) {
            if (i != axis) {
                output_shape[i] = c->Dim(x_shape, i);
            } else {
                c->Add(c->Dim(x_shape, axis), c->Dim(indices_shape, 0), &output_shape[axis]);
            }
        }
        c->set_output(0, c->MakeShape(output_shape));
        return Status::OK();
    })
    .Doc(R"doc(
    Op to insert slices of a given value along a given axis.

    Insert mimics the numpy.insert operation from NumPy. The mapping between the
    naming of the arguments in np.insert to this op is as follows:

        x = arr in np
        indices = obj in np
        value = values in np
        axis = axis in np

    A difference between np.insert and the current implementation of this op is
    that we require value to be a scalar, i.e. you can only insert the same
    constant.

    Example: Assume that x is a 3-by-3 matrix and we want to add a column of
    zeros after the first column (i.e. before the second), this is done with:

        insert(x, [1], 0, axis=1)

    Similarily if we want to add a row of zeros before the first row and after
    the last row, this would be done with:

        insert(x, [0, 3], 0, axis=0)

    Arguments:
        x: A tensor to add slices to.
        indices: A vector valued tensor specifying the indices before, which we
            should insert slices.
        value: A scalar tensor containing the values in the slice.

    Returns:
        A tensor with the same shape as the input x except that along axis, the
        size of the tensor has grown by the length of indices.
    )doc");

template <typename Device, typename T, typename Index>
class InsertOp : public OpKernel {
 public:
    explicit InsertOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& Tx = ctx->input(0);
        const Tensor& Tindices = ctx->input(1);
        const Tensor& Tvalue = ctx->input(2);

        //
        // Validate inputs and compute output shape.
        //
        OP_REQUIRES(ctx, TensorShapeUtils::IsVector(Tindices.shape()),
                    errors::InvalidArgument("indices must be a vector, got shape ",
                                            Tindices.shape().DebugString()));
        OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(Tvalue.shape()),
                    errors::InvalidArgument("value must be a scalar, got shape ",
                                            Tvalue.shape().DebugString()));
        OP_REQUIRES(ctx, axis_ < Tx.dims(),
                    errors::InvalidArgument("axis must be >= 0 and < rank(x)"));

        TensorShape output_shape{Tx.shape()};
        int64 output_axis_size = Tx.dim_size(axis_) + Tindices.dim_size(0);
        output_shape.set_dim(axis_, output_axis_size);

        //
        // Check that indices are within range.
        //
        auto indices_raw = Tindices.flat<Index>();
        for (decltype(indices_raw.size()) i = 0; i < indices_raw.size(); ++i) {
            OP_REQUIRES(
                ctx, indices_raw(i) >= 0 && indices_raw(i) <= Tx.dim_size(axis_),
                errors::InvalidArgument("indices must be >= 0 and < size of axis dimension"));
        }

        // Cast to size_t to avoid templating code on index type.
        std::vector<std::size_t> indices(indices_raw.data(),
                                         indices_raw.data() + indices_raw.size());

        //
        // Allocate output and prefill output with "value".
        //
        Tensor* Toutput = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &Toutput));

        T value = Tvalue.scalar<T>()();
        auto flat_output = Toutput->flat<T>();
        for (int64 i = 0; i < flat_output.size(); ++i) {
            flat_output(i) = value;
        }

        //
        // Compute scatter index tables.
        //
        auto sizes = Tx.shape().dim_sizes();
        auto max_size = *std::max_element(sizes.begin(), sizes.end());

        // On all axis except scatter axis we output to the same as the original index.
        std::vector<std::size_t> identity_map(max_size);
        std::iota(identity_map.begin(), identity_map.end(), 0);
        std::vector<std::size_t> scatter_indices = ComputeScatterIndices(indices, sizes[axis_]);

        // Compute output tables up to rank 5.
        std::array<const std::vector<std::size_t>*, 5> output_table;
        output_table.fill(&identity_map);
        output_table[axis_] = &scatter_indices;

        switch (Tx.dims()) {
            case 1: {
                ScatterCopy1<T>(Tx, output_table, Toutput);
                break;
            }
            case 2: {
                ScatterCopy2<T>(Tx, output_table, Toutput);
                break;
            }
            case 3: {
                ScatterCopy3<T>(Tx, output_table, Toutput);
                break;
            }
            case 4: {
                ScatterCopy4<T>(Tx, output_table, Toutput);
                break;
            }
            case 5: {
                ScatterCopy5<T>(Tx, output_table, Toutput);
                break;
            }
            default:
                OP_REQUIRES(ctx, false,
                            errors::InvalidArgument("Insert only supports input tensors"
                                                    " of rank <= 5, got tensor of rank ",
                                                    std::to_string(Tx.dims())));
        }
    }

 private:
    int32 axis_ = 0;
};

#define REGISTER_KERNEL(D, TYPE)                                   \
    REGISTER_KERNEL_BUILDER(Name("Insert")                         \
                                .Device(DEVICE_##D)                \
                                .TypeConstraint<TYPE>("T")         \
                                .TypeConstraint<int32>("Tindices") \
                                .HostMemory("indices"),            \
                            InsertOp<D##Device, TYPE, int32>);     \
    REGISTER_KERNEL_BUILDER(Name("Insert")                         \
                                .Device(DEVICE_##D)                \
                                .TypeConstraint<TYPE>("T")         \
                                .TypeConstraint<int64>("Tindices") \
                                .HostMemory("indices"),            \
                            InsertOp<D##Device, TYPE, int64>);

REGISTER_KERNEL(CPU, float);
REGISTER_KERNEL(CPU, double);

#undef REGISTER_KERNEL

}  // namespace tensorflow
