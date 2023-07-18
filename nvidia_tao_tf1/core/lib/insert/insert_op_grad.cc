// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

#include <algorithm>
#include <array>
#include <numeric>
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

template <typename T>
void InsertGrad1(const Tensor& Tgrad,
                 const std::array<const std::vector<std::size_t>*, 5>& output_table, Tensor* Tdx) {
    auto dx = Tdx->vec<T>();
    auto grad = Tgrad.vec<T>();

    for (std::size_t i = 0; i < static_cast<std::size_t>(dx.size()); ++i) {
        dx(i) = grad((*output_table[0])[i]);
    }
}

template <typename T>
void InsertGrad2(const Tensor& Tgrad,
                 const std::array<const std::vector<std::size_t>*, 5>& output_table, Tensor* Tdx) {
    auto dx = Tdx->tensor<T, 2>();
    auto grad = Tgrad.tensor<T, 2>();

    auto sizes_raw = Tdx->shape().dim_sizes();
    std::array<std::size_t, 2> sizes = {static_cast<std::size_t>(sizes_raw[0]),
                                        static_cast<std::size_t>(sizes_raw[1])};

    for (std::size_t i = 0; i < sizes[0]; ++i) {
        for (std::size_t j = 0; j < sizes[1]; ++j) {
            dx(i, j) = grad((*output_table[0])[i], (*output_table[1])[j]);
        }
    }
}

template <typename T>
void InsertGrad3(const Tensor& Tgrad,
                 const std::array<const std::vector<std::size_t>*, 5>& output_table, Tensor* Tdx) {
    auto dx = Tdx->tensor<T, 3>();
    auto grad = Tgrad.tensor<T, 3>();

    auto sizes_raw = Tdx->shape().dim_sizes();
    std::array<std::size_t, 3> sizes = {static_cast<std::size_t>(sizes_raw[0]),
                                        static_cast<std::size_t>(sizes_raw[1]),
                                        static_cast<std::size_t>(sizes_raw[2])};

    for (std::size_t i = 0; i < sizes[0]; ++i) {
        for (std::size_t j = 0; j < sizes[1]; ++j) {
            for (std::size_t k = 0; k < sizes[2]; ++k) {
                dx(i, j, k) =
                    grad((*output_table[0])[i], (*output_table[1])[j], (*output_table[2])[k]);
            }
        }
    }
}

template <typename T>
void InsertGrad4(const Tensor& Tgrad,
                 const std::array<const std::vector<std::size_t>*, 5>& output_table, Tensor* Tdx) {
    auto dx = Tdx->tensor<T, 4>();
    auto grad = Tgrad.tensor<T, 4>();

    auto sizes_raw = Tdx->shape().dim_sizes();
    std::array<std::size_t, 4> sizes = {
        static_cast<std::size_t>(sizes_raw[0]), static_cast<std::size_t>(sizes_raw[1]),
        static_cast<std::size_t>(sizes_raw[2]), static_cast<std::size_t>(sizes_raw[3])};

    for (std::size_t i = 0; i < sizes[0]; ++i) {
        for (std::size_t j = 0; j < sizes[1]; ++j) {
            for (std::size_t k = 0; k < sizes[2]; ++k) {
                for (std::size_t l = 0; l < sizes[3]; ++l) {
                    dx(i, j, k, l) = grad((*output_table[0])[i], (*output_table[1])[j],
                                          (*output_table[2])[k], (*output_table[3])[l]);
                }
            }
        }
    }
}

template <typename T>
void InsertGrad5(const Tensor& Tgrad,
                 const std::array<const std::vector<std::size_t>*, 5>& output_table, Tensor* Tdx) {
    auto dx = Tdx->tensor<T, 5>();
    auto grad = Tgrad.tensor<T, 5>();

    auto sizes_raw = Tdx->shape().dim_sizes();
    std::array<std::size_t, 5> sizes = {
        static_cast<std::size_t>(sizes_raw[0]), static_cast<std::size_t>(sizes_raw[1]),
        static_cast<std::size_t>(sizes_raw[2]), static_cast<std::size_t>(sizes_raw[3]),
        static_cast<std::size_t>(sizes_raw[4])};

    for (std::size_t i = 0; i < sizes[0]; ++i) {
        for (std::size_t j = 0; j < sizes[1]; ++j) {
            for (std::size_t k = 0; k < sizes[2]; ++k) {
                for (std::size_t l = 0; l < sizes[3]; ++l) {
                    for (std::size_t m = 0; m < sizes[4]; ++m) {
                        dx(i, j, k, l, m) = grad((*output_table[0])[i], (*output_table[1])[j],
                                                 (*output_table[2])[k], (*output_table[3])[l],
                                                 (*output_table[4])[m]);
                    }
                }
            }
        }
    }
}

}  // anonymous namespace

REGISTER_OP("InsertGrad")
    .Input("grad: T")
    .Input("indices: Tindices")
    .Attr("T: type")
    .Attr("Tindices: {int32, int64} = DT_INT32")
    .Attr("axis: int >= 0 = 0")
    .Output("dx: T")
    .SetShapeFn(shape_inference::UnknownShape);

template <typename Device, typename T, typename Index>
class InsertOpGrad : public OpKernel {
 public:
    explicit InsertOpGrad(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& Tgrad = ctx->input(0);
        const Tensor& Tindices = ctx->input(1);

        //
        // Validate inputs and compute output shape.
        //
        OP_REQUIRES(ctx, TensorShapeUtils::IsVector(Tindices.shape()),
                    errors::InvalidArgument("indices must be a vector, got shape ",
                                            Tindices.shape().DebugString()));
        OP_REQUIRES(ctx, axis_ < Tgrad.dims(),
                    errors::InvalidArgument("axis must be >= 0 and < rank(x)"));

        TensorShape dx_shape{Tgrad.shape()};
        int64 output_axis_size = Tgrad.dim_size(axis_) - Tindices.dim_size(0);
        dx_shape.set_dim(axis_, output_axis_size);

        //
        // Check that indices are within range.
        //
        auto indices_raw = Tindices.flat<Index>();
        for (decltype(indices_raw.size()) i = 0; i < indices_raw.size(); ++i) {
            OP_REQUIRES(
                ctx, indices_raw(i) >= 0 && indices_raw(i) <= dx_shape.dim_size(axis_),
                errors::InvalidArgument("indices must be >= 0 and < size of axis Insert input"));
        }
        // Cast to size_t to avoid templating code on index type.
        std::vector<std::size_t> indices(indices_raw.data(),
                                         indices_raw.data() + indices_raw.size());

        Tensor* Tdx = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, dx_shape, &Tdx));

        //
        // Compute scatter index tables.
        //
        auto sizes = dx_shape.dim_sizes();
        auto max_size = *std::max_element(sizes.begin(), sizes.end());

        // On all axis except scatter axis we output to the same as the original index.
        std::vector<std::size_t> identity_map(max_size);
        std::iota(identity_map.begin(), identity_map.end(), 0);
        std::vector<std::size_t> scatter_indices = ComputeScatterIndices(indices, sizes[axis_]);

        // Compute output tables up to rank 5.
        std::array<const std::vector<std::size_t>*, 5> output_table;
        output_table.fill(&identity_map);
        output_table[axis_] = &scatter_indices;

        // Call handler for current rank.
        switch (Tgrad.dims()) {
            case 1: {
                InsertGrad1<T>(Tgrad, output_table, Tdx);
                break;
            }
            case 2: {
                InsertGrad2<T>(Tgrad, output_table, Tdx);
                break;
            }
            case 3: {
                InsertGrad3<T>(Tgrad, output_table, Tdx);
                break;
            }
            case 4: {
                InsertGrad4<T>(Tgrad, output_table, Tdx);
                break;
            }
            case 5: {
                InsertGrad5<T>(Tgrad, output_table, Tdx);
                break;
            }
            default:
                OP_REQUIRES(ctx, false,
                            errors::InvalidArgument("Insert only supports input tensors"
                                                    " of rank <= 5, got tensor of rank ",
                                                    std::to_string(Tgrad.dims())));
        }
    }

 private:
    int32 axis_ = 0;
};

#define REGISTER_KERNEL(D, TYPE)                                   \
    REGISTER_KERNEL_BUILDER(Name("InsertGrad")                     \
                                .Device(DEVICE_##D)                \
                                .TypeConstraint<TYPE>("T")         \
                                .TypeConstraint<int32>("Tindices") \
                                .HostMemory("indices"),            \
                            InsertOpGrad<D##Device, TYPE, int32>); \
    REGISTER_KERNEL_BUILDER(Name("InsertGrad")                     \
                                .Device(DEVICE_##D)                \
                                .TypeConstraint<TYPE>("T")         \
                                .TypeConstraint<int64>("Tindices") \
                                .HostMemory("indices"),            \
                            InsertOpGrad<D##Device, TYPE, int64>);

REGISTER_KERNEL(CPU, float);
REGISTER_KERNEL(CPU, double);

#undef REGISTER_KERNEL

}  // namespace tensorflow
