#pragma once

// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.

/*
Very basic tensor implementation that works around Eigen's
weird need to materialize a sliced tensor before you can
perform element access operations
*/

#include <algorithm>
#include <sstream>
#include <string>
#include <utility>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace std;
using namespace tensorflow;

/*
Checks to see if the values of a given row in two different
matrices match up to a given column.
*/
template <typename EigenTensor>
bool IsSubDimEqual(int64 num_cols, const EigenTensor &a, int64 a_row, const EigenTensor &b,
                   int64 b_row) {
    if (num_cols > a.dimension(1) || num_cols > b.dimension(1))
        throw runtime_error(
            "The number of columns to check cannot exceed "
            "the size of the matrices!");

    auto ptr_a_row = &a(a_row, 0);
    auto ptr_b_row = &b(b_row, 0);

    for (int64 col = 0; col < num_cols; ++col) {
        if (ptr_a_row[col] != ptr_b_row[col]) {
            return false;
        }
    }
    return true;
}

/*
returns the size of the inside dimensions of the tensor.
For example, if the tensor is rank 2, and start_dim = 1,
then it will return the number of columns. If it's rank 3,
and start_dim = 1, then it will return the product of the
sizes of the second and third dimensions.
*/
template <typename EigenTensor>
int64 InnerSize(const EigenTensor &t, int64 start_dim) {
    int64 ret = 1;
    for (; start_dim < t.NumDimensions; ++start_dim) {
        ret *= t.dimension(start_dim);
    }
    return ret;
}

/*
tensorflow::Tensor has a similar function, but it doesn't
print more than just a couple values, and there doesn't seem
to be a good way to control how many values to print. This
function will print an eigen tensor of up to rank 2,
and will print both the size and the values.
*/
template <typename EigenTensor>
string DebugString(const EigenTensor &t) {
    ostringstream oss;
    oss << "(";
    if (t.NumDimensions > 0) {
        oss << t.dimension(0);
        for (int i = 1; i < t.NumDimensions; ++i) {
            oss << ", " << t.dimension(i);
        }
    }
    oss << ") - ";

    auto ptr_data = t.data();

    if (t.NumDimensions == 0) {
        oss << ptr_data[0];
    } else if (t.NumDimensions == 1) {
        oss << "[";
        if (t.dimension(0) > 0) {
            oss << ptr_data[0];
            for (int64 i = 1; i < t.dimension(0); ++i) {
                oss << ", " << ptr_data[i];
            }
        }
        oss << "]";
    } else if (t.NumDimensions == 2) {
        if (t.dimension(0) == 0 || t.dimension(1) == 0) {
            oss << "[[]]";
        } else {
            oss << "[" << endl;
            int64 num_cols = t.dimension(1);
            for (int64 row = 0; row < t.dimension(0); ++row) {
                oss << "\t[" << ptr_data[row * num_cols];
                for (int64 col = 1; col < num_cols; ++col) {
                    oss << ", " << ptr_data[row * num_cols + col];
                }
                oss << "]" << endl;
            }
            oss << "]" << endl;
        }
    } else {
        throw runtime_error("Unsupported Tensor Dimensions!");
    }
    return oss.str();
}

/*
Slices the input tensor along the outer dimension. It is mandatory that
consecutive elements in the input tensor are in contiguous memory. The
returned tensor will point to the same memory as the input tensor.
*/
template <typename EigenTensor>
EigenTensor Slice(const EigenTensor &mat, int64 start_row, int64 end_row) {
    if (start_row < 0 || start_row >= mat.dimension(0)) throw runtime_error("Invalid start row!");
    if (end_row < start_row || end_row > mat.dimension(0)) throw runtime_error("Invalid end row!");

    const int64 inner_size = InnerSize(mat, 1);
    typename EigenTensor::Scalar *ptr_data = mat.data();
    auto ptr_ret = ptr_data + start_row * inner_size;
    typename EigenTensor::Dimensions new_dims = mat.dimensions();
    new_dims[0] = end_row - start_row;

    return EigenTensor{ptr_ret, new_dims};
}

/*
Assigns the provided values beginning at the specified "row" or outer-most
dimension. Values can be a mix of scalars, as well as a pair<T*,size> that
provides a range of values.
*/
template <typename EigenTensor, typename... ValTypes>
inline void InnerAssign(const EigenTensor &mat, int64 row, ValTypes... vals);

template <typename ScalarType, typename ValType>
inline void _InnerAssign(ScalarType **data, ValType val) {
    **data = val;
    *data += 1;
}

template <typename ScalarType, typename ValType, typename SizeType>
inline void _InnerAssign(ScalarType **data, pair<const ValType *, SizeType> val) {
    copy(val.first, val.first + val.second, *data);
    *data += val.second;
}

template <typename ScalarType, typename ValType, typename... ValTypes>
inline void _InnerAssign(ScalarType **data, ValType val, ValTypes... vals) {
    _InnerAssign(data, val);
    _InnerAssign(data, vals...);
}

template <typename EigenTensor, typename... ValTypes>
inline void InnerAssign(const EigenTensor &mat, int64 row, ValTypes... vals) {
    if (row < 0 || row >= mat.dimension(0))
        throw runtime_error(
            "Invalid row specified. Must be within the size of the "
            "outer shape of the tensor!");

    const int64 inner_size = InnerSize(mat, 1);
    typename EigenTensor::Scalar *data = mat.data() + row * inner_size;

    _InnerAssign(&data, vals...);
}

template <typename EigenIndicesTensor, typename EigenValuesTensor, typename EigenShapeTensor>
void PrintSparseTensor(ostream &os, const EigenIndicesTensor &indices,
                       const EigenValuesTensor &values, const EigenShapeTensor &dense_shape) {
    os << "Indices:" << endl << DebugString(indices) << endl;
    os << "Values:" << endl << DebugString(values) << endl;
    os << "Shape:" << endl << DebugString(dense_shape) << endl;
}
