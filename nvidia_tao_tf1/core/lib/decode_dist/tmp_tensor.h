// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

#ifndef TMP_TENSOR_H_
#define TMP_TENSOR_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.h"
using namespace tensorflow;

// ValueType  must be 'memcopable', supported by tensorflow::Tensor
template <class ValueType>
class TmpDeviceTensor {
 protected:
    Tensor tensor_;
    OpKernelContext* context_;
    DataType data_type_;  // tensorflow datatype such as DT_INT16, etc.
    bool is_initialized_;

 public:
    TmpDeviceTensor(OpKernelContext* context, TensorShape tensor_shape)
        : context_(context), data_type_(DataTypeToEnum<ValueType>::value) {
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<ValueType>::value,
                                                       tensor_shape, &tensor_));
        is_initialized_ = true;
    }

    TmpDeviceTensor()
        : data_type_(DataTypeToEnum<ValueType>::value), context_(nullptr), is_initialized_(false) {}

    void init(OpKernelContext* context, TensorShape tensor_shape) {
        if (!is_initialized_) {
            context_ = context;
            OP_REQUIRES_OK(context, context->allocate_temp(data_type_, tensor_shape, &tensor_));
            is_initialized_ = true;
        }
    }

    ValueType* get_dptr() {
        if (is_initialized_)
            return tensor_.flat<ValueType>().data();
        else
            return nullptr;
    }

    size_t get_dim0() {
        if (is_initialized_)
            return tensor_.dim_size(0);
        else
            return 0;
    }

    Tensor get_tensor() { return tensor_; }
};

#endif /* TMP_TENSOR_H_ */
