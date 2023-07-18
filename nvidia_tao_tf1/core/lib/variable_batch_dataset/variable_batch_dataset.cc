/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Modifications: Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#include <map>

#include <iostream>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/random/random.h"

// Below includes are checked into ai-infra repo.
#include "tensorflow/core/kernels/data/captured_function.h"
#include "tensorflow/core/kernels/data/window_dataset.h"

namespace tensorflow {
namespace {

// See documentation in ../ops/dataset_ops.cc for a high-level
// description of the following op.
class VariableBatchDatasetOp : public UnaryDatasetOpKernel {
 public:
    explicit VariableBatchDatasetOp(OpKernelConstruction* ctx)
        : UnaryDatasetOpKernel(ctx), graph_def_version_(ctx->graph_def_version()) {
        OP_REQUIRES_OK(ctx, data::FunctionMetadata::Create(ctx, "key_func", /*params=*/{},
                                                           &key_func_metadata_));
        OP_REQUIRES_OK(ctx, data::FunctionMetadata::Create(ctx, "reduce_func", /*params=*/{},
                                                           &reduce_func_metadata_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
    }

    void MakeDataset(OpKernelContext* ctx, DatasetBase* input, DatasetBase** output) override {
        // Get captured inputs for the key, reduce, and window_size functions.
        OpInputList key_func_other_argument_inputs;
        OP_REQUIRES_OK(
            ctx, ctx->input_list("key_func_other_arguments", &key_func_other_argument_inputs));
        std::vector<Tensor> key_func_other_arguments;
        key_func_other_arguments.reserve(key_func_other_argument_inputs.size());
        for (const Tensor& t : key_func_other_argument_inputs) {
            key_func_other_arguments.push_back(t);
        }
        OpInputList reduce_func_other_argument_inputs;
        OP_REQUIRES_OK(ctx, ctx->input_list("reduce_func_other_arguments",
                                            &reduce_func_other_argument_inputs));
        std::vector<Tensor> reduce_func_other_arguments;
        reduce_func_other_arguments.reserve(reduce_func_other_argument_inputs.size());
        for (const Tensor& t : reduce_func_other_argument_inputs) {
            reduce_func_other_arguments.push_back(t);
        }
        OpInputList window_size_func_other_argument_inputs;

        // TODO(mrry): Refactor CapturedFunction to share the runtime
        // state between multiple functions?
        std::unique_ptr<CapturedFunction> captured_key_func;
        OP_REQUIRES_OK(
            ctx, CapturedFunction::Create(ctx, key_func_metadata_,
                                          std::move(key_func_other_arguments), &captured_key_func));
        std::unique_ptr<CapturedFunction> captured_reduce_func;
        OP_REQUIRES_OK(ctx, CapturedFunction::Create(ctx, reduce_func_metadata_,
                                                     std::move(reduce_func_other_arguments),
                                                     &captured_reduce_func));

        *output = new Dataset(ctx, input, key_func_metadata_->func(), reduce_func_metadata_->func(),
                              std::move(captured_key_func), std::move(captured_reduce_func),
                              output_types_, output_shapes_);
    }

 private:
    class Dataset : public DatasetBase {
     public:
        Dataset(OpKernelContext* ctx, const DatasetBase* input, const NameAttrList& key_func,
                const NameAttrList& reduce_func,
                std::unique_ptr<CapturedFunction> captured_key_func,
                std::unique_ptr<CapturedFunction> captured_reduce_func,
                const DataTypeVector& output_types,
                const std::vector<PartialTensorShape>& output_shapes)
            : DatasetBase(DatasetContext(ctx)),
              input_(input),
              key_func_(key_func),
              reduce_func_(reduce_func),
              captured_key_func_(std::move(captured_key_func)),
              captured_reduce_func_(std::move(captured_reduce_func)),
              output_types_(output_types),
              output_shapes_(output_shapes) {
            input_->Ref();
        }

        ~Dataset() override { input_->Unref(); }

        std::unique_ptr<IteratorBase> MakeIteratorInternal(const string& prefix) const override {
            return std::unique_ptr<IteratorBase>(
                new Iterator({this, strings::StrCat(prefix, "::VariableBatch")}));
        }

        const DataTypeVector& output_dtypes() const override { return output_types_; }
        const std::vector<PartialTensorShape>& output_shapes() const override {
            return output_shapes_;
        }

        string DebugString() const override { return "VariableBatchDatasetOp::Dataset"; }

     protected:
        Status AsGraphDefInternal(SerializationContext* ctx, DatasetGraphDefBuilder* b,
                                  Node** output) const override {
            Node* input_graph_node = nullptr;
            TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

            std::vector<Node*> key_func_other_arguments_node;
            DataTypeVector key_func_other_arguments_types;
            TF_RETURN_IF_ERROR(captured_key_func_->AddToGraph(
                ctx, b, &key_func_other_arguments_node, &key_func_other_arguments_types));

            std::vector<Node*> reduce_func_other_arguments_node;
            DataTypeVector reduce_func_other_arguments_types;
            TF_RETURN_IF_ERROR(captured_reduce_func_->AddToGraph(
                ctx, b, &reduce_func_other_arguments_node, &reduce_func_other_arguments_types));

            AttrValue key_func;
            b->BuildAttrValue(key_func_, &key_func);
            AttrValue reduce_func;
            b->BuildAttrValue(reduce_func_, &reduce_func);

            AttrValue key_func_other_arguments_types_attr;
            b->BuildAttrValue(key_func_other_arguments_types, &key_func_other_arguments_types_attr);
            AttrValue reduce_func_other_arguments_types_attr;
            b->BuildAttrValue(reduce_func_other_arguments_types,
                              &reduce_func_other_arguments_types_attr);

            TF_RETURN_IF_ERROR(b->AddDataset(
                this, {{0, input_graph_node}},
                {
                    {1, key_func_other_arguments_node}, {2, reduce_func_other_arguments_node},
                },
                {
                    {"key_func", key_func},
                    {"reduce_func", reduce_func},
                    {"Tkey_func_other_arguments", key_func_other_arguments_types_attr},
                    {"Treduce_func_other_arguments", reduce_func_other_arguments_types_attr},
                },
                output));
            return Status::OK();
        }

     private:
        class Iterator : public DatasetIterator<Dataset> {
         public:
            explicit Iterator(const Params& params) : DatasetIterator<Dataset>(params) {}

            Status Initialize(IteratorContext* ctx) override {
                TF_RETURN_IF_ERROR(dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_));
                TF_RETURN_IF_ERROR(
                    dataset()->captured_key_func_->Instantiate(ctx, &instantiated_key_func_));
                TF_RETURN_IF_ERROR(
                    dataset()->captured_reduce_func_->Instantiate(ctx, &instantiated_reduce_func_));
                return Status::OK();
            }

            Status GetNextInternal(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                                   bool* end_of_sequence) override {
                mutex_lock l(mu_);

                std::vector<std::vector<Tensor>> group;
                std::unique_ptr<IteratorBase> current_group_iterator;

                std::vector<Tensor> next_input_element;
                bool end_of_input;
                TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx, &next_input_element, &end_of_input));

                if (end_of_input) {
                    *end_of_sequence = true;
                    return Status::OK();
                }

                std::vector<Tensor> key_func_output;
                TF_RETURN_IF_ERROR(instantiated_key_func_->RunWithBorrowedArgs(
                    ctx, next_input_element, &key_func_output));

                if (key_func_output.size() != 2 || key_func_output[0].dtype() != DT_INT64 ||
                    key_func_output[0].NumElements() != 1 ||
                    key_func_output[1].dtype() != DT_INT64 ||
                    key_func_output[1].NumElements() != 1) {
                    return errors::InvalidArgument(
                        "`key_func` must return two scalar int64, batch size and stride.");
                }
                const int64 batch_size = key_func_output[0].scalar<int64>()();
                const int64 stride = key_func_output[1].scalar<int64>()();

                for (int i = 0; i < batch_size; i++) {
                    // Push first element to the group, then loop over the
                    // batch_size elements, discarding all strided elements.
                    group.push_back(std::move(next_input_element));
                    if (i != (batch_size - 1)) {
                        for (int s = 0; s < std::abs(stride); s++) {
                            TF_RETURN_IF_ERROR(
                                input_impl_->GetNext(ctx, &next_input_element, &end_of_input));
                            if (end_of_input) {
                                *end_of_sequence = true;
                                return Status::OK();
                            }
                            // This is an in-between element, clear it.
                            if (s != std::abs(stride) - 1) {
                                next_input_element.clear();
                            }
                        }
                    }
                }

                // Reverse time if stride is negative.
                if (stride < 0) {
                    std::reverse(group.begin(), group.end());
                }

                // Flush the group.
                DatasetBase* group_dataset;
                TF_RETURN_IF_ERROR(NewWindowDataset(group, dataset()->input_->output_dtypes(),
                                                    dataset()->input_->output_shapes(),
                                                    &group_dataset));

                Tensor group_dataset_arg(DT_VARIANT, TensorShape({}));
                TF_RETURN_IF_ERROR(StoreDatasetInVariantTensor(group_dataset, &group_dataset_arg));

                std::vector<Tensor> args({std::move(group_dataset_arg)});
                std::vector<Tensor> return_values;
                TF_RETURN_IF_ERROR(
                    instantiated_reduce_func_->Run(ctx, std::move(args), &return_values));

                if (!(return_values.size() == 1 && return_values[0].dtype() == DT_VARIANT &&
                      TensorShapeUtils::IsScalar(return_values[0].shape()))) {
                    return errors::InvalidArgument(
                        "`reduce_func` must return a single scalar of dtype "
                        "DT_VARIANT.");
                }

                // Retrieve the dataset that was created in `f`.
                // `returned_dataset` is borrowed from the `return_values[0]`.
                DatasetBase* returned_dataset;
                TF_RETURN_IF_ERROR(
                    GetDatasetFromVariantTensor(return_values[0], &returned_dataset));

                // Create an iterator for the dataset that was returned by `f`.
                returned_dataset->MakeIterator(ctx, prefix(), &current_group_iterator);

                bool end_of_group;
                TF_RETURN_IF_ERROR(
                    current_group_iterator->GetNext(ctx, out_tensors, &end_of_group));
                assert(end_of_group == True);

                *end_of_sequence = false;
                return Status::OK();
            }

         private:
            mutex mu_;
            std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
            std::unique_ptr<data::InstantiatedCapturedFunction> instantiated_key_func_;
            std::unique_ptr<data::InstantiatedCapturedFunction> instantiated_reduce_func_;
        };

        const DatasetBase* const input_;
        const NameAttrList key_func_;
        const NameAttrList reduce_func_;
        const std::unique_ptr<CapturedFunction> captured_key_func_;
        const std::unique_ptr<CapturedFunction> captured_reduce_func_;
        const DataTypeVector output_types_;
        const std::vector<PartialTensorShape> output_shapes_;
    };

    const int graph_def_version_;
    DataTypeVector output_types_;
    std::vector<PartialTensorShape> output_shapes_;
    std::shared_ptr<data::FunctionMetadata> key_func_metadata_ = nullptr;
    std::shared_ptr<data::FunctionMetadata> reduce_func_metadata_ = nullptr;
};

REGISTER_OP("VariableBatchDataset")
    .Input("input_dataset: variant")
    .Input("key_func_other_arguments: Tkey_func_other_arguments")
    .Input("reduce_func_other_arguments: Treduce_func_other_arguments")
    .Output("handle: variant")
    .Attr("key_func: func")
    .Attr("reduce_func: func")
    .Attr("Tkey_func_other_arguments: list(type) >= 0")
    .Attr("Treduce_func_other_arguments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_KERNEL_BUILDER(Name("VariableBatchDataset").Device(DEVICE_CPU), VariableBatchDatasetOp);

}  // namespace
}  // namespace tensorflow
