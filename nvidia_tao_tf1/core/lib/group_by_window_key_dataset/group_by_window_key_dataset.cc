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
class GroupByWindowKeyDatasetOp : public UnaryDatasetOpKernel {
 public:
    explicit GroupByWindowKeyDatasetOp(OpKernelConstruction* ctx)
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
                new Iterator({this, strings::StrCat(prefix, "::GroupByWindowKey")}));
        }

        const DataTypeVector& output_dtypes() const override { return output_types_; }
        const std::vector<PartialTensorShape>& output_shapes() const override {
            return output_shapes_;
        }

        string DebugString() const override { return "GroupByWindowKeyDatasetOp::Dataset"; }

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
                do {
                    if (current_group_iterator_) {
                        // We are currently processing a group, so try to get the
                        // next element.
                        bool end_of_group;
                        TF_RETURN_IF_ERROR(
                            current_group_iterator_->GetNext(ctx, out_tensors, &end_of_group));
                        if (!end_of_group) {
                            // Produce the subelement as output.
                            *end_of_sequence = false;
                            return Status::OK();
                        }
                        // We have reached the end of the current group, so maybe move on
                        // to the next group.
                        current_group_iterator_.reset();
                        groups_.erase(flush_key_);
                    }

                    // Iterate through the input dataset until we get a full
                    // group, or reach the end.
                    while (!end_of_input_) {
                        std::vector<Tensor> next_input_element;
                        TF_RETURN_IF_ERROR(
                            input_impl_->GetNext(ctx, &next_input_element, &end_of_input_));

                        if (!end_of_input_) {
                            // Run the key function on the input element to identify its
                            // group.
                            std::vector<Tensor> key_func_output;
                            TF_RETURN_IF_ERROR(instantiated_key_func_->RunWithBorrowedArgs(
                                ctx, next_input_element, &key_func_output));

                            if (key_func_output.size() != 1 ||
                                key_func_output[0].dtype() != DT_INT64 ||
                                key_func_output[0].NumElements() != 1) {
                                // TODO(mrry): Support non-int64 keys.
                                return errors::InvalidArgument(
                                    "`key_func` must return a scalar int64.");
                            }
                            const int64 key = key_func_output[0].scalar<int64>()();

                            if (groups_.empty()) {
                                // Initial assignment.
                                current_key_ = key;
                            }

                            std::vector<std::vector<Tensor>>& group = groups_[key];
                            group.push_back(std::move(next_input_element));

                            if (key != current_key_) {
                                flush_key_ = current_key_;
                                TF_RETURN_IF_ERROR(StartFlushingGroup(ctx, flush_key_));
                                current_key_ = key;
                                break;
                            }
                        }
                    }

                    if (end_of_input_) {
                        if (!groups_.empty()) {
                            // We have consumed all of the input, so flush an
                            // arbitrarily chosen group.
                            flush_key_ = groups_.begin()->first;
                            TF_RETURN_IF_ERROR(StartFlushingGroup(ctx, flush_key_));
                        }
                    }
                } while (current_group_iterator_ || !end_of_input_);

                *end_of_sequence = true;
                return Status::OK();
            }

         protected:
            Status SaveInternal(IteratorStateWriter* writer) override {
                mutex_lock l(mu_);
                TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));

                if (end_of_input_) {
                    TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("end_of_input"), ""));
                }

                // Saving groups_
                if (!groups_.empty()) {
                    TF_RETURN_IF_ERROR(
                        writer->WriteScalar(full_name("groups_size"), groups_.size()));
                    int idx = 0;
                    for (auto it = groups_.begin(); it != groups_.end(); it++) {
                        int64 key = it->first;
                        TF_RETURN_IF_ERROR(writer->WriteScalar(
                            full_name(strings::StrCat("groups_[", idx, "]->key")), key));
                        TF_RETURN_IF_ERROR(SaveGroup(
                            writer, full_name(strings::StrCat("groups_[", idx, "]")), it->second));
                        idx++;
                    }
                }

                if (current_group_iterator_) {
                    TF_RETURN_IF_ERROR(SaveInput(writer, current_group_iterator_));

                    // Saving flush_key_
                    TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("flush_key"), flush_key_));
                    // Saving current_key_
                    TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("current_key"), current_key_));
                } else {
                    TF_RETURN_IF_ERROR(
                        writer->WriteScalar(full_name("current_iterator_not_initialized"), ""));
                }

                return Status::OK();
            }

            Status RestoreInternal(IteratorContext* ctx, IteratorStateReader* reader) override {
                mutex_lock l(mu_);
                TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));

                if (reader->Contains(full_name("end_of_input"))) end_of_input_ = true;

                // Restoring groups
                if (reader->Contains(full_name("groups_size"))) {
                    int64 size;
                    TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("groups_size"), &size));
                    for (int idx = 0; idx < size; idx++) {
                        int64 key;
                        TF_RETURN_IF_ERROR(reader->ReadScalar(
                            full_name(strings::StrCat("groups_[", idx, "]->key")), &key));
                        std::vector<std::vector<Tensor>> group;
                        TF_RETURN_IF_ERROR(RestoreGroup(
                            reader, full_name(strings::StrCat("groups_[", idx, "]")), &group));
                        groups_[key] = group;
                    }
                }

                if (reader->Contains(full_name("current_iterator_not_initialized"))) {
                    current_group_iterator_.reset();
                } else {
                    // Restore flush_key_
                    TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("flush_key"), &flush_key_));
                    // Restore current_key_
                    TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("current_key"), &current_key_));

                    // Initialize current_group_iterator_
                    TF_RETURN_IF_ERROR(StartFlushingGroup(ctx, flush_key_));
                    // Restore current_group_iterator_ state
                    TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, current_group_iterator_));
                }
                return Status::OK();
            }

         private:
            Status SaveGroup(IteratorStateWriter* writer, const string& name,
                             const std::vector<std::vector<Tensor>>& group)
                EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                TF_RETURN_IF_ERROR(
                    writer->WriteScalar(strings::StrCat(name, "_size"), group.size()));
                for (size_t i = 0; i < group.size(); i++) {
                    TF_RETURN_IF_ERROR(writer->WriteScalar(strings::StrCat(name, "[", i, "]_size"),
                                                           group[i].size()));
                    for (size_t j = 0; j < group[i].size(); j++) {
                        TF_RETURN_IF_ERROR(writer->WriteTensor(
                            strings::StrCat(name, "[", i, "][", j, "]"), group[i][j]));
                    }
                }
                return Status::OK();
            }

            Status RestoreGroup(IteratorStateReader* reader, const string& name,
                                std::vector<std::vector<Tensor>>* group)
                EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                int64 group_size;
                TF_RETURN_IF_ERROR(reader->ReadScalar(strings::StrCat(name, "_size"), &group_size));
                group->resize(group_size);
                for (int i = 0; i < group_size; i++) {
                    int64 vector_size;
                    TF_RETURN_IF_ERROR(
                        reader->ReadScalar(strings::StrCat(name, "[", i, "]_size"), &vector_size));
                    group->at(i).resize(vector_size);
                    for (int j = 0; j < vector_size; j++) {
                        TF_RETURN_IF_ERROR(reader->ReadTensor(
                            strings::StrCat(name, "[", i, "][", j, "]"), &group->at(i)[j]));
                    }
                }
                return Status::OK();
            }

            Status StartFlushingGroup(IteratorContext* ctx, int64 key)
                EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                DatasetBase* group_dataset;
                TF_RETURN_IF_ERROR(
                    NewWindowDataset(groups_[key], dataset()->input_->output_dtypes(),
                                     dataset()->input_->output_shapes(), &group_dataset));

                Tensor key_arg(DT_INT64, TensorShape({}));
                key_arg.scalar<int64>()() = key;

                Tensor size_arg(DT_INT64, TensorShape({}));
                size_arg.scalar<int64>()() = static_cast<int64>(groups_[key].size());

                Tensor group_dataset_arg(DT_VARIANT, TensorShape({}));
                TF_RETURN_IF_ERROR(StoreDatasetInVariantTensor(group_dataset, &group_dataset_arg));

                std::vector<Tensor> args(
                    {std::move(key_arg), std::move(size_arg), std::move(group_dataset_arg)});
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
                return returned_dataset->MakeIterator(ctx, prefix(), &current_group_iterator_);
            }

            mutex mu_;
            std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(mu_);
            // TODO(mrry): Optimize for dense key space if appropriate.
            bool end_of_input_ GUARDED_BY(mu_) = false;
            int64 flush_key_ GUARDED_BY(mu_);
            int64 current_key_ GUARDED_BY(mu_);
            std::map<int64, std::vector<std::vector<Tensor>>> groups_ GUARDED_BY(mu_);
            std::unique_ptr<IteratorBase> current_group_iterator_ GUARDED_BY(mu_);
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

REGISTER_OP("GroupByWindowKeyDataset")
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

REGISTER_KERNEL_BUILDER(Name("GroupByWindowKeyDataset").Device(DEVICE_CPU),
                        GroupByWindowKeyDatasetOp);

}  // namespace
}  // namespace tensorflow
