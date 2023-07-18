// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

#include <float.h>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/logging.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "point.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;

class _ClusterOneSweep : public OpKernel {
 protected:
    // These variables are set based on input attributes.
    int up_scale_factor_;
    float cluster_radius_;
    float cluster_distance_;
    float thre_min_mask_;
    float thre_max_mask_;
    float thre_min_vote_;
    float thre_max_vote_;

    // These variables are set based on input dimensions.
    int num_pred_channels_;
    int batch_size_;
    int height_;
    int width_;

    static const int DIM_FOR_PR = 2;         // Dimension of precision and recall.
    static const int NUM_MASK_CHANNELS = 1;  // Number of channels for cluster_mask.

    // Color coefficients for converting prediction channels to colors.
    const float color_table_[3][4] = {
        {0, 0.4, 0.6, 0}, {0, 0.4, 0, 0.6}, {0.6, 0.4, 0, 0},
    };

    struct cluster_node {
        PointNf center = PointNf(0);
        int pointCount;
        bool valid = true;
    };

    std::vector<cluster_node> cluster_list_;

    void _set_member_variables(const int n_channels, const int n_batches, const int height,
                               const int width) {
        num_pred_channels_ = n_channels;
        batch_size_ = n_batches;
        height_ = height;
        width_ = width;
    }

    void _point_add_to_new_cluster(const PointNf& point) {
        cluster_node new_node{};
        new_node.center = point;
        new_node.pointCount = 1;
        cluster_list_.push_back(new_node);
    }

    void _point_add_to_exist_cluster(const PointNf& point, int idx) {
        cluster_node& current_node = cluster_list_[idx];
        int num_points = current_node.pointCount;
        current_node.center = (current_node.center * static_cast<float>(num_points) + point) *
                              static_cast<float>(1.0 / (num_points + 1));
        current_node.pointCount += 1;
    }

    // Check whether the points are voted.
    bool _check_vote_mask(const int* cluster_mask_start, int width, int i, int j) {
        int k = 0;
        while (k < NUM_MASK_CHANNELS) {
            if (cluster_mask_start[i * width * NUM_MASK_CHANNELS + j * NUM_MASK_CHANNELS + k] > 0)
                return true;
            k++;
        }
        return false;
    }

    /* Determine the color of the cluster based on encoded_blob input.
    This is just for demonstration and may not display the difference when the
    number of raw clustering channels is greater than 3. */
    int _color_transform(const PointNf& p, int channel) {
        float res = 0;
        for (int i = 0; i < num_pred_channels_; i++) {
            res += color_table_[channel][i % 4] * p.x_[i] * 255.0;
        }
        return static_cast<int>(res < 0 ? 0 : (res > 255 ? 255 : res));
    }

    // Helper function for descending sort.
    static bool _compare_func(const std::pair<int, int>& a, const std::pair<int, int>& b) {
        return a.second >= b.second;
    }

 public:
    explicit _ClusterOneSweep(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("up_scale_factor", &up_scale_factor_));
        OP_REQUIRES_OK(context, context->GetAttr("cluster_radius", &cluster_radius_));
        OP_REQUIRES_OK(context, context->GetAttr("cluster_distance", &cluster_distance_));
        OP_REQUIRES_OK(context, context->GetAttr("thre_min_mask", &thre_min_mask_));
        OP_REQUIRES_OK(context, context->GetAttr("thre_max_mask", &thre_max_mask_));
        OP_REQUIRES_OK(context, context->GetAttr("thre_min_vote", &thre_min_vote_));
        OP_REQUIRES_OK(context, context->GetAttr("thre_max_vote", &thre_max_vote_));

        OP_REQUIRES(
            context, up_scale_factor_ > 0,
            errors::InvalidArgument("`up_scale_factor`:", up_scale_factor_, " must be positive."));
        OP_REQUIRES(
            context, cluster_radius_ > 0,
            errors::InvalidArgument("`cluster_radius`:", cluster_radius_, " must be positive."));
        OP_REQUIRES(context, cluster_distance_ > 0,
                    errors::InvalidArgument("`cluster_distance`:", cluster_distance_,
                                            " must be positive."));
        OP_REQUIRES(context, cluster_radius_ <= cluster_distance_,
                    errors::InvalidArgument("`cluster_radius`:", cluster_radius_,
                                            " must be no greater than cluster_distance:",
                                            cluster_distance_));
        OP_REQUIRES(
            context, thre_min_mask_ > 0,
            errors::InvalidArgument("`thre_min_mask`:", thre_min_mask_, " must be positive."));
        OP_REQUIRES(
            context, thre_max_mask_ > 0,
            errors::InvalidArgument("`thre_max_mask`:", thre_max_mask_, " must be positive."));
        OP_REQUIRES(
            context, thre_max_mask_ > thre_min_mask_,
            errors::InvalidArgument("`thre_max_mask`:", thre_max_mask_,
                                    " must be greater than thre_min_mask:", thre_min_mask_));

        OP_REQUIRES(
            context, thre_min_vote_ > 0,
            errors::InvalidArgument("`thre_min_mask`:", thre_min_vote_, " must be positive."));
        OP_REQUIRES(
            context, thre_max_vote_ > 0,
            errors::InvalidArgument("`thre_max_vote`:", thre_max_vote_, " must be positive."));
        OP_REQUIRES(
            context, thre_max_vote_ > thre_min_vote_,
            errors::InvalidArgument("`thre_max_mask`:", thre_max_vote_,
                                    " must be greater than thre_min_vote:", thre_min_vote_));
    }

    // Perform one-sweep clustering.
    virtual void _render_cluster_one_sweep(const float* encoded_blobs_start,
                                           const int* cluster_mask_start, const int up_scale_factor,
                                           const bool flag_check_mask) = 0;

    // Filter out extreme cases of clustering results.
    virtual void _render_cluster_filter(const float* encoded_blobs_start,
                                        const int* cluster_mask_start, const int up_scale_factor,
                                        const bool flag_check_mask, const float cluster_thres,
                                        const float background_thres) = 0;

    // Render cluster ID map and cluster colormap.
    virtual void _render_cluster_id_colormap(int* output_cluster_id_map_start,
                                             int* output_cluster_color_map_start,
                                             const float* encoded_blobs_start,
                                             const int* cluster_mask_start,
                                             const int up_scale_factor,
                                             const bool flag_check_mask) = 0;

    // Core function for one-sweep clustering.
    virtual void _render_cluster_core(int* output_cluster_id_map_start,
                                      int* output_cluster_color_map_start,
                                      const float* encoded_blobs_start,
                                      const int* cluster_mask_start, const int up_scale_factor,
                                      const bool flag_check_mask, const float cluster_thres,
                                      const float background_thres) = 0;

    // Core function for evaluation (Precision and recall).
    virtual void _compute_pr_core(int* output_cluster_id_map_start,
                                  int* output_cluster_color_map_start,
                                  float* output_binary_pr_start,
                                  const int* cluster_id_gt_start) = 0;

    // This function calls _render_cluster_core() and _compute_pr_core().
    virtual void _cluster_core(int* output_cluster_id_map, int* output_cluster_color_map,
                               int* output_cluster_id_map_vote, int* output_cluster_color_map_vote,
                               float* output_binary_pr, const float* encoded_blobs,
                               const int* cluster_id_gt, const int* cluster_mask) = 0;

    void Compute(OpKernelContext* context) override {
        // Process input.
        const Tensor& tensor_encoded_blobs = context->input(0);  // NCHW
        const Tensor& tensor_gt = context->input(1);             // NHWC (C=1)
        const Tensor& tensor_mask = context->input(2);           // NHWC (C=1)
        auto encoded_blobs = tensor_encoded_blobs.flat<float>();
        auto cluster_id_gt = tensor_gt.flat<int>();
        auto cluster_mask = tensor_mask.flat<int>();

        // Check assumptions.
        OP_REQUIRES(context, tensor_encoded_blobs.dims() == 4,
                    errors::InvalidArgument("The rank of the encoded_blobs should be 4"));

        const int batch_size = tensor_encoded_blobs.dim_size(0);
        const int num_pred_channels = tensor_encoded_blobs.dim_size(1);
        const int height = tensor_encoded_blobs.dim_size(2);
        const int width = tensor_encoded_blobs.dim_size(3);

        OP_REQUIRES(context, tensor_gt.shape().dim_size(1) == height,
                    errors::InvalidArgument("tensor_gt.shape().dim_size(1) == height, got ",
                                            tensor_gt.shape().dim_size(1), " and  ", height));
        OP_REQUIRES(context, tensor_gt.shape().dim_size(2) == width,
                    errors::InvalidArgument("tensor_gt.shape().dim_size(2) == width, got ",
                                            tensor_gt.shape().dim_size(2), " and  ", width));
        OP_REQUIRES(
            context, height * up_scale_factor_ == tensor_mask.shape().dim_size(1),
            errors::InvalidArgument("`input height`(", height, ") multiply by `up_scale_factor_`=",
                                    up_scale_factor_, ") should be equal to `mask height`=",
                                    tensor_mask.shape().dim_size(1)));
        OP_REQUIRES(
            context, width * up_scale_factor_ == tensor_mask.shape().dim_size(2),
            errors::InvalidArgument("`input width`(", width, ") multiply by `up_scale_factor_`=",
                                    up_scale_factor_, ") should be equal to `mask width`=",
                                    tensor_mask.shape().dim_size(2)));

        // Set up output tensor.
        Tensor* output_tensor = NULL;

        // Create an output tensor.
        TensorShape output_shape1({batch_size, height, width, 1});
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape1, &output_tensor));
        auto output_cluster_id_map = output_tensor->flat<int>();

        TensorShape output_shape2({batch_size, height, width, 3});
        OP_REQUIRES_OK(context, context->allocate_output(1, output_shape2, &output_tensor));
        auto output_cluster_color_map = output_tensor->flat<int>();

        TensorShape output_shape3(
            {batch_size, height * up_scale_factor_, width * up_scale_factor_, 1});
        OP_REQUIRES_OK(context, context->allocate_output(2, output_shape3, &output_tensor));
        auto output_cluster_id_map_vote = output_tensor->flat<int>();

        TensorShape output_shape4(
            {batch_size, height * up_scale_factor_, width * up_scale_factor_, 3});
        OP_REQUIRES_OK(context, context->allocate_output(3, output_shape4, &output_tensor));
        auto output_cluster_color_map_vote = output_tensor->flat<int>();

        TensorShape output_shape5({batch_size, 1, 1, DIM_FOR_PR});
        OP_REQUIRES_OK(context, context->allocate_output(4, output_shape5, &output_tensor));
        auto binary_pr = output_tensor->flat<float>();

        _set_member_variables(num_pred_channels, batch_size, height, width);
        _cluster_core(output_cluster_id_map.data(), output_cluster_color_map.data(),
                      output_cluster_id_map_vote.data(), output_cluster_color_map_vote.data(),
                      binary_pr.data(), encoded_blobs.data(), cluster_id_gt.data(),
                      cluster_mask.data());
    }
};
