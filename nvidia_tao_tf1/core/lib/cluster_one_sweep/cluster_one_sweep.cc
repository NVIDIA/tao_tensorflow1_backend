// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#define EIGEN_USE_THREADS
#include <float.h>
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cluster_one_sweep.h"

// We should register only once (CPU)
REGISTER_OP("ClusterOneSweep")
    .Input("encoded_blobs: float")
    .Input("cluster_id_gt: int32")
    .Input("cluster_mask: int32")
    .Output("cluster_id_map: int32")
    .Output("cluster_color_map: int32")
    .Output("cluster_id_map_vote: int32")
    .Output("cluster_color_map_vote: int32")
    .Output("precision_recall: float")
    .Attr("up_scale_factor: int")
    .Attr("cluster_radius: float")
    .Attr("cluster_distance: float")
    .Attr("thre_min_mask: float")
    .Attr("thre_max_mask: float")
    .Attr("thre_min_vote: float")
    .Attr("thre_max_vote: float")
    .SetShapeFn([](::shape_inference::InferenceContext* c) {
        return ::shape_inference::UnchangedShapeWithRankAtLeast(c, 4);
    })
    .Doc(R"doc(
      Operator that draws clustering results, including cluster ids and cluster colormaps for
      the entire field-of-view (FOV) or only voted points from previous decoder outputs.
      Basically points in the FOV are selected one by one, and added to the same cluster
      (if the distance to existing cluster centers is smaller than cluster_radius) or added to a
      new cluster (if the distance to existing cluster centers is greater than cluster_distance).
      Support at least 3 raw clustering channels (channel IDs are defined in constants).

      References:
        [1] https://confluence.nvidia.com/pages/viewpage.action?pageId=154634270

      Arguments:
        encoded_blobs: a fp32 tensor with shape 'NCHW' used for clustering.
            N: batch size, C: number of clustering channels, H: height, W:width.

        cluster_id_gt: an int32 tensor with shape 'NHWC' indicating the ground truth
            clustering result (C=1).

        cluster_mask: an int32 tensor with shape 'NHWC' indicating the points to be
            clustered (C=1 indicates the color channel).

      Outputs:
        cluster_id_map: a 4D int32 tensor containing the cluster ids with shape
            `(batch_size, input_height, input_width, 1)`.
        cluster_color_map: a 4D int32 tensor containing the colormaps indicating
            different clusters with shape `(batch_size, input_height, input_width,
            channels)`
        cluster_id_map_vote: a 4D int32 tensor containing the cluster ids with shape
            `(batch_size, target_height, target_width, 1)` for decoded (voted) points
            only.
        cluster_color_map_vote: a 4D int32 tensor containing the colormaps indicating
            different clusters with shape `(batch_size, target_height, target_width,
            channels)` for decoded (voted) points only.
        cluster_pr: a 4D float32 tensor containing the precision and recall of
            clustering with shape (batch_size, 1, 1, 2).

      Attributes:
        cluster_radius: radius of each cluster. If the distance between two points
            is greater than this value, then they don't belong to the same cluster.
        cluster_distance: distance between two clusters. If the distance between
            two points is greater than this value, then create a new cluster.
        thre_min_mask: Minimum number of points for valid clusters (entire FOV).
        thre_max_mask: Maximum number of points for valid clusters (entire FOV).
        thre_min_vote: Minimum number of points for valid clusters (decoded points only).
        thre_max_vote: Maximum number of points for valid clusters (decoded points only).

      )doc");

class ClusterOneSweep : public _ClusterOneSweep {
 protected:
    // Perform one-sweep clustering.
    void _render_cluster_one_sweep(const float* encoded_blobs_start, const int* cluster_mask_start,
                                   const int up_scale_factor, const bool flag_check_mask) {
        PointNf p = PointNf(num_pred_channels_);
        const int output_width = width_ * up_scale_factor;
        const int output_height = height_ * up_scale_factor;
        cluster_list_.clear();
        for (int i = 0; i < output_height; i++) {
            for (int j = 0; j < output_width; j++) {
                int ii = i / up_scale_factor;
                int jj = j / up_scale_factor;
                if ((flag_check_mask == false) ||
                    (_check_vote_mask(cluster_mask_start, output_width, i, j))) {
                    for (int k = 0; k < num_pred_channels_; k++) {
                        p.x_[k] = encoded_blobs_start[k * width_ * height_ + ii * width_ + jj];
                    }
                    float dist = FLT_MAX;
                    int clusterIndex = 0;
                    for (int k = 0; k < static_cast<int>(cluster_list_.size()); k++) {
                        float d = point_to_point_distance(cluster_list_[k].center, p);
                        if (d < dist) {
                            dist = d;
                            clusterIndex = k;
                        }
                    }
                    if (dist > cluster_distance_) {
                        _point_add_to_new_cluster(p);
                    } else if (dist < cluster_radius_) {
                        _point_add_to_exist_cluster(p, clusterIndex);
                    }
                }
            }
        }
    }

    // Filter out extreme cases in clustering results.
    void _render_cluster_filter(const float* encoded_blobs_start, const int* cluster_mask_start,
                                const int up_scale_factor, const bool flag_check_mask,
                                const float cluster_thres, const float background_thres) {
        PointNf p = PointNf(num_pred_channels_);
        const int output_width = width_ * up_scale_factor;
        const int output_height = height_ * up_scale_factor;
        for (int k = 0; k < static_cast<int>(cluster_list_.size()); k++) {
            int numPoints = 0;
            for (int i = 0; i < output_height; i++) {
                for (int j = 0; j < output_width; j++) {
                    int ii = i / up_scale_factor;
                    int jj = j / up_scale_factor;
                    if ((flag_check_mask == false) ||
                        (_check_vote_mask(cluster_mask_start, output_width, i, j))) {
                        for (int k = 0; k < num_pred_channels_; k++) {
                            p.x_[k] = encoded_blobs_start[k * width_ * height_ + ii * width_ + jj];
                        }
                        if (point_to_point_distance(p, cluster_list_[k].center) < cluster_radius_) {
                            numPoints += 1;
                        }
                    }
                }
            }

            // Filter 1, cluster size should not be too large (> background_thres).
            if (numPoints > background_thres) {
                cluster_list_[k].valid = false;
                continue;
            }
            // Filter 2, cluster size should not be too small (< cluster_thres).
            if (numPoints < cluster_thres) {
                cluster_list_[k].valid = false;
                continue;
            }
        }
    }

    // Render cluster ID map and cluster colormap.
    void _render_cluster_id_colormap(int* output_cluster_id_map_start,
                                     int* output_cluster_color_map_start,
                                     const float* encoded_blobs_start,
                                     const int* cluster_mask_start, const int up_scale_factor,
                                     const bool flag_check_mask) {
        PointNf p = PointNf(num_pred_channels_);
        const int output_width = width_ * up_scale_factor;
        const int output_height = height_ * up_scale_factor;
        int cluster_id = 0;
        for (int k = 0; k < static_cast<int>(cluster_list_.size()); k++) {
            if (!cluster_list_[k].valid) {
                continue;
            }
            cluster_id++;
            for (int i = 0; i < output_height; i++) {
                for (int j = 0; j < output_width; j++) {
                    int ii = i / up_scale_factor;
                    int jj = j / up_scale_factor;
                    if ((flag_check_mask == false) ||
                        (_check_vote_mask(cluster_mask_start, output_width, i, j))) {
                        for (int k = 0; k < num_pred_channels_; k++) {
                            p.x_[k] = encoded_blobs_start[k * width_ * height_ + ii * width_ + jj];
                        }

                        if (point_to_point_distance(p, cluster_list_[k].center) < cluster_radius_) {
                            output_cluster_id_map_start[i * output_width + j] = cluster_id;
                            for (int c = 0; c < 3; c++) {
                                output_cluster_color_map_start[i * output_width * 3 + j * 3 + c] =
                                    _color_transform(cluster_list_[k].center, c);
                            }
                        }
                    }
                }
            }
        }
    }

    // Core function for one-sweep clustering.
    void _render_cluster_core(int* output_cluster_id_map_start, int* output_cluster_color_map_start,
                              const float* encoded_blobs_start, const int* cluster_mask_start,
                              const int up_scale_factor, const bool flag_check_mask,
                              const float cluster_thres, const float background_thres) {
        // Step 1, one sweep to get cluster list, with center and pointCount.
        _render_cluster_one_sweep(encoded_blobs_start, cluster_mask_start, up_scale_factor,
                                  flag_check_mask);

        // Step 2, apply filters.
        _render_cluster_filter(encoded_blobs_start, cluster_mask_start, up_scale_factor,
                               flag_check_mask, cluster_thres, background_thres);

        // Step 3, render cluster IDs and color maps.
        _render_cluster_id_colormap(output_cluster_id_map_start, output_cluster_color_map_start,
                                    encoded_blobs_start, cluster_mask_start, up_scale_factor,
                                    flag_check_mask);
    }

    // Core function for evaluation (Precision and recall).
    void _compute_pr_core(int* output_cluster_id_map_start, int* output_cluster_color_map_start,
                          float* output_binary_pr_start, const int* cluster_id_gt_start) {
        int max_cluster_id_pred = 0;
        int max_cluster_id_gt = 0;
        int area_pred = 0;
        int area_gt = 0;

        // Get number of clusters info and total cluster areas.
        for (int i = 0; i < height_; i++) {
            for (int j = 0; j < width_; j++) {
                int cur_id_gt = cluster_id_gt_start[i * width_ + j];
                int cur_id_pred = output_cluster_id_map_start[i * width_ + j];
                if (cur_id_pred > 0) {
                    area_pred++;
                }
                if (cur_id_gt > 0) {
                    area_gt++;
                }
                if (cur_id_pred > max_cluster_id_pred) {
                    max_cluster_id_pred = cur_id_pred;
                }
                if (cur_id_gt > max_cluster_id_gt) {
                    max_cluster_id_gt = cur_id_gt;
                }
            }
        }

        // Do calculations only when both the predicted and the ground truth clusters exist.
        if (max_cluster_id_gt > 0 && max_cluster_id_pred > 0) {
            std::vector<int> cluster_gt_area(max_cluster_id_gt, 0);
            std::vector<int> cluster_gt_id(max_cluster_id_gt, 0);

            for (int i = 0; i < height_; i++) {
                for (int j = 0; j < width_; j++) {
                    int cur_id_gt = cluster_id_gt_start[i * width_ + j];
                    if (cur_id_gt > 0) {
                        cluster_gt_area[cur_id_gt - 1] += 1;
                    }
                }
            }

            // Sort cluster_id_gt (ranking areas from large to small).
            std::vector<std::pair<int, int>> cluster_gt_pair;
            for (int i = 0; i < max_cluster_id_gt; i++) {
                cluster_gt_pair.push_back({i, cluster_gt_area[i]});
            }
            std::sort(cluster_gt_pair.begin(), cluster_gt_pair.end(), _compare_func);

            for (int i = 0; i < max_cluster_id_gt; i++) {
                cluster_gt_id[i] = cluster_gt_pair[i].first;
            }

            // Get overlap area (here list_area_overlap is representing a 2D matrix).
            std::vector<int> list_area_overlap(max_cluster_id_pred * max_cluster_id_gt, 0);
            for (int i = 0; i < height_; i++) {
                for (int j = 0; j < width_; j++) {
                    int cur_id_gt = cluster_id_gt_start[i * width_ + j] - 1;
                    int cur_id_pred = output_cluster_id_map_start[i * width_ + j] - 1;
                    if (cur_id_gt >= 0 && cur_id_pred >= 0) {
                        list_area_overlap[cur_id_gt * max_cluster_id_pred + cur_id_pred] += 1;
                    }
                }
            }

            /* Find the biggest overlap which means at these pixels,
            the predicted clusters match with the ground truth. */
            std::vector<int> cluster_pred_id;
            for (int i = 0; i < max_cluster_id_gt; i++) {
                int gt_id = cluster_gt_id[i];
                int pred_id = 0;
                int area_tmp = 0;
                for (int j = 0; j < max_cluster_id_pred; j++) {
                    if (list_area_overlap[gt_id * max_cluster_id_pred + j] > area_tmp) {
                        area_tmp = list_area_overlap[gt_id * max_cluster_id_pred + j];
                        pred_id = j;
                    }
                }
                // To avoid the same pred_cluster being counted twice.
                for (int j = 0; j < static_cast<int>(cluster_pred_id.size()); j++) {
                    if (cluster_pred_id[j] == pred_id) {
                        pred_id = 0;
                        break;
                    }
                }
                cluster_pred_id.push_back(pred_id);
            }

            // Get the total overlapped area, which is the numerator of the calculation.
            int area_overlap = 0;
            for (int i = 0; i < max_cluster_id_gt; i++) {
                area_overlap +=
                    list_area_overlap[cluster_gt_id[i] * max_cluster_id_pred + cluster_pred_id[i]];
            }
            output_binary_pr_start[0] = static_cast<float>(area_overlap) / std::max(area_pred, 1);
            output_binary_pr_start[1] = static_cast<float>(area_overlap) / std::max(area_gt, 1);
        } else if (max_cluster_id_gt == 0 && max_cluster_id_pred == 0) {
            output_binary_pr_start[0] = 1.0;
            output_binary_pr_start[1] = 1.0;
        } else {
            output_binary_pr_start[0] = 0.0;
            output_binary_pr_start[1] = 0.0;
        }
    }

 public:
    explicit ClusterOneSweep(OpKernelConstruction* context) : _ClusterOneSweep(context) {}

    // This function calls _render_cluster_core() and _compute_pr_core().
    void _cluster_core(int* output_cluster_id_map, int* output_cluster_color_map,
                       int* output_cluster_id_map_vote, int* output_cluster_color_map_vote,
                       float* output_binary_pr, const float* encoded_blobs,
                       const int* cluster_id_gt, const int* cluster_mask) {
        memset(output_cluster_id_map, 0, sizeof(int) * batch_size_ * height_ * width_);
        memset(output_cluster_color_map, 0, sizeof(int) * batch_size_ * height_ * width_ * 3);
        memset(output_cluster_id_map_vote, 0,
               sizeof(int) * batch_size_ * height_ * width_ * up_scale_factor_ * up_scale_factor_);
        memset(output_cluster_color_map_vote, 0, sizeof(int) * batch_size_ * height_ * width_ *
                                                     up_scale_factor_ * up_scale_factor_ * 3);

        for (int b = 0; b < batch_size_; b++) {
            const float* encoded_blobs_start =
                encoded_blobs + height_ * width_ * num_pred_channels_ * b;
            const int* cluster_id_gt_start = cluster_id_gt + height_ * width_ * b;
            const int* cluster_mask_start =
                cluster_mask +
                height_ * up_scale_factor_ * width_ * up_scale_factor_ * NUM_MASK_CHANNELS * b;
            int* output_cluster_id_map_start = output_cluster_id_map + height_ * width_ * b;
            int* output_cluster_color_map_start =
                output_cluster_color_map + height_ * width_ * 3 * b;
            int* output_cluster_id_map_vote_start =
                output_cluster_id_map_vote +
                height_ * width_ * up_scale_factor_ * up_scale_factor_ * b;
            int* output_cluster_color_map_vote_start =
                output_cluster_color_map_vote +
                height_ * width_ * up_scale_factor_ * up_scale_factor_ * 3 * b;
            float* output_binary_pr_start = output_binary_pr + DIM_FOR_PR * b;

            // Perform clustering for all points. Also used for evaluation (precision and recall).
            _render_cluster_core(output_cluster_id_map_start, output_cluster_color_map_start,
                                 encoded_blobs_start, cluster_mask_start, 1, false, thre_min_mask_,
                                 thre_max_mask_);
            // Perform clustering for only voted points.
            _render_cluster_core(output_cluster_id_map_vote_start,
                                 output_cluster_color_map_vote_start, encoded_blobs_start,
                                 cluster_mask_start, up_scale_factor_, true, thre_min_vote_,
                                 thre_max_vote_);
            // Evaluate based on results without votes.
            _compute_pr_core(output_cluster_id_map_start, output_cluster_color_map_start,
                             output_binary_pr_start, cluster_id_gt_start);
        }
    }
};

#pragma message("Registering CPU kernel")
REGISTER_KERNEL_BUILDER(Name("ClusterOneSweep")
                            .Device(DEVICE_CPU)
                            .HostMemory("encoded_blobs")
                            .HostMemory("cluster_id_gt")
                            .HostMemory("cluster_mask"),
                        ClusterOneSweep);
