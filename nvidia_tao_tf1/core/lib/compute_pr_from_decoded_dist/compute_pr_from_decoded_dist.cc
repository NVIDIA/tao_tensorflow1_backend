// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

#undef EIGEN_USE_GPU
#include <float.h>

#include "compute_pr_from_decoded_dist.h"

// We should register only once (CPU)
REGISTER_OP("ComputePRFromDecodedDist")
    .Input("label: int32")
    .Input("prediction: int32")
    .Input("input_image: int32")
    .Output("precision_recall: float")
    .Output("precision_recall_per_distance: float")
    .Output("multiclass_precision_recall: float")
    .Output("multiclass_precision_recall_per_distance: float")
    .Output("input_image_with_stat: int32")
    .Attr("n_classes: int")
    .Attr("height: int")
    .Attr("width: int")
    .Attr("draw_metrics: bool")
    .Attr("search_radius: int")
    .Attr("bottom_ratio: float")
    .Attr("top_ratio: float")
    .Attr("verbose: bool = false")
    .SetShapeFn([](::shape_inference::InferenceContext* c) {
        // Inherit the output shape from the attributes.
        int n_classes;
        TF_RETURN_IF_ERROR(c->GetAttr("n_classes", &n_classes));
        std::vector<::shape_inference::DimensionHandle> dims_out;

        // Batch_size x 1 x 1 x 2 tensor.
        dims_out.push_back(c->UnknownDim());
        dims_out.push_back(c->MakeDim(1));
        dims_out.push_back(c->MakeDim(1));
        dims_out.push_back(c->MakeDim(DIM_FOR_PR));
        c->set_output(0, c->MakeShape(dims_out));

        // Batch_size x 3 (regions) x 1 x 2 tensor.
        dims_out[1] = c->MakeDim(N_REGIONS);
        c->set_output(1, c->MakeShape(dims_out));

        // Batch_size x 1 x n_classes x 2 tensor.
        dims_out[1] = c->MakeDim(1);
        dims_out[2] = c->MakeDim(n_classes);
        c->set_output(2, c->MakeShape(dims_out));

        // Batch_size x 3 (regions) x n_classes x 2 tensor.
        dims_out[1] = c->MakeDim(N_REGIONS);
        c->set_output(3, c->MakeShape(dims_out));
        return Status::OK();
    })
    .Doc(R"doc(
        Compute PR (Precision Recall) From Decoded Dist Op.

        Summary:
            Provided decoded distance blob from label and predictions, this operator
            computes custom precision and recall, also precisions recall by distance
            regions, by classes. Optionally, it draws the metrics with decoded stat on images
            for visualization. (For details on decoded blobs please see Decode Dist Op Docs.)

            The operator takes pixels from label and prediction, within search radius of
            choice, to compare the class id from label and prediction. It accumulates true
            positive(tp) counts (matched from both label and prediction), pixels from groundtruth
            not found in prediction (false negatives), pixels from prediction not found in
            groundtruth (false positives), etc. Then the operator computes precision and recall.
            Similarly, metrics are computed by class id, or by distance range or
            by combination of class id and distance range.

            This operator also provides an option(`draw_metrics`) to visualize metrics over `input_image`.

        References:
            [1] https://confluence.nvidia.com/display/AV/Line+Regressor+Encoding

        Arguments:
            label: an int32 tensor ('NCHW', N:batch size, C:channels, H:height, W:width, C=1)
                of ground-truth pixels with class id for label type.
            prediction: a int32 ('NCHW', C=1) tensor of predicted pixels with class id
                for label type.
            input_image: an int32 ('NHWC', C=3) tensor of input images to draw metrics
                for visualization. Precision and recall for first 10 classes will be drawn.

        Attributes:
            n_classes: number of classes for labels. Must be positive.
            width: input label width.
            height: input label height.
            draw_metrics: if enabled, this op will write the following information on the image.
                `CA:P:70.00 R:90.00`
                `C0:P:50.00 R:80.00`
                `C1:P:30.00 R:70.00` on upper left corner output image.
                `CA:P:70.00 R:90.00` means overall precision=0.7, recall=0.9.
                `C0:P:50.00 R:80.00` means class 1 has precision=0.5, recall=0.8.
                Notice that the legend 'C0' corresponds to class id=1, 'C1' corresponds to class
                id=2, in general, `CN` corresponds to class id=N+1.
                If disabled, the output of the image visualisation tensor is undefined (garbage).
            search_radius: radius to search around the ground-truth label pixels for evaluation.
            bottom_ratio: ratio (from 0.0 to 1.0) to define distance ranges (of image height).
                Ranges are defined as:
                1. (bottom_ratio*height, height]
                2. (top_ratio*height, bottom_ratio*height]
                3. [0, top_ratio*height]
                Metrics would be evaluated by these ranges based on pixel location in y-coordinates.
            top_ratio: see above.
            verbose: if enabled, debug information will be printed.

        Returns:
            precision_recall: a fp32 of shape batch_size x 1 x 1 x 2 tensor with
                overall precision and recall.
            precision_recall_per_distance: a fp32 of shape batch_size x 3 (regions) x 1 x 2 tensor of
                precision and recall by different distance regions.
                (see bottom_ratio, top_ratio).
            multiclass_precision_recall: a fp32 of shape batch_size x 1 x n_classes x 2 tensor
                with precision and recall by different classes.
            multiclass_precision_recall_per_distance:  a fp32 of shape
                batch_size x 3 (regions) x  n_classes x 2 tensor precision recall
                by different combinations of classes and distance regions.
            input_image_with_stat: an int32 image of shape  batch_size x height x width x 3 tensor
                to have metrics drawn on if turn on `draw_metricss`, if `draw_metricss` is false,
                the image tensor output is undefined (garbage).
        )doc");

class ComputePRFromDecodedDist : public _ComputePRFromDecodedDist {
 public:
    explicit ComputePRFromDecodedDist(OpKernelConstruction* context)
        : _ComputePRFromDecodedDist(context) {}
    void evaluate_core(float* binary_pr, float* binary_pr_per_distance, float* multiclass_pr,
                       float* multiclass_pr_per_distance, const int* tensor_GT_blobs,
                       const int* tensor_pred_blobs, const int height, const int width,
                       const int search_radius) {
        // From point of prediction we check.
        uint32_t n_TP_for_precision = 0;
        uint32_t n_TP_for_recall = 0;

        uint32_t n_total_prediction = 0;
        uint32_t n_total_gt = 0;

        std::unique_ptr<uint32_t[]> multiclass_TP_for_precision;
        std::unique_ptr<uint32_t[]> multiclass_TP_for_recall;
        std::unique_ptr<uint32_t[]> binary_per_distance_TP_for_precision;
        std::unique_ptr<uint32_t[]> multiclass_per_distance_TP_for_precision;
        std::unique_ptr<uint32_t[]> binary_per_distance_TP_for_recall;
        std::unique_ptr<uint32_t[]> multiclass_per_distance_TP_for_recall;

        std::unique_ptr<uint32_t[]> total_prediction_per_distance;
        std::unique_ptr<uint32_t[]> total_GT_per_distance;

        std::unique_ptr<uint32_t[]> multiclass_total_prediction;
        std::unique_ptr<uint32_t[]> multiclass_total_GT;

        std::unique_ptr<uint32_t[]> multiclass_total_predictionper_distance;
        std::unique_ptr<uint32_t[]> multiclass_total_GTper_distance;

        binary_per_distance_TP_for_precision.reset(new uint32_t[N_REGIONS]);
        multiclass_per_distance_TP_for_precision.reset(new uint32_t[N_REGIONS * n_classes_]);

        binary_per_distance_TP_for_recall.reset(new uint32_t[N_REGIONS]);
        multiclass_per_distance_TP_for_recall.reset(new uint32_t[N_REGIONS * n_classes_]);

        multiclass_TP_for_precision.reset(new uint32_t[n_classes_]);
        multiclass_TP_for_recall.reset(new uint32_t[n_classes_]);

        multiclass_total_prediction.reset(new uint32_t[n_classes_]);
        multiclass_total_GT.reset(new uint32_t[n_classes_]);

        total_prediction_per_distance.reset(new uint32_t[N_REGIONS]);
        multiclass_total_predictionper_distance.reset(new uint32_t[N_REGIONS * n_classes_]);

        total_GT_per_distance.reset(new uint32_t[N_REGIONS]);
        multiclass_total_GTper_distance.reset(new uint32_t[N_REGIONS * n_classes_]);

        compute_core(&n_total_prediction, total_prediction_per_distance.get(),
                     multiclass_total_prediction.get(),
                     multiclass_total_predictionper_distance.get(), &n_TP_for_precision,
                     binary_per_distance_TP_for_precision.get(), multiclass_TP_for_precision.get(),
                     multiclass_per_distance_TP_for_precision.get(), tensor_pred_blobs,
                     tensor_GT_blobs, bottom_ratio_, top_ratio_, height, width, search_radius,
                     n_classes_);

        // Precision.
        if (n_total_prediction > 0) {
            binary_pr[0] = static_cast<float>(n_TP_for_precision) / n_total_prediction;
        } else {
            binary_pr[0] = 1;
        }

        // Precision per distance.
        for (int i = 0; i < N_REGIONS; i++) {
            if (total_prediction_per_distance[i] > 0) {
                binary_pr_per_distance[i * DIM_FOR_PR] =
                    static_cast<float>(binary_per_distance_TP_for_precision[i]) /
                    total_prediction_per_distance[i];
            } else {
                binary_pr_per_distance[i * DIM_FOR_PR] = 1;
            }
        }
        // Precision per class.
        for (int i = 0; i < n_classes_; i++) {
            if (multiclass_total_prediction[i] > 0) {
                multiclass_pr[i * DIM_FOR_PR] = static_cast<float>(multiclass_TP_for_precision[i]) /
                                                multiclass_total_prediction[i];
            } else {
                multiclass_pr[i * DIM_FOR_PR] = 1;
            }
        }
        // Precision per class per distance.
        for (int j = 0; j < N_REGIONS; j++) {
            for (int i = 0; i < n_classes_; i++) {
                if (multiclass_total_predictionper_distance[j * n_classes_ + i] > 0) {
                    multiclass_pr_per_distance[j * DIM_FOR_PR * n_classes_ + i * DIM_FOR_PR] =
                        static_cast<float>(
                            multiclass_per_distance_TP_for_precision[j * n_classes_ + i]) /
                        multiclass_total_predictionper_distance[j * n_classes_ + i];
                } else {
                    multiclass_pr_per_distance[j * DIM_FOR_PR * n_classes_ + i * DIM_FOR_PR] = 1;
                }
            }
        }

        compute_core(&n_total_gt, total_GT_per_distance.get(), multiclass_total_GT.get(),
                     multiclass_total_GTper_distance.get(), &n_TP_for_recall,
                     binary_per_distance_TP_for_recall.get(), multiclass_TP_for_recall.get(),
                     multiclass_per_distance_TP_for_recall.get(), tensor_GT_blobs,
                     tensor_pred_blobs, bottom_ratio_, top_ratio_, height, width, search_radius,
                     n_classes_);

        // Recall.
        if (n_total_gt > 0) {
            binary_pr[1] = static_cast<float>(n_TP_for_recall) / n_total_gt;
        } else {
            binary_pr[1] = 1;
        }

        // Recall per distance.
        for (int i = 0; i < N_REGIONS; i++) {
            if (total_GT_per_distance[i] > 0) {
                binary_pr_per_distance[i * DIM_FOR_PR + 1] =
                    static_cast<float>(binary_per_distance_TP_for_recall[i]) /
                    total_GT_per_distance[i];
            } else {
                binary_pr_per_distance[i * DIM_FOR_PR + 1] = 1;
            }
        }
        // Recall per class.
        for (int i = 0; i < n_classes_; i++) {
            if (multiclass_total_GT[i] > 0) {
                multiclass_pr[i * DIM_FOR_PR + 1] =
                    static_cast<float>(multiclass_TP_for_recall[i]) / multiclass_total_GT[i];
            } else {
                multiclass_pr[i * DIM_FOR_PR + 1] = 1;
            }
        }
        // Recall per class per distance.
        for (int j = 0; j < N_REGIONS; j++) {
            for (int i = 0; i < n_classes_; i++) {
                if (multiclass_total_GTper_distance[j * n_classes_ + i] > 0) {
                    multiclass_pr_per_distance[j * DIM_FOR_PR * n_classes_ + i * DIM_FOR_PR + 1] =
                        static_cast<float>(
                            multiclass_per_distance_TP_for_recall[j * n_classes_ + i]) /
                        multiclass_total_GTper_distance[j * n_classes_ + i];
                } else {
                    multiclass_pr_per_distance[j * DIM_FOR_PR * n_classes_ + i * DIM_FOR_PR + 1] =
                        1;
                }
            }
        }
    }

    void compute_core(uint32_t* total_count, uint32_t* total_count_per_distance,
                      uint32_t* multiclass_total_count,
                      uint32_t* multiclass_total_count_per_distance, uint32_t* binary_tp,
                      uint32_t* binary_tp_per_distance, uint32_t* multiclass_tp,
                      uint32_t* multiclass_tp_per_distance, const int* probing_blob,
                      const int* search_target_blob, const float bottom_ratio,
                      const float top_ratio, const int height, const int width,
                      const int search_radius, const int n_classes) {
        int quad_radius = search_radius * search_radius;
        // Len 1: total count.
        total_count[0] = 0;
        // Count per region.
        memset(total_count_per_distance, 0, sizeof(uint32_t) * N_REGIONS);
        // Count per class.
        memset(multiclass_total_count, 0, sizeof(uint32_t) * n_classes);
        // Count per class and per distance.
        memset(multiclass_total_count_per_distance, 0, sizeof(uint32_t) * N_REGIONS * n_classes);

        binary_tp[0] = 0;
        memset(multiclass_tp, 0, sizeof(uint32_t) * n_classes);
        memset(binary_tp_per_distance, 0, sizeof(uint32_t) * N_REGIONS);
        memset(multiclass_tp_per_distance, 0, sizeof(uint32_t) * N_REGIONS * n_classes);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int probing_class_id = probing_blob[y * width + x];
                if (probing_class_id > 0) {
                    // Total valid pixel count.
                    // This will be denominator for precision and recall.
                    total_count[0]++;
                    // Check before setting multiclass_total_count because prediction is sometimes
                    // more than n_classes
                    if (probing_class_id <= n_classes)
                        multiclass_total_count[probing_class_id - 1]++;
                    if (y > bottom_ratio * height) {
                        total_count_per_distance[0]++;
                        if (probing_class_id <= n_classes)
                            multiclass_total_count_per_distance[probing_class_id - 1]++;
                    } else if (y > top_ratio * height) {
                        total_count_per_distance[1]++;
                        if (probing_class_id <= n_classes)
                            multiclass_total_count_per_distance[n_classes + probing_class_id - 1]++;
                    } else {
                        total_count_per_distance[2]++;
                        if (probing_class_id <= n_classes)
                            multiclass_total_count_per_distance[2 * n_classes + probing_class_id -
                                                                1]++;
                    }
                    bool found = false;
                    bool class_found = false;
                    bool stop_flag = (found && class_found);
                    for (int dx = -search_radius; dx <= search_radius && stop_flag == false; dx++) {
                        for (int dy = -search_radius; dy <= search_radius && stop_flag == false;
                             dy++) {
                            if (dx * dx + dy * dy <= quad_radius) {
                                if (x + dx >= 0 && y + dy >= 0 && x + dx < width &&
                                    y + dy < height) {
                                    int search_target_class_id =
                                        search_target_blob[(y + dy) * width + x + dx];
                                    if (search_target_class_id > 0) {
                                        found = true;
                                    }
                                    if (probing_class_id == search_target_class_id) {
                                        class_found = true;
                                    }
                                }
                            }
                            stop_flag = (found & class_found);
                        }
                    }
                    if (found) {
                        binary_tp[0]++;
                        if (y > bottom_ratio * height) {
                            binary_tp_per_distance[0]++;
                        } else if (y > top_ratio * height) {
                            binary_tp_per_distance[1]++;
                        } else {
                            binary_tp_per_distance[2]++;
                        }
                    }
                    if (class_found) {
                        multiclass_tp[probing_class_id - 1]++;
                        if (y > bottom_ratio * height) {
                            multiclass_tp_per_distance[probing_class_id - 1]++;
                        } else if (y > top_ratio * height) {
                            multiclass_tp_per_distance[n_classes + probing_class_id - 1]++;
                        } else {
                            multiclass_tp_per_distance[2 * n_classes + probing_class_id - 1]++;
                        }
                    }
                }
            }
        }
    }

    void evaluate(OpKernelContext* context, const int batch_size, const int* tensor_GT_blobs,
                  const int* tensor_pred_blobs, const int* input_image, float* binary_pr,
                  float* binary_pr_per_distance, float* multiclass_pr,
                  float* multiclass_pr_per_distance, int* input_image_with_stat) {
        for (int32_t i = 0; i < batch_size * DIM_FOR_PR; i++) binary_pr[i] = 0.0f;
        for (int32_t i = 0; i < batch_size * N_REGIONS * DIM_FOR_PR; i++)
            binary_pr_per_distance[i] = 0.0f;
        for (int32_t i = 0; i < batch_size * n_classes_ * DIM_FOR_PR; i++) multiclass_pr[i] = 0.0f;
        for (int32_t i = 0; i < batch_size * N_REGIONS * n_classes_ * DIM_FOR_PR; i++)
            multiclass_pr_per_distance[i] = 0.0f;
        if (draw_metrics_) {
            memcpy(input_image_with_stat, input_image,
                   sizeof(int) * batch_size * height_ * width_ * 3);
        }
        Color recall_color(0, 255, 0);
        Color prec_color(255, 0, 0);
        Color bg_color(0, 0, 0);
        Color class_color(255, 255, 0);
        const float scale = 1.5f;
        const float thickness = 1.5f;
        const float class_scale = 1.2f;
        const float class_thickness = 1.3f;
        for (int i = 0; i < batch_size; i++) {
            const int* gt_start = tensor_GT_blobs + height_ * width_ * i;
            const int* pred_start = tensor_pred_blobs + height_ * width_ * i;
            float* binary_pr_start = binary_pr + i * DIM_FOR_PR;

            float* binary_pr_per_distance_start =
                binary_pr_per_distance + i * DIM_FOR_PR * N_REGIONS;

            float* multiclass_pr_start = multiclass_pr + i * DIM_FOR_PR * n_classes_;

            float* multiclass_pr_per_distance_start =
                multiclass_pr_per_distance + i * DIM_FOR_PR * n_classes_ * N_REGIONS;

            evaluate_core(binary_pr_start, binary_pr_per_distance_start, multiclass_pr_start,
                          multiclass_pr_per_distance_start, gt_start, pred_start, height_, width_,
                          search_radius_);
            // We can draw statistics on image, as option.
            if (draw_metrics_) {
                int* input_image_with_stat_start = input_image_with_stat + height_ * width_ * i * 3;

                // Each drawing function would return pixel location
                // starting from `original_offset` so that
                // all the drawings are nicely spaced.
                Point<float> original_offset(10, 10);
                Point<float> offset = original_offset;

                // Write precision first.
                Point<float> pixel_loc =
                    draw_apis_.drawClassId(input_image_with_stat_start, height_, width_, offset,
                                           scale, thickness, bg_color, -1);
                draw_apis_.drawClassId(input_image_with_stat_start, height_, width_, offset, scale,
                                       thickness - 1, class_color, -1);
                offset.x_ += pixel_loc.x_;
                pixel_loc = draw_apis_.drawPrecision(input_image_with_stat_start, height_, width_,
                                                     binary_pr_start[0], offset, scale, thickness,
                                                     bg_color);
                draw_apis_.drawPrecision(input_image_with_stat_start, height_, width_,
                                         binary_pr_start[0], offset, scale, thickness - 1,
                                         prec_color);
                offset.x_ += pixel_loc.x_;
                // Write recall.
                pixel_loc =
                    draw_apis_.drawRecall(input_image_with_stat_start, height_, width_,
                                          binary_pr_start[1], offset, scale, thickness, bg_color);
                draw_apis_.drawRecall(input_image_with_stat_start, height_, width_,
                                      binary_pr_start[1], offset, scale, thickness - 1,
                                      recall_color);
                // New line and get back to original x.
                offset.y_ += pixel_loc.y_;
                offset.x_ = original_offset.x_;

                // Visualize metrics with classes.
                for (int c = 0; c < n_classes_; ++c) {
                    pixel_loc =
                        draw_apis_.drawClassId(input_image_with_stat_start, height_, width_, offset,
                                               class_scale, class_thickness, bg_color, 1);
                    draw_apis_.drawClassId(input_image_with_stat_start, height_, width_, offset,
                                           class_scale, class_thickness - 1, class_color, c);
                    offset.x_ += pixel_loc.x_;
                    // Write precision for current class id.
                    draw_apis_.drawPrecision(input_image_with_stat_start, height_, width_,
                                             multiclass_pr_start[c * DIM_FOR_PR], offset,
                                             class_scale, class_thickness, bg_color);
                    pixel_loc =
                        draw_apis_.drawPrecision(input_image_with_stat_start, height_, width_,
                                                 multiclass_pr_start[c * DIM_FOR_PR], offset,
                                                 class_scale, class_thickness - 1, prec_color);
                    offset.x_ += pixel_loc.x_;
                    // Write recall for current class id.
                    draw_apis_.drawRecall(input_image_with_stat_start, height_, width_,
                                          multiclass_pr_start[c * DIM_FOR_PR + 1], offset,
                                          class_scale, class_thickness, bg_color);
                    pixel_loc =
                        draw_apis_.drawRecall(input_image_with_stat_start, height_, width_,
                                              multiclass_pr_start[c * DIM_FOR_PR + 1], offset,
                                              class_scale, class_thickness - 1, recall_color);
                    // New line and get back to original x.
                    offset.y_ += pixel_loc.y_;
                    offset.x_ = original_offset.x_;
                }
            }
        }
    }

 protected:
    DrawCharacters draw_apis_;
};

REGISTER_KERNEL_BUILDER(Name("ComputePRFromDecodedDist")
                            .Device(DEVICE_CPU)
                            .HostMemory("label")
                            .HostMemory("prediction")
                            .HostMemory("input_image"),
                        ComputePRFromDecodedDist);
