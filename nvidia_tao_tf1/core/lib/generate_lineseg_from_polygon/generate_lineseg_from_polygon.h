// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
// This header file implements common functionality of generate_lineseg_from_polygon op in Maglev

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/logging.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "line.h"
#include "point.h"
#include "polygon.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

class _GenerateLinesegFromPolygon : public OpKernel {
 protected:
    // Number of output channels (13) is fixed in this op:
    static constexpr int NUM_OUTPUT_CHANNELS = 13;
    static constexpr int FIX_V_SHAPE_POLYGON = 1;

    // class members for op attributes
    bool verbose_;
    bool inverse_;
    uint32_t skip_conversion_class_mask_;
    uint32_t scanline_conversion_class_mask_;
    enum TYPES { POLYGON = 0, LINE_SEC = 1 };

 public:
    explicit _GenerateLinesegFromPolygon(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("verbose", &verbose_));
        int32_t tmp;
        OP_REQUIRES_OK(context, context->GetAttr("skip_conversion_class_mask", &tmp));
        skip_conversion_class_mask_ = static_cast<uint32_t>(tmp);

        OP_REQUIRES_OK(context, context->GetAttr("scanline_conversion_class_mask", &tmp));
        scanline_conversion_class_mask_ = static_cast<uint32_t>(tmp);
    }

    void Compute(OpKernelContext* context) override { ComputeEncode(context); }

    virtual void Convert(OpKernelContext* context, const int nvertices, const int npolygons,
                         const int batch_size, const int len_class_ids_per_polygon,
                         const int* vertex_counts_per_polygon, const int* polygons_per_image,
                         const int* class_ids_per_polygon, const bool* is_polyline_per_polygon,
                         const float* input_polygon_vertices, const int* width_data,
                         const int* height_data, std::vector<Line2<float>>* line_segments,
                         std::vector<int>* line_segments_count_per_image,
                         std::vector<int>* cluster_count_per_image) = 0;

    // image > polygon > vertex
    void ComputeEncode(OpKernelContext* context) {
        //
        // Grab the input tensor
        //
        // widths
        const Tensor& width_tensor = context->input(0);
        auto width_data = width_tensor.flat<int>();

        // heights
        const Tensor& height_tensor = context->input(1);
        auto height_data = height_tensor.flat<int>();

        // all the vertices of polygons in all images
        const Tensor& polygon_vertices_tensor = context->input(2);
        auto input_polygon_vertices = polygon_vertices_tensor.flat<float>();

        const Tensor& vertex_counts_per_polygon_tensor = context->input(3);
        auto vertex_counts_per_polygon = vertex_counts_per_polygon_tensor.flat<int>();

        const Tensor& class_ids_per_polygon_tensor = context->input(4);
        auto class_ids_per_polygon = class_ids_per_polygon_tensor.flat<int>();

        // number of polygons for each image
        const Tensor& polygons_per_image_tensor = context->input(5);
        auto polygons_per_image = polygons_per_image_tensor.flat<int>();

        const Tensor& is_polyline_per_polygon_tensor = context->input(6);
        auto is_polyline_per_polygon = is_polyline_per_polygon_tensor.flat<bool>();

        OP_REQUIRES(context, 2 == polygon_vertices_tensor.shape().dims(),
                    errors::InvalidArgument("polygon tensor must have 2 dimensions,", "shape is: ",
                                            polygon_vertices_tensor.shape().DebugString(), "."));
        OP_REQUIRES(context, 1 == vertex_counts_per_polygon_tensor.shape().dims(),
                    errors::InvalidArgument(
                        "vertex_counts_per_polygon must be a 1 dimensional vector,", " shape is: ",
                        vertex_counts_per_polygon_tensor.shape().DebugString(), "."));
        OP_REQUIRES(context, 1 == class_ids_per_polygon_tensor.shape().dims(),
                    errors::InvalidArgument(
                        "class_ids_per_polygon must be a 1 dimensional vector,", " shape is: ",
                        class_ids_per_polygon_tensor.shape().DebugString(), "."));
        OP_REQUIRES(context, 1 == is_polyline_per_polygon_tensor.shape().dims(),
                    errors::InvalidArgument(
                        "is_polyline_per_polygon must be a 1 dimensional vector,", " shape is: ",
                        is_polyline_per_polygon_tensor.shape().DebugString(), "."));
        OP_REQUIRES(context, 1 == polygons_per_image_tensor.shape().dims(),
                    errors::InvalidArgument("polygons_per_image must be a 1 dimensional vector,",
                                            " shape is: ",
                                            polygons_per_image_tensor.shape().DebugString(), "."));

        // total number of vertices of all polygons in all images
        int nvertices = polygon_vertices_tensor.shape().dim_size(0);

        // total number of polygons in all images
        int npolygons = vertex_counts_per_polygon_tensor.shape().dim_size(0);

        // total number of lane class IDs
        int len_class_ids_per_polygon = class_ids_per_polygon_tensor.shape().dim_size(0);

        // total number of lane is_polyline flags.
        int len_is_polyline_per_polygon = is_polyline_per_polygon_tensor.shape().dim_size(0);

        OP_REQUIRES(
            context, npolygons == len_class_ids_per_polygon,
            errors::InvalidArgument("vertex_counts_per_polygon vector and ",
                                    "class_ids_per_polygon ", "vector shapes are not equal; ",
                                    vertex_counts_per_polygon_tensor.shape().DebugString(), " and ",
                                    class_ids_per_polygon_tensor.shape().DebugString(), "."));

        OP_REQUIRES(
            context, npolygons == len_is_polyline_per_polygon,
            errors::InvalidArgument("vertex_counts_per_polygon vector and ",
                                    "is_polyline_per_polygon ", "vector shapes are not equal; ",
                                    vertex_counts_per_polygon_tensor.shape().DebugString(), " and ",
                                    is_polyline_per_polygon_tensor.shape().DebugString(), "."));

        OP_REQUIRES(
            context, nvertices >= npolygons,
            errors::InvalidArgument("Number of polygons ", npolygons,
                                    " is larger than the number of vertices ", nvertices, "."));

        // total number of images
        int batch_size = polygons_per_image_tensor.shape().dim_size(0);

        // Check if number of polygons in the image match number of polygons in the batch
        if (batch_size == 1) {
            OP_REQUIRES(context, npolygons == polygons_per_image.data()[0],
                        errors::InvalidArgument("Number of polygons ", npolygons, " not matching ",
                                                polygons_per_image.data()[0], "."));
        }

        if (verbose_) {
            std::cout << "Number of images: " << batch_size << std::endl;
        }
        std::vector<Line2<float>> line_segment;
        std::vector<int> line_segments_count_per_image;
        std::vector<int> cluster_count_per_image;

        Convert(context, nvertices, npolygons, batch_size, len_class_ids_per_polygon,
                vertex_counts_per_polygon.data(), polygons_per_image.data(),
                class_ids_per_polygon.data(), is_polyline_per_polygon.data(),
                input_polygon_vertices.data(), width_data.data(), height_data.data(), &line_segment,
                &line_segments_count_per_image, &cluster_count_per_image);
        OP_REQUIRES(context,
                    line_segments_count_per_image.size() == static_cast<size_t>(batch_size),
                    errors::InvalidArgument("Number of images ", batch_size, " should be equal to ",
                                            line_segments_count_per_image.size(), "."));
        OP_REQUIRES(context, cluster_count_per_image.size() == static_cast<size_t>(batch_size),
                    errors::InvalidArgument("Number of images ", batch_size, " should be equal to ",
                                            cluster_count_per_image.size(), "."));

        // Create an output tensor
        int total_line_segments = line_segment.size();

        Tensor* output_tensor;
        // 1st output blob
        TensorShape output_shape0({batch_size});
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape0, &output_tensor));

        auto output0 = output_tensor->template flat<int>();

        // 2nd output blob
        // expected output channels orders are:
        // x1, y1, x2, y2, angle, angle_top, angle_bottom, classid, clusterid,
        // width_left_top, width_right_top, width_left_bottom, width_right_bottom
        TensorShape output_shape1({total_line_segments, NUM_OUTPUT_CHANNELS});
        OP_REQUIRES_OK(context, context->allocate_output(1, output_shape1, &output_tensor));
        auto output1 = output_tensor->template flat<float>();

        OP_REQUIRES_OK(context, context->allocate_output(2, output_shape0, &output_tensor));
        auto cluster_count_per_img = output_tensor->template flat<int>();

        // fill the content.
        CopyToTensor(output0.data(), cluster_count_per_img.data(), batch_size, output1.data(),
                     total_line_segments, NUM_OUTPUT_CHANNELS, line_segments_count_per_image,
                     cluster_count_per_image, line_segment);
    }

    void CopyToTensor(int* output0, int* cluster_count_per_img, int cols0, float* output1,
                      int rows1, int cols1, const std::vector<int>& line_segments_count_per_image,
                      const std::vector<int>& cluster_count_per_image,
                      const std::vector<Line2<float>>& line_segments) {
        for (size_t i = 0; i < line_segments_count_per_image.size(); i++) {
            output0[i] = line_segments_count_per_image[i];
            cluster_count_per_img[i] = cluster_count_per_image[i];
        }
        for (size_t i = 0; i < line_segments.size(); i++) {
            output1[i * cols1] = line_segments[i].top_.x;
            output1[i * cols1 + 1] = line_segments[i].top_.y;
            output1[i * cols1 + 2] = line_segments[i].bottom_.x;
            output1[i * cols1 + 3] = line_segments[i].bottom_.y;
            output1[i * cols1 + 4] = line_segments[i].angle_;
            output1[i * cols1 + 5] = line_segments[i].angle_top_;
            output1[i * cols1 + 6] = line_segments[i].angle_bottom_;
            output1[i * cols1 + 7] = line_segments[i].class_id_;
            output1[i * cols1 + 8] = line_segments[i].cluster_id_;
            output1[i * cols1 + 9] = line_segments[i].width_left_top_;
            output1[i * cols1 + 10] = line_segments[i].width_right_top_;
            output1[i * cols1 + 11] = line_segments[i].width_left_bottom_;
            output1[i * cols1 + 12] = line_segments[i].width_right_bottom_;
        }
    }
};
