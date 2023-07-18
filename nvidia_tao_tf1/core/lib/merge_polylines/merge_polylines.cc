// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#include <iostream>
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

// We should register only once (CPU)
REGISTER_OP("MergePolylines")
    .Input("polyline_vertices: float")
    .Input("vertex_count_per_polyline: int32")
    .Input("class_identifier: int32")
    .Input("width: int32")
    .Output("output_polyline: float")
    .Output("output_labels: int32")
    .Doc(R"doc(
    Op that joins vertices of a Polyline to form a one dimensional array of Polyline
    using linear interpolation between vertices. It also outputs the class_identifier corresponding
    to the object covered by the Polyline. The size of the one dimensional output Boundary and
    label tensor is equal to the width of the image.

    This Op takes a tensor of polyline vertices as input. Vertices of all the polylines in the image
    are appended together in this tensor which is in the form of list of lists. It also takes
    vertex count of every polyline as input.

    Algorithm to Merge polylines into a single Polyline:
        For every two consecutive vertices in the polyline, a line equation is derived. Now for all the
        columns (x) that lie between these two vertices, a row value (y) is calculated using the
        line equation. Thus, all the consecutive vertices in a polylines are linearly interpolated to
        account for all the columns of the image that lie between the y values of the starting and
        ending vertex of the column. Two separate polylines, however, are not connected by the line and
        all the row values for the columns between the polylines are assigned a value of -1.
        Finally, an output boundary array is returned whose size is equal to the width of the image.
        For every column(x) of the image, the calculated row(y) value is assigned to this output polyline
        array.

    Label creation Algorithm:
        Labels output array is created alongside the Merge Polylines algorithm. All the interpolated
        vertices of a polyline contain the same class_identifier. Thus, for every column of the image,
        this array contains the class_identifier corresponding to the polyline whose vertices covers
        that column.

    Arguments:
        polyline_vertices: a tensor in the form of a list of lists. The top-level list contains
            sub-lists with 2 elements each; each sub-list contains absolute x/y coordinates
            (in that order) of a single vertex of a single polyline for a single image.
            The length of the top-level list is therefore equal to the total
            number of vertices over all polylines in the image.
        vertex_count_per_polyline: a tensor in the form of a flat list. The elements of the list
            are the vertex counts for each polyline. Thus, the length of this list is equal to
            the number of polylines we will draw, and if we were to sum all the values in this list,
            the sum should equal the length of the ``polyline_vertices`` list above.
        class_identifier: an int32 tensor as a flat list containing labels for each polylines in the image.
        width: Width of the output polyline and label points.

    Returns:
        output_polyline: A tensor as a flat list containing y values for points on the merged polyline.
        output_labels: Class labels corresponding to the merged polyline.
    )doc");

class MergePolylinesOp : public OpKernel {
 public:
    using Tensor_type_float =
        Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int64_t>, 16, Eigen::MakePointer>;
    using Tensor_type_int =
        Eigen::TensorMap<Eigen::Tensor<int, 1, 1, int64_t>, 16, Eigen::MakePointer>;
    using Tensor_type_const_int =
        Eigen::TensorMap<Eigen::Tensor<const int, 1, 1, int64_t>, 16, Eigen::MakePointer>;
    explicit MergePolylinesOp(OpKernelConstruction* context) : OpKernel(context) {}

    void CheckRange(float* variable, float low, float high) {
        if (*variable > high) {
            *variable = high - 1;
        }
        if (*variable < low) {
            *variable = low;
        }
    }
    void joinVerticesByStraightLine(Tensor_type_float* output_polyline,
                                    Tensor_type_int* output_label,
                                    Tensor_type_const_int class_labels, float vertex1_x,
                                    float vertex1_y, float vertex2_x, float vertex2_y,
                                    float image_width, int start_col, int end_col,
                                    int current_polyline) {
        // Find the range of columns of image that this polyline covers.
        int range_col = static_cast<int>(end_col - start_col);
        int curr_col = 0, curr_row = 0;

        // range_col == -1 is for the case where Vertex1 and Vertex2 have same x
        // location on boundary. This is because the start_col of the vertex
        // is incremented by 1 (in order to calculate linear points within the range)
        if (range_col == -1) {
            if (vertex2_y >= output_polyline->data()[end_col]) {
                output_polyline->data()[end_col] = vertex2_y;
                output_label->data()[end_col] = class_labels.data()[current_polyline];
            }
        } else if (range_col == 0) {
            // Vertex1 and Vertex2 belong to adjacent columns. Include both of them if valid.
            if (vertex1_y >= output_polyline->data()[start_col - 1]) {
                output_polyline->data()[start_col - 1] = vertex1_y;
                output_label->data()[start_col - 1] = class_labels.data()[current_polyline];
            }
            if (vertex2_y >= output_polyline->data()[end_col]) {
                output_polyline->data()[end_col] = vertex2_y;
                output_label->data()[end_col] = class_labels.data()[current_polyline];
            }
        } else if (range_col > 0) {
            // Find all points between vertex1 and vertex2.
            // Line equation between vertex1 and vertex2.
            float slope = (vertex2_y - vertex1_y) / (vertex2_x - vertex1_x);
            float constant = (vertex1_y - (slope * vertex1_x));
            // Add the first point.
            if (vertex1_y >= output_polyline->data()[start_col - 1]) {
                output_polyline->data()[start_col - 1] = vertex1_y;
                output_label->data()[start_col - 1] = class_labels.data()[current_polyline];
            }
            // Add intermediate points.
            for (int range_ind = 0; range_ind < range_col; range_ind++) {
                if ((start_col + range_ind) >= image_width) {
                    curr_col = image_width - 1;
                } else {
                    curr_col = start_col + range_ind;
                }
                curr_row = slope * curr_col + constant;
                output_polyline->data()[curr_col] = curr_row;
                output_label->data()[curr_col] = class_labels.data()[current_polyline];
            }
            // Add last point.
            if (vertex2_y >= output_polyline->data()[end_col]) {
                output_polyline->data()[end_col] = vertex2_y;
                output_label->data()[end_col] = class_labels.data()[current_polyline];
            }
        }
    }
    void Compute(OpKernelContext* context) override {
        const Tensor& vertices_tensor = context->input(0);
        auto vertices = vertices_tensor.flat<float>();

        const Tensor& vertex_count_per_polyline_tensor = context->input(1);
        auto vertex_count_per_polyline = vertex_count_per_polyline_tensor.flat<int>();

        const Tensor& class_labels_tensor = context->input(2);
        auto class_labels = class_labels_tensor.flat<int>();

        const Tensor& width_tensor = context->input(3);
        auto width = width_tensor.flat<int>();

        // Test that the sum of the `vertex_count_per_polyline_tensor` vector adds
        // up to `nvertices`.
        const int64 nvertices = vertices_tensor.dim_size(0);
        const int64 npolylines = vertex_count_per_polyline_tensor.dim_size(0);
        const int64 nclass_labels = class_labels_tensor.dim_size(0);
        int nvertices_from_vertex_counts = 0;

        for (int i = 0; i < npolylines; i++) {
            nvertices_from_vertex_counts += vertex_count_per_polyline.data()[i];
        }
        OP_REQUIRES(context, nvertices_from_vertex_counts == nvertices,
                    errors::InvalidArgument(
                        "Sum of points_per_polyline_tensor", nvertices_from_vertex_counts,
                        " over all polylines does not add up to nvertices ", nvertices, "."));

        // Test that size of vertex count per polyline tensor and size of class labels
        // tensor matches.
        OP_REQUIRES(context, npolylines == nclass_labels,
                    errors::InvalidArgument("Size of vertex count per polyline tensor", npolylines,
                                            " and size of class labels tensor", nclass_labels,
                                            " are not equal."));

        // Create polyline outputs.
        Tensor* output_polyline_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({width.data()[0]}),
                                                         &output_polyline_tensor));
        auto output_polyline = output_polyline_tensor->flat<float>();

        for (int i = 0; i < output_polyline.size(); i++) {
            output_polyline(i) = -1;
        }

        // Create label outputs.
        Tensor* output_label_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({width.data()[0]}),
                                                         &output_label_tensor));
        auto output_label = output_label_tensor->flat<int>();
        for (int i = 0; i < output_label.size(); i++) {
            output_label(i) = -1;
        }

        int current_polyline = 0;
        int current_vertices = 0;
        float image_width = width.data()[0];
        int vertices_in_polyline = vertex_count_per_polyline.data()[current_polyline];

        // Since vertices_tensor is flattened, every vertex is covered by 2 array elements
        // in vertices array, i.e. vertices[0] will be the x point and vertices[1] will be the
        // y point for the first vertex. Hence, we skip the vert_ind by 2. Since we access vertex
        // indices from vert_ind to vert_ind+3, our stopping criteria is vertices.size() - 3 to
        // avoid accessing index out of range.
        for (int vert_ind = 0; vert_ind < vertices.size() - 3; vert_ind += 2) {
            // Get two vertices from the array of vertices.
            float vertex1_x = vertices.data()[vert_ind];
            float vertex1_y = vertices.data()[vert_ind + 1];
            float vertex2_x = vertices.data()[vert_ind + 2];
            float vertex2_y = vertices.data()[vert_ind + 3];
            int vertex1_x_pixel = static_cast<int>(vertex1_x);
            int vertex2_x_pixel = static_cast<int>(vertex2_x);

            // Find minimum among two vertices and swap.
            if ((vertex1_x_pixel == vertex2_x_pixel && vertex1_y > vertex2_y) ||
                vertex1_x > vertex2_x) {
                std::swap(vertex1_x, vertex2_x);
                std::swap(vertex1_y, vertex2_y);
            }

            // Check if the columns are within range (0, image_width-1).
            float start_col = vertex1_x + 1;
            float end_col = vertex2_x;
            CheckRange(&start_col, 0, image_width);
            CheckRange(&end_col, 0, image_width);
            int start_col_int = static_cast<int>(start_col);
            int end_col_int = static_cast<int>(end_col);
            joinVerticesByStraightLine(&output_polyline, &output_label, class_labels, vertex1_x,
                                       vertex1_y, vertex2_x, vertex2_y, image_width, start_col_int,
                                       end_col_int, current_polyline);

            // Count Vertices in current polyline.
            current_vertices += 1;
            // Update the current polyline to the next one, if the count of current vertices
            // reaches total vertices in the polyline. vert_ind is also updated so that
            // we don't connect vertices in previous polyline to the next polyline.
            if (current_vertices == (vertices_in_polyline - 1)) {
                current_polyline++;
                current_vertices = 0;
                vert_ind += 2;
                if (current_polyline >= npolylines) break;
                vertices_in_polyline = vertex_count_per_polyline.data()[current_polyline];
            }
        }
    }
};
REGISTER_KERNEL_BUILDER(Name("MergePolylines").Device(DEVICE_CPU), MergePolylinesOp);
}  // namespace tensorflow
