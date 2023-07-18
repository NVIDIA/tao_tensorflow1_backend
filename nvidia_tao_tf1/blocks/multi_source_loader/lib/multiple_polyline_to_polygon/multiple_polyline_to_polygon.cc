// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.

/*
Polyline to Polygon conversion algorithm

*/
#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>
#include <unordered_set>
#include <vector>
#include "../utils/point2.h"
#include "../utils/tensor_utils.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_types.h"

using namespace std;
using namespace tensorflow;

const int64 DEFAULT_EMPTY_ATTRIBUTE_ID = -1;

REGISTER_OP("MultiplePolylineToPolygon")
    .Input("polygon_indices: int64")
    .Input("polygon_values: float32")
    .Input("polygon_dense_shape: int64")
    .Input("attribute_indices: int64")
    .Input("attribute_values: int32")
    .Input("attribute_shape: int64")
    .Input("attribute_id_list: int32")
    .Input("class_id_list: int32")
    .Output("output_polygon_indices: int64")
    .Output("output_polygon_values: float32")
    .Output("output_polygon_dense_shape: int64")
    .Output("output_class_indices: int64")
    .Output("output_class_values: int32")
    .Output("output_class_shape: int64")
    .Doc(R"doc(
        Multiple Polyline to Polygon conversion op.
        Summary:
            Takes in three SparseTensor[s] describing a set of polylines to convert.

            polygon_dense_shape must be >2D ([NT]PVC), where N is
            batch dimension, T is temporal dimension, P is polygons, V vertices, and
            C coordinate index (0 or 1).

            polygon_values is a flat fp32 list of interleaved vertex (x, y) coordinates.

            polygon_indices is a 2d tensor with dimension 0 the size of the
            polygons.values tensor, and dimension 1 is size 5.
            Based on attributes combines multiple polylines / polygons. Polygons/Polylines
            sharing same attribute value will be combined.

        Tensor Arguments:
            polygon_indices: indices field of a SparseTensor describing the input polygons
            polygon_values: values field of a SparseTensor describing the input polygons.
            polygon_dense_shape: dense_shape field of a SparseTensor describing the input
                                 polygons.
            attribute_indices: indices field of a SparseTensor describing the attributes of each
                               polygon/polyline
            attribute_values: values field of a SparseTensor describing the attributes of each
                              polygon/polyline
            attribute_shape: dense_shape field of a SparseTensor describing the attributes of
                             each polygon/polyline
            attribute_id_list: Tensor with attribute ids.
            class_id_list: Tensor with class ids.

        Returns:
            output_polygon_indices: same format as polygon_indices
            output_polygon_values: same format as polygon_values
            output_polygon_dense_shape: same format as polygon_dense_shape
            output_class_indices: same format as class_ids_indices
            output_class_values: same format as class_ids_values
            output_class_shape: same format as class_ids_shape
     )doc");

class MultiplePolylineToPolygonOp : public OpKernel {
 public:
    explicit MultiplePolylineToPolygonOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override;
};

#pragma message("Registering CPU kernel")
REGISTER_KERNEL_BUILDER(Name("MultiplePolylineToPolygon").Device(DEVICE_CPU),
                        MultiplePolylineToPolygonOp);
/*
Maps Attributes to a set of polylines/polygons that share attribute ID.
i.e. If polygon/polyline 1, 5, 6 have same attribute 0 then the map would contain
a mapping as 0 -> {1, 5, 6}. A polyline/polygon can be a included in different
attribute IDs as well.
Arguments:
    attribute_values: Vector containing attribute values.
    attribute_ranges: Vector containing attribute ranges for polygons/polylines.
                      Attribute ranges for polygon/polyline i, will lie between
                      2*i and 2*i+1.
    polygon_ranges: Vector containing polygon ranges.
    start_range: Starting polygon/polyline (represented by integers).
    end_range: Ending polyline/polygon (represented by integers)
Returns:
    attribute_to_polygon_map: An unordered map between attributes and polylines/
                              polygons that have those attributes within the
                              given range of polylines/polygons. Polylines having
                              similar attributes are grouped together.
    output_polygon_count: Total number of unique polylines/polygons left within
                          the given range after the polylines/polygons with similar
                          attributes are combined.
    max_vertices_count: Maximum number of vertices in a polygon after combining
                        polylines/polygons.
*/
tuple<unordered_map<int64, set<int64>>, int64, int64, int64> GetAttributeToPolygonMap(
    const TTypes<int32>::ConstVec &attribute_values, const vector<int64> &attribute_ranges,
    const vector<int64> &polygon_ranges, int64 start_range, int64 end_range) {
    int64 start_attribute = 0;
    int64 end_attribute = 0;
    int64 output_polygon_count = 0;
    int64 output_indices_count = 0;

    unordered_map<int64, set<int64>> attribute_to_polygon_map;
    unordered_map<int64, int64> attribute_to_vertex_count;

    // Iterate only over the range of polygons provided.
    // start_range and end_range point to polygon numbers.
    for (int64 i = start_range; i < end_range; i++) {
        // Attributes for polygon i is between attribute_ranges(2*i) to
        // attribute_ranges(2*i+1).
        start_attribute = attribute_ranges[2 * i];
        end_attribute = attribute_ranges[2 * i + 1];
        // Check if all attribute values are not equal to DEFAULT_EMPTY_ATTRIBUTE_ID.
        // If they are then, this polyline / polygon will not be combined with others.
        bool all_empty_attributes = true;
        for (int64 j = start_attribute; j < end_attribute; j++) {
            if (attribute_values(j) != DEFAULT_EMPTY_ATTRIBUTE_ID) {
                all_empty_attributes = false;
            }
        }
        // If start_attribute and end_attribute is DEFAULT_EMPTY_ATTRIBUTE_ID, then no
        // attributes were provided for this polygon. All such polygons/polylines
        // will not be combined. Though we will map them to DEFAULT_EMPTY_ATTRIBUTE_ID.
        if (start_attribute == DEFAULT_EMPTY_ATTRIBUTE_ID ||
            end_attribute == DEFAULT_EMPTY_ATTRIBUTE_ID || all_empty_attributes) {
            auto it = attribute_to_polygon_map.find(DEFAULT_EMPTY_ATTRIBUTE_ID);
            if (it != attribute_to_polygon_map.end()) {
                // Check if polygon/polyline i has not been included in the set of
                // polylines/polygons. If not included then add it and increment the
                // number of polygons as well as vertices
                if (it->second.count(i) == 0) {
                    it->second.insert(i);
                    output_polygon_count += 1;
                    output_indices_count +=
                        static_cast<int64>((polygon_ranges[i + 1] - polygon_ranges[i]));
                    auto vertex_count =
                        static_cast<int64>((polygon_ranges[i + 1] - polygon_ranges[i]) / 2);
                    attribute_to_vertex_count[DEFAULT_EMPTY_ATTRIBUTE_ID] =
                        max(attribute_to_vertex_count[DEFAULT_EMPTY_ATTRIBUTE_ID], vertex_count);
                }
            } else {
                attribute_to_polygon_map[DEFAULT_EMPTY_ATTRIBUTE_ID].insert(i);
                // Polygons with DEFAULT_EMPTY_ATTRIBUTE_ID as attribute are not combined,
                // hence increment num_output_polygon.
                output_polygon_count += 1;
                output_indices_count +=
                    static_cast<int64>((polygon_ranges[i + 1] - polygon_ranges[i]));
                auto vertex_count =
                    static_cast<int64>((polygon_ranges[i + 1] - polygon_ranges[i]) / 2);

                attribute_to_vertex_count[DEFAULT_EMPTY_ATTRIBUTE_ID] = vertex_count;
            }
        } else {
            // If start_attribute and end_attribute is not equal to DEFAULT_ATTRIBUTE_EMPTY_ID
            // (i.e. attributes exist), then these polylines/polygons will be combined.
            for (int64 j = start_attribute; j < end_attribute; j++) {
                int64 key = attribute_values(j);
                if (key == DEFAULT_EMPTY_ATTRIBUTE_ID) {
                    continue;
                } else {
                    // Find attribute value in the map, if exists then include the polygon/polyline
                    // in the set.
                    auto it = attribute_to_polygon_map.find(key);
                    if (it != attribute_to_polygon_map.end()) {
                        if (it->second.count(i) == 0) {
                            it->second.insert(i);
                            output_indices_count +=
                                static_cast<int64>((polygon_ranges[i + 1] - polygon_ranges[i]));
                            auto vertex_count =
                                static_cast<int64>((polygon_ranges[i + 1] - polygon_ranges[i]) / 2);
                            attribute_to_vertex_count[key] += vertex_count;
                        }
                    } else {
                        attribute_to_polygon_map[key].insert(i);
                        output_polygon_count += 1;
                        output_indices_count +=
                            static_cast<int64>((polygon_ranges[i + 1] - polygon_ranges[i]));
                        auto vertex_count =
                            static_cast<int64>((polygon_ranges[i + 1] - polygon_ranges[i]) / 2);
                        attribute_to_vertex_count[key] = vertex_count;
                    }
                }
            }
        }
    }

    int64 max_vertices_count = 0;
    for (auto map : attribute_to_vertex_count) {
        max_vertices_count = max(max_vertices_count, map.second);
    }
    return make_tuple(attribute_to_polygon_map, output_indices_count, output_polygon_count,
                      max_vertices_count);
}
/*
Sorts the polylines such that consecutive polylines form a polygon. For every polyline,
this function finds the next polyline that is closest to it's endpoint. It checks the
first as well as the last endpoint of every polyline, if the first endpoint is closest then,
no need to reverse the polyline but if last endpoint is closer then the polyline coordinates
have to be reversed.

Arguments:
    polygon_ranges: Represents range of polygon indices. For example, vertices for
                    ith polygon are between polygon_ranges[i] and polygon_ranges[i+1]
    polygon_values: Vector of polygon vertices.
    polygon_set: Set of unordered polylines (represented by integers) that form a polygon.

Returns:
    ordered_polygon_vector: Vector of ordered polylines (represented by int) that form polygon.
    reverse_vector: A vector of 0/1 for each polyline in the polygon, to indicate whether the
                    polyline should be reversed or not.
*/
tuple<vector<int64>, vector<int64>> SortPolygons(const vector<int64> &polygon_ranges,
                                                 const TTypes<float>::ConstVec &polygon_values,
                                                 set<int64> polygon_set) {
    vector<int64> ordered_polygon_vector(polygon_set.size());
    vector<int64> reverse_vector(polygon_set.size());

    int64 curr_polygon = *polygon_set.begin();
    polygon_set.erase(polygon_set.begin());

    ordered_polygon_vector[0] = curr_polygon;
    reverse_vector[0] = 0;

    int64 ordered_polygon_counter = 0;
    float min_dist;
    set<int64>::iterator min_index;
    int64 reverse = 0;

    while (!polygon_set.empty()) {
        // Check if reverse is zero, if zero then use the ending point of the current
        // polyline, else use the starting point of the current polyline as the endpoint.
        Point2<float> curr_polygon_endpoint{0, 0};
        if (reverse == 0) {
            int64 polygon_index = polygon_ranges[curr_polygon + 1];
            curr_polygon_endpoint.x = polygon_values(polygon_index - 2);
            curr_polygon_endpoint.y = polygon_values(polygon_index - 1);
        } else {
            int64 polygon_index = polygon_ranges[curr_polygon];
            curr_polygon_endpoint.x = polygon_values(polygon_index);
            curr_polygon_endpoint.y = polygon_values(polygon_index + 1);
        }

        // Declare minimum default value.
        min_dist = numeric_limits<float>::max();

        // Check every polyline that remains in the set for the polyline that is
        // closest to the endpoint of current polyline.
        for (auto it = polygon_set.begin(); it != polygon_set.end(); it++) {
            int64 next_polygon = *it;
            int64 next_polygon_start = polygon_ranges[next_polygon];
            int64 next_polygon_end = polygon_ranges[next_polygon + 1];
            // Start point of the next polyline.
            Point2<float> next_polygon_startpoint{polygon_values(next_polygon_start),
                                                  polygon_values(next_polygon_start + 1)};
            // End point of the next polyline.
            Point2<float> next_polygon_endpoint{polygon_values(next_polygon_end - 2),
                                                polygon_values(next_polygon_end - 1)};
            // Calculate distance for both start and end point of next polyline with
            // current polylines endpoint.
            float dist1 = distance(curr_polygon_endpoint, next_polygon_startpoint);
            float dist2 = distance(curr_polygon_endpoint, next_polygon_endpoint);
            // If start point of next polyline is closest, then no need to reverse
            // the next polyline.
            if (dist1 < min_dist) {
                min_dist = dist1;
                min_index = it;
                reverse = 0;
            }
            // If end point of next polyline is closest, then reverse the next polyline.
            if (dist2 < min_dist) {
                min_dist = dist2;
                min_index = it;
                reverse = 1;
            }
        }
        ordered_polygon_counter++;
        // Assign the polyline with minimum distance as the next polyline.
        ordered_polygon_vector[ordered_polygon_counter] = *min_index;
        // Assign whether the next polyline should be reversed or not.
        reverse_vector[ordered_polygon_counter] = reverse;
        // Next polyline now becomes current polyline.
        curr_polygon = *min_index;
        // Remove the polyline from the set.
        polygon_set.erase(min_index);
    }
    return make_tuple(ordered_polygon_vector, reverse_vector);
}

/*
Finds the ranges within the polygon_indices tensor that represent the
individual polygons. If you want to get the slice of indices for a given polygon "i",
then the start index is out_ranges(i) and the non-inclusive end index is out_ranges(i + 1).

This Function also finds the polygons ranges that belong to each frame. For example;
If, out_frame_ranges(j) = k and out_frame_ranges(j+1) = m, then jth frame contains,
polygons/polylines from k to m. Their ranges in polygon_indices tensor can be found out using
out_ranges(k) and out_ranges(m).
NOTE: polygon_indices MUST be lexicographically sorted from the left column to the right
Arguments:
    polygon_indices: Indices of sparse input polygon tensor.
    num_frames: Total number of frames in this batch of input labels.
Returns:
    ranges: Vector containing ranges for polygons.
    frame_ranges: Vector containing ranges for frames.
*/
tuple<vector<int64>, vector<int64>> GetPolygonAndFrameRanges(
    const TTypes<int64>::ConstMatrix &polygon_indices) {
    // Create a vector that can hold the row ranges for each polygon.
    // These ranges can be used to slice both the indices and values
    // tensors to a specific polygon.
    vector<int64> ranges;
    vector<int64> frame_ranges;

    const int64 num_indices = polygon_indices.dimension(0);
    // The final two columns of the index tensor are what actually
    // define the coordinates of the tensor. All preceding dimensions are specifying
    // which polygon we're dealing with. Essentially, all but the last 2 columns
    // form an "id" tuple that uniquely identifies a polygon.
    const int64 num_id_columns = polygon_indices.dimension(1) - 2;

    // First polygon starts at index 0.
    ranges.push_back(0);
    frame_ranges.push_back(0);
    // The index of the start of the last polygon.
    int64 prev_index = 0;
    // The index of last frame.
    int64 prev_frame_index = 0;
    // The index of the range we're currently dealing with.
    int64 curr_range = 1;
    // Coordinates always come in pairs, so we can skip every other row.
    for (int64 i = 2; i < num_indices; i += 2) {
        // Check to see if the id tuple of the current indice matches that of the previous frame
        // index.
        // If they don't match, then this is the start of a new frame.
        if (!IsSubDimEqual(num_id_columns - 1, polygon_indices, prev_frame_index, polygon_indices,
                           i)) {
            // Note the index in ranges array that corresponds to frame change.
            frame_ranges.push_back(curr_range);
            prev_frame_index = i;
        }
        // Check to see if the id tuple of the current indice matches that of the previous one.
        // If they don't match, then this is the start of a new polygon.
        if (!IsSubDimEqual(num_id_columns, polygon_indices, prev_index, polygon_indices, i)) {
            ranges.push_back(i);
            prev_index = i;
            ++curr_range;
        }
    }
    ranges.push_back(num_indices);
    frame_ranges.push_back(curr_range);
    return make_tuple(ranges, frame_ranges);
}

/*
Converts ID dimensions to string, for example if the ID of polygon/polyline or frame is
0 1 0 then this function would return a string "0 1 0"
Arguments:
    indices: Indices Matrix whose row will be used to create ID.
    num_id_columns: Number of columns starting from first column that will be used to create ID.
    row_index: Index of row in indices array for which the ID is to be created.
Returns:
    string_id: ID created for row at row_index of Indices matrix using (0 - num_id_columns-1)
               columns.
*/
string GetStringID(const TTypes<int64>::ConstMatrix &indices, int64 num_id_columns,
                   int64 row_index) {
    ostringstream id;

    for (int64 j = 0; j < num_id_columns; j++) {
        id << indices(row_index * indices.dimension(1) + j) << " ";
    }
    string string_id = id.str();
    return string_id;
}

/*
Returns a mapping between string ID and polygon/polyline number. For example if "0 1 2"
represents 15th polygon/polyline of the batch then the map will contain "0 1 2" --> 15
as a mapping.
Arguments:
    ranges: ranges representing polygons/polylines for which IDMap is to be created.
    indices: Indices representing polyline/polygon.
    num_id_columns: Number of columns from indices matrix that will be used to create ID.
Returns:
    map_id: An unordered map that maps string ID to polygon/polyline represented by
            integers.
*/
unordered_map<string, int64> GetPolygonIDMap(const vector<int64> &ranges,
                                             const TTypes<int64>::ConstMatrix &indices,
                                             int64 num_id_columns) {
    unordered_map<string, int64> map_id;
    for (uint64 i = 0; i < ranges.size(); i++) {
        string id = GetStringID(indices, num_id_columns, ranges[i]);
        map_id.emplace(move(id), i);
    }
    return map_id;
}

/*
Returns a mapping between attribute id and the class id.
Arguments:
    attribute_id_list: vector containing attribute_id's.
    class_id_list: vector contaibubg class_id's.
Returns:
    attribute_class_map: An unordered map that maps attribute_ids to class_ids.
*/
unordered_map<int32, int32> GetAttributeToClassIDMap(
    const TTypes<int32>::ConstVec &attribute_id_list,
    const TTypes<int32>::ConstVec &class_id_list) {
    unordered_map<int32, int32> attribute_class_map;
    for (int i = 0; i < attribute_id_list.dimension(0); i++) {
        attribute_class_map.emplace(attribute_id_list(i), class_id_list(i));
    }
    return attribute_class_map;
}

/*
For every polygon/polyline, get the range of attribute indices that corresponds to that.
For polyline/polygon i, the starting range for attributes will be at 2*i and the
ending range will be at 2*i+1. If the polyline/polygon has no attributes, then the
starting and ending range will both be assigned to DEFAULT_EMPTY_ATTRIBUTE_ID.
Arguments:
    attribute_indices: Indices of sparse tensor for attributes.
    polygon_indices: Indices of sparse tensor for polygons.
    polygon_map: Map between indices as ID of polygon (represented as string) and polygon/
                 polylines represented as integer.
    num_polygons: Total number of polygons in the input labels.
Returns:
    ranges: Vector that will contain attribute ranges with size as 2*num_polygons.
            Attributes ranges for ith polygon will lie between 2*i and 2*i+1
*/
vector<int64> GetAttributeRanges(const TTypes<int64>::ConstMatrix &attribute_indices,
                                 const TTypes<int64>::ConstMatrix &polygon_indices,
                                 const unordered_map<string, int64> &polygon_map,
                                 int64 num_polygons) {
    int64 prev_index = 0;
    int64 range_index = 0;
    int64 num_id_columns = attribute_indices.dimension(1) - 1;
    // Initialize the ranges to DEFAULT_EMPTY_ATTRIBUTE_ID. DEFAULT_EMPTY_ATTRIBUTE_ID
    // in the start and end range represents polygons with no attributes.
    vector<int64> ranges(2 * num_polygons, DEFAULT_EMPTY_ATTRIBUTE_ID);

    // Assign the initial prev_index and range index for polygon with attributes.
    string polygon_id = GetStringID(attribute_indices, num_id_columns, 0);
    auto it = polygon_map.find(polygon_id);
    if (it != polygon_map.end()) {
        range_index = 2 * it->second;
    } else {
        throw runtime_error("Attribute polygon ID not found in polygon map!");
    }
    ranges[range_index] = 0;

    for (int64 j = 1; j < attribute_indices.dimension(0); j++) {
        // Find the change in polygonID for attribute indices. Assign ending range by incrementing,
        // range_index. This will always be 2*i+1 where i is the polygon number.
        if (!IsSubDimEqual(num_id_columns, attribute_indices, prev_index, attribute_indices, j)) {
            range_index++;
            ranges[range_index] = j;
            // Get the polygonID as string and find the polygon number from the map. Next
            // range_index, starts from j for polygon i and range index 2*i.
            polygon_id = GetStringID(attribute_indices, num_id_columns, j);
            it = polygon_map.find(polygon_id);
            if (it != polygon_map.end()) {
                range_index = 2 * it->second;
            } else {
                throw runtime_error("Attribute polygon ID not found in polygon map!");
            }
            ranges[range_index] = j;
            prev_index = j;
        }
    }
    ranges[range_index + 1] = attribute_indices.dimension(0);
    return ranges;
}

/*
Assigns Polygon Indices and Values within start range and end range to output indices
and values. The index is created using input indices id columns, polygon_value, vertex_value
and the rest of the input indices.
Arguments:
    out_indices: Pointer to matrix for Output indices that needs to be assigned.
    out_values: Pointer to vector for Output values that needs to be assigned.
    input_indices: Matrix of Input polygon/polyline indices.
    input_values: Vector of Input polygon/polyline values.
    num_id_columns: Number of columns in the indices matrix that represent the ID of polygons.
    polygon_value: Polygon number to be assigned for the given range of vertices.
    vertex_value: Starting vertex value for this range of vertices.
    start_range: Starting index (w.r.t input_indices) for the data to be assigned to output.
    end_range: Ending index (w.r.t input_indices) for the data to be assigned to output.
    reverse: Whether to assign this range of data in reverse order or not.
    out_index: Index within output_indices/output_values from where assigning of data begins.
*/
void AssignPolygonIndicesAndValuesToOutput(TTypes<int64>::Matrix *out_indices,
                                           TTypes<float>::Vec *out_values,
                                           const TTypes<int64>::ConstMatrix &input_indices,
                                           const TTypes<float>::ConstVec &input_values,
                                           int64 num_id_columns, int64 polygon_value,
                                           int64 vertex_value, int64 start_range, int64 end_range,
                                           int64 reverse, int64 out_index) {
    int64 i;
    int64 incr;
    if (reverse == 1) {
        i = end_range - 2;
        incr = -2;
    } else {
        i = start_range;
        incr = 2;
    }
    while (i >= start_range && i < end_range - 1) {
        // Slice every input index and assign it to out_indices.
        auto input_indices_per_row_x = Slice(input_indices, i, i + 1);
        auto input_indices_per_row_y = Slice(input_indices, i + 1, i + 2);
        InnerAssign(*out_indices, out_index,
                    make_pair(input_indices_per_row_x.data(), num_id_columns), polygon_value,
                    vertex_value, input_indices_per_row_x(num_id_columns + 2));
        (*out_values)(out_index) = input_values(i);
        out_index++;
        InnerAssign(*out_indices, out_index,
                    make_pair(input_indices_per_row_y.data(), num_id_columns), polygon_value,
                    vertex_value, input_indices_per_row_y(num_id_columns + 2));
        (*out_values)(out_index) = input_values(i + 1);
        out_index++;
        vertex_value++;
        i += incr;
    }
}
/*
Returns a vector of attribute to polygon maps where each map corresponds to a frame
in the batch. For every frame, get the polygon numbers that correspond to the frame,
and call GetAttributeToPolygonMap for that range.
Arguments:
    attribute_values: Vector containing Attribute values corresponding to input
                      sparse attribute tensor.
    attribute_ranges: Vector containing attribute ranges for polygons/polylines.
                      Attribute ranges for polygon/polyline i, will lie between
                      2*i and 2*i+1.
    frame_ranges: Vector representing range of polylines/polygons that lie within a
                  frame. For example, Polygons/Polylines ranging between
                  frame_ranges(i) and frame_ranges(i+1) correspond to ith frame.
    polygon_ranges: Represents range of polygon indices. For example, vertices for
                    ith polygon are between polygon_ranges(i) and polygon_ranges(i+1).
Returns:
    map_per_frame: Vector of unordered maps (attributes to polygon set) for each frame
                   in the input labels.
    total_polygons: Total number of polygons in the output.
    max_polygons: Maximum number of polygons in any frame.
    max_vertices: Maximum number vertices in any polygon after combining them.
*/
tuple<vector<unordered_map<int64, set<int64>>>, int64, int64, int64, int64>
GetAttributeToPolygonMapsPerFrame(const TTypes<int32>::ConstVec &attribute_values,
                                  const vector<int64> &attribute_ranges,
                                  const vector<int64> &frame_ranges,
                                  const vector<int64> &polygon_ranges) {
    int64 start_frame_range = 0;
    int64 end_frame_range = 0;
    int64 polygons_per_frame = 0;
    int64 max_vertices_per_frame = 0;
    int64 indices_per_frame = 0;
    int64 total_indices = 0;
    int64 total_polygons = 0;
    int64 max_polygons = 0;
    int64 max_vertices = 0;
    vector<unordered_map<int64, set<int64>>> map_per_frame;
    unordered_map<int64, set<int64>> attribute_polygon_map;
    for (unsigned int j = 0; j < frame_ranges.size() - 1; j++) {
        start_frame_range = frame_ranges[j];
        end_frame_range = frame_ranges[j + 1];
        tie(attribute_polygon_map, indices_per_frame, polygons_per_frame, max_vertices_per_frame) =
            GetAttributeToPolygonMap(attribute_values, attribute_ranges, polygon_ranges,
                                     start_frame_range, end_frame_range);
        total_indices += indices_per_frame;
        total_polygons += polygons_per_frame;
        max_polygons = max(max_polygons, polygons_per_frame);
        max_vertices = max(max_vertices, max_vertices_per_frame);
        map_per_frame.push_back(move(attribute_polygon_map));
    }
    return make_tuple(map_per_frame, total_indices, total_polygons, max_polygons, max_vertices);
}

/*
Merge Polylines/Polygons into Polygons based on attributes.
Arguments:
    polygon_indices: Matrix representing indices for sparse input polygon tensor.
    polygon_values: Vector representing polygon vertex values for sparse input
                    polygon tensor.
    polygon_ranges: Represents range of polygon indices. For example, vertices for
                    ith polygon are between polygon_ranges(i) and polygon_ranges(i+1).
    attribute_polygon_map_per_frame: Vector of unordered maps (attributes to polygon set)
                                     for each frame in the input labels.
    out_indices: Pointer to Matrix of indices corresponding to output sparse polygon
                 tensor.
    out_values: Pointer to vector of values corresponding to output sparse polygon
                tensor.
    out_class_indices: Pointer to matrix of output indices for sparse class tensor.
    out_class_values: Pointer to vector of output values for sparse class tensor.
*/
void MergePolylines(const TTypes<int64>::ConstMatrix &polygon_indices,
                    const TTypes<float>::ConstVec &polygon_values,
                    const vector<int64> &polygon_ranges,
                    const vector<unordered_map<int64, set<int64>>> &attribute_polygon_map_per_frame,
                    const unordered_map<int32, int32> &attribute_class_id_map,
                    TTypes<int64>::Matrix *out_indices, TTypes<float>::Vec *out_values,
                    TTypes<int64>::Matrix *out_class_indices,
                    TTypes<int32>::Vec *out_class_values) {
    int64 polygon_counter_per_frame, out_indices_index = 0, total_polygon_counter = 0;
    // Iterate over attribute_polygon_map_per_frame. It contains a map between attributes and
    // polygons for a particular frame.
    for (const auto &attribute_polygon_map : attribute_polygon_map_per_frame) {
        polygon_counter_per_frame = 0;
        // Iterate over each mapping of attribute to polygon set
        for (const auto &attribute_pair : attribute_polygon_map) {
            int64 attribute_key = attribute_pair.first;
            const auto &polygon_set = attribute_pair.second;
            if (attribute_key == DEFAULT_EMPTY_ATTRIBUTE_ID) {
                // If attribute is DEFAULT_EMPTY_ATTRIBUTE_ID, don't combine them.
                for (auto polygon : polygon_set) {
                    // int64 polygon = polygon_set(i);
                    int64 start_polygon_range = polygon_ranges[polygon];
                    int64 end_polygon_range = polygon_ranges[polygon + 1];
                    // Assign Polygon Indices and Values.
                    AssignPolygonIndicesAndValuesToOutput(
                        out_indices, out_values, polygon_indices, polygon_values,
                        polygon_indices.dimension(1) - 3, polygon_counter_per_frame, 0,
                        start_polygon_range, end_polygon_range, 0, out_indices_index);
                    out_indices_index += (end_polygon_range - start_polygon_range);
                    // Since we are not combining these polygons, Assign class indices and values,
                    // for every polyline/polygon.
                    auto polygon_id =
                        Slice(polygon_indices, start_polygon_range, start_polygon_range + 1);
                    InnerAssign(*out_class_indices, total_polygon_counter,
                                make_pair(polygon_id.data(), polygon_indices.dimension(1) - 3),
                                polygon_counter_per_frame, 0);
                    (*out_class_values)(total_polygon_counter) = attribute_key;
                    polygon_counter_per_frame++;
                    total_polygon_counter++;
                }
            } else {
                // If Attribute is not DEFAULT_EMPTY_ATTRIBUTE_ID, then combine them.
                vector<int64> polygon_vec;
                vector<int64> reverse_vec;
                tie(polygon_vec, reverse_vec) =
                    SortPolygons(polygon_ranges, polygon_values, polygon_set);
                int64 polygon_vertex_num = 0;
                int64 start_polygon_range = 0, end_polygon_range = 0;
                for (unsigned int i = 0; i < polygon_vec.size(); i++) {
                    int64 polygon = polygon_vec[i];
                    int64 reverse = reverse_vec[i];
                    start_polygon_range = polygon_ranges[polygon];
                    end_polygon_range = polygon_ranges[polygon + 1];
                    // Assign Polygon Indices and Values.
                    AssignPolygonIndicesAndValuesToOutput(
                        out_indices, out_values, polygon_indices, polygon_values,
                        polygon_indices.dimension(1) - 3, polygon_counter_per_frame,
                        polygon_vertex_num, start_polygon_range, end_polygon_range, reverse,
                        out_indices_index);
                    out_indices_index += (end_polygon_range - start_polygon_range);
                    // Since we have two coordinates, out_indices_index will always be a
                    // multiple of 2.
                    polygon_vertex_num += (end_polygon_range - start_polygon_range) / 2;
                }
                // Slice a row from polygon_indices to get initial columns of polygon id i.e. [NT]
                // from the original format of [NT] PVC (N: Batch dimension, T: time dimension,
                // P: Polygon no, V: Vertex no, C: Coordinate number. [NT] will be common for
                // all polygons in this polygon_set. Hence, using start_polygon_range of the
                // last polygon in the set to get the id.
                auto polygon_id =
                    Slice(polygon_indices, start_polygon_range, start_polygon_range + 1);
                // Every polygon in the polygon_set has been converted into one polygon,
                // hence, only one class needs to be assigned. The class assigned is the
                // attribute key.
                InnerAssign(*out_class_indices, total_polygon_counter,
                            make_pair(polygon_id.data(), polygon_indices.dimension(1) - 3),
                            polygon_counter_per_frame, 0);
                auto find_mapping = attribute_class_id_map.find(attribute_key);
                if (find_mapping == attribute_class_id_map.end()) {
                    throw runtime_error("Attribute mapping for attribute id: " +
                                        to_string(attribute_key) +
                                        " does not exist in the config!");
                }
                (*out_class_values)(total_polygon_counter) = find_mapping->second;
                polygon_counter_per_frame++;
                total_polygon_counter++;
            }
        }
    }
}

void MultiplePolylineToPolygonOp::Compute(OpKernelContext *context) {
    const Tensor &tf_indices = context->input(0);
    OP_REQUIRES(context, tf_indices.dims() == 2 && tf_indices.dim_size(1) > 2,
                errors::InvalidArgument("polygon_indices must be a 2D tensor"
                                        " with >2 columns. Shape is ",
                                        tf_indices.shape().DebugString()));
    const auto &indices = tf_indices.matrix<int64>();

    const Tensor &tf_values = context->input(1);
    OP_REQUIRES(context, tf_values.dims() == 1,
                errors::InvalidArgument("polygon_values must be 1D. Shape is ",
                                        tf_values.shape().DebugString()));
    const auto &values = tf_values.vec<float>();

    const Tensor &tf_dense_shape = context->input(2);
    OP_REQUIRES(context, tf_dense_shape.dims() == 1,
                errors::InvalidArgument("polygon_dense_shape must be 1D. Shape is ",
                                        tf_dense_shape.shape().DebugString()));
    const auto &dense_shape = tf_dense_shape.vec<int64>();

    const Tensor &tf_attribute_indices = context->input(3);
    OP_REQUIRES(context, tf_attribute_indices.dims() == 2 &&
                             tf_attribute_indices.dim_size(1) == tf_indices.dim_size(1) - 1,
                errors::InvalidArgument("attribute_indices must be a 2D tensor "
                                        " with 1 less column than polygon_indices. Shape is ",
                                        tf_attribute_indices.shape().DebugString(), " vs. ",
                                        tf_indices.shape().DebugString()));
    const auto &attribute_indices = tf_attribute_indices.matrix<int64>();

    const Tensor &tf_attribute_values = context->input(4);
    OP_REQUIRES(context, tf_attribute_values.dims() == 1,
                errors::InvalidArgument("attribute_values must be 1D. Shape is ",
                                        tf_attribute_values.shape().DebugString()));
    const auto &attribute_values = tf_attribute_values.vec<int32>();

    const Tensor &tf_attribute_shape = context->input(5);
    OP_REQUIRES(context, tf_attribute_shape.dims() == 1,
                errors::InvalidArgument("attribute_shape must be 1D. Shape is ",
                                        tf_attribute_shape.shape().DebugString()));
    const auto &attribute_shape = tf_attribute_shape.vec<int64>();

    const Tensor &tf_attribute_id_list = context->input(6);
    OP_REQUIRES(context, tf_attribute_id_list.dims() == 1,
                errors::InvalidArgument("attribute_id_list must be 1D. Shape is ",
                                        tf_attribute_id_list.shape().DebugString()));
    const auto &attribute_id_list = tf_attribute_id_list.vec<int32>();

    const Tensor &tf_class_id_list = context->input(7);
    OP_REQUIRES(context, tf_class_id_list.dims() == 1,
                errors::InvalidArgument("class_id_list must be 1D. Shape is ",
                                        tf_class_id_list.shape().DebugString()));
    const auto &class_id_list = tf_class_id_list.vec<int32>();

    OP_REQUIRES(context, attribute_id_list.dimension(0) == class_id_list.dimension(0),
                errors::InvalidArgument("attribute_id_list should contain the same number"
                                        " of entries as class_id_list! The attribute_id_list"
                                        "dimension ",
                                        to_string(attribute_id_list.dimension(0)),
                                        " class_id_list dimension ",
                                        to_string(class_id_list.dimension(0)), " do not match!"));
    try {
        const int64 num_example_dims = indices.dimension(1) - 3;
        // Polygon ranges to specify starting and ending indices of a polygon.
        // Frame ranges to specify which polygons belong to which frame.
        vector<int64> polygon_ranges;
        vector<int64> polygon_frame_ranges;
        tie(polygon_ranges, polygon_frame_ranges) = GetPolygonAndFrameRanges(indices);
        // Get total number of polygons in the input labels.
        int64 input_num_polygons = polygon_ranges.size() - 1;
        // polygon_map consists of polygon id converted to string and is mapped to
        // corresponding polygon no.
        const auto polygon_map = GetPolygonIDMap(polygon_ranges, indices, indices.dimension(1) - 2);
        // Attribute ranges specify the range of attributes for every polygon/polyline
        // in this batch. If polygon i, then attribute_ranges(2*i) gives the starting index
        // of attributes and attribute_ranges(2*i+1) gives ending index of attributes for this
        // polygon. If attribute_ranges(2*i)=DEFAULT_EMPTY_ATTRIBUTE_ID, then no attributes
        // exist for that polygon/polyline and they will not be combined.
        vector<int64> attribute_ranges;
        attribute_ranges =
            GetAttributeRanges(attribute_indices, indices, polygon_map, input_num_polygons);

        // Get a vector of maps (attribute to polygon maps) for every frame in this batch.
        // Each map contains unique attributes per frame as keys and are mapped to an unordered
        // set of polygons. i.e. We group together polygons with same attributes.
        int64 output_num_indices = 0;
        int64 output_num_polygons = 0;
        int64 max_num_polygons = 0;
        int64 max_num_vertices = 0;
        vector<unordered_map<int64, set<int64>>> attribute_map_per_frame;
        tie(attribute_map_per_frame, output_num_indices, output_num_polygons, max_num_polygons,
            max_num_vertices) =
            GetAttributeToPolygonMapsPerFrame(attribute_values, attribute_ranges,
                                              polygon_frame_ranges, polygon_ranges);

        // Get Attribute to Class ID Map.
        unordered_map<int32, int32> attribute_class_id_map =
            GetAttributeToClassIDMap(attribute_id_list, class_id_list);
        // Merge Polylines/Polygons according to attributes.
        // Output declaration
        // Output polygon indices
        Tensor *tf_output_indices = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
                                    0, TensorShape{output_num_indices, tf_indices.dim_size(1)},
                                    &tf_output_indices));
        auto output_polygon_indices = tf_output_indices->matrix<int64>();

        // Output polygon values
        Tensor *tf_output_values = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{output_num_indices},
                                                         &tf_output_values));
        auto output_polygon_values = tf_output_values->vec<float>();

        // Output polygon shape
        Tensor *tf_output_dense_shape = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape{dense_shape.dimension(0)},
                                                         &tf_output_dense_shape));
        auto output_polygon_dense_shape = tf_output_dense_shape->vec<int64>();
        InnerAssign(output_polygon_dense_shape, 0, make_pair(dense_shape.data(), num_example_dims),
                    max_num_polygons, max_num_vertices, output_num_polygons > 0 ? 2 : 0);

        // Output class indices
        Tensor *tf_output_class_indices = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(
                           3, TensorShape{output_num_polygons, tf_attribute_indices.dim_size(1)},
                           &tf_output_class_indices));
        auto output_class_indices = tf_output_class_indices->matrix<int64>();

        // Output class values
        Tensor *tf_output_class_values = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(4, TensorShape{output_num_polygons},
                                                         &tf_output_class_values));
        auto output_class_values = tf_output_class_values->vec<int32>();

        // Output class shape
        Tensor *tf_output_class_shape = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(5, TensorShape{attribute_shape.dimension(0)},
                                                &tf_output_class_shape));
        auto output_class_shape = tf_output_class_shape->vec<int64>();
        InnerAssign(output_class_shape, 0, make_pair(attribute_shape.data(), num_example_dims),
                    max_num_polygons, 1);

        MergePolylines(indices, values, polygon_ranges, attribute_map_per_frame,
                       attribute_class_id_map, &output_polygon_indices, &output_polygon_values,
                       &output_class_indices, &output_class_values);
    } catch (exception &ex) {
        OP_REQUIRES_OK(context, errors::Internal(ex.what()));
    }
}
