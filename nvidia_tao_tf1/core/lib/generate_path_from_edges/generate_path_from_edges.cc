// Copyright (c) 2017-2018, NVIDIA CORPORATION.  All rights reserved.
/*
  Path generation algorithm
    This Op encodes labeled paths as deviations from a set of pre-defined path priors.
    The general flow of the algorithm is this:
       Gather the input (see next section).
       Loop over each image.
       For each labeled path in this image, interpolate the path to the same
       number of points as the priors.
       Then, for each interpolated path, find the closest prior by summing the
       distances between their (now equal number of) points.
       Assign each path the prior it was closest to.
       Handle collisions where multiple paths are assigned the same prior.
       Recode the labeled path as the difference between the interpolated points
       and the path prior points.
  Input gathering
    N images, each contains P(N) polylines, each of which contains V(P(N)) vertices.
    M priors, each contains Q vertices, where is predefined and sent as an input.
  Path Generation kernel
    Currently not using and computing everything on the CPU.
    For a paths training dataset you will have NumImages*NumPriors*NumLabeledPaths*PointsPerPath
    points to compute on. NumImages will be < 500, NumPriors will be on the order of 1000,
    NumLabeledPaths will be on the order of 20 and PointsPerPath will be on the order of 10,
    making the total number of points on the order of 100 million points per batch.
    To speed this up, we should use the GPU to calculate the distance matrices between the priors
    and the paths. Will most likely require launching one kernel per image since the number of
    labeled paths will be different in each image.
*/

#include <float.h>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
using namespace std;

static const float PI = 3.14159265;

// edge_attribute.
static const int Left_Edge = -1;
static const int Right_Edge = 1;

// Left_prior_group(DP_Minus_1_fork_left, DP_0_fork_left).
static const int Left_prior_group = 0;
// Middle_prior_group(DP_Minus_1, DP_0, DP_1).
static const int Middle_prior_group = 1;
// Right_prior_group(DP_0_fork_right, DP_1_fork_right).
static const int Right_prior_group = 2;

// The number of feature maps for angle ratio sum.
static const int NUM_SUM_ANGLE_RATIOS = 1;

// The number of attributes used to determine classes, i.e., left_edge/right_edge/center_rail.
static const int NUM_ATTRIBUTES_4_CLASS = 1;

typedef Eigen::ThreadPoolDevice CPUDevice;

// We should register only once (CPU)
REGISTER_OP("GeneratePathFromEdges")
    .Input("polyline_vertices: float")
    .Input("path_prior_vertices: float")
    .Input("vertex_counts_per_polyline: int32")
    .Input("class_ids_per_polyline: int32")
    .Input("attributes_per_polyline: int32")
    .Input("polylines_per_image: int32")
    .Input("width: int32")
    .Input("height: int32")
    .Output("output_images: float")
    .Attr("nclasses: int = 1")
    .Attr("nall_priors: int = 1")
    .Attr("points_per_prior: int = 1")
    .Attr("npath_attributes: int = 0")
    .Attr("prior_threshold: float = 1.0")
    .Attr("equal_spacing: bool = true")
    .Attr("prior_assignment_constraint: bool = false")
    .Attr("using_invalid_path_class: bool = false")
    .Attr("edges_per_path: int = 2")
    .Attr("verbose: bool = false")
    .SetShapeFn([](::shape_inference::InferenceContext* c) {
        // Inherit the output shape from the attributes
        int nclasses;
        TF_RETURN_IF_ERROR(c->GetAttr("nclasses", &nclasses));
        int nall_priors;
        TF_RETURN_IF_ERROR(c->GetAttr("nall_priors", &nall_priors));
        int points_per_prior;
        TF_RETURN_IF_ERROR(c->GetAttr("points_per_prior", &points_per_prior));
        int npath_attributes;
        TF_RETURN_IF_ERROR(c->GetAttr("npath_attributes", &npath_attributes));
        float prior_threshold;
        TF_RETURN_IF_ERROR(c->GetAttr("prior_threshold", &prior_threshold));
        bool equal_spacing;
        TF_RETURN_IF_ERROR(c->GetAttr("equal_spacing", &equal_spacing));
        bool prior_assignment_constraint;
        TF_RETURN_IF_ERROR(c->GetAttr("prior_assignment_constraint", &prior_assignment_constraint));
        bool using_invalid_path_class;
        TF_RETURN_IF_ERROR(c->GetAttr("using_invalid_path_class", &using_invalid_path_class));
        int edges_per_path;
        TF_RETURN_IF_ERROR(c->GetAttr("edges_per_path", &edges_per_path));
        bool verbose;
        TF_RETURN_IF_ERROR(c->GetAttr("verbose", &verbose));
        int num_coordinates_per_point = edges_per_path * 2;
        vector<::shape_inference::DimensionHandle> dims_out;
        dims_out.push_back(c->UnknownDim());          // Number of images.
        dims_out.push_back(c->MakeDim(nall_priors));  // Number of path priors.
        // Number of points in the path + number of classes + 1 sum angle ratio
        // + number of path attributes.
        dims_out.push_back(c->MakeDim(NUM_SUM_ANGLE_RATIOS + nclasses + npath_attributes +
                                      points_per_prior * num_coordinates_per_point));
        c->set_output(0, c->MakeShape(dims_out));
        return Status::OK();
    })
    .Doc(R"doc(
      Path generator op.
      @TODO
      )doc");

struct float2 {
    float x;
    float y;
};

struct Path_Edges {
    vector<int> left_vertices;
    vector<int> right_vertices;
    vector<int> center_rail_vertices;

    int left_interpolated_start_vertex;
    int right_interpolated_start_vertex;
    int center_interpolated_start_vertex;

    int image_id = 0;

    float sum_angle_ratio = -1.0;

    // Class ids will range from about -10 to 10, where negative numbers indicate
    // paths to the left of the ego path, which has and id of 0, and positive numbers
    // indicate paths to the right of the ego path.
    // The ego path is the path that an agent, e.g. a car, is currently following.
    int class_id = -1;  // Class index of -1 means the polyline will be skipped.

    // Path attributes except the left edge, right edge and center rail attributes.
    vector<int> path_attribute;
};

// @yifangx add comments.
struct Image {
    vector<int> invalid_path_groups;
    int start_path_group = -1;
    int npath_edge_groups = 0;
};

class GeneratePathFromEdgesOp : public OpKernel {
 private:
    int nclasses_;
    int nall_priors_;
    int points_per_prior_;
    int npath_attributes_;
    float prior_threshold_;
    bool equal_spacing_;
    bool prior_assignment_constraint_;
    bool using_invalid_path_class_;
    int edges_per_path_;
    bool verbose_;
    int coordinates_per_point_;

    // class_id for all the 7 classes if using prior assignment constraint.
    int DP_0 = 0;
    int DP_Minus_1 = 1;
    int DP_1 = 2;
    int DP_Minus_1_fork_left = 3;
    int DP_0_fork_left = 4;
    int DP_0_fork_right = 5;
    int DP_1_fork_right = 6;

    // Initialize attribute numbers for Forward and Opposite Edges.
    // These attributes enums will be modified later when 'center_rail' is added
    // if edges_per_path == 3
    int Path_Attribute_Opposite_Edge = 2;
    int Path_Attribute_Forward_Edge = 3;

    // Initialize attribute number for center rail to -inf.
    // Will be changed in the constructor.
    int Center_Rail = INT_MIN;

    // Valid attributes and count per class for a path edge group.
    unordered_map<int, int> valid_attribute_count_{{Left_Edge, 1}, {Right_Edge, 1}};

 public:
    explicit GeneratePathFromEdgesOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("nclasses", &nclasses_));
        OP_REQUIRES_OK(context, context->GetAttr("nall_priors", &nall_priors_));
        OP_REQUIRES_OK(context, context->GetAttr("points_per_prior", &points_per_prior_));
        OP_REQUIRES_OK(context, context->GetAttr("npath_attributes", &npath_attributes_));
        OP_REQUIRES_OK(context, context->GetAttr("prior_threshold", &prior_threshold_));
        OP_REQUIRES_OK(context, context->GetAttr("equal_spacing", &equal_spacing_));
        OP_REQUIRES_OK(context, context->GetAttr("prior_assignment_constraint",
                                                 &prior_assignment_constraint_));
        OP_REQUIRES_OK(context,
                       context->GetAttr("using_invalid_path_class", &using_invalid_path_class_));
        OP_REQUIRES_OK(context, context->GetAttr("edges_per_path", &edges_per_path_));

        /*
        // Adjust the ids of allowed classes for prior constraint if invalid_path class is used.
        // Since class with id 0 will have name "invalid", move all the rest of the class by 1.
        */
        if (using_invalid_path_class_) {
            DP_Minus_1 += 1;
            DP_0 += 1;
            DP_1 += 1;
            DP_Minus_1_fork_left += 1;
            DP_0_fork_left += 1;
            DP_0_fork_right += 1;
            DP_1_fork_right += 1;
        }

        // Set coordinates per point to number of edges in a path.
        coordinates_per_point_ = edges_per_path_ * 2;

        // If edges_per_path == 3, we add another required attibute aka center_rail to the
        // valid_attribute_count map. We also update the IDs of opposite_edge and forward_edge
        // to match that of the label processor in the spec.
        if (edges_per_path_ == 3) {
            Center_Rail = 2;
            valid_attribute_count_.insert({{Center_Rail, 1}});
            Path_Attribute_Opposite_Edge = 3;
            Path_Attribute_Forward_Edge = 4;
        }

        OP_REQUIRES(context, nclasses_ > 0,
                    errors::InvalidArgument("Need nclasses > 0, got ", nclasses_));
        OP_REQUIRES(context, nall_priors_ > 0,
                    errors::InvalidArgument("Need nall_priors > 0, got ", nall_priors_));
        OP_REQUIRES(context, nall_priors_ >= nclasses_,
                    errors::InvalidArgument("Number of classes: ", nclasses_,
                                            " is larger than the total number of path priors ",
                                            nall_priors_, "."));
        OP_REQUIRES(context, points_per_prior_ > 0,
                    errors::InvalidArgument("Need points_per_prior > 0, got ", points_per_prior_));
        OP_REQUIRES(context, prior_threshold_ >= 0,
                    errors::InvalidArgument("Need prior_threshold >= 0.0, got ", prior_threshold_));
        OP_REQUIRES(context, prior_threshold_ <= 1,
                    errors::InvalidArgument("Need prior_threshold <= 1.0, got ", prior_threshold_));
        OP_REQUIRES(context, npath_attributes_ >= 0,
                    errors::InvalidArgument("Need npath_attributes >= 0, got ", npath_attributes_));
        OP_REQUIRES(
            context, (edges_per_path_ == 2 || edges_per_path_ == 3),
            errors::InvalidArgument("Need edges_per_path to be 2 or 3, got ", edges_per_path_));
        OP_REQUIRES_OK(context, context->GetAttr("verbose", &verbose_));
    }

    /*
    // _write_output
    //   loops through the output vertices and assigns them to the output tensor.
    //
    // out:                 pointer to output tensor (B * AP * (1 + C + A + N * E * D).
    // image_id:            image number.
    // nclasses:            number of path classes (C).
    // nall_priors:         number of path priors in each image (AP).
    // points_per_prior:    number of points in each path prior (N).
    // npath_attributes:    number of attributes (A).
    // edges_per_path:      number of edges per path (E).
    // sum_angle_ratios:    sum_angle_ratio associated with each prior (B * AP).
    // classes: The class   id associated with each prior (B * AP).
    // path_attributes:     path attributes (B * AP * A)
    // path_delta_vertices: The encoded path vertices represented as a delta between the prior and
    //                      the interpolated labeled path (B * AP * E * N * D) where D = 2 is the
    //                      number of coordinates in a vertex.
    */
    void _write_output(float* out[], int image_id, int nclasses, int nall_priors,
                       int points_per_prior, int npath_attributes, const float* sum_angle_ratios,
                       const int* classes, const vector<vector<int>> path_attributes,
                       const float2* path_delta_vertices) {
        int out_index = 0;

        // Display path_attributes values.
        if (verbose_) {
            printf("---\n");
            printf("   path_attributes has %lu rows\n", path_attributes.size());
            printf("   Each row has %lu column(s)\n", path_attributes[0].size());
            for (uint j = 0; j < path_attributes.size(); j++) {
                for (uint k = 0; k < path_attributes[0].size(); k++) {
                    printf("   path_attributes[%d][%d] = %d\n", j, k, path_attributes[j][k]);
                }
            }
        }

        // Populate the output tensor.
        for (int prior = 0; prior < nall_priors; prior++) {
            // Write the sum angle ratio.
            int sum_angle_ratio_idx = image_id * nall_priors + prior;
            (*out)[out_index++] = sum_angle_ratios[sum_angle_ratio_idx];

            // Write the class for this prior as a one hot vector.
            for (int cls = 0; cls < nclasses; cls++) {
                int class_idx = image_id * nall_priors + prior;
                if (classes[class_idx] == cls) {
                    (*out)[out_index++] = 1.0;
                } else {
                    (*out)[out_index++] = 0.0;
                }
            }

            // Write path attibutes.
            if (npath_attributes > 0) {
                int path_attribute_idx = image_id * nall_priors + prior;
                for (int j = 0; j < npath_attributes; j++) {
                    if (path_attributes[path_attribute_idx][j] == Path_Attribute_Opposite_Edge) {
                        (*out)[out_index++] = 1.0;
                    } else {
                        (*out)[out_index++] = 0.0;
                    }
                }
            }

            // Write the Locations.
            for (int point = 0; point < points_per_prior; point++) {
                // Index of the left and right vertices for this point.
                int leftv =
                    image_id * nall_priors + prior * edges_per_path_ * points_per_prior + point;
                int rightv = leftv + points_per_prior;
                int centerv = rightv + points_per_prior;

                // Write the result, incrementing the output tensor pointer
                (*out)[out_index++] = path_delta_vertices[leftv].x;
                (*out)[out_index++] = path_delta_vertices[leftv].y;
                (*out)[out_index++] = path_delta_vertices[rightv].x;
                (*out)[out_index++] = path_delta_vertices[rightv].y;
                if (edges_per_path_ == 3) {
                    (*out)[out_index++] = path_delta_vertices[centerv].x;
                    (*out)[out_index++] = path_delta_vertices[centerv].y;
                }
            }
        }
    }

    /*
    // _distance_path_edge_to_prior
    //    computes the distance between each prior/path_edge pair by summing the distances between
    each matched pair of
    //    points.
    //
    // Inputs:
    //   path_edge_groups:       Path_Edges struct with the vertices for each path edge group.
    //   interpolated_vertices:  interpolated vertices for all paths and images.
    //   path_prior_vertices:    path prior vertices (AP * N * 2).
    //   npath_edge_groups:      number of labeled paths in each image.
    //   nall_priors:            number of path priors in each image (AP).
    //   points_per_prior:       number of points in each path prior (N).
    //
    // distance_vector:        average distance from path edges to priors, (AP * npath_edge_groups).
    */
    vector<float> _distance_path_edge_to_prior(vector<Path_Edges> path_edge_groups,
                                               vector<float2> interpolated_vertices,
                                               vector<float2> path_prior_vertices,
                                               const int npath_edge_groups, const int nall_priors,
                                               const int points_per_prior) {
        // initialize the output distance vector.
        vector<float> distance_vector(nall_priors * npath_edge_groups);

        const int edges_in_prior = 2;  // The priors passed in only have 2 edges.

        // We are only computing the distance between priors and the left and right edge of a path.
        // This is no design restriction and center rail can be added too.
        for (int prior = 0; prior < nall_priors; prior++) {
            int path_edge_group_index = 0;
            for (auto& path_edge_group : path_edge_groups) {
                // Sum up the distances between the left and right points for this
                // pair of prior and labeled path edges
                float dist = 0.0;
                for (int point = 0; point < points_per_prior; point++) {
                    // get the path prior vertex indices
                    int left_prior_vertex = edges_in_prior * (prior * points_per_prior + point);
                    int right_prior_vertex = left_prior_vertex + 1;

                    // get the path edge vertex indices
                    int left_path_edge_group_vertex =
                        path_edge_group.left_interpolated_start_vertex + point;
                    int right_path_edge_group_vertex =
                        path_edge_group.right_interpolated_start_vertex + point;

                    // accumulate left and right distances
                    dist += _distance(path_prior_vertices[left_prior_vertex],
                                      interpolated_vertices[left_path_edge_group_vertex]);
                    dist += _distance(path_prior_vertices[right_prior_vertex],
                                      interpolated_vertices[right_path_edge_group_vertex]);
                }
                distance_vector[prior * npath_edge_groups + path_edge_group_index] =
                    dist / (edges_in_prior * points_per_prior);
                path_edge_group_index += 1;
            }
        }

        return distance_vector;
    }

    /*
    // _reorder_vector uses a list of indices to reorder a vector in place.
    // v:     Vector to reorder.
    // order: Vector of indices that indicate the new order.
    */
    template <class T>
    vector<T> _reorder_vector(vector<T> v, vector<int> order) {
        for (unsigned int s = 1, d; s < order.size(); s++) {
            for (d = order[s]; d < s; d = order[d]) {
            }
            if (d == s) {
                while (d = order[d], d != s) swap(v[s], v[d]);
            }
        }
        return v;
    }

    /*
    // _class_path_type returns the path type, the type of priors it belongs to.
    // class_id: The class id of the path edge group.
    // Return the path type id.
    //
    // Currently the prior assignment constraint only supports 7 classes, with name
    // and class_id specified in the const variables in the beginning.
    // Divide it into three groups specified below according to the priors
    // it can be assigned to.
    */
    int _class_path_type(int class_id) {
        if (class_id == DP_Minus_1 || class_id == DP_0 || class_id == DP_1)
            return Middle_prior_group;
        else if (class_id == DP_Minus_1_fork_left || class_id == DP_0_fork_left)
            return Left_prior_group;
        else if (class_id == DP_0_fork_right || class_id == DP_1_fork_right)
            return Right_prior_group;
        else
            LOG(ERROR) << "\n--------------------- ERROR ---------------------\n"
                       << "Prior assignment constraint currently only hard-coded"
                       << "for 7 classes model, which should have class_id ranging from "
                       << "0 to 6, but got " << class_id << " here";
        exit(1);
    }

    /*
    // _prior_path_type returns the prior type according to the prior assignment constraint
    // scheme.
    // prior_id: The index of the distance in the distances_to_priors vector.
    // returns the prior type id.
    //
    // Based on the fact that, if you are using 3 priors per location(currently required for
    // prior assignment constraint), the prior_id is continous number from left prior to
    // right prior at every location(from left top to the right bottom), so the prior_id % 3
    // will return the prior type at each location, in other words, 0 means left prior, 1 means
    // middle prior, 2 means right prior.
    */
    int _prior_path_type(int prior_id) { return prior_id % 3; }

    /*
    // Description of prior_assignment_constraint algorithm:
    // This algorithm constrains priors to such that the prior path type must be the
    // same as the class path type. e.g. Maintaining classes can only be assigned maintaining
    // priors. For all priors that do not meet their distance is set to FLT_MAX so
    // they can be ignored.
    //
    // Apply prior assignment constraint when assign ground truth to priors.
    //          Goals: To enforce the prior assignment within designed constraint, the
    //          distances_to_priors tensor is modified to only contain distances for
    //          valid [path_edge_group, prior] pairs.
    //          Steps: Perform a pre-processing step to modify distances_to_priors if the prior
    //          assignment constraint is enabled. Specifically, for every distance in the
    //          distances_to_priors tensor, if the path edge group can't be assigned to that
    //          prior according to the prior assignment constraint scheme, then disable
    //          (set to FLT_MAX) that corresponding distances_to_priors element.
    // distances_to_priors:  Vector of distances between each pair of priors
    //                       and path edge groups for this image.
    // path_edge_groups:     Path_Edges struct with the vertices for each path edge group.
    // still_valid_distances: Still valid distance counts in the distances_to_priors tensor.
    // npath_edge_groups:    The total number of path edge groups in this image.
    // ndistances:           The total number of the elements in the distances_to_priors vector.
    */
    void _apply_prior_assignment_constraint(vector<float>* distances_to_priors,
                                            vector<Path_Edges> path_edge_groups,
                                            int* still_valid_distances, int npath_edge_groups,
                                            int ndistances) {
        for (unsigned distance_id = 0; distance_id < ndistances; distance_id++) {
            int prior_to_assign = distance_id / npath_edge_groups;
            int path_to_assign_prior_to = distance_id % npath_edge_groups;
            int class_id = path_edge_groups[path_to_assign_prior_to].class_id;
            // According to the class_id and the index of the prior_to_assign
            // to determine whether it meets the prior assignment constraint scheme, if not then
            // disable that distance (set to FLT_MAX) in the distance_to_priors vector.
            if (_class_path_type(class_id) != _prior_path_type(prior_to_assign)) {
                (*distances_to_priors)[distance_id] = FLT_MAX;
                *still_valid_distances -= 1;
            }
        }
    }

    /*
    // _assign_paths_to_priors assigns N unique priors to M paths. The default is
    //                         1 prior for each path.
    // min_prior_idxs:         Reference to the vector of prior indices.
    // min_prior_dists:        Reference to the vector of prior distances.
    // distances_to_priors:    Vector of distances between each pair of priors
    //                         and path edge groups for this image.
    // nall_priors:                The total number of path priors.
    // npath_edge_groups:            The total number of path edge groups in this image.
    // prior_assignment_constraint:  If true, enable the prior assignment constraint,
    //                   If false, don't enable the prior assignment constraint.
    //                   By default, it's false.
    */
    void _assign_paths_to_priors(vector<int>* min_prior_idxs, vector<float>* min_prior_dists,
                                 vector<float> distances_to_priors, int nall_priors,
                                 int npath_edge_groups, int priors_per_path,
                                 vector<Path_Edges> path_edge_groups,
                                 bool prior_assignment_constraint = false) {
        // TODO(blythe): ensure method can handle more than 1 prior_per_path.

        priors_per_path = 1;

        int max_assignments = npath_edge_groups * priors_per_path;
        int nassignments = 0;
        int still_valid_distances = distances_to_priors.size();
        int ndistances = distances_to_priors.size();
        vector<int> nall_priors_per_path(npath_edge_groups, 0);
        if (prior_assignment_constraint) {
            _apply_prior_assignment_constraint(&distances_to_priors, path_edge_groups,
                                               &still_valid_distances, npath_edge_groups,
                                               ndistances);
        }

        // While there are still paths to be assigned priors (rows in the distance matrix)
        // continue the assignment process.
        while ((still_valid_distances > 0) && (nassignments < max_assignments)) {
            nassignments += 1;

            // Find the minimum distance in the distance matrix and make the assignment
            // between the associated path and prior.
            int index_of_min = 0;
            for (unsigned i = 0; i < distances_to_priors.size(); ++i) {
                if (distances_to_priors[i] < distances_to_priors[index_of_min]) {
                    index_of_min = i;
                }
            }

            if (distances_to_priors[index_of_min] == FLT_MAX) {
                LOG(ERROR) << "\n--------------------- ERROR ---------------------\n"
                           << "No valid priors for the paths. This should never happen.";
                return;
            }

            int prior_to_assign = index_of_min / npath_edge_groups;          // row number
            int path_to_assign_prior_to = index_of_min % npath_edge_groups;  // column number

            (*min_prior_idxs)[path_to_assign_prior_to] = prior_to_assign;
            (*min_prior_dists)[path_to_assign_prior_to] = distances_to_priors[index_of_min];
            nall_priors_per_path[path_to_assign_prior_to] += 1;

            // Get indices to remove assigned prior from the distance matrix
            // [prior_to_assign:(prior_to_assign + npath_edge_groups)].
            set<int> all_indices_to_remove;
            for (int ind = 0; ind < npath_edge_groups; ind++) {
                all_indices_to_remove.insert((prior_to_assign * npath_edge_groups) + ind);
            }

            // Get indices to remove path given the assignment from the distance matrix
            // if we've reached the maximum number of priors_per_path.
            // [path_to_assign_prior_to:npath_edge_groups:end].
            if (nall_priors_per_path[path_to_assign_prior_to] == priors_per_path) {
                for (int j = 0; j < nall_priors; ++j) {
                    all_indices_to_remove.insert(path_to_assign_prior_to + j * npath_edge_groups);
                }
            }

            // Setting the distance value to FLT_MAX ensures it won't be chosen as the min.
            // Effectively removes values from consideration without dangerous resizing.
            for (const auto& index_to_remove : all_indices_to_remove) {
                // Check if the distance has been disabled (for example,
                // due to prior assignment constraint). If true, then ignore this
                // distance to avoid double counting the still_valid_distances.
                if ((prior_assignment_constraint &&
                     distances_to_priors[index_to_remove] != FLT_MAX) ||
                    (!prior_assignment_constraint)) {
                    distances_to_priors[index_to_remove] = FLT_MAX;
                    still_valid_distances -= 1;
                }
            }
        }
    }

    /*
    // _ensure_path_edge_groups_close_enough_to_priors checks if the associated
    //                      path group and prior pairs are within a reasonable
    //                      distance from each other, where reasonable is
    //                      determined by a threshold. If not, print a warning.
    // min_prior_idxs:      Pointer to the vector of prior indices.
    // min_prior_dists:     Pointer to the vector of prior distances.
    // image_id:            The image id. Only used for the output message.
    // threshold:           The threshold in pixels.
    */
    void _ensure_path_edge_groups_close_enough_to_priors(vector<int> min_prior_idxs,
                                                         vector<float> min_prior_dists,
                                                         int image_id, float threshold) {
        for (unsigned int path_edge_group = 0; path_edge_group < min_prior_idxs.size();
             path_edge_group++) {
            if (min_prior_dists[path_edge_group] > threshold) {
                LOG(WARNING) << "\n--------------------- WARNING ---------------------\n"
                             << "image number " << image_id << ", path edge group "
                             << path_edge_group << " the closest prior \n (after collision "
                             << "handling) is a distance of " << min_prior_dists[path_edge_group]
                             << " pixels \n away which is greater than the tolerated threshold "
                             << "of " << threshold << " \n"
                             << "Adjust your priors to accomodate paths edges such as this. \n\n";
            }
        }
    }

    /*
    // _distance - computes the Euclidean distance between two points.
    //
    // a: float2 point
    // b: float2 point
    */
    float _distance(const float2& a, const float2& b) {
        const float dx = b.x - a.x;
        const float dy = b.y - a.y;
        const float lsq = dx * dx + dy * dy;
        return sqrt(lsq);
    }

    /*
    // _linear_curve_length - computes the arc length of the polyline defined by points.
    //
    // points: a vector of float2 points
    */
    float _linear_curve_length(vector<float2> const& points) {
        auto start = points.begin();
        if (start == points.end()) return 0;
        auto finish = start + 1;
        float sum = 0;
        while (finish != points.end()) {
            sum += _distance(*start, *finish);
            start = finish++;
        }

        return sum;
    }

    /*
    // _interpolate - resamples the polyline defined by input_points at npoints locations.
    // input_points:    Vector of float2s containing the points to interpolate between.
    // npoints:         number of desired interpolated points.
    // equal_spacing:   use equal arc length spacing, else use log spacing.
    */
    vector<float2> _interpolate(vector<float2> const& input_points, int npoints,
                                bool equal_spacing) {
        vector<float2> points = input_points;
        vector<float2> interpolated_points(npoints);

        if (points.size() < 2 || npoints < 2) {
            // degenerate points vector or npoints value
            // for simplicity, this returns an empty vector
            // but special cases may be handled when appropriate for the application
            vector<float2> no_points;

            return no_points;
        }

        // Flip the y-coordinates if not ordered from bottom to top. Don't sort the individual
        // coordinates because you can have lines that are non-monotonic in y.
        if (points[0].y < points[points.size() - 1].y) {
            reverse(points.begin(), points.end());
        }

        // total_length is the total length along a linear interpolation
        // of the points points.
        const float total_length = _linear_curve_length(points);

        // segment_length is the length between interpolated_points points, taken as
        // distance traveled between these points on a linear interpolation
        // of the points points.  The actual Euclidean distance between
        // points in the interpolated_points vector can vary, and is always less than
        // or equal to segment_length.
        vector<float> segment_lengths(npoints - 1);
        if (equal_spacing) {
            const float segment_length = total_length / (npoints - 1);
            segment_lengths[1] = segment_length;
            for (int i = 2; i < (npoints - 1); i++) {
                segment_lengths[i] = segment_length + segment_lengths[i - 1];
            }
        } else {  // If not equal spacing, use logarithmic spacing.
            const float scale = total_length / log(npoints);
            for (int i = 0; i < (npoints - 1); i++) {
                segment_lengths[i] = scale * log(i + 1);
            }
        }

        // start and finish are the current points segment's endpoints
        auto start = points.begin();
        auto finish = start + 1;

        // src_segment_offset is the distance along a linear interpolation
        // of the points curve from its first point to the start of the current
        // points segment.
        float src_segment_offset = 0;

        // src_segment_length is the length of a line connecting the current
        // points segment's start and finish points.
        float src_segment_length = _distance(*start, *finish);

        // The first point in the interpolated_points is the same as the first point
        // in the points.
        interpolated_points[0] = *start;

        for (int i = 1; i < npoints - 1; ++i) {
            // next_offset is the distance along a linear interpolation
            // of the points curve from its beginning to the location
            // of the i'th point in the interpolated_points.
            // segment_length is multiplied by i here because iteratively
            // adding segment_length could accumulate error.
            const float next_offset = segment_lengths[i];

            // Check if next_offset lies inside the current points segment.
            // If not, move to the next points segment and update the
            // points segment offset and length variables.
            while (src_segment_offset + src_segment_length < next_offset) {
                src_segment_offset += src_segment_length;
                start = finish++;
                src_segment_length = _distance(*start, *finish);
            }
            // part_offset is the distance into the current points segment
            // associated with the i'th point's offset.
            const float part_offset = next_offset - src_segment_offset;

            // part_ratio is part_offset's normalized distance into the
            // points segment. Its value is between 0 and 1,
            // where 0 locates the next point at "start" and 1
            // locates it at "finish".  In-between values represent a
            // weighted location between these two extremes.
            // Add FLT_MIN to denominator to prevent 0-division.
            float part_ratio = 0;
            if (src_segment_length != 0) {
                part_ratio = part_offset / src_segment_length;
            }

            // Use part_ratio to calculate the next point's components
            // as weighted averages of components of the current
            // points segment's points.
            interpolated_points[i] = {start->x + part_ratio * (finish->x - start->x),
                                      start->y + part_ratio * (finish->y - start->y)};
        }

        // The first and last points of the interpolated_points are exactly
        // the same as the first and last points from the input,
        // so the iterated calculation above skips calculating
        // the last point in the interpolated_points, which is instead copied
        // directly from the points vector here.
        interpolated_points[npoints - 1] = points.back();

        return interpolated_points;
    }

    /*
    // _vector_angle Returns the angle in degrees between two vectors.
    // vector1:  float2 vector.
    // vector2:  float2 vector.
    */
    float _vector_angle(float2 vector1, float2 vector2) {
        // Norms.

        float vector1_norm = sqrt(pow(vector1.x, 2) + pow(vector1.y, 2));
        float vector2_norm = sqrt(pow(vector2.x, 2) + pow(vector2.y, 2));
        float norm = vector1_norm * vector2_norm;
        // Dot product.
        float cos_angle = ((vector1.x * vector2.x) + (vector1.y * vector2.y)) / (norm + FLT_MIN);
        // Cross product.
        float sin_angle = ((vector1.x * vector2.y) - (vector1.y * vector2.x)) / (norm + FLT_MIN);

        return atan2(sin_angle, cos_angle) * (180.0 / PI);
    }

    /*
    // _get_sum_angle_ratio Returns the ratio of the sum of angles between adjacent line segments
                            and the sum of straight angles.
    //                      sum_angle_ratio = sum(angle_i / 180.0, i=[1:N]) / N * 180.0,
    //                      where angle_i is the interior angle between two adjacent lines segments.
    //                      This is a measure of curvature.
    // vertices:  Vector of float2s containing the points to measure the sum angle between.
    // npoints:   Number of points in the vertices vector.
    */
    float _get_sum_angle_ratio(vector<float2> vertices, int npoints) {
        // Subtract 2 from the number of points because we have one angle per line segment.

        float sum_straight_angles = (npoints - 2) * 180.0;
        float sum_path_angles = 0.0;
        float2 vector1;
        float2 vector2;
        float vector_angle;

        // For each adjacent pair of line segments, calculate the angle between them and
        // sum with previous.
        for (int i = 1; i < (npoints - 1); i++) {
            vector1.x = vertices[i].x - vertices[i - 1].x;
            vector1.y = vertices[i].y - vertices[i - 1].y;

            vector2.x = vertices[i].x - vertices[i + 1].x;
            vector2.y = vertices[i].y - vertices[i + 1].y;

            vector_angle = _vector_angle(vector1, vector2);

            sum_path_angles += min(static_cast<float>(360) - abs(vector_angle), abs(vector_angle));
        }

        float sum_angle_ratio = sum_path_angles / sum_straight_angles;

        return sum_angle_ratio;
    }

    /*
    // _check_if_valid_attribute_per_class: Checks if the attribute passed belongs to allowed
                                            attributes in valid_attribute_count_ table. Also if the
                                            class_id passed has matching attribute counts to the
                                            table.
    // class_to_attribute_counter_map: Hashmap holding the counts of attribute per class
    // class_id:                       (int) Id of the class to be checked.
    // attribute:                      (int) Id of the attribute to be checked.
    */
    bool _check_if_valid_attribute_in_class(
        const map<int, map<int, int>>& class_to_attribute_counter_map, int class_id,
        int attribute) {
        if (valid_attribute_count_.count(attribute) == 0) return false;

        auto attribute_counter_map_iter = class_to_attribute_counter_map.find(class_id);
        if (attribute_counter_map_iter == class_to_attribute_counter_map.end()) {
            LOG(ERROR) << "The class " << class_id << " wasn't in the attribute counter map.";
            exit(1);
        }

        for (auto& element : valid_attribute_count_) {
            auto attribute_counter_iter = attribute_counter_map_iter->second.find(element.first);
            if (attribute_counter_iter == attribute_counter_map_iter->second.end()) return false;
            if (attribute_counter_iter->second != element.second) return false;
        }

        return true;
    }

    /*
    // _extract_path_edges_groups_with_valid_attributes: Extracts the path edge groups in a image
    // allowing only those classes with edge groups satisfying the VALID_ATTRIBUTES_COUNT table.
    //
    // Output:
    //   path_edge_groups:            Vector of type Path_Edges holding the indices of coordinates
    of
                                      valid path groups. The coordinates are saved in "vertices"
                                      vector.
    //   vertices:                    Vector holding the coordinates of the valid path groups.
    //   npath_edge_groups_per_image: Number of valid path edge groups per image.
    //
    // Input:
    //   nimages:                    (int) Number of images (B).
    //   npath_attributes:           (int) Number of attributes (A).
    //   width:                      (int) Width of the image (IW).
    //   height:                     (int) Height of the image (IH).
    //   polylines_per_image:        Flat Tensor holding number of polylines per image.
    //   class_ids_per_polyline:     Flat Tensor holding class_id of each polyline.
    //   attributes_per_polyline:    Flat tensor holding attribute of each polyline.
    //   vertex_counts_per_polyline: Flat tensor holding number of points in each polyline.
    //   input_polyline_vertices:    Flat tensor holding the coordinates of vertices in each
    polyline.
    */
    void _extract_path_edges_groups_with_valid_attributes(
        vector<Path_Edges>* path_edge_groups, vector<float2>* vertices,
        vector<int>* npath_edge_groups_per_image, const int nimages, const int npath_attributes,
        const int width, const int height, const TTypes<const int>::Flat& polylines_per_image,
        const TTypes<const int>::Flat& class_ids_per_polyline,
        const TTypes<const int>::Flat& attributes_per_polyline,
        const TTypes<const int>::Flat& vertex_counts_per_polyline,
        const TTypes<const float>::Flat& input_polyline_vertices) {
        int v = 0;
        int p = 0;
        int max_path_edge_group_index = -1;
        // Store left edge, right edge and center rail attributes.
        // Check the length = class_ids_per_polyline length later.
        std::vector<int> edge_attributes;
        // Store attributes except left edge, right edge and center rail.
        // Check the length = class_ids_per_polyline length later.
        std::vector<vector<int>> path_attributes;
        for (int image_id = 0; image_id < nimages; image_id++) {
            int npoly_image = polylines_per_image(image_id);
            int last_image = p + npoly_image;

            int npath_groups_image = 0;
            int npath_edges_image = 0;

            map<int, map<int, int>> class_to_attribute_counter_map;
            map<int, int> class_to_path_edge_group_index_map;

            // Loop through polylines within one image.
            // Fill the class to attribute counters.
            for (int i = p; i < last_image; i++) {
                int class_id = class_ids_per_polyline(i);
                int edge_attribute = attributes_per_polyline(i);
                edge_attributes.push_back(
                    edge_attribute);  // add edge attrbiutes for this polyline to edge_attributes
                class_to_attribute_counter_map[class_id][edge_attribute]++;  // number of edge
                                                                             // attributes per
                                                                             // class_id in this
                                                                             // image

                // add path attributes for this edge to path_attributes
                vector<int> path_attribute_per_edge;
                if (npath_attributes > 0) {
                    for (int j = 0; j < npath_attributes; j++) {
                        path_attribute_per_edge.push_back(
                            attributes_per_polyline((j + 1) * npoly_image + i));
                    }
                } else {
                    path_attribute_per_edge.push_back(-1);
                }
                path_attributes.push_back(path_attribute_per_edge);
            }

            // Loop through polylines within one image (again).
            for (int base_p = p; p < last_image; p++) {
                int nvertices_poly = vertex_counts_per_polyline(p);
                int class_id = class_ids_per_polyline(p);
                int last_polyline = v + nvertices_poly;
                int edge_attribute = edge_attributes[p - base_p];

                vector<int> path_attribute;
                path_attribute.push_back(path_attributes[p - base_p][0]);
                if (npath_attributes > 1) {
                    for (int j = 1; j < npath_attributes; j++) {
                        path_attribute.push_back(path_attributes[p - base_p][j]);
                    }
                }

                // If the path edge has a valid attribute for the class
                // and if the class has valid attribute counts.
                if (_check_if_valid_attribute_in_class(class_to_attribute_counter_map, class_id,
                                                       edge_attribute)) {
                    // Do not reset the max_path_edge_group_index.
                    // Keep incrementing it across the images.
                    if (class_to_path_edge_group_index_map.count(class_id) == 0) {
                        max_path_edge_group_index++;
                        // class_to_path_edge_group_index_map[class_id] = max_path_edge_group_index;
                        class_to_path_edge_group_index_map.emplace(class_id,
                                                                   max_path_edge_group_index);
                    }

                    int path_edge_group_index = class_to_path_edge_group_index_map[class_id];

                    for (; v < last_polyline; v++) {
                        // Extract data from input arrays.
                        int d = v * 2;
                        float x = input_polyline_vertices(d) * width;
                        float y = input_polyline_vertices(d + 1) * height;

                        // Collect data in vertices array.
                        vertices->at(v) = {x, y};

                        // Collect the vertices in path edge groups.
                        if (edge_attribute == Left_Edge) {
                            path_edge_groups->at(path_edge_group_index).left_vertices.push_back(v);
                        } else if (edge_attribute == Right_Edge) {
                            path_edge_groups->at(path_edge_group_index).right_vertices.push_back(v);
                        } else if (edge_attribute == Center_Rail) {
                            path_edge_groups->at(path_edge_group_index)
                                .center_rail_vertices.push_back(v);
                        }
                    }
                    npath_edges_image++;

                    // Collect other data about this path from the original data.
                    path_edge_groups->at(path_edge_group_index).image_id = image_id;
                    path_edge_groups->at(path_edge_group_index).class_id = class_id;
                    path_edge_groups->at(path_edge_group_index).path_attribute = path_attribute;
                } else if (verbose_) {
                    // Throw away path edges with attribute 0.
                    LOG(WARNING) << "\n--------------------- WARNING ---------------------\n"
                                 << "image number " << image_id << " has "
                                 << class_to_attribute_counter_map[class_id][edge_attribute]
                                 << " path edges which have the same class id: " << class_id
                                 << " and attribute: " << edge_attribute
                                 << ". Throwing away those paths edges. \n\n";
                }
                v = last_polyline;
            }

            if (npath_edges_image % 2 != 0) {
                LOG(ERROR) << "\n--------------------- ERROR ---------------------\n"
                           << "Odd number of edges found in image " << image_id << " \n\n";
            }
            npath_groups_image = npath_edges_image / edges_per_path_;
            npath_edge_groups_per_image->at(image_id) = npath_groups_image;
        }
    }

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& polyline_vertices_tensor = context->input(0);
        auto input_polyline_vertices = polyline_vertices_tensor.flat<float>();

        const Tensor& path_prior_vertices_tensor = context->input(1);
        auto input_path_prior_vertices = path_prior_vertices_tensor.flat<float>();

        const Tensor& vertex_counts_per_polyline_tensor = context->input(2);
        auto vertex_counts_per_polyline = vertex_counts_per_polyline_tensor.flat<int>();

        const Tensor& class_ids_per_polyline_tensor = context->input(3);
        auto class_ids_per_polyline = class_ids_per_polyline_tensor.flat<int>();

        const Tensor& attributes_per_polyline_tensor = context->input(4);
        auto attributes_per_polyline = attributes_per_polyline_tensor.flat<int>();

        const Tensor& polylines_per_image_tensor = context->input(5);
        auto polylines_per_image = polylines_per_image_tensor.flat<int>();

        const Tensor& width_tensor = context->input(6);
        int width = *(width_tensor.flat<int>().data());

        const Tensor& height_tensor = context->input(7);
        int height = *(height_tensor.flat<int>().data());

        // check conditions for variables
        OP_REQUIRES(context, 2 == polyline_vertices_tensor.shape().dims(),
                    errors::InvalidArgument("polyline_vertices must be a 2 dimensional tensor, ",
                                            "shape is: ",
                                            polyline_vertices_tensor.shape().DebugString(), "."));
        OP_REQUIRES(context, 2 == polyline_vertices_tensor.shape().dim_size(1),
                    errors::InvalidArgument("polyline_vertices tensor last dimension must be 2, ",
                                            "shape is: ",
                                            polyline_vertices_tensor.shape().DebugString(), "."));

        int nvertices = polyline_vertices_tensor.shape().dim_size(0);

        OP_REQUIRES(context, 2 == path_prior_vertices_tensor.shape().dims(),
                    errors::InvalidArgument("path_prior_vertices must be a 2 dimensional tensor, ",
                                            "shape is: ",
                                            path_prior_vertices_tensor.shape().DebugString(), "."));
        OP_REQUIRES(context, 2 == path_prior_vertices_tensor.shape().dim_size(1),
                    errors::InvalidArgument("path_prior_vertices tensor last dimension must be 2,",
                                            "shape is: ",
                                            path_prior_vertices_tensor.shape().DebugString(), "."));

        OP_REQUIRES(context, 1 == vertex_counts_per_polyline_tensor.shape().dims(),
                    errors::InvalidArgument(
                        "vertex_counts_per_polyline must be a 1 dimensional vector,", " shape is: ",
                        vertex_counts_per_polyline_tensor.shape().DebugString(), "."));
        OP_REQUIRES(context, 1 == class_ids_per_polyline_tensor.shape().dims(),
                    errors::InvalidArgument(
                        "class_ids_per_polyline must be a 1 dimensional vector,", " shape is: ",
                        class_ids_per_polyline_tensor.shape().DebugString(), "."));
        OP_REQUIRES(context, 1 == attributes_per_polyline_tensor.shape().dims(),
                    errors::InvalidArgument(
                        "attributes_per_polyline must be a 1 dimensional vector,", " shape is: ",
                        attributes_per_polyline_tensor.shape().DebugString(), "."));
        OP_REQUIRES(context, 1 == polylines_per_image_tensor.shape().dims(),
                    errors::InvalidArgument("polylines_per_image must be a 1 dimensional vector,",
                                            " shape is: ",
                                            polylines_per_image_tensor.shape().DebugString(), "."));

        int npolylines = vertex_counts_per_polyline_tensor.shape().dim_size(0);
        int len_class_ids_per_polyline = class_ids_per_polyline_tensor.shape().dim_size(0);
        int len_attributes_per_polyline = attributes_per_polyline_tensor.shape().dim_size(0);
        int nimages = polylines_per_image_tensor.shape().dim_size(0);

        OP_REQUIRES(context, npolylines == len_class_ids_per_polyline,
                    errors::InvalidArgument(
                        "vertex_counts_per_polyline vector and ", "class_ids_per_polyline ",
                        "vector shapes are not equal; ",
                        vertex_counts_per_polyline_tensor.shape().DebugString(), " and ",
                        class_ids_per_polyline_tensor.shape().DebugString(), "."));

        OP_REQUIRES(context, len_attributes_per_polyline ==
                                 (npath_attributes_ + NUM_ATTRIBUTES_4_CLASS) * npolylines,
                    errors::InvalidArgument(
                        "vertex_counts_per_polyline and attributes_per_polyline ",
                        "vector shapes are not compatible; ",
                        // (vertex_counts_per_polyline_tensor).tensor<int32, (1)>(), " and ",
                        // attributes_per_polyline_tensor.tensor<int32, (1)>(), "."));
                        (vertex_counts_per_polyline_tensor).DebugString(), " and ",
                        attributes_per_polyline_tensor.DebugString(), "."));

        OP_REQUIRES(
            context, nvertices >= npolylines,
            errors::InvalidArgument("Number of polylines ", npolylines,
                                    " is larger than the number of vertices ", nvertices, "."));

        // Test that the sum of the `vertex_counts_per_polyline` vector adds up to `nvertices`
        int nvertices_from_vertex_counts = 0;
        for (int i = 0; i < npolylines; i++) {
            nvertices_from_vertex_counts += vertex_counts_per_polyline(i);
        }
        OP_REQUIRES(context, nvertices_from_vertex_counts == nvertices,
                    errors::InvalidArgument(
                        "Sum of vertex_counts_per_polyline", nvertices_from_vertex_counts,
                        " over all polylines does not add up to nvertices ", nvertices, "."));

        // Test that the sum of the `polylines_per_image` vector adds up to `npolylines`
        int npolylines_from_images = 0;
        for (int i = 0; i < nimages; i++) {
            npolylines_from_images += polylines_per_image(i);
        }

        OP_REQUIRES(
            context, npolylines_from_images == npolylines,
            errors::InvalidArgument("Sum of `polylines_from_images` over all images does not ",
                                    "add up to `npolylines`"));

        // Check value assumptions of input vertices.
        // No polyline vertices is allowed.
        // TODO(blythe): Consider using the tensorflow Check_Numeric ops function for infs and nans.
        if (input_polyline_vertices.size() > 0) {
            for (int i = 0; i < input_polyline_vertices.size(); i++) {
                OP_REQUIRES(
                    context, isinf(input_polyline_vertices(i)) == false,
                    errors::InvalidArgument("Polyline vertex ", i, "is infinite. ",
                                            "Operation expects real valued input vertices."));
                OP_REQUIRES(
                    context, isnan(input_polyline_vertices(i)) == false,
                    errors::InvalidArgument("Polyline vertex ", i, "is nan. ",
                                            "Operation expects real valued input vertices."));
            }
        }

        if (input_path_prior_vertices.size() > 0) {
            for (int i = 0; i < input_path_prior_vertices.size(); i++) {
                OP_REQUIRES(
                    context, isinf(input_path_prior_vertices(i)) == false,
                    errors::InvalidArgument("Prior vertex ", i, "is infinite. ",
                                            "Operation expects real valued input vertices."));
                OP_REQUIRES(
                    context, isnan(input_path_prior_vertices(i)) == false,
                    errors::InvalidArgument("Prior vertex ", i, "is nan. ",
                                            "Operation expects real valued input vertices."));
            }
        }

        if (len_attributes_per_polyline != (npath_attributes_ + 1) * npolylines) {
            LOG(ERROR) << "\n--------------------- ERROR ---------------------\n"
                       << "Invalid vertex_counts_per_polyline vector "
                       << "and/or len_attributes_per_polyline \n"
                       << "vertex_counts_per_polyline = " << npolylines
                       << "len_attributes_per_polyline = " << len_attributes_per_polyline << "\n\n";
            exit(1);
        }

        // Initialize vertices, path_edges, and images struct.
        vector<float2> vertices(nvertices);
        vector<float2> interpolated_vertices(npolylines * points_per_prior_);
        vector<float2> path_prior_vertices(nall_priors_ * 2 * points_per_prior_);

        // Initial value is equal to the normalized width and height of the image, a value you
        // should never have in the output.
        vector<float2> path_delta_vertices(
            nimages * nall_priors_ * edges_per_path_ * points_per_prior_, {-1, -1});
        // Upper limit on the number of path edge groups is one edge per class.
        vector<Path_Edges> path_edge_groups((nclasses_ + 1) * nimages);
        vector<Image> images(nimages);

        // Vector for path classes. If invalid_path_class is used, the default class assigned to
        // a prior is class 0, else none of the classes.
        int default_class_index = -1;
        if (using_invalid_path_class_) default_class_index = 0;
        vector<int> path_classes(nimages * nall_priors_, default_class_index);

        // Vector for sum angle ratios.
        vector<float> path_sum_angle_ratios(nimages * nall_priors_, -1);
        int row = (npath_attributes_ > 1) ? npath_attributes_ : 1;
        vector<int> path_attributes_row(row, -1);
        // path_attributes' size is (nimages * nall_priors_ x (npath_attributes+1)).
        vector<vector<int>> path_attributes(nimages * nall_priors_, path_attributes_row);

        vector<int> npath_edge_groups_per_image(nimages, 0);

        if (nvertices > 0) {
            // Collect the prior vertices in the prior_path_vertices vector.
            int d = 0;
            for (int prior = 0; prior < nall_priors_; prior++) {
                for (int pv = 0; pv < 2 * points_per_prior_; pv++, d++) {
                    path_prior_vertices[d].x = input_path_prior_vertices(2 * d) * width;
                    path_prior_vertices[d].y = input_path_prior_vertices(2 * d + 1) * height;
                }
            }

            if (verbose_) {
                // print the prior vertices.
                printf("Prior Vertices: \n");
                for (int prior = 0; prior < nall_priors_; prior++) {
                    printf("---\n");
                    for (int v = 0; v < points_per_prior_; v++) {
                        int leftv, rightv;
                        leftv = prior * 2 * points_per_prior_ + 2 * v;
                        rightv = prior * 2 * points_per_prior_ + (2 * v + 1);
                        printf(
                            "    prior %d: vertex (%d, %d)"
                            "\tleft: %.2f, %.2f \tright: %.2f, %.2f\n",
                            prior, leftv, rightv, path_prior_vertices[leftv].x,
                            path_prior_vertices[leftv].y, path_prior_vertices[rightv].x,
                            path_prior_vertices[rightv].y);
                    }
                }
            }

            _extract_path_edges_groups_with_valid_attributes(
                &path_edge_groups, &vertices, &npath_edge_groups_per_image, nimages,
                npath_attributes_, width, height, polylines_per_image, class_ids_per_polyline,
                attributes_per_polyline, vertex_counts_per_polyline, input_polyline_vertices);

            // Interpolate the left, right and center path edges separately.
            int interpolated_vertices_index = 0;
            int n_interpolated_paths = 0;
            for (unsigned int pe = 0; pe < path_edge_groups.size(); pe++) {
                float sum_angle_ratio_left = -1;
                float sum_angle_ratio_right = -1;
                if (path_edge_groups[pe].left_vertices.size() > 1) {
                    // Gather the vertices for this edge in a vector.
                    int nleft = path_edge_groups[pe].left_vertices.size();
                    vector<float2> original_left(nleft);
                    for (int v = 0; v < nleft; v++) {
                        int leftv = path_edge_groups[pe].left_vertices[v];
                        original_left[v].x = vertices[leftv].x;
                        original_left[v].y = vertices[leftv].y;
                    }
                    // Perform the interpolation.
                    vector<float2> interpolated_left =
                        _interpolate(original_left, points_per_prior_, equal_spacing_);
                    // Store the interpolated data in a new vector and store the index.
                    for (int v = 0; v < points_per_prior_; v++) {
                        interpolated_vertices[interpolated_vertices_index++] = interpolated_left[v];
                        if (v == 0) {
                            path_edge_groups[pe].left_interpolated_start_vertex =
                                n_interpolated_paths * points_per_prior_;
                            n_interpolated_paths++;
                        }
                    }
                    // Calculate the sum angle ratio for the left edge.
                    sum_angle_ratio_left =
                        _get_sum_angle_ratio(interpolated_left, points_per_prior_);
                }

                if (path_edge_groups[pe].right_vertices.size() > 1) {
                    // Gather the vertices for this edge in a vector.
                    int nright = path_edge_groups[pe].right_vertices.size();
                    vector<float2> original_right(nright);
                    for (int v = 0; v < nright; v++) {
                        int rightv = path_edge_groups[pe].right_vertices[v];
                        original_right[v].x = vertices[rightv].x;
                        original_right[v].y = vertices[rightv].y;
                    }
                    // Perform the interpolation.
                    vector<float2> interpolated_right =
                        _interpolate(original_right, points_per_prior_, equal_spacing_);
                    // Store the interpolated data in a new vector and store the index.
                    for (int v = 0; v < points_per_prior_; v++) {
                        interpolated_vertices[interpolated_vertices_index++] =
                            interpolated_right[v];
                        if (v == 0) {
                            path_edge_groups[pe].right_interpolated_start_vertex =
                                n_interpolated_paths * points_per_prior_;
                            n_interpolated_paths++;
                        }
                    }
                    // Calculate the sum angle ratio for the right edge.
                    sum_angle_ratio_right =
                        _get_sum_angle_ratio(interpolated_right, points_per_prior_);
                }

                // Average the sum angle ratios to get one per path.
                // Note that we are taking sum angle ratio of left and right edge only.
                // This is not by any design restriction and center rail too can be added if
                // present.
                if ((sum_angle_ratio_left >= 0) && (sum_angle_ratio_right >= 0)) {
                    path_edge_groups[pe].sum_angle_ratio =
                        (sum_angle_ratio_left + sum_angle_ratio_right) / 2.0;
                }

                // Calculate interpolated vertices for center rail too.
                if (path_edge_groups[pe].center_rail_vertices.size() > 1 && edges_per_path_ == 3) {
                    // Gather the vertices for this edge in a vector.
                    int ncenter = path_edge_groups[pe].center_rail_vertices.size();
                    vector<float2> original_center(ncenter);
                    for (int v = 0; v < ncenter; v++) {
                        int centerv = path_edge_groups[pe].center_rail_vertices[v];
                        original_center[v].x = vertices[centerv].x;
                        original_center[v].y = vertices[centerv].y;
                    }
                    // Perform the interpolation.
                    vector<float2> interpolated_center =
                        _interpolate(original_center, points_per_prior_, equal_spacing_);
                    // Store the interpolated data in a new vector and store the index.
                    for (int v = 0; v < points_per_prior_; v++) {
                        interpolated_vertices[interpolated_vertices_index++] =
                            interpolated_center[v];
                        if (v == 0) {
                            path_edge_groups[pe].center_interpolated_start_vertex =
                                n_interpolated_paths * points_per_prior_;
                            n_interpolated_paths++;
                        }
                    }
                }
            }

            // Collect the polylines and paths in an Image vector.
            unsigned int start_path_group = 0;
            // Loop over the images (batch dim).
            for (int image_id = 0; image_id < nimages; image_id++) {
                int npath_groups_image = npath_edge_groups_per_image[image_id];

                // Fill in the path_edges in the vector of images (batch_dim * nclasses).
                for (unsigned int p = start_path_group;
                     p < start_path_group + path_edge_groups.size(); p++) {
                    // Skip (don't include) path_edges with negative class indices or
                    // without both left and right vertices or with negative sum_angle_ratio.
                    if ((path_edge_groups[p].class_id < 0) ||
                        ((path_edge_groups[p].left_vertices.size() < 2) ||
                         (path_edge_groups[p].right_vertices.size() < 2)) ||
                        (path_edge_groups[p].sum_angle_ratio < 0)) {
                        images[image_id].invalid_path_groups.push_back(p);
                        continue;
                    }

                    // Skip (don't include) path edges without center rail vertices if
                    // edges_per_path is 3
                    if (edges_per_path_ == 3 &&
                        path_edge_groups[p].center_rail_vertices.size() < 2) {
                        images[image_id].invalid_path_groups.push_back(p);
                        continue;
                    }

                    images[image_id].npath_edge_groups += 1;
                    // Only save the start_path on first occurance
                    if (images[image_id].start_path_group == -1) {
                        images[image_id].start_path_group = p;
                    }
                }
                start_path_group += npath_groups_image;
            }

            // Remove invalid path edge groups. Go backwards to keep indices from interfering.
            for (int image_id = nimages - 1; image_id >= 0; image_id--) {
                if (images[image_id].invalid_path_groups.size() > 0) {
                    for (int i = images[image_id].invalid_path_groups.size() - 1; i >= 0; i--) {
                        path_edge_groups.erase(path_edge_groups.begin() +
                                               images[image_id].invalid_path_groups[i]);
                    }
                }
            }

            // For each image, assign the path_edges to a path_prior.
            for (int image_id = 0; image_id < nimages; image_id++) {
                // get the sum distance between all the points in each prior/path_edge pair
                vector<float> distances_to_priors(nall_priors_ *
                                                  images[image_id].npath_edge_groups);

                distances_to_priors = _distance_path_edge_to_prior(
                    path_edge_groups, interpolated_vertices, path_prior_vertices,
                    images[image_id].npath_edge_groups, nall_priors_, points_per_prior_);

                // For each path_edge, identify the closest prior.
                vector<int> min_prior_idxs(images[image_id].npath_edge_groups);
                vector<float> min_prior_dists(images[image_id].npath_edge_groups);
                // TODO(blythe): make priors_per_path an input argument.
                int priors_per_path = 1;
                _assign_paths_to_priors(&min_prior_idxs, &min_prior_dists, distances_to_priors,
                                        nall_priors_, images[image_id].npath_edge_groups,
                                        priors_per_path, path_edge_groups,
                                        prior_assignment_constraint_);
                if (verbose_) {
                    for (int path_edge_group = 0;
                         path_edge_group < images[image_id].npath_edge_groups; path_edge_group++) {
                        printf(
                            "For path edge group %d assigned prior is "
                            "%d with a distance of %.2f\n",
                            path_edge_group, min_prior_idxs[path_edge_group],
                            min_prior_dists[path_edge_group]);
                    }
                }

                // Indentify any path edge groups that are more than a specified
                // distance away from their closest priors. Threshold is programatically
                // set to be the number of path points * 20% of the image in both the
                // x and y directions. This is to ensure that each point in a path edge group
                // differs from their matched prior points by more than 10% of the image are
                // caught and the user is notified so they can adjust their prior set.
                float threshold =
                    sqrt(pow((prior_threshold_ * width), 2) + pow((prior_threshold_ * height), 2));
                _ensure_path_edge_groups_close_enough_to_priors(min_prior_idxs, min_prior_dists,
                                                                image_id, threshold);

                // Transform the interpolated vertices for each path
                // edge group (path_edge_group) into a delta representation from the matched prior.
                float interpolated_x, path_prior_x, interpolated_y, path_prior_y;
                for (int path_edge_group = 0; path_edge_group < images[image_id].npath_edge_groups;
                     path_edge_group++) {
                    for (int point = 0; point < points_per_prior_; point++) {
                        // get the path prior vertex indices
                        int left_prior_vertex =
                            min_prior_idxs[path_edge_group] * 2 * points_per_prior_ + 2 * point;
                        int right_prior_vertex =
                            min_prior_idxs[path_edge_group] * 2 * points_per_prior_ +
                            (2 * point + 1);

                        // get the path edge group vertex indices
                        int left_path_edge_group_vertex =
                            path_edge_groups[path_edge_group].left_interpolated_start_vertex +
                            point;
                        int right_path_edge_group_vertex =
                            path_edge_groups[path_edge_group].right_interpolated_start_vertex +
                            point;
                        int center_path_edge_group_vertex =
                            path_edge_groups[path_edge_group].center_interpolated_start_vertex +
                            point;

                        // left points: normalize and subtract labeled path from prior
                        int left_path_idx =
                            image_id * nall_priors_ +
                            min_prior_idxs[path_edge_group] * edges_per_path_ * points_per_prior_ +
                            point;

                        interpolated_x = interpolated_vertices[left_path_edge_group_vertex].x;
                        path_prior_x = path_prior_vertices[left_prior_vertex].x;
                        path_delta_vertices[left_path_idx].x =
                            (interpolated_x - path_prior_x) / width;

                        interpolated_y = interpolated_vertices[left_path_edge_group_vertex].y;
                        path_prior_y = path_prior_vertices[left_prior_vertex].y;
                        path_delta_vertices[left_path_idx].y =
                            (interpolated_y - path_prior_y) / height;

                        // right points: normalize and subtract labeled path from prior
                        int right_path_idx =
                            image_id * nall_priors_ +
                            min_prior_idxs[path_edge_group] * edges_per_path_ * points_per_prior_ +
                            points_per_prior_ + point;

                        interpolated_x = interpolated_vertices[right_path_edge_group_vertex].x;
                        path_prior_x = path_prior_vertices[right_prior_vertex].x;
                        path_delta_vertices[right_path_idx].x =
                            (interpolated_x - path_prior_x) / width;

                        interpolated_y = interpolated_vertices[right_path_edge_group_vertex].y;
                        path_prior_y = path_prior_vertices[right_prior_vertex].y;
                        path_delta_vertices[right_path_idx].y =
                            (interpolated_y - path_prior_y) / height;

                        // center points: normalize and subtract labeled path from prior
                        if (edges_per_path_ == 3) {
                            int center_path_idx = image_id * nall_priors_ +
                                                  min_prior_idxs[path_edge_group] *
                                                      edges_per_path_ * points_per_prior_ +
                                                  2 * points_per_prior_ + point;

                            interpolated_x = interpolated_vertices[center_path_edge_group_vertex].x;
                            path_prior_x = (path_prior_vertices[left_prior_vertex].x +
                                            path_prior_vertices[right_prior_vertex].x) /
                                           2.0;
                            path_delta_vertices[center_path_idx].x =
                                (interpolated_x - path_prior_x) / width;

                            interpolated_y = interpolated_vertices[center_path_edge_group_vertex].y;
                            path_prior_y = (path_prior_vertices[left_prior_vertex].y +
                                            path_prior_vertices[right_prior_vertex].y) /
                                           2.0;
                            path_delta_vertices[center_path_idx].y =
                                (interpolated_y - path_prior_y) / height;
                        }
                    }

                    // Encode the class and sum angle ratio for this prior edge group.
                    int path_idx = image_id * nall_priors_ + min_prior_idxs[path_edge_group];
                    path_classes[path_idx] = path_edge_groups[path_edge_group].class_id;
                    path_attributes[path_idx] = path_edge_groups[path_edge_group].path_attribute;
                    path_sum_angle_ratios[path_idx] =
                        path_edge_groups[path_edge_group].sum_angle_ratio;
                }
            }
        }  // if (nvertices > 0)

        // Verbose summary
        if (verbose_) {
            // print general stats
            printf(
                "nimages = %d, npolylines = %d, nvertices = %d, "
                "width = %d, height = %d\n",
                nimages, npolylines, nvertices, width, height);
            for (int image_id = 0; image_id < nimages; image_id++) {
                printf("for image %d there are %d path edge groups or %lu\n", image_id,
                       images[image_id].npath_edge_groups, path_edge_groups.size());
                int nump = images[image_id].npath_edge_groups;
                if (nump > 0) {
                    for (int path_edge_group = 0;
                         path_edge_group < images[image_id].npath_edge_groups; path_edge_group++) {
                        printf(" path %d: \n", path_edge_group);
                        int nleft = path_edge_groups[path_edge_group].left_vertices.size();
                        int nright = path_edge_groups[path_edge_group].right_vertices.size();
                        int ncenter = path_edge_groups[path_edge_group].center_rail_vertices.size();
                        int leftv, rightv, centerv;
                        // print path edge vertices
                        for (int v = 0; v < nleft; v++) {
                            leftv = path_edge_groups[path_edge_group].left_vertices[v];
                            printf("    left vertex %d: %.2f, %.2f\n", leftv, vertices[leftv].x,
                                   vertices[leftv].y);
                        }
                        for (int v = 0; v < nright; v++) {
                            rightv = path_edge_groups[path_edge_group].right_vertices[v];
                            printf("    right vertex %d: %.2f, %.2f\n", rightv, vertices[rightv].x,
                                   vertices[rightv].y);
                        }
                        for (int v = 0; v < ncenter; v++) {
                            centerv = path_edge_groups[path_edge_group].center_rail_vertices[v];
                            printf("    center vertex %d: %.2f, %.2f\n", centerv,
                                   vertices[centerv].x, vertices[centerv].y);
                        }
                        // print interpolated path edge vertices
                        for (int v = 0; v < points_per_prior_; v++) {
                            leftv =
                                path_edge_groups[path_edge_group].left_interpolated_start_vertex +
                                v;
                            printf("    left interpolated vertex %d: %.2f, %.2f\n", leftv,
                                   interpolated_vertices[leftv].x, interpolated_vertices[leftv].y);
                        }
                        for (int v = 0; v < points_per_prior_; v++) {
                            rightv =
                                path_edge_groups[path_edge_group].right_interpolated_start_vertex +
                                v;
                            printf("    right interpolated vertex %d: %.2f, %.2f\n", rightv,
                                   interpolated_vertices[rightv].x,
                                   interpolated_vertices[rightv].y);
                        }
                        if (edges_per_path_ == 3) {
                            for (int v = 0; v < points_per_prior_; v++) {
                                centerv = path_edge_groups[path_edge_group]
                                              .center_interpolated_start_vertex +
                                          v;
                                printf("    center interpolated vertex %d: %.2f, %.2f\n", centerv,
                                       interpolated_vertices[centerv].x,
                                       interpolated_vertices[centerv].y);
                            }
                        }
                        // Print path attribute vector.
                        int path_attribute_size_per_edge =
                            path_edge_groups[path_edge_group].path_attribute.size();
                        for (int j = 0; j < path_attribute_size_per_edge; j++) {
                            printf("    path_edge_group.path_attribute[%d] = %d \n", j,
                                   path_edge_groups[path_edge_group].path_attribute[j]);
                        }
                    }
                    // print associated delta edge vertices.
                    printf("Delta Vertices: \n");
                    int leftv, rightv, centerv;
                    for (int prior = 0; prior < nall_priors_; prior++) {
                        printf("---\n");
                        for (int v = 0; v < points_per_prior_; v++) {
                            leftv = image_id * nall_priors_ +
                                    prior * edges_per_path_ * points_per_prior_ + v;
                            rightv = image_id * nall_priors_ +
                                     prior * edges_per_path_ * points_per_prior_ +
                                     points_per_prior_ + v;
                            if (edges_per_path_ == 3) {
                                centerv = image_id * nall_priors_ +
                                          prior * edges_per_path_ * points_per_prior_ +
                                          2 * points_per_prior_ + v;
                                printf(
                                    "    prior %d: delta vertex (%d, %d)"
                                    "\tleft: %.6f, %.6f \tright: %.6f, %.6f \tcenter: %.6f, %.6f\n",
                                    prior, leftv, rightv, path_delta_vertices[leftv].x,
                                    path_delta_vertices[leftv].y, path_delta_vertices[rightv].x,
                                    path_delta_vertices[rightv].y, path_delta_vertices[centerv].x,
                                    path_delta_vertices[centerv].y);
                            } else {
                                printf(
                                    "    prior %d: delta vertex (%d, %d)"
                                    "\tleft: %.6f, %.6f \tright: %.6f, %.6f\n",
                                    prior, leftv, rightv, path_delta_vertices[leftv].x,
                                    path_delta_vertices[leftv].y, path_delta_vertices[rightv].x,
                                    path_delta_vertices[rightv].y);
                            }
                        }
                    }
                }
            }
        }  // if (verbose_)

        // Create the output tensor.
        int layers = NUM_SUM_ANGLE_RATIOS + nclasses_ + npath_attributes_ +
                     points_per_prior_ * edges_per_path_ * 2;
        TensorShape output_shape({nimages, nall_priors_, layers});

        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
        auto output_paths = output_tensor->template flat<float>();

        // Write the output.
        for (int image_id = 0; image_id < static_cast<int>(images.size()); image_id++) {
            // Get pointer to the start of this image's data.
            float* out = output_paths.data() + image_id * nall_priors_ * layers;
            // Write it!
            _write_output(&out, image_id, nclasses_, nall_priors_, points_per_prior_,
                          npath_attributes_, &path_sum_angle_ratios[0], &path_classes[0],
                          path_attributes, &path_delta_vertices[0]);
        }

        if (verbose_) {
            printf("done\n");
        }
    }
};

#pragma message("Registering CPU kernel")
REGISTER_KERNEL_BUILDER(Name("GeneratePathFromEdges")
                            .Device(DEVICE_CPU)
                            .HostMemory("polyline_vertices")
                            .HostMemory("path_prior_vertices")
                            .HostMemory("vertex_counts_per_polyline")
                            .HostMemory("class_ids_per_polyline")
                            .HostMemory("attributes_per_polyline")
                            .HostMemory("polylines_per_image")
                            .HostMemory("width")
                            .HostMemory("height"),
                        GeneratePathFromEdgesOp);
