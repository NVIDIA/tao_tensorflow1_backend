# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BpNet Post-processor."""

import numpy as np
import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import TAOObject
import nvidia_tao_tf1.cv.bpnet.inferencer.utils as inferencer_utils

# Enable eager execution
# tf.compat.v1.enable_eager_execution()


class BpNetPostprocessor(TAOObject):
    """BpNet Postprocessor class."""

    def __init__(
            self,
            topology,
            num_parts,
            heatmap_threshold=0.1,
            paf_threshold=0.05,
            heatmap_gaussian_sigma=3,
            heatmap_gaussian_kernel=5,
            line_integral_samples_num=10,
            line_integral_count_threshold=0.8,
            num_parts_thresh=4,
            overall_score_thresh=0.4,
            use_tf_postprocess=True):
        """Init.

        Args:
            topology (np.ndarray): N x 4 array where N is the number of
                connections, and the columns are (start_paf_idx, end_paf_idx,
                start_conn_idx, end_conn_idx)
            num_parts (int): number of keypoints in the given model
            heatmap_threshold (float): Threshold value to use for
                filtering peaks after Non-max supression.
            paf_threshold (float): Threshold value to use for
                suppressing connection in part affinity fields.
            heatmap_gaussian_sigma (float): sigma to use for gaussian filter
                used for smoothing the heatmap
            heatmap_gaussian_kernel (int): kernel size to use for gaussian filter
                used for smoothing the heatmap
            line_integral_samples_num (int): number of sampling points (N) along each vector
            line_integral_count_threshold (float): Threshold on the ratio of qualified points
                to total sample points (line_integral_samples_num)
            num_parts_thresh (int): Minimum number of parts needed to qualify as a detection
            overall_score_thresh (float): Minimum overall score needed to qualify as a detection
            use_tf_postprocess (bool): Enable use of tensorflow based find peaks.
                If False, reverts to numpy and cv2 based ops.
        """
        self.topology = topology
        self.num_parts = num_parts
        self.num_connections = topology.shape[0]
        self.heatmap_threshold = heatmap_threshold
        self.paf_threshold = paf_threshold
        self.num_parts_thresh = num_parts_thresh
        self.overall_score_thresh = overall_score_thresh
        self.heatmap_gaussian_sigma = heatmap_gaussian_sigma
        self.heatmap_gaussian_kernel = heatmap_gaussian_kernel
        self.line_integral_count_threshold = line_integral_count_threshold
        self.line_integral_samples_num = line_integral_samples_num
        self.use_tf_postprocess = use_tf_postprocess
        # array of points used to sample along each vector and
        # it has shape (1, N, 1)
        self.line_integral_samples = np.arange(
            line_integral_samples_num,
            dtype=np.float32).reshape(1, -1, 1)
        # Initialize gaussian kernel used for peak smoothing
        self.gaussian_kernel = inferencer_utils.get_gaussian_kernel(
            heatmap_gaussian_kernel, self.num_parts, sigma=self.heatmap_gaussian_sigma)

        # Initialize peak nms graph
        # NOTE: There is bug when using tensorflow graph alongside tensorrt engine excution
        # Model inference gives completely wrong results. So disable this section when using
        # tensorrt engine for inference.
        if use_tf_postprocess:
            self.graph = tf.compat.v1.get_default_graph()
            self.persistent_sess = tf.Session(graph=self.graph)
            self.heatmap_tf = tf.placeholder(
                tf.float32,
                shape=(1, None, None, self.num_parts))
            self.peaks_map_tf = self.peak_nms_tf(self.heatmap_tf)
            # Dry run with dummy input
            self.persistent_sess.run(tf.global_variables_initializer())
            self.persistent_sess.run(
                self.peaks_map_tf,
                feed_dict={
                    self.heatmap_tf: [np.ndarray(shape=(256, 256, self.num_parts),
                                      dtype=np.float32)]
                }
            )

    def peak_nms_tf(self, heatmap):
        """Peak non-max suppresion using tensorflow.

        Steps:
            a. Refine the heatmap using gaussian smoothing
            b. Find the local maximums using window of size K
                and substitute the center pixel with max using maxpool
            c. Compare this with the smoothed heatmap and retain the
                original heatmap values where they match. Other pixel
                locations (non-maximum) are suppressed to 0.

        Args:
            heatmap (tf.Tensor): heatmap tensor with shape (N, H, W, C)
                where C is the number of parts.

        Returns:
            peaks_map (tf.Tensor): heatmap after NMS
        """
        # Define gaussian kernel
        with tf.compat.v1.variable_scope('postprocess'):
            gaussian_kernel_tf = tf.Variable(
                tf.convert_to_tensor(self.gaussian_kernel),
                name='gauss_kernel')

        # Apply depthwise conv with gaussian kernel
        smoothed_heatmap = tf.nn.depthwise_conv2d(
            heatmap,
            filter=gaussian_kernel_tf,
            strides=[1, 1, 1, 1],
            padding='SAME')
        # Non-max suppresion by using maxpool on the smoothed heatmap
        maxpool_kernel_size = (3, 3)
        maxpooled_heatmap = tf.nn.pool(
            smoothed_heatmap,
            window_shape=maxpool_kernel_size,
            pooling_type='MAX',
            padding='SAME')
        peaks_map = tf.where(
            tf.equal(smoothed_heatmap, maxpooled_heatmap),
            heatmap,
            tf.zeros_like(heatmap))

        return peaks_map

    def peak_nms(self, heatmap):
        """Peak non-max suppresion.

        Steps:
            a. Refine the heatmap using gaussian smoothing
            b. Find the local maximums using window of size K
                and substitute the center pixel with max using maxpool
            c. Compare this with the smoothed heatmap and retain the
                original heatmap values where they match. Other pixel
                locations (non-maximum) are suppressed to 0.

        Args:
            heatmap (np.ndarray): heatmap tensor with shape (H, W, C)
                where C is the number of parts.

        Returns:
            peaks_map (np.ndarray): heatmap after NMS
        """
        smoothed_heatmap = inferencer_utils.apply_gaussian_smoothing(
            heatmap,
            kernel_size=self.heatmap_gaussian_kernel,
            sigma=self.heatmap_gaussian_sigma,
            backend="cv")

        return inferencer_utils.nms_np(smoothed_heatmap)

    def find_peaks(self, heatmap):
        """Find peak candidates using the heatmap.

        Steps:
            a. Smooth the heatmap and apply Non-max suppression
            b. Further suppress the peaks with scores below defined
                `heatmap_threshold`
            c. Gather peaks accorind to keypoint ordering

        Args:
            heatmap (np.ndarray): heatmap array with shape (H, W, C)
                where C is the number of parts.

        Returns:
            peaks (list): List of candidate peaks per keypoint
        """
        if self.use_tf_postprocess:
            # Drop the last channel which corresponds to background
            # Expand dims before passing into the tensorflow graph
            heatmap = np.expand_dims(heatmap[:, :, :-1], axis=0)
            # Run non-max suppression using tensorflow ops
            peaks_map = self.persistent_sess.run(
                [self.peaks_map_tf], feed_dict={self.heatmap_tf: heatmap})
            peaks_map = np.squeeze(peaks_map)
            heatmap = heatmap[0]

        else:
            # Drop the last channel which corresponds to background
            heatmap = heatmap[:, :, :-1]
            # Run non-max suppression
            peaks_map = self.peak_nms(heatmap)

        # Further suppress the peaks with scores below defined threshold
        peak_ys, peak_xs, kpt_idxs = np.where(
            peaks_map > self.heatmap_threshold)
        # Sort the peaks based on the kpt ordering
        sorted_indices = kpt_idxs.argsort()
        kpt_idxs = kpt_idxs[sorted_indices]
        peak_ys = peak_ys[sorted_indices]
        peak_xs = peak_xs[sorted_indices]

        # Gather the peaks according to their keypoint index
        peak_counter = 0
        peaks = [[] for kpt_idx in range(0, self.num_parts)]
        for (kpt_idx, peak_y, peak_x) in zip(kpt_idxs, peak_ys, peak_xs):
            peak_score = heatmap[peak_y, peak_x, kpt_idx]
            peaks[kpt_idx].append((peak_x, peak_y, peak_score, peak_counter))
            peak_counter += 1

        return peaks, peak_counter

    @staticmethod
    def get_bipartite_graph(conn_start, conn_end, n_start, n_end):
        """Get the bipartite graph for candidate limb connections.

        The vertices represent the keypoint candidates and the edges
        represent the connection candidates.

        Args:
            conn_start (np.ndarray): keypoint candidates for source keypoint
            conn_end (np.ndarray): keypoint candidates for end keypoint
            n_start (int): number of keypoint candidates for source keypoint
            n_end (int): number of keypoint candidates for end keypoint

        Returns:
            out (np.ndarray): bipartite graph of shape (n_end, n_start, 2)
        """
        # Expand dims to (1, nA, 2)
        kpts_start = np.expand_dims(conn_start[:, :2], axis=0)
        # Expand dims to (nB, 1, 2)
        kpts_end = np.expand_dims(conn_end[:, :2], axis=1)
        # Broadcast nB times along first dim
        kpts_start = np.broadcast_to(kpts_start, (n_end, n_start, 2))
        # Return the bipartite graph of vectors
        return (kpts_end - kpts_start), kpts_start

    def compute_line_integral(
            self, bipartite_graph, connection_paf, kpts_start):
        """Compute the line integral along the vector of each candidate limbs.

        This gives each connection a score which will be used for
        the assignment step.

        Args:
            bipartite_graph (np.ndarray): contains candidate limbs
                of shape (nB, nA, 2)
            connection_paf (np.ndarray): part affinity field for candidate limb
                connecting two keypoints with shape (H, W, 2)

        Returns:
            weighted_bipartite_graph (np.ndarray): scores of the candidate connections
        """

        # Calculate unit step size along the vector
        bipartite_graph = bipartite_graph.reshape(-1, 1, 2)
        steps = (1 / (self.line_integral_samples_num - 1) * bipartite_graph)
        # Sample N points along every candidate limb vector
        points = steps * self.line_integral_samples + \
            kpts_start.reshape(-1, 1, 2)
        points = points.round().astype(dtype=np.int32)
        x = points[..., 0].ravel()
        y = points[..., 1].ravel()

        # Get part afifnity vector fields along the limb sample points
        paf_vectors = connection_paf[y, x].reshape(
            -1, self.line_integral_samples_num, 2)
        # Compute the candidate limb unit vectors
        limb_vec_norm = np.linalg.norm(
            bipartite_graph, ord=2, axis=-1, keepdims=True)
        limb_unit_vec = bipartite_graph / (limb_vec_norm + 1e-6)
        # Compute the dot prodcut at each location of the candidate limbs
        # with the part affinity vectors at that location
        scores = (paf_vectors * limb_unit_vec).sum(-1).reshape(-1,
                                                               self.line_integral_samples_num)
        # Suppress scores below given threshold
        valid_scores_mask = scores > self.paf_threshold
        num_qualified_points = valid_scores_mask.sum(1)
        # Compute the line integral / weighted bipartite graph by summing
        # over the scores of valid points
        weighted_bipartite_graph = (
            scores * valid_scores_mask).sum(1) / (num_qualified_points + 1e-6)

        return weighted_bipartite_graph, num_qualified_points

    @staticmethod
    def assignment(valid_candidate_limb_pairs, weighted_bipartite_graph):
        """Assignment algorithm to obtain final connections with maximum score.

        Steps:
            a. Sort each possible connection by its score.
            b. The connection with the highest score is chosen as final connection.
            c. Move to next possible connection. If no parts of this connection
                have been assigned to a final connection before, this is a final connection.
            d. Repeat the step 3 until we are done.

        Args:
            valid_candidate_limb_pairs (list): list of arrays with start and end conn indices
            weighted_bipartite_graph (np.ndarray): scores of the candidate connections

        Returns:
            conn_start_idx (np.ndarray): start indices of the final connections
            conn_end_idx (np.ndarray): end indices of the final connections
            weighted_bipartite_graph (np.ndarray): scores of the final connections
        """
        # Sort based on scores
        order = weighted_bipartite_graph.argsort()[::-1]
        weighted_bipartite_graph = weighted_bipartite_graph[order]
        conn_start_idx = valid_candidate_limb_pairs[1][order]
        conn_end_idx = valid_candidate_limb_pairs[0][order]
        idx = []
        has_start_kpt = set()
        has_end_kpt = set()
        # Start assignment from the largest score
        for t, (i, j) in enumerate(zip(conn_start_idx, conn_end_idx)):
            if i not in has_start_kpt and j not in has_end_kpt:
                idx.append(t)
                has_start_kpt.add(i)
                has_end_kpt.add(j)
        idx = np.asarray(idx, dtype=np.int32)

        return conn_start_idx[idx], conn_end_idx[idx], weighted_bipartite_graph[idx]

    def find_connections(self, peaks, paf, image_width):
        """Find connection candidates using the part affinity fields.

        Steps:
            a. Obtain the bipartite graph vectors between each pairs of
                keypoint connection
            b. Compute line integral over the part affinity fileds along
                the candidate connection vectors to obtain the weighted
                bipartite graph
            c. Suppress the candidates that don't meet given criterions
            d. Assigment: find the connections that maximize the total
                score when matching the bipartite graph.

        Args:
            peaks (list): List of candidate peaks per keypoint
            paf (np.ndarray): part affinity fields with shape (H, W, C)
                where C is the (number of connections * 2)
            image_width (int): width of the image

        Returns:
            connection_all (list): List of all detected connections for
                each part/limb.
        """
        connection_all = []

        for k in range(self.num_connections):
            connection_paf = paf[:, :, self.topology[k][:2]]
            conn_start = np.array(peaks[self.topology[k][2]])
            conn_end = np.array(peaks[self.topology[k][3]])
            n_start = len(conn_start)
            n_end = len(conn_end)
            if (n_start == 0 or n_end == 0):
                connection_all.append([])
                continue

            # Get the bipartite graph - all possible connections between two
            # types of candidate keypoints
            bipartite_graph, kpts_start = self.get_bipartite_graph(
                conn_start, conn_end, n_start, n_end)

            # Get weighted bipartite graph using line integral over the part
            # affinity fields
            weighted_bipartite_graph, num_qualified_points = self.compute_line_integral(
                bipartite_graph, connection_paf, kpts_start)

            # Suppress the candidate limbs that don't meet the following
            # criterion
            num_thresh_points = self.line_integral_count_threshold * \
                self.line_integral_samples_num
            is_criterion_met = np.logical_and(
                weighted_bipartite_graph > 0,
                num_qualified_points > num_thresh_points)
            valid_condidate_limb_idxs = np.where(is_criterion_met)[0]
            if len(valid_condidate_limb_idxs) == 0:
                connection_all.append([])
                continue

            valid_candidate_limb_pairs = np.divmod(
                valid_condidate_limb_idxs, n_start)
            weighted_bipartite_graph = weighted_bipartite_graph[valid_condidate_limb_idxs]

            # Assignment algorithm to get final connections
            conn_start_idx, conn_end_idx, connection_scores = self.assignment(
                valid_candidate_limb_pairs, weighted_bipartite_graph)
            connections = list(zip(conn_start[conn_start_idx, 3].astype(np.int32),
                                   conn_end[conn_end_idx, 3].astype(np.int32),
                                   connection_scores))
            connection_all.append(np.array(connections))
        return connection_all

    def connect_parts(self, connection_all, peaks_all, topology):
        """Connect the parts to build the final skeleton(s).

        Steps:
            a. Iterate through every connection and start with assigning to a
                new human everytime.
            b. This initial human will be updated with the connection end
                every time a start part in current connection is already
                part of the human.
            c. If two humans share the same part index with the same coordintes,
                but have disjoint connections, they are merged into one.
            d. We iterate b and c until all connections are exhausted.
            c. Suppress the humans that don't meet certain criterion.

        Args:
            connection_all (list): List of all detected connections for
                each part/limb.
            peaks_all (list): List of all candidate peaks per keypoint
            topology (np.ndarray): N x 4 array where N is the number of
                connections, and the columns are (start_paf_idx, end_paf_idx,
                start_conn_idx, end_conn_idx)

        Returns:
            humans (np.ndarray): array with final skeletons of shape (N, M)
                where N is the number of skeletons and M is (num_parts + 2)
            candidate_peaks (np.ndarray): array with all candidate peaks of
                shape (N, 4) where N is number of peaks
        """
        # Initialize humans array with (N, num_parts + 2)
        # Last column: total parts for person corresponding to that row
        # Second last column: score of the overall configuration
        humans = -1 * np.ones((0, self.num_parts + 2))
        # Concat all peaks into an (N x 4) array
        candidate_peaks = np.array(
            [item for sublist in peaks_all for item in sublist])

        # Iterate through each edge
        for pidx in range(self.num_connections):
            if not len(connection_all[pidx]):
                continue
            kpts_start = connection_all[pidx][:, 0]
            kpts_end = connection_all[pidx][:, 1]
            start_idx, end_idx = np.array(topology[pidx][2:])
            # Iterate through all connections corresponding to current edge
            for cidx in range(len(connection_all[pidx])):
                # Check if multiple humans share the same connection
                humans_sharing_kpts = []
                for j in range(len(humans)):
                    is_present = (humans[j][start_idx] == kpts_start[cidx] or
                                  humans[j][end_idx] == kpts_end[cidx])
                    if is_present:
                        humans_sharing_kpts.append(j)
                # If only one row/human shares the part index, assign the part index
                # end to the human
                if (len(humans_sharing_kpts) == 1):
                    j = humans_sharing_kpts[0]
                    if (humans[j][end_idx] != kpts_end[cidx]):
                        humans[j][end_idx] = kpts_end[cidx]
                        humans[j][-1] += 1
                        humans[j][-2] += \
                            candidate_peaks[kpts_end[cidx]
                                            .astype(int), 2] + connection_all[pidx][cidx][2]
                # If two rows/humans share the part index, and the union of
                # the connections are disjoint, then merge them
                # else, just assign the end keypoint to current row
                # similar to the case 1
                elif (len(humans_sharing_kpts) == 2):
                    j1, j2 = humans_sharing_kpts
                    membership = ((humans[j1] >= 0).astype(
                        int) + (humans[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        humans[j1][:-2] += (humans[j2][:-2] + 1)
                        humans[j1][-2:] += humans[j2][-2:]
                        humans[j1][-2] += connection_all[pidx][cidx][2]
                        humans = np.delete(humans, j2, 0)
                    else:
                        humans[j1][end_idx] = kpts_end[cidx]
                        humans[j1][-1] += 1
                        humans[j1][-2] += \
                            candidate_peaks[kpts_end[cidx]
                                            .astype(int), 2] + connection_all[pidx][cidx][2]
                # If the start index is not in any row/humans, create a new row/human
                # Idea is that everytime there is a new connection, we assign it to a
                # new human, and later merge them together as above.
                elif not len(humans_sharing_kpts) and (pidx < self.num_connections - 2):
                    row = -1 * np.ones((self.num_parts + 2))
                    row[start_idx] = kpts_start[cidx]
                    row[end_idx] = kpts_end[cidx]
                    row[-1] = 2
                    row[-2] = sum(
                        candidate_peaks[connection_all[pidx][cidx, :2]
                                        .astype(int), 2]) + connection_all[pidx][cidx][2]
                    humans = np.vstack([humans, row])
        # Suppress the humans/rows based on the following criterion:
        # 1. Parts fewer than given threshold
        # 2. Overall score lesser than given threshold
        invalid_idx = []
        for hidx in range(len(humans)):
            if humans[hidx][-1] < self.num_parts_thresh or \
                    humans[hidx][-2] / humans[hidx][-1] < self.overall_score_thresh:
                invalid_idx.append(hidx)
        humans = np.delete(humans, invalid_idx, axis=0)
        return humans, candidate_peaks

    def get_final_keypoints(
            self,
            humans,
            candidate_peaks,
            scale_factor,
            offset_factor):
        """Get final scaled keypoints.

        Args:
            humans (np.ndarray): array with final skeletons of shape (N, M)
                where N is the number of skeletons and M is (num_parts + 2)
            candidate_peaks (np.ndarray): array with all candidate peaks of
                shape (N, 4) where N is number of peaks
            scale_factor (list): scale factor with format (fx, fy)
            offset_factor (list): offset factor with format (oy, ox)

        Returns:
            keypoints (list): List of lists containing keypoints per skeleton
            scores (list): List of scores per skeleton
        """
        keypoints = []
        scores = []
        for human in humans:
            keypoint_indexes = human[0:self.num_parts]
            person_keypoint_coordinates = []
            # This is the sum of all keypoint and paf scores
            person_score = human[-2]
            for index in keypoint_indexes:
                # No candidates for keypoint
                if index == -1:
                    X, Y = 0, 0
                else:
                    X = scale_factor[1] * \
                        candidate_peaks[index.astype(int), 0] + offset_factor[0]
                    Y = scale_factor[0] * \
                        candidate_peaks[index.astype(int), 1] + offset_factor[1]
                person_keypoint_coordinates.append([X, Y])
            keypoints.append(person_keypoint_coordinates)
            # Scale the scores between 0 and 1
            # TODO: how is this used by COCO eval
            scores.append(1 - 1.0 / person_score)
        return keypoints, scores
