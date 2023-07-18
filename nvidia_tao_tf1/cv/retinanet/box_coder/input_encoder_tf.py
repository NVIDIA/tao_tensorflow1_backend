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

"""
RetinaNet label encoder.

Code partially from GitHub (Apache v2 license):
https://github.com/pierluigiferrari/ssd_keras/tree/3ac9adaf3889f1020d74b0eeefea281d5e82f353
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from nvidia_tao_tf1.cv.ssd.utils.box_utils import (
    bipartite_match_row,
    corners_to_centroids,
    iou,
    multi_match
)
from nvidia_tao_tf1.cv.ssd.utils.box_utils import np_convert_coordinates
from nvidia_tao_tf1.cv.ssd.utils.tensor_utils import tensor_slice_replace, tensor_strided_replace


class InputEncoderTF:
    '''
    Encoder class.

    Transforms ground truth labels for object detection in images
    (2D bounding box coordinates and class labels) to the format required for
    training an SSD or RetinaNet model.

    In the process of encoding the ground truth labels, a template of anchor boxes
    is being built, which are subsequently matched to the ground truth boxes
    via an intersection-over-union threshold criterion.
    '''

    def __init__(self,
                 img_height,
                 img_width,
                 n_classes,
                 predictor_sizes,
                 min_scale=0.1,
                 max_scale=0.9,
                 scales=None,
                 aspect_ratios_global=None,
                 aspect_ratios_per_layer=None,
                 two_boxes_for_ar1=True,
                 steps=None,
                 n_anchor_levels=3,
                 offsets=None,
                 clip_boxes=False,
                 variances=None,
                 pos_iou_threshold=0.5,
                 neg_iou_limit=0.3,
                 normalize_coords=True,
                 gt_normalized=False,
                 class_weights=None):
        '''
        Init encoder.

        Arguments:
            img_height (int): The height of the input images.
            img_width (int): The width of the input images.
            n_classes (int): The number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO.
            predictor_sizes (list): A list of int-tuples of the format `(height, width)`
                containing the output heights and widths of the convolutional predictor layers.
            min_scale (float, optional): The smallest scaling factor for the size of the anchor
                boxes as a fraction of the shorter side of the input images. Note that you should
                set the scaling factors such that the resulting anchor box sizes correspond to the
                sizes of the objects you are trying to detect. Must be >0.
            max_scale (float, optional): The largest scaling factor for the size of the anchor boxes
                as a fraction of the shorter side of the input images. All scaling factors between
                the smallest and the largest will be linearly interpolated. Note that the second to
                last of the linearly interpolated scaling factors will actually be the scaling
                factor for the last predictor layer, while the last scaling factor is used for the
                second box for aspect ratio 1 in the last predictor layer if `two_boxes_for_ar1` is
                `True`. Note that you should set the scaling factors such that the resulting anchor
                box sizes correspond to the sizes of the objects you are trying to detect. Must be
                greater than or equal to `min_scale`.
            scales (list, optional): A list of floats >0 containing scaling factors per
                convolutional predictor layer. This list must be one element longer than the number
                of predictor layers. The first `k` elements are the scaling factors for the `k`
                predictor layers, while the last element is used for the second box for aspect ratio
                1 in the last predictor layer if `two_boxes_for_ar1` is `True`. This additional last
                scaling factor must be passed either way, even if it is not being used. If a list is
                passed, this argument overrides `min_scale` and `max_scale`. All scaling factors
                must be greater than zero. Note that you should set the scaling factors such that
                the resulting anchor box sizes correspond to the sizes of the objects you are trying
                to detect.
            aspect_ratios_global (list, optional): The list of aspect ratios for which anchor boxes
                are to be generated. This list is valid for all prediction layers. Note that you
                should set the aspect ratios such that the resulting anchor box shapes roughly
                correspond to the shapes of the objects you are trying to detect.
            aspect_ratios_per_layer (list, optional): A list containing one aspect ratio list for
                each prediction layer. If a list is passed, it overrides `aspect_ratios_global`.
                Note that you should set the aspect ratios such that the resulting anchor box shapes
                very roughly correspond to the shapes of the objects you are trying to detect.
            two_boxes_for_ar1 (bool, optional): Only relevant for aspect ratios lists that contain
                1. Will be ignored otherwise. If `True`, two anchor boxes will be generated for
                aspect ratio 1. The first will be generated using the scaling factor for the
                respective layer, the second one will be generated using geometric mean of said
                scaling factor and next bigger scaling factor.
            steps (list, optional): `None` or a list with as many elements as there are predictor
                layers. The elements can be either ints/floats or tuples of two ints/floats. These
                numbers represent for each predictor layer how many pixels apart the anchor box
                center points should be vertically and horizontally along the spatial grid over the
                image. If the list contains ints/floats, then that value will be used for both
                spatial dimensions. If the list contains tuples of two ints/floats, then they
                represent `(step_height, step_width)`. If no steps are provided, then they will be
                computed such that the anchor box center points will form an equidistant grid within
                the image dimensions.
            offsets (list, optional): `None` or a list with as many elements as there are predictor
                layers. The elements can be either floats or tuples of two floats. These numbers
                represent for each predictor layer how many pixels from the top and left boarders of
                the image the top-most and left-most anchor box center points should be as a
                fraction of `steps`. The last bit is important: The offsets are not absolute pixel
                values, but fractions of the step size specified in the `steps` argument. If the
                list contains floats, then that value will be used for both spatial dimensions. If
                the list contains tuples of two floats, then they represent
                `(vertical_offset, horizontal_offset)`. If no offsets are provided, then they will
                default to 0.5 of the step size.
            clip_boxes (bool, optional): If `True`, limits the anchor box coordinates to stay within
                image boundaries.
            variances (list, optional): A list of 4 floats >0. The anchor box offset for each
                coordinate will be divided by its respective variance value.
            pos_iou_threshold (float, optional): The intersection-over-union similarity threshold
                that must be met in order to match a given ground truth box to a given anchor box.
            neg_iou_limit (float, optional): The maximum allowed intersection-over-union similarity
                of an anchor box with any ground truth box to be labeled a negative
                (i.e. background) box. If an anchor box is neither a positive, nor a negative box,
                it will be ignored during training.
            normalize_coords (bool, optional): If `True`, the encoder uses relative instead of
                absolute coordinates. This means instead of using absolute tartget coordinates, the
                encoder will scale all coordinates to be within [0,1]. This way learning becomes
                independent of the input image size.
        '''
        predictor_sizes = np.array(predictor_sizes)
        if predictor_sizes.ndim == 1:
            predictor_sizes = np.expand_dims(predictor_sizes, axis=0)

        ##################################################################################
        # Handle exceptions.
        ##################################################################################

        if (min_scale is None or max_scale is None) and scales is None:
            raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")

        if scales:
            if (len(scales) != predictor_sizes.shape[0] + 1):
                # Must be two nested `if` statements since `list` and `bool` can't be combined by &
                raise ValueError("It must be either scales is None or len(scales) == \
len(predictor_sizes)+1, but len(scales) == {} and len(predictor_sizes)+1 == {}"
                                 .format(len(scales), len(predictor_sizes)+1))
            scales = np.array(scales)
            if np.any(scales <= 0):
                raise ValueError("All values in `scales` must be greater than 0, but the passed \
list of scales is {}".format(scales))
        else:
            # If no scales passed, we make sure that `min_scale` and `max_scale` are valid values.
            if not 0 < min_scale <= max_scale:
                raise ValueError("It must be 0 < min_scale <= max_scale, but it is min_scale = {} \
and max_scale = {}".format(min_scale, max_scale))

        if not (aspect_ratios_per_layer is None):
            if (len(aspect_ratios_per_layer) != predictor_sizes.shape[0]):
                # Must be two nested `if` statements since `list` and `bool` can't be combined by &
                raise ValueError("It must be either aspect_ratios_per_layer is None or \
len(aspect_ratios_per_layer) == len(predictor_sizes), but len(aspect_ratios_per_layer) == {} \
and len(predictor_sizes) == {}".format(len(aspect_ratios_per_layer), len(predictor_sizes)))
            for aspect_ratios in aspect_ratios_per_layer:
                if np.any(np.array(aspect_ratios) <= 0):
                    raise ValueError("All aspect ratios must be greater than zero.")
        else:
            if (aspect_ratios_global is None):
                raise ValueError("At least one of `aspect_ratios_global` and \
`aspect_ratios_per_layer` must not be `None`.")
            if np.any(np.array(aspect_ratios_global) <= 0):
                raise ValueError("All aspect ratios must be greater than zero.")

        if len(variances) != 4:
            raise ValueError("4 variance values must be pased, but {} values were received."
                             .format(len(variances)))

        variances = np.array(variances)

        if np.any(variances <= 0):
            raise ValueError("All variances must be >0, but the variances given are {}"
                             .format(variances))

        if (not (steps is None)) and (len(steps) != predictor_sizes.shape[0]):
            raise ValueError("You must provide at least one step value per predictor layer.")

        if (not (offsets is None)) and (len(offsets) != predictor_sizes.shape[0]):
            raise ValueError("You must provide at least one offset value per predictor layer.")

        ##################################################################################
        # Set or compute members.
        ##################################################################################

        self.img_height = float(img_height)
        self.img_width = float(img_width)
        self.n_classes = n_classes
        self.predictor_sizes = predictor_sizes
        self.min_scale = min_scale
        self.max_scale = max_scale
        # If `scales` is None, compute the scaling factors by linearly interpolating between
        # `min_scale` and `max_scale`. If an explicit list of `scales` is given, however,
        # then it takes precedent over `min_scale` and `max_scale`.
        if (scales is None):
            self.scales = np.linspace(self.min_scale, self.max_scale, len(self.predictor_sizes)+1)
        else:
            # If a list of scales is given explicitly, we'll use that instead of computing it from
            # `min_scale` and `max_scale`.
            self.scales = scales
        # If `aspect_ratios_per_layer` is None, then we use the same list of aspect ratios
        # `aspect_ratios_global` for all predictor layers. If `aspect_ratios_per_layer` is given,
        # however, then it takes precedent over `aspect_ratios_global`.
        if (aspect_ratios_per_layer is None):
            self.aspect_ratios = [aspect_ratios_global] * predictor_sizes.shape[0]
        else:
            # If aspect ratios are given per layer, we'll use those.
            self.aspect_ratios = aspect_ratios_per_layer

        self.two_boxes_for_ar1 = two_boxes_for_ar1
        if not (steps is None):
            self.steps = steps
        else:
            self.steps = [None] * predictor_sizes.shape[0]
        if not (offsets is None):
            self.offsets = offsets
        else:
            self.offsets = [None] * predictor_sizes.shape[0]
        self.clip_boxes = clip_boxes
        self.variances = variances
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_limit = neg_iou_limit
        self.normalize_coords = normalize_coords
        self.gt_normalized = gt_normalized

        # Compute the number of boxes per spatial location for each predictor layer.
        # For example, if a predictor layer has three different aspect ratios, [1.0, 0.5, 2.0], and
        # is supposed to predict two boxes of slightly different size for aspect ratio 1.0, then
        # that predictor layer predicts a total of four boxes at every spatial location across the
        # feature map.
        if not (aspect_ratios_per_layer is None):
            self.n_boxes = []
            for aspect_ratios in aspect_ratios_per_layer:
                if (1 in aspect_ratios) & two_boxes_for_ar1:
                    self.n_boxes.append(len(aspect_ratios) + 1)
                else:
                    self.n_boxes.append(len(aspect_ratios))
        else:
            if (1 in aspect_ratios_global) & two_boxes_for_ar1:
                self.n_boxes = n_anchor_levels * (len(aspect_ratios_global) + 1)
            else:
                self.n_boxes = n_anchor_levels * len(aspect_ratios_global)
        self.n_anchor_levels = n_anchor_levels
        self.anchor_sizes = np.power(2, np.linspace(0, 1, 1+n_anchor_levels))[:-1]

        ##################################################################################
        # Compute the anchor boxes for each predictor layer.
        ##################################################################################

        # Compute the anchor boxes for each predictor layer. We only have to do this once
        # since the anchor boxes depend only on the model configuration, not on the input data.
        # For each predictor layer (i.e. for each scaling factor) the tensors for that layer's
        # anchor boxes will have the shape `(feature_map_height, feature_map_width, n_boxes, 4)`.

        boxes_list = []  # This will store the anchor boxes for each predicotr layer.

        boxes_corner = []

        # Iterate over all predictor layers and compute the anchor boxes for each one.
        for i in range(len(self.predictor_sizes)):
            boxes, box_corner = self.__anchor_layer(feature_map_size=self.predictor_sizes[i],
                                                    aspect_ratios=self.aspect_ratios[i],
                                                    this_scale=self.scales[i],
                                                    next_scale=self.scales[i+1],
                                                    this_steps=self.steps[i],
                                                    this_offsets=self.offsets[i])
            boxes_list.append(boxes)
            boxes_corner.append(box_corner)

        anchor_box_list_np = np.concatenate([i.reshape((-1, 4)) for i in boxes_corner], axis=0)
        self.anchorbox_tensor = tf.convert_to_tensor(anchor_box_list_np, dtype=tf.float32)

        self.encoding_template_tensor = tf.convert_to_tensor(self.__encode_template(boxes_list),
                                                             dtype=tf.float32)

        if class_weights is None:
            self.class_weights = np.ones(self.n_classes, dtype=np.float32)
        else:
            self.class_weights = np.array(class_weights, dtype=np.float32)

        self.class_weights = tf.constant(self.class_weights, dtype=tf.float32)

    def __anchor_layer(self,
                       feature_map_size,
                       aspect_ratios,
                       this_scale,
                       next_scale,
                       this_steps=None,
                       this_offsets=None):
        '''
        Generate numpy anchors for each layer.

        Computes an array of the spatial positions and sizes of the anchor boxes for one predictor
        layer of size `feature_map_size == [feature_map_height, feature_map_width]`.

        Arguments:
            feature_map_size (tuple): A list or tuple `[feature_map_height, feature_map_width]` with
                the spatial dimensions of the feature map for which to generate the anchor boxes.
            aspect_ratios (list): A list of floats, the aspect ratios for which anchor boxes are to
                be generated. All list elements must be unique.
            this_scale (float): A float in [0, 1], the scaling factor for the size of the generate
                anchor boxes as a fraction of the shorter side of the input image.
            next_scale (float): A float in [0, 1], the next larger scaling factor. Only relevant if
                `self.two_boxes_for_ar1 == True`.

        Returns:
            A 4D np tensor of shape `(feature_map_height, feature_map_width, n_boxes_per_cell, 4)`
            where the last dimension contains `(xmin, xmax, ymin, ymax)` for each anchor box in each
            cell of the feature map.
        '''
        # Compute box width and height for each aspect ratio.

        # The shorter side of the image will be used to compute `w` and `h` using `scale` and
        # `aspect_ratios`.
        size = min(self.img_height, self.img_width)
        # Compute the box widths and and heights for all aspect ratios
        wh_list = []

        for scale_augment in self.anchor_sizes:
            # 2^0, 2^(1/3), 2^(2/3), used in the original paper
            for ar in aspect_ratios:
                if (ar == 1):
                    # Compute the regular anchor box for aspect ratio 1.
                    box_height = this_scale * size * scale_augment
                    box_width = this_scale * size * scale_augment
                    wh_list.append((box_width, box_height))
                    if self.two_boxes_for_ar1:
                        # Compute one slightly larger version using the geometric mean of this scale
                        # value and the next.
                        box_height = np.sqrt(this_scale * next_scale) * size * scale_augment
                        box_width = np.sqrt(this_scale * next_scale) * size * scale_augment
                        wh_list.append((box_width, box_height))
                else:
                    box_width = this_scale * size * scale_augment * np.sqrt(ar)
                    box_height = this_scale * size * scale_augment / np.sqrt(ar)
                    wh_list.append((box_width, box_height))

        wh_list = np.array(wh_list)
        n_boxes = len(wh_list)

        # Compute the grid of box center points. They are identical for all aspect ratios.

        # Compute the step sizes, i.e. how far apart the anchor box center points will be vertically
        # and horizontally.
        if (this_steps is None):
            step_height = self.img_height / feature_map_size[0]
            step_width = self.img_width / feature_map_size[1]
        else:
            if isinstance(this_steps, (list, tuple)) and (len(this_steps) == 2):
                step_height = this_steps[0]
                step_width = this_steps[1]
            elif isinstance(this_steps, (int, float)):
                step_height = this_steps
                step_width = this_steps
        # Compute the offsets, i.e. at what pixel values the first anchor box center point will be
        # from the top and from the left of the image.
        if (this_offsets is None):
            offset_height = 0.5
            offset_width = 0.5
        else:
            if isinstance(this_offsets, (list, tuple)) and (len(this_offsets) == 2):
                offset_height = this_offsets[0]
                offset_width = this_offsets[1]
            elif isinstance(this_offsets, (int, float)):
                offset_height = this_offsets
                offset_width = this_offsets
        # Now that we have the offsets and step sizes, compute the grid of anchor box center points.
        cy = np.linspace(offset_height * step_height,
                         (offset_height + feature_map_size[0] - 1) * step_height,
                         feature_map_size[0])
        cx = np.linspace(offset_width * step_width,
                         (offset_width + feature_map_size[1] - 1) * step_width, feature_map_size[1])
        cx_grid, cy_grid = np.meshgrid(cx, cy)

        # This is necessary for np.tile() to do what we want further down
        cx_grid = np.expand_dims(cx_grid, -1)
        # This is necessary for np.tile() to do what we want further down
        cy_grid = np.expand_dims(cy_grid, -1)

        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # where the last dimension will contain `(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_size[0], feature_map_size[1], n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes))  # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes))  # Set cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0]  # Set w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1]  # Set h

        # Convert `(cx, cy, w, h)` to `(xmin, ymin, xmax, ymax)`
        boxes_tensor = np_convert_coordinates(boxes_tensor,
                                              start_index=0,
                                              conversion='centroids2corners')

        # If `clip_boxes` is enabled, clip the coordinates to lie within the image boundaries
        if self.clip_boxes:
            x_coords = boxes_tensor[:, :, :, [0, 2]]
            x_coords[x_coords >= self.img_width] = self.img_width - 1
            x_coords[x_coords < 0] = 0
            boxes_tensor[:, :, :, [0, 2]] = x_coords
            y_coords = boxes_tensor[:, :, :, [1, 3]]
            y_coords[y_coords >= self.img_height] = self.img_height - 1
            y_coords[y_coords < 0] = 0
            boxes_tensor[:, :, :, [1, 3]] = y_coords

        # `normalize_coords` is enabled, normalize the coordinates to be within [0,1]
        if self.normalize_coords:
            boxes_tensor[:, :, :, [0, 2]] /= self.img_width
            boxes_tensor[:, :, :, [1, 3]] /= self.img_height

        box_tensor_corner = np.array(boxes_tensor)

        # Convert `(xmin, ymin, xmax, ymax)` back to `(cx, cy, w, h)`.
        boxes_tensor = np_convert_coordinates(boxes_tensor,
                                              start_index=0,
                                              conversion='corners2centroids')

        return boxes_tensor, box_tensor_corner

    def __encode_template(self, boxes_list):
        # Tile the anchor boxes for each predictor layer across all batch items.
        boxes_batch = []
        for boxes in boxes_list:
            boxes = np.reshape(boxes, (-1, 4))
            boxes_batch.append(boxes)

        # Concatenate the anchor tensors from the individual layers to one.
        boxes_tensor = np.concatenate(boxes_batch, axis=0)

        # 3: Create a template tensor to hold the one-hot class encodings of shape
        # `(batch, #boxes, #classes)`. It will contain all zeros for now, the classes will be set in
        #  the matching process that follows
        classes_tensor = np.zeros((boxes_tensor.shape[0], self.n_classes))
        classes_tensor[:, 0] = 1

        classes_weights = np.ones((boxes_tensor.shape[0], 1))

        # 4: Create a tensor to contain the variances. This tensor has the same shape as
        #    `boxes_tensor` and simply contains the same 4 variance values for every position in the
        # last axis.
        variances_tensor = np.zeros_like(boxes_tensor)
        variances_tensor += self.variances  # Long live broadcasting
        self.variances_tensor = tf.convert_to_tensor(variances_tensor, dtype=tf.float32)

        # 4: Concatenate the classes, boxes and variances tensors to get our final template for
        #    y_encoded. We also need another tensor of the shape of `boxes_tensor` as a space filler
        #    so that `y_encoding_template` has the same shape as the SSD model output tensor. The
        #    content of this tensor is irrelevant, we'll just use `boxes_tensor` a second time.
        #    Add class weights
        y_encoding_template = np.concatenate((classes_weights, classes_tensor,
                                              boxes_tensor, boxes_tensor, variances_tensor), axis=1)

        return y_encoding_template

    def __call__(self, ground_truth_labels):
        '''
        Converts ground truth bounding box data into a suitable format to train an SSD model.

        Arguments:
            ground_truth_labels (list): A python list of length `batch_size` that contains one 2D
                Numpy array for each batch image. Each such array has `k` rows for the `k` ground
                truth bounding boxes belonging to the respective image, and the data for each ground
                truth bounding box has the format `(class_id, xmin, ymin, xmax, ymax)` (i.e. the
                'corners' coordinate format), and `class_id` must be an integer greater than 0 for
                all boxes as class ID 0 is reserved for the background class.


        Returns:
            `y_encoded`, a 3D numpy array of shape `(batch_size, #boxes, #classes + 4 + 4 + 4)` that
            serves as the ground truth label tensor for training, where `#boxes` is the total number
            of boxes predicted by the model per image, and the classes are one-hot-encoded. The four
            elements after the class vecotrs in the last axis are the box coordinates, the next four
            elements after that are just dummy elements, and the last four elements are the
            variances.
        '''

        y_encoded = []

        ##################################################################################
        # Match ground truth boxes to anchor boxes.
        ##################################################################################

        # Match the ground truth boxes to the anchor boxes. Every anchor box that does not have
        # a ground truth match and for which the maximal IoU overlap with any ground truth box is
        # less than or equal to `neg_iou_limit` will be a negative (background) box.

        for gt_label in ground_truth_labels:  # For each batch item...
            match_y = tf.cond(tf.equal(tf.shape(gt_label)[0], 0),
                              lambda: self.encoding_template_tensor,
                              lambda label=gt_label: self.__calc_matched_anchor_gt(label))
            y_encoded.append(match_y)
        y_encoded = tf.stack(y_encoded, axis=0)
        ##################################################################################
        # Convert box coordinates to anchor box offsets.
        ##################################################################################
        y_encoded = self.__tf_convert_anchor_to_offset(y_encoded)
        return y_encoded

    def __tf_convert_anchor_to_offset(self, tensor):
        return tensor_strided_replace(tensor, (-12, -8), tf.concat([
                tf.truediv(tensor[..., -12:-10] - tensor[..., -8:-6],
                           tensor[..., -6:-4] * self.variances_tensor[..., -4:-2]),
                tf.truediv(tf.log(tf.truediv(tensor[..., -10:-8], tensor[..., -6:-4])),
                           self.variances_tensor[..., -2:])
            ], axis=-1), axis=-1)

    def __calc_matched_anchor_gt(self, gt_label):
        gt_label = tf.cast(gt_label, tf.float32)
        # Maybe normalize the box coordinates.
        confs = tf.gather(gt_label, tf.constant(0, dtype=tf.int32), axis=-1)
        if self.normalize_coords:
            # gt_label is [class_id, xmin, ymin, xmax, ymax]
            """
            gt_label = tf.stack([gt_label[:, 0],
                                 tf.truediv(gt_label[:, 1], self.img_width),
                                 tf.truediv(gt_label[:, 2], self.img_height),
                                 tf.truediv(gt_label[:, 3], self.img_width),
                                 tf.truediv(gt_label[:, 4], self.img_height),
                                 ], axis=-1)
            """
            if not self.gt_normalized:
                x_mins = tf.gather(gt_label, tf.constant(1, dtype=tf.int32), axis=-1)
                y_mins = tf.gather(gt_label, tf.constant(2, dtype=tf.int32), axis=-1)
                x_maxs = tf.gather(gt_label, tf.constant(3, dtype=tf.int32), axis=-1)
                y_maxs = tf.gather(gt_label, tf.constant(4, dtype=tf.int32), axis=-1)
                gt_label = tf.stack([confs,
                                     tf.truediv(x_mins, self.img_width),
                                     tf.truediv(y_mins, self.img_height),
                                     tf.truediv(x_maxs, self.img_width),
                                     tf.truediv(y_maxs, self.img_height)], axis=-1)
        """
        classes_one_hot = tf.one_hot(tf.reshape(tf.cast(gt_label[:, 0], tf.int32), [-1]),
                                     self.n_classes)
        """
        classes_one_hot = tf.one_hot(tf.reshape(tf.cast(confs, tf.int32), [-1]),
                                     self.n_classes)

        classes_weights = tf.gather(self.class_weights, tf.cast(gt_label[:, 0:1], tf.int32))

        # Compute the IoU similarities between all anchor boxes and all ground truth boxes for this
        # batch item. This is a matrix of shape `(num_ground_truth_boxes, num_anchor_boxes).
        gt_xys = tf.gather(gt_label, tf.range(1, 5), axis=-1)
        # similarities = iou(gt_label[..., 1:], self.anchorbox_tensor)
        similarities = iou(gt_xys, self.anchorbox_tensor)

        # Maybe convert the box coordinate format.
        # gt_label = corners_to_centroids(gt_label, start_index=1)
        gt_centroid = corners_to_centroids(gt_label, start_index=1)
        gt_centroid = tf.gather(gt_centroid, tf.range(1, 5), axis=-1)
        # labels_one_hot = tf.concat([classes_one_hot, gt_label[:, 1:]], axis=-1)
        labels_one_hot = tf.concat([classes_weights, classes_one_hot, gt_centroid], axis=-1)

        # First: Do bipartite matching, i.e. match each ground truth box to the one anchor box with
        #        the highest IoU.
        #        This ensures that each ground truth box will have at least one good match.

        # For each ground truth box, get the anchor box to match with it.
        bipartite_matches = bipartite_match_row(similarities)

        # Write the ground truth data to the matched anchor boxes.
        y_encoded_cls_box = self.encoding_template_tensor[:, :-8]
        match_y = tensor_slice_replace(y_encoded_cls_box,
                                       labels_one_hot,
                                       bipartite_matches,
                                       tf.range(tf.shape(labels_one_hot)[0]))

        # Write the highest IOU flag:
        end_flag = tf.expand_dims(self.encoding_template_tensor[:, -1], -1)
        end_flag_on = tf.fill(tf.shape(end_flag), 1024.0)
        end_flag = tensor_slice_replace(end_flag,
                                        end_flag_on,
                                        bipartite_matches,
                                        bipartite_matches)

        # Set the columns of the matched anchor boxes to zero to indicate that they were matched.
        sim_trans = tf.transpose(similarities)
        sim_trans_zero = tf.zeros_like(sim_trans)
        sim_trans_replace = tensor_slice_replace(sim_trans,
                                                 sim_trans_zero,
                                                 bipartite_matches,
                                                 bipartite_matches)
        similarities = tf.transpose(sim_trans_replace)

        # Second: Maybe do 'multi' matching, where each remaining anchor box will be matched to its
        #         most similar ground truth box with an IoU of at least `pos_iou_threshold`, or not
        #         matched if there is no such ground truth box.
        # Get all matches that satisfy the IoU threshold.
        matches = multi_match(similarities, self.pos_iou_threshold)
        # Write the ground truth data to the matched anchor boxes.
        match_y = tensor_slice_replace(match_y,
                                       labels_one_hot,
                                       matches[1],
                                       matches[0])

        # Set the columns of the matched anchor boxes to zero to indicate that they were matched.
        sim_trans_replace = tensor_slice_replace(sim_trans_replace,
                                                 sim_trans_zero,
                                                 matches[1],
                                                 matches[1])

        similarities = tf.transpose(sim_trans_replace)

        # Third: Now after the matching is done, all negative (background) anchor boxes that have
        #        an IoU of `neg_iou_limit` or more with any ground truth box will be set to netral,
        #        i.e. they will no longer be background boxes. These anchors are "too close" to a
        #        ground truth box to be valid background boxes.
        # +1 on background_id for class_weights

        max_background_similarities = tf.reduce_max(similarities, axis=0)
        neutral_boxes = tf.reshape(tf.where(max_background_similarities >= self.neg_iou_limit),
                                   [-1])
        neutral_boxes = tf.cast(neutral_boxes, tf.int32)
        match_y_bg_only = tf.expand_dims(match_y[:, 1], -1)
        match_y_bg_only_zero = tf.zeros_like(match_y_bg_only)
        match_y_bg_only = tensor_slice_replace(match_y_bg_only,
                                               match_y_bg_only_zero,
                                               neutral_boxes,
                                               neutral_boxes)
        match_y = tensor_strided_replace(match_y, (1, 2), match_y_bg_only, axis=-1)
        match_y = tf.concat([match_y, self.encoding_template_tensor[:, -8:-1], end_flag], axis=-1)
        return match_y
