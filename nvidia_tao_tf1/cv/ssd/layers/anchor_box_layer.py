'''
A custom Keras layer to generate anchor boxes.

Copyright (C) 2019 Pierluigi Ferrari

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import division
import keras.backend as K
from keras.engine.topology import InputSpec, Layer
import numpy as np
import tensorflow as tf
from nvidia_tao_tf1.cv.ssd.utils.box_utils import np_convert_coordinates


class AnchorBoxes(Layer):
    '''
    AnchorBoxes layer.

    This is a keras custom layer for SSD. Code is from GitHub and with Apache-2 license. Link:
    https://github.com/pierluigiferrari/ssd_keras/tree/3ac9adaf3889f1020d74b0eeefea281d5e82f353

    A Keras layer to create an output tensor containing anchor box coordinates
    and variances based on the input tensor and the passed arguments.

    A set of 2D anchor boxes of different aspect ratios is created for each spatial unit of
    the input tensor. The number of anchor boxes created per unit depends on the arguments
    `aspect_ratios` and `two_boxes_for_ar1`, in the default case it is 4. The boxes
    are parameterized by the coordinate tuple `(xmin, xmax, ymin, ymax)`.

    The logic implemented by this layer is identical to the logic in the module
    `ssd_box_encode_decode_utils.py`.

    The purpose of having this layer in the network is to make the model self-sufficient
    at inference time. Since the model is predicting offsets to the anchor boxes
    (rather than predicting absolute box coordinates directly), one needs to know the anchor
    box coordinates in order to construct the final prediction boxes from the predicted offsets.
    If the model's output tensor did not contain the anchor box coordinates, the necessary
    information to convert the predicted offsets back to absolute coordinates would be missing
    in the model output. The reason why it is necessary to predict offsets to the anchor boxes
    rather than to predict absolute box coordinates directly is explained in `README.md`.

    Input shape:
        4D tensor of shape `(batch, channels, height, width)` if `dim_ordering = 'th'`
        or `(batch, height, width, channels)` if `dim_ordering = 'tf'`.

    Output shape:
        5D tensor of shape `(batch, height, width, n_boxes, 8)`. The last axis contains
        the four anchor box coordinates and the four variance values for each box.
    '''

    def __init__(self,
                 img_height,
                 img_width,
                 this_scale,
                 next_scale,
                 aspect_ratios=None,
                 two_boxes_for_ar1=True,
                 this_steps=None,
                 this_offsets=None,
                 clip_boxes=False,
                 variances=None,
                 **kwargs):
        '''
        init function.

        All arguments need to be set to the same values as in the box encoding process,
        otherwise the behavior is undefined. Some of these arguments are explained in
        more detail in the documentation of the `SSDBoxEncoder` class.

        Arguments:
            img_height (int): The height of the input images.
            img_width (int): The width of the input images.
            this_scale (float): A float in [0, 1], the scaling factor for the size of
                the generated anchor boxes as a fraction of the shorter side of the input image.
            next_scale (float): A float in [0, 1], the next larger scaling factor. Only
                relevant if self.two_boxes_for_ar1 == True`.
            aspect_ratios (list, optional): The list of aspect ratios for which default
                boxes are to be generated for this layer.
            two_boxes_for_ar1 (bool, optional): Only relevant if `aspect_ratios` contains 1.
                If `True`, two default boxes will be generated for aspect ratio 1. The first
                will be generated using the scaling factor for the respective layer, the second
                one will be generated using geometric mean of said scaling factor and next bigger
                scaling factor.
            clip_boxes (bool, optional): If `True`, clips the anchor box coordinates to stay within
                image boundaries.
            variances (list, optional): A list of 4 floats >0. The anchor box offset for
                each coordinate will be divided by its respective variance value.
        '''
        if K.backend() != 'tensorflow':
            raise TypeError("This layer only supports TF at the moment, but you are using the {}."
                            .format(K.backend()))

        if (this_scale < 0) or (next_scale < 0) or (this_scale > 1):
            raise ValueError("`this_scale` must be in [0, 1] and `next_scale` must be >0, \
            but `this_scale` == {}, `next_scale` == {}"
                             .format(this_scale, next_scale))

        if len(variances) != 4:
            raise ValueError("4 variance values must be pased, but {} values were received."
                             .format(len(variances)))
        variances = np.array(variances)
        if np.any(variances <= 0):
            raise ValueError("All variances must be >0, but the variances given are {}"
                             .format(variances))

        self.img_height = img_height
        self.img_width = img_width
        self.this_scale = this_scale
        self.next_scale = next_scale
        self.aspect_ratios = aspect_ratios
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.this_steps = this_steps
        self.this_offsets = this_offsets
        self.clip_boxes = clip_boxes
        self.variances = variances
        # Compute the number of boxes per cell
        if (1 in aspect_ratios) and two_boxes_for_ar1:
            self.n_boxes = len(aspect_ratios) + 1
        else:
            self.n_boxes = len(aspect_ratios)
        super(AnchorBoxes, self).__init__(**kwargs)

    def build(self, input_shape):
        """Layer build function."""

        self.input_spec = [InputSpec(shape=input_shape)]

        size = min(self.img_height, self.img_width)
        # Compute the box widths and and heights for all aspect ratios
        wh_list = []
        for ar in self.aspect_ratios:
            if (ar == 1):
                # Compute the regular anchor box for aspect ratio 1.
                box_height = box_width = self.this_scale * size
                wh_list.append((box_width, box_height))
                if self.two_boxes_for_ar1:
                    # Compute one slightly larger version using the geometric mean.
                    box_height = box_width = np.sqrt(self.this_scale * self.next_scale) * size
                    wh_list.append((box_width, box_height))
            else:
                box_height = self.this_scale * size / np.sqrt(ar)
                box_width = self.this_scale * size * np.sqrt(ar)
                wh_list.append((box_width, box_height))
        wh_list = np.array(wh_list)

        _, _, feature_map_height, feature_map_width = input_shape

        # Compute the grid of box center points. They are identical for all aspect ratios.

        # Compute the step sizes
        if (self.this_steps is None):
            step_height = self.img_height / feature_map_height
            step_width = self.img_width / feature_map_width
        else:
            if isinstance(self.this_steps, (list, tuple)) and (len(self.this_steps) == 2):
                step_height = self.this_steps[0]
                step_width = self.this_steps[1]
            elif isinstance(self.this_steps, (int, float)):
                step_height = self.this_steps
                step_width = self.this_steps
        # Compute the offsets.
        if (self.this_offsets is None):
            offset_height = 0.5
            offset_width = 0.5
        else:
            if isinstance(self.this_offsets, (list, tuple)) and (len(self.this_offsets) == 2):
                offset_height = self.this_offsets[0]
                offset_width = self.this_offsets[1]
            elif isinstance(self.this_offsets, (int, float)):
                offset_height = self.this_offsets
                offset_width = self.this_offsets
        # Now that we have the offsets and step sizes, compute the grid of anchor box center points.
        cy = np.linspace(offset_height * step_height,
                         (offset_height + feature_map_height - 1) * step_height, feature_map_height)
        cx = np.linspace(offset_width * step_width,
                         (offset_width + feature_map_width - 1) * step_width, feature_map_width)
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1)  # This is necessary for np.tile()
        cy_grid = np.expand_dims(cy_grid, -1)  # This is necessary for np.tile()

        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # where the last dimension will contain `(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_height, feature_map_width, self.n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, self.n_boxes))  # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, self.n_boxes))  # Set cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0]  # Set w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1]  # Set h

        # Convert `(cx, cy, w, h)` to `(xmin, xmax, ymin, ymax)`
        boxes_tensor = np_convert_coordinates(boxes_tensor, start_index=0,
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

        boxes_tensor[:, :, :, [0, 2]] /= self.img_width
        boxes_tensor[:, :, :, [1, 3]] /= self.img_height

        # AnchorBox layer will output `(xmin,ymin,xmax,ymax)`. The ground truth is
        # `(cx,cy,logw,logh)`. However, we don't need to further convert to centroids here since
        # this layer will not carry any gradient backprob. The following command will do the
        # convertion if we eventually want.
        # boxes_tensor = np_convert_coordinates(boxes_tensor,
        #                                     start_index=0, conversion='corners2centroids')

        # Create a tensor to contain the variances and append it to `boxes_tensor`.
        # This tensor has the same shape as `boxes_tensor` and simply contains the same
        # 4 variance values for every position in the last axis.
        variances_tensor = np.zeros_like(boxes_tensor)  # `(height, width, n_boxes, 4)`
        variances_tensor += self.variances  # Long live broadcasting
        # Now `boxes_tensor` becomes a tensor of shape `(height, width, n_boxes, 8)`
        boxes_tensor = np.concatenate((boxes_tensor, variances_tensor), axis=-1)

        # Below to make tensor 4D.
        # (feature_map, n_boxes, 8)
        boxes_tensor = boxes_tensor.reshape((-1, self.n_boxes, 8))

        # Now prepend one dimension to `boxes_tensor` to account for the batch size and tile it.
        # The result will be a 5D tensor of shape `(batch_size, height, width, n_boxes, 8)`
        boxes_tensor = np.expand_dims(boxes_tensor, axis=0)

        self.boxes_tensor = K.constant(boxes_tensor, dtype='float32')
        # (feature_map, n_boxes, 8)
        super(AnchorBoxes, self).build(input_shape)

    def call(self, x, mask=None):
        '''
        Return an anchor box tensor based on the shape of the input tensor.

        Note that this tensor does not participate in any graph computations at runtime.
        It is being created as a constant once during graph creation and is just being
        output along with the rest of the model output during runtime. Because of this,
        all logic is implemented as Numpy array operations and it is sufficient to convert
        the resulting Numpy array into a Keras tensor at the very end before outputting it.

        Arguments:
            x (tensor): 4D tensor of shape `(batch, channels, height, width)` if
                `dim_ordering = 'th'` or `(batch, height, width, channels)` if
                `dim_ordering = 'tf'`. The input for this layer must be the output
                of the localization predictor layer.
        '''

        # Compute box width and height for each aspect ratio
        # The shorter side of the image will be used to compute `w` and `h`.
        box_tensor_dup = tf.identity(self.boxes_tensor)
        with tf.name_scope(None, 'FirstDimTile'):
            x_dup = tf.identity(x)
            boxes_tensor = K.tile(box_tensor_dup, (K.shape(x_dup)[0], 1, 1, 1))

        return boxes_tensor

    def compute_output_shape(self, input_shape):
        '''Layer output shape function.'''
        batch_size, _, feature_map_height, feature_map_width = input_shape
        return (batch_size, feature_map_height*feature_map_width, self.n_boxes, 8)

    def get_config(self):
        '''Layer get_config function.'''
        config = {
            'img_height': self.img_height,
            'img_width': self.img_width,
            'this_scale': self.this_scale,
            'next_scale': self.next_scale,
            'aspect_ratios': list(self.aspect_ratios),
            'two_boxes_for_ar1': self.two_boxes_for_ar1,
            'clip_boxes': self.clip_boxes,
            'variances': list(self.variances)
        }
        base_config = super(AnchorBoxes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
