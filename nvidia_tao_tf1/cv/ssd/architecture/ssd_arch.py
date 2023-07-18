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

"""IVA SSD base architecture."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Concatenate, Conv2D, Input, Permute, Reshape, Softmax
from keras.models import Model

import numpy as np

from nvidia_tao_tf1.core.models.quantize_keras_model import create_quantized_keras_model
from nvidia_tao_tf1.cv.ssd.layers.anchor_box_layer import AnchorBoxes
from nvidia_tao_tf1.cv.ssd.models.base_model import get_base_model


def ssd(image_size,
        n_classes,
        is_dssd,
        kernel_regularizer=None,
        freeze_blocks=None,
        freeze_bn=None,
        min_scale=None,
        max_scale=None,
        scales=None,
        aspect_ratios_global=None,
        aspect_ratios_per_layer=None,
        two_boxes_for_ar1=False,
        steps=None,
        offsets=None,
        clip_boxes=False,
        variances=None,
        confidence_thresh=0.01,
        iou_threshold=0.45,
        top_k=200,
        nms_max_output_size=400,
        arch="resnet",
        nlayers=10,
        pred_num_channels=0,
        input_tensor=None,
        qat=True):
    '''
    Build a Keras model with SSD300 architecture, see references.

    The base network is a reduced atrous VGG-16, extended by the SSD architecture,
    as described in the paper.

    Most of the arguments that this function takes are only needed for the anchor
    box layers. In case you're training the network, the parameters passed here must
    be the same as the ones used to set up `SSDBoxEncoder`. In case you're loading
    trained weights, the parameters passed here must be the same as the ones used
    to produce the trained weights.

    Some of these arguments are explained in more detail in the documentation of the
    `SSDBoxEncoder` class.

    Note: Requires Keras v2.0 or later. Currently works only with the
    TensorFlow backend (v1.0 or later).

    Arguments:
        image_size (tuple): The input image size in the format `(channels, height, width)`.
        n_classes (int): The number of positive classes plus background class,
                         e.g. 21 for Pascal VOC, 81 for MS COCO.
        is_dssd (bool): Is it a DSSD model or SSD model.
        arch (string): Network architecture
        nlayers (int): Property of the arch
        pred_num_channels (int): number of channels for convs inside pred module.
        kernel_regularizer (float, optional): Applies to all convolutional layers.
        freeze_blocks (list, optional): The indices of the freezed subblocks
        freeze_bn (boolean, optional): Whether to freeze the BN layers
        min_scale (float, optional): The smallest scaling factor for the size of the anchor
            boxes as a fraction of the shorter side of the input images.
        max_scale (float, optional): The largest scaling factor for the size of the anchor boxes
            as a fraction of the shorter side of the input images. All scaling factors between
            the smallest and the largest will be linearly interpolated. Note that the second to
            last of the linearly interpolated scaling factors will actually be the scaling factor
            for the last predictor layer, while the last scaling factor is used for the second box
            for aspect ratio 1 in the last predictor layer if `two_boxes_for_ar1` is `True`.
        scales (list, optional): A list of floats containing scaling factors per convolutional
            predictor layer. This list must be one element longer than the number of predictor
            layers. The first `k` elements are the scaling factors for the `k` predictor layers,
            while the last element is used for the second box for aspect ratio 1 in the last
            predictor layer if `two_boxes_for_ar1` is `True`. This additional last scaling factor
            must be passed either way, even if it is not being used. If a list is passed, this
            argument overrides `min_scale` and `max_scale`. All scaling factors must be greater
            than zero.
        aspect_ratios_global (list, optional): The list of aspect ratios for which anchor boxes are
            to be generated. This list is valid for all prediction layers.
        aspect_ratios_per_layer (list, optional): A list containing one aspect ratio list for each
            prediction layer. This allows you to set the aspect ratios for each predictor layer
            individually, which is the case for the original SSD300 implementation. If a list is
            passed, it overrides `aspect_ratios_global`.
        two_boxes_for_ar1 (bool, optional): Only relevant for aspect ratio lists that contain 1.
            Will be ignored otherwise. If `True`, two anchor boxes will be generated for aspect
            ratio 1. The first will be generated using the scaling factor for the respective layer,
            the second one will be generated using geometric mean of said scaling factor and next
            bigger scaling factor.
        steps (list, optional): `None` or a list with as many elements as there are pred layers.
            The elements can be either ints/floats or tuples of two ints/floats. These numbers
            represent for each predictor layer how many pixels apart the anchor box center points
            should be vertically and horizontally along the spatial grid over the image. If the
            list contains ints/floats, then that value will be used for both spatial dimensions.
            If the list contains tuples of two ints/floats, then they represent
            `(step_height, step_width)`. If no steps are provided, then they will be computed such
            that the anchor box center points will form an equidistant grid within the image
            dimensions.
        offsets (list, optional): `None` or a list with as many elements as there are predictor
            layers. The elements can be either floats or tuples of two floats. These numbers
            represent for each predictor layer how many pixels from the top and left boarders of
            the image the top-most and left-most anchor box center points should be as a fraction of
            `steps`. The last bit is important: The offsets are not absolute pixel values, but
            fractions of the step size specified in the `steps` argument. If the list contains
            floats, then that value will be used for both spatial dimensions. If the list contains
            tuples of two floats, then they represent `(vertical_offset, horizontal_offset)`. If no
            offsets are provided, then they will default to 0.5 of the step size.
        clip_boxes (bool, optional): If `True`, clips the anchor box coordinates to stay within
            image boundaries.
        variances (list, optional): A list of 4 floats >0. The anchor box offset for each coordinate
            will be divided by its respective variance value.
        divide_by_stddev (array-like, optional): `None` or an array-like object of non-zero integers
            or floating point values of any shape that is broadcast-compatible with the image shape.
            The image pixel intensity values will be divided by the elements of this array. For
            example, pass a list of three integers to perform per-channel standard deviation
            normalization for color images.
        swap_channels (list, optional): Either `False` or a list of integers representing the
            desired order in which the input image channels should be swapped.
        confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence
            in a specific positive class in order to be considered for the non-maximum suppression
            stage for the respective class. A lower value will result in a larger part of the
            selection process being done by the non-maximum suppression stage, while a larger value
            will result in a larger part of the selection process happening in the confidence
            thresholding stage.
        iou_threshold (float, optional): A float in [0,1]. All boxes that have a Jaccard similarity
            of greater than `iou_threshold` with a locally maximal box will be removed from the set
            of predictions for a given class, where 'maximal' refers to the box's confidence score.
        top_k (int, optional): The number of highest scoring predictions to be kept for each batch
            item after the non-maximum suppression stage.
        nms_max_output_size (int, optional): The maximal number of predictions that will be left
            over after the NMS stage.
        return_predictor_sizes (bool, optional): If `True`, this function not only returns the
            model, but also a list containing the spatial dimensions of the predictor layers. This
            isn't strictly necessary since you can always get their sizes easily via the Keras API,
            but it's convenient and less error-prone to get them this way. They are only relevant
            for training anyway (SSDBoxEncoder needs to know the spatial dimensions of the predictor
            layers), for inference you don't need them.
        qat (bool): If `True`, build an quantization aware model.

    Returns:
        model: The Keras SSD300 model.
        predictor_sizes (optional): A Numpy array containing the `(height, width)` portion
            of the output tensor shape for each convolutional predictor layer. During
            training, the generator function needs this in order to transform
            the ground truth labels into tensors of identical structure as the
            output tensors of the model, which is in turn needed for the cost
            function.

    References:
        https://arxiv.org/abs/1512.02325v5
    '''

    # n_predictor conv layers in the network is 6 for the original SSD300.
    n_predictor_layers = 6
    img_channels, img_height, img_width = image_size[0], image_size[1], image_size[2]

    ############################################################################
    # Get a few exceptions out of the way.
    ############################################################################

    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError("`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None.\
         At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError("It must be either aspect_ratios_per_layer is None or \
                len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}."
                             .format(n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError(
            "Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers+1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but \
                len(scales) == {}.".format(n_predictor_layers+1, len(scales)))
    else:  # If no explicit list of scaling factors was passed, compute the list of scaling factors
        scales = np.linspace(min_scale, max_scale, n_predictor_layers+1)

    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received."
                         .format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}"
                         .format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError(
            "You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError(
            "You must provide at least one offset value per predictor layer.")

    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                # +1 for the second box for aspect ratio 1
                n_boxes.append(len(ar) + 1)
            else:
                n_boxes.append(len(ar))
    else:
        # If only a global aspect ratio list was passed, then the number of boxes is the same
        # for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    ############################################################################
    # Build the network.
    ############################################################################
    if input_tensor is None:
        x = Input(shape=(img_channels, img_height, img_width), name="Input")
    else:
        x = Input(tensor=input_tensor, name="Input")

    feature_map_list = get_base_model(x, arch, nlayers, is_dssd, pred_num_channels,
                                      kernel_regularizer=kernel_regularizer,
                                      freeze_blocks=freeze_blocks,
                                      freeze_bn=freeze_bn)
    if len(feature_map_list) != 6:
        raise ValueError('Need 6 feature maps from base model')

    conf_list = []
    loc_list = []
    anchor_list = []

    for idx, feature_map in enumerate(feature_map_list):
        conf = Conv2D(n_boxes[idx] * n_classes, (3, 3),
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=kernel_regularizer,
                      name='ssd_conf_'+str(idx))(feature_map)
        loc = Conv2D(n_boxes[idx] * 4, (3, 3),
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=kernel_regularizer,
                     name='ssd_loc_'+str(idx))(feature_map)

        conf_list.append(conf)
        loc_list.append(loc)

    if qat:
        raw_model = Model(inputs=x, outputs=conf_list+loc_list)
        qat_model = create_quantized_keras_model(raw_model)
        conf_list = []
        loc_list = []
        for idx in range(len(feature_map_list)):
            conf_list.append(qat_model.get_layer('ssd_conf_'+str(idx)).output)
            loc_list.append(qat_model.get_layer('ssd_loc_'+str(idx)).output)

    for idx, loc in enumerate(loc_list):
        anchor = AnchorBoxes(img_height, img_width,
                             this_scale=scales[idx],
                             next_scale=scales[idx+1],
                             aspect_ratios=aspect_ratios[idx],
                             two_boxes_for_ar1=two_boxes_for_ar1,
                             this_steps=steps[idx],
                             this_offsets=offsets[idx],
                             clip_boxes=clip_boxes,
                             variances=variances,
                             name='ssd_anchor_'+str(idx))(loc)

        anchor_list.append(anchor)

    conf_list = [Reshape((-1, 1, n_classes),
                         name='conf_reshape_'+str(idx)
                         )(Permute((2, 3, 1))(x)) for idx, x in enumerate(conf_list)]
    loc_list = [Reshape((-1, 1, 4),
                        name='loc_reshape_'+str(idx)
                        )(Permute((2, 3, 1))(x)) for idx, x in enumerate(loc_list)]
    anchor_list = [Reshape((-1, 1, 8),
                           name='anchor_reshape_'+str(idx)
                           )(x) for idx, x in enumerate(anchor_list)]

    # Concatenate the predictions from the different layers

    # Axis0 (batch) and axis2 (n_classes or 4, respectively) are identical for all layer predictions
    # so we want to concatenate along axis 1, the number of boxes per layer
    # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
    mbox_conf = Concatenate(axis=1, name='mbox_conf')(conf_list)

    # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
    mbox_loc = Concatenate(axis=1, name='mbox_loc')(loc_list)

    # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')(anchor_list)

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    # @TODO(tylerz): Using softmax for class prediction
    mbox_conf = Permute((3, 2, 1), name="before_softmax_permute")(mbox_conf)
    mbox_conf_softmax = Softmax(axis=1, name='mbox_conf_softmax_')(mbox_conf)
    mbox_conf_softmax = Permute(
        (3, 2, 1), name='mbox_conf_softmax')(mbox_conf_softmax)

    # Concatenate the class and box predictions and the anchors to one large predictions vector
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    # @TODO(tylerz): Using softmax for class prediction
    predictions = Concatenate(
        axis=-1)([mbox_conf_softmax, mbox_loc, mbox_priorbox])
    predictions = Reshape((-1, n_classes+4+8),
                          name='ssd_predictions')(predictions)
    return Model(inputs=x, outputs=predictions, name=('dssd_' if is_dssd else 'ssd_')+arch)
