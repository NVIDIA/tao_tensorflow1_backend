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

"""IVA RetinaNet base architecture."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras.layers import Activation, Concatenate, Conv2D, Input, Permute, Reshape
from keras.models import Model

import numpy as np

from nvidia_tao_tf1.core.models.quantize_keras_model import create_quantized_keras_model
from nvidia_tao_tf1.cv.retinanet.initializers.prior_prob import PriorProbability
from nvidia_tao_tf1.cv.retinanet.layers.anchor_box_layer import RetinaAnchorBoxes
from nvidia_tao_tf1.cv.retinanet.models.fpn import FPN


class Subnet(object):
    """Subnet Base Class."""

    def __init__(self, name):
        """Init."""
        self.name = name
        self.subnet = {}
        self.regressor = None

    def __call__(self, feature_map):
        """Build model graph."""
        x = feature_map
        for i in range(len(self.subnet)):
            x = self.subnet[self.name + '_' + str(i)](x)
        return self.regressor(x)


class ClassSubnet(Subnet):
    """Subnet for classification branch."""

    def __init__(self, n_anchors, n_classes, prior_prob, n_kernels,
                 feature_size=256, kernel_regularizer=None, name='class_submodule'):
        """Initialize layers in the subnet."""
        self.n_anchors = n_anchors
        self.n_classes = n_classes
        self.prior_prob = prior_prob
        self.n_kernels = n_kernels
        self.feature_size = feature_size
        self.name = name
        options = {
            'padding' : 'same',
            'data_format' : 'channels_first',
            'kernel_initializer' : keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            'kernel_regularizer' : kernel_regularizer
        }
        self.subnet = {}
        for i in range(n_kernels):
            self.subnet[name + '_' + str(i)] = Conv2D(feature_size, (3, 3),
                                                      activation='relu',
                                                      name='retinanet_class_subn_'+str(i),
                                                      bias_initializer='zeros',
                                                      **options)
        self.regressor = Conv2D(n_anchors * n_classes, (3, 3),
                                bias_initializer=PriorProbability(probability=prior_prob),
                                name='retinanet_conf_regressor',
                                **options)


class LocSubnet(Subnet):
    """Subnet for localization branch."""

    def __init__(self, n_anchors, n_kernels, feature_size=256,
                 kernel_regularizer=None, name='loc_submodule'):
        """Initialize layers in the subnet."""
        self.n_anchors = n_anchors
        self.n_kernels = n_kernels
        self.feature_size = feature_size
        self.name = name
        options = {
            'kernel_size' : 3,
            'strides' : 1,
            'padding'  : 'same',
            'data_format' : 'channels_first',
            'kernel_initializer' : keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            'kernel_regularizer' : kernel_regularizer,
            'bias_initializer' : 'zeros'
        }
        self.subnet = {}
        for i in range(n_kernels):
            self.subnet[name + '_' + str(i)] = Conv2D(feature_size, activation='relu',
                                                      name='retinanet_loc_subn_'+str(i),
                                                      **options)
        self.regressor = Conv2D(n_anchors * 4,
                                name='retinanet_loc_regressor',
                                **options)


def retinanet(image_size,
              n_classes,
              kernel_regularizer=None,
              bias_regularizer=None,
              freeze_blocks=None,
              freeze_bn=None,
              use_batch_norm=True,
              use_pooling=False,
              use_bias=False,
              all_projections=True,
              dropout=None,
              min_scale=None,
              max_scale=None,
              scales=None,
              aspect_ratios_global=None,
              aspect_ratios_per_layer=None,
              two_boxes_for_ar1=False,
              steps=None,
              n_anchor_levels=3,
              offsets=None,
              clip_boxes=False,
              variances=None,
              arch="resnet",
              nlayers=None,
              n_kernels=2,
              feature_size=256,
              input_tensor=None,
              qat=False):
    '''
    Build a Keras model with RetinaNet architecture, see references.

    Arguments:
        image_size (tuple): The input image size in the format `(channels, height, width)`.
        n_classes (int): The number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO.
        kernel_regularizer (float, optional): Applies to all convolutional layers.
        freeze_blocks (list, optional): The indices of the freezed subblocks.
        freeze_bn (boolean, optional): Whether to freeze the BN layers.
        use_batch_norm (boolean, optional): Whether to use BN in the feature extractor.
        use_pooling (boolean, optional): Whether to use max pooling in the feature extractor.
        use_bias (boolean, optional): Whether to use bias in conv layers of the feature extractor.
        all_projections (boolean, optional): Whether to use projection in some feature extractors.
        dropout (float, optional): dropout ratio for squeezenet.
        n_kernels (int, optional): number of conv kernels in the submodule.
        feature_size (int, optional): number of channels in FPN and submodule.
        nlayers (int, optional): number of layers in ResNets or Vggs.
        arch (string): name of the feature extractor.
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
            individually. If a list is
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
        return_predictor_sizes (bool, optional): If `True`, this function not only returns the
            model, but also a list containing the spatial dimensions of the predictor layers. This
            isn't strictly necessary since you can always get their sizes easily via the Keras API,
            but it's convenient and less error-prone to get them this way. They are only relevant
            for training anyway, for inference you don't need them.
        qat (bool): If `True`, build an quantization aware model.

    Returns:
        model: The Keras RetinaNet model.
        predictor_sizes (optional): A Numpy array containing the `(height, width)` portion
            of the output tensor shape for each convolutional predictor layer. During
            training, the generator function needs this in order to transform
            the ground truth labels into tensors of identical structure as the
            output tensors of the model, which is in turn needed for the cost
            function.

    References:
        https://arxiv.org/abs/1708.02002
    '''

    n_predictor_layers = 5  # n_predictor conv layers in the network is 5 for retinanet.
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
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers+1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but \
                len(scales) == {}.".format(n_predictor_layers+1, len(scales)))
    else:  # If no explicit list of scaling factors was passed, compute the list of scaling factors
        scales = np.linspace(min_scale, max_scale, n_predictor_layers+1)

    if len(variances) != 4:
        raise ValueError("4 variance values must be passed, but {} values were received."
                         .format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}"
                         .format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")

    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # If only a global aspect ratio list was passed, then the number of boxes is the same
    # for each predictor layer
    if (1 in aspect_ratios_global) & two_boxes_for_ar1:
        n_boxes = n_anchor_levels * (len(aspect_ratios_global) + 1)
    else:
        n_boxes = n_anchor_levels * (len(aspect_ratios_global))

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

    retinanet_fpn = FPN(x, arch,
                        nlayers=nlayers,
                        data_format='channels_first',
                        use_batch_norm=use_batch_norm,
                        use_pooling=use_pooling,
                        use_bias=use_bias,
                        all_projections=all_projections,
                        dropout=dropout,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        freeze_blocks=freeze_blocks,
                        freeze_bn=freeze_bn,
                        force_relu=False)
    if arch in ['vgg', 'resnet']:
        assert nlayers > 0, "Please specify the number of layers for VGGNets or ResNets."

    feature_map_list = retinanet_fpn.generate(feature_size=feature_size,
                                              kernel_regularizer=kernel_regularizer)
    if len(feature_map_list) != 5:
        raise ValueError('Need 5 feature maps from base model')

    conf_list = []
    loc_list = []
    anchor_list = []

    classification_subnet = ClassSubnet(n_boxes, n_classes, 0.01, n_kernels,
                                        feature_size=feature_size,
                                        kernel_regularizer=kernel_regularizer)
    localization_subnet = LocSubnet(n_boxes, n_kernels,
                                    feature_size=feature_size,
                                    kernel_regularizer=kernel_regularizer)

    if qat:
        raw_model = Model(inputs=x, outputs=feature_map_list)
        qat_model = create_quantized_keras_model(raw_model)
        feature_map_list = [
            qat_model.get_layer('P3_relu').output,
            qat_model.get_layer('P4_relu').output,
            qat_model.get_layer('P5_relu').output,
            qat_model.get_layer('P6_relu').output,
            qat_model.get_layer('P7_relu').output]

    for idx, feature_map in enumerate(feature_map_list):

        conf = classification_subnet(feature_map)
        loc = localization_subnet(feature_map)

        anchor = RetinaAnchorBoxes(img_height, img_width,
                                   this_scale=scales[idx],
                                   next_scale=scales[idx+1],
                                   aspect_ratios=aspect_ratios[idx],
                                   two_boxes_for_ar1=two_boxes_for_ar1,
                                   this_steps=steps[idx],
                                   this_offsets=offsets[idx],
                                   clip_boxes=clip_boxes,
                                   variances=variances,
                                   n_anchor_levels=n_anchor_levels,
                                   name='retinanet_anchor_'+str(idx))(loc)
        conf = Reshape((-1, 1, n_classes), name='conf_reshape_'+str(idx))(Permute((2, 3, 1))(conf))
        loc = Reshape((-1, 1, 4), name='loc_reshape_'+str(idx))(Permute((2, 3, 1))(loc))
        anchor = Reshape((-1, 1, 8), name='anchor_reshape_'+str(idx))(anchor)
        conf_list.append(conf)
        loc_list.append(loc)
        anchor_list.append(anchor)

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
    mbox_conf_sigmoid = Activation('sigmoid', name='mbox_conf_sigmoid')(mbox_conf)

    # Concatenate the class and box predictions and the anchors to one large predictions vector
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=-1)([mbox_conf_sigmoid, mbox_loc, mbox_priorbox])
    predictions = Reshape((-1, n_classes+4+8), name='retinanet_predictions')(predictions)
    return Model(inputs=x, outputs=predictions, name='retinanet_'+arch)
