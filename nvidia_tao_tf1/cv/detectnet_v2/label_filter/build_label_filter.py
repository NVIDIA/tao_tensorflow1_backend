# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.

"""Builder for BaseLabelFilter and child classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf1.cv.detectnet_v2.label_filter.bbox_crop_label_filter import BboxCropLabelFilter
from nvidia_tao_tf1.cv.detectnet_v2.label_filter.bbox_dimensions_label_filter import (
    BboxDimensionsLabelFilter
)
from nvidia_tao_tf1.cv.detectnet_v2.label_filter.source_class_label_filter import (
    SourceClassLabelFilter
)


ONE_OF_KEY = "label_filter_params"

FILTER_MAPPING = {
    "bbox_dimensions_label_filter": BboxDimensionsLabelFilter,
    "bbox_crop_label_filter": BboxCropLabelFilter,
    "source_class_label_filter": SourceClassLabelFilter,
}


def build_label_filter(label_filter_config):
    """Build the necessary filter.

    Args:
        label_filter_config: nvidia_tao_tf1.cv.detectnet_v2.proto.LabelFilter proto message.

    Returns:
        label_filter: BaseLabelFilter or child class.
    """
    # First, determine what kind of label filter the config pertains to.
    one_of_name = label_filter_config.WhichOneof(ONE_OF_KEY)
    # Then, extract the required kwargs.
    label_filter_kwargs = {
        # Here you notice proto_field.name has to be the same as the kwarg for the __init__
        #  of the <XYZ>LabelFilter. Note that .ListFields() only lists fields in the message
        #  which are not empty.
        proto_field.name: arg_value for proto_field, arg_value in
        getattr(label_filter_config, one_of_name).ListFields()
    }

    # Return appropriate class and kwargs.
    return FILTER_MAPPING[one_of_name](**label_filter_kwargs)
