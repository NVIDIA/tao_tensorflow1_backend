# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.

"""PostProcessingConfig class that holds postprocessing parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from nvidia_tao_tf1.cv.detectnet_v2.postprocessor.clustering_config import build_clustering_config
from nvidia_tao_tf1.cv.detectnet_v2.postprocessor.clustering_config import build_clustering_proto
from nvidia_tao_tf1.cv.detectnet_v2.postprocessor.confidence_config import build_confidence_config
from nvidia_tao_tf1.cv.detectnet_v2.postprocessor.confidence_config import build_confidence_proto
from nvidia_tao_tf1.cv.detectnet_v2.proto.postprocessing_config_pb2 import PostProcessingConfig as\
    PostProcessingProto


def build_postprocessing_config(postprocessing_proto):
    """Build PostProcessingConfig from a proto.

    Args:
        postprocessing_proto: proto.postprocessing_config proto message.

    Returns:
        configs: A dict of PostProcessingConfig instances indexed by target class name.
    """
    configs = {}
    for class_name, config in six.iteritems(postprocessing_proto.target_class_config):
        clustering_config = build_clustering_config(config.clustering_config)
        confidence_config = build_confidence_config(config.confidence_config)
        configs[class_name] = PostProcessingConfig(clustering_config, confidence_config)
    return configs


class PostProcessingConfig(object):
    """Hold the post-processing parameters for one class."""

    def __init__(self, clustering_config, confidence_config):
        """Constructor.

        Args:
            clustering_config (ClusteringConfig object): Built clustering configuration object.
            confidence_config (ConfidenceConfig object): Built confidence configuration object.
        """
        self.clustering_config = clustering_config
        self.confidence_config = confidence_config


def build_postprocessing_proto(postprocessing_config):
    """Build proto from a PostProcessingConfig dictionary.

    Args:
        postprocessing_config: A dict of PostProcessingConfig instances indexed by target class
            name.

    Returns:
        postprocessing_proto: proto.postprocessing_config proto message.
    """
    proto = PostProcessingProto()

    for target_class_name, target_class in six.iteritems(postprocessing_config):
        proto.target_class_config[target_class_name].clustering_config.CopyFrom(
            build_clustering_proto(target_class.clustering_config))
        proto.target_class_config[target_class_name].confidence_config.CopyFrom(
            build_confidence_proto(target_class.confidence_config))

    return proto
