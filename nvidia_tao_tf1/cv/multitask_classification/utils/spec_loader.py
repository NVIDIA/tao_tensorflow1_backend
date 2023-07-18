# Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.

"""Load an experiment spec file to run MultiTaskNet training, evaluation, pruning."""

from google.protobuf.text_format import Merge as merge_text_proto

import nvidia_tao_tf1.cv.multitask_classification.proto.experiment_pb2 as experiment_pb2


def load_experiment_spec(spec_path=None):
    """Load experiment spec from a .txt file and return an experiment_pb2.Experiment object.

    Args:
        spec_path (str): location of a file containing the custom experiment spec proto.

    Returns:
        experiment_spec: protocol buffer instance of type experiment_pb2.Experiment.
    """
    experiment_spec = experiment_pb2.Experiment()
    merge_text_proto(open(spec_path, "r").read(), experiment_spec)
    return experiment_spec
