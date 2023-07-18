"""dataset utils for unit test."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


def int64_feature(value):
    """int64_feature."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    """int64_list_feature."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    """bytes_feature."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    """bytes_list_feature."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    """float_feature."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    """float_list_feature."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
