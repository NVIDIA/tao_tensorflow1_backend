"""Tensor related utility functions."""
import tensorflow as tf


def get_init_ops():
    """Return all ops required for initialization."""
    return tf.group(tf.local_variables_initializer(),
                    tf.tables_initializer(),
                    *tf.get_collection('iterator_init'))
