"""Generate image shape tensors for multi-scale training."""
import numpy as np
import tensorflow as tf


def global_var_with_init(init_value):
    """global variable with initialization."""
    with tf.variable_scope("global_step", reuse=tf.AUTO_REUSE):
        v = tf.get_variable(
            "global_step_var",
            trainable=False,
            dtype=tf.int32,
            initializer=init_value
        )
    return v


def gen_random_shape_tensors(
    T,
    h_min,
    h_max,
    w_min,
    w_max
):
    """Generate random tensors for multi-scale training."""
    # make sure it is the output shape is a multiple of 32 to
    # align feature map shape for Upsample2D and Concatenate
    divider = 32
    h_min = h_min / divider
    h_max = h_max / divider
    w_min = w_min / divider
    w_max = w_max / divider
    # random size: uniform distribution in [size_min, size_max]
    rand_h = tf.cast(
        h_min + tf.random.uniform([]) * (h_max - h_min),
        tf.int32
    )
    rand_w = tf.cast(
        w_min + tf.random.uniform([]) * (w_max - w_min),
        tf.int32
    )
    # moving sum to repeat the size T times
    h_buffer = tf.Variable(
        np.zeros((T,), dtype=np.int32),
        trainable=False,
        dtype=tf.int32
    )
    w_buffer = tf.Variable(
        np.zeros((T,), dtype=np.int32),
        trainable=False,
        dtype=tf.int32
    )
    # global step
    global_step = global_var_with_init(-1)
    assign_gstep = tf.assign(global_step, global_step + 1)
    with tf.control_dependencies([assign_gstep]):
        # upsampled random size
        rand_h = tf.cond(
            tf.equal(tf.math.floormod(global_step, T), 0),
            true_fn=lambda: rand_h,
            false_fn=lambda: tf.zeros([], dtype=tf.int32)
        )
        rand_w = tf.cond(
            tf.equal(tf.math.floormod(global_step, T), 0),
            true_fn=lambda: rand_w,
            false_fn=lambda: tf.zeros([], dtype=tf.int32)
        )
        h_buffer_updated = tf.concat(
            [h_buffer[1:], [rand_h]],
            axis=-1
        )
        w_buffer_updated = tf.concat(
            [w_buffer[1:], [rand_w]],
            axis=-1
        )
        assign_h_buffer = tf.assign(h_buffer, h_buffer_updated)
        assign_w_buffer = tf.assign(w_buffer, w_buffer_updated)
        with tf.control_dependencies([assign_h_buffer, assign_w_buffer]):
            repeated_rand_hsize = tf.reduce_sum(h_buffer, axis=-1)
            repeated_rand_wsize = tf.reduce_sum(w_buffer, axis=-1)
            rh = repeated_rand_hsize * divider
            rw = repeated_rand_wsize * divider
            return tf.reshape(rh, []), tf.reshape(rw, [])
