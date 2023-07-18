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

"""YOLO v3 data augmentations."""

import cv2
import numpy as np
import tensorflow as tf


def aug_hsv_api(img, h=0.1, s=1.5, v=1.5):
    """Apply HSV augmentation using tf.image.

    Args:
        img: HWC RGB image
        h (float): Change hue at most h * 180
        s, v (float): change sv at most s, v, 1/s, 1/v times
    Returns:
        aug_img: HWC RGB img after augmentation
    """
    img = tf.image.random_hue(img, h/2)
    img = tf.image.random_saturation(img, 1.0/s, s)
    img = tf.image.random_brightness(img, v)
    return img


def aug_hsv(img, h=0.1, s=1.5, v=1.5, depth=8):
    """Apply HSV augmentation.

    Args:
        img: HWC RGB image
        h (float): Change hue at most h * 180
        s, v (float): change sv at most s, v, 1/s, 1/v times
        depth(int): Number of bits per pixel per channel of the image
    Returns:
        aug_img: HWC RGB img after augmentation
    """

    def rand_inv(x):
        return tf.cond(
            tf.random.uniform([]) < 0.5,
            true_fn=lambda: x,
            false_fn=lambda: 1.0 / x
        )

    max_limit = (2 ** depth - 1)
    sv_mul = tf.random.uniform([2]) * tf.constant([s - 1.0, v - 1.0]) + 1.0
    sv_mul = tf.reshape(tf.map_fn(rand_inv, sv_mul), (1, 1, 2))
    hsv = tf.image.rgb_to_hsv(img / max_limit) * tf.constant([180., max_limit, max_limit])
    hsv = tf.concat(
        [
            hsv[..., 0:1] + (tf.random.uniform([]) * 2. - 1.) * h * 180.,
            hsv[..., 1:] * sv_mul
        ],
        axis=-1
    )
    hsv = tf.cast(tf.math.round(hsv), tf.int32)
    hsv = tf.concat(
        [tf.math.floormod(hsv[..., 0:1], 180),
         tf.clip_by_value(hsv[..., 1:], 0, max_limit)],
        axis=-1
    )
    hsv = tf.cast(hsv, tf.float32)
    return tf.image.hsv_to_rgb(hsv * tf.constant([1/180., 1/max_limit, 1/max_limit])) * max_limit


def random_hflip(image, prob, seed):
    """Random horizontal flip.

    Args:
        image(Tensor): The input image in (H, W, C).
        prob(float): The probability for horizontal flip.
        seed(int): The random seed.

    Returns:
        out_images(Tensor): The output image.
        flipped(boolean Tensor): A boolean scalar tensor to indicate whether flip is
        applied or not. This can be used to manipulate the labels accordingly.
    """

    val = tf.random.uniform([], maxval=1.0, seed=seed)
    is_flipped = tf.less_equal(val, prob)
    # flip and to CHW
    flipped_image = tf.image.flip_left_right(image)
    out_images = tf.cond(
        is_flipped,
        true_fn=lambda: flipped_image,
        false_fn=lambda: image
    )
    return out_images, is_flipped


def hflip_bboxes(boxes, flipped, xmax=1.0):
    """Flip the bboxes horizontally.

    Args:
        boxes(Tensor): (N, 4) shaped bboxes in [x1, y1, x2, y2] normalized coordinates.

    Returns:
        out_boxes(Tensor): horizontally flipped boxes.
    """

    # x1 becomes new x2, while x2 becomes new x1
    # (N,)
    x1_new = xmax - boxes[:, 2]
    x2_new = xmax - boxes[:, 0]
    # (N, 4)
    flipped_boxes = tf.stack([x1_new, boxes[:, 1], x2_new, boxes[:, 3]], axis=1)
    out_boxes = tf.cond(
        flipped,
        true_fn=lambda: flipped_boxes,
        false_fn=lambda: boxes
    )
    return out_boxes


def aug_hflip(img, gt_labels, prob=0.5, xmax=1.0):
    """random horizontal flip of image and bboxes."""
    img, flipped = random_hflip(img, prob, 42)
    # x1, y1, x2, y2
    flipped_boxes = hflip_bboxes(gt_labels, flipped, xmax=xmax)
    return img, flipped_boxes


def _aug_flip_np(img, boxes, ftype=1):
    """Apply flip.

    Args:
        img: RGB image in numpy array
        boxes: (N, 4) numpy arrays (xmin, ymin, xmax, ymax) containing bboxes. {x,y}{min,max} is
            in [0, 1] range.
        ftype (0 or 1): 0: vertical flip. 1: horizontal flip
    Returns:
        aug_img: img after flip
        aug_boxes: boxes after flip
    """

    if ftype == 0:
        ymin = 1.0 - boxes[:, 3]
        ymax = 1.0 - boxes[:, 1]
        xmin = boxes[:, 0]
        xmax = boxes[:, 2]
    elif ftype == 1:
        ymin = boxes[:, 1]
        ymax = boxes[:, 3]
        xmin = 1.0 - boxes[:, 2]
        xmax = 1.0 - boxes[:, 0]
    else:
        raise ValueError("Use ftype 0 for vertical flip and 1 for horizontal flip.")

    return cv2.flip(img, ftype), np.stack([xmin, ymin, xmax, ymax], axis=-1)


def aug_flip_np(img, boxes):
    """aug flip np."""
    if np.random.rand() < 0.5:
        img = np.clip(img, 0., 255.).astype(np.uint8)
        img, boxes = _aug_flip_np(img, boxes)
        img = img.astype(np.float32)
    return img, boxes


def _update_dx_wide(w, h, ratio, dl, dr, dt, db):
    # first try to decrease new_width
    ar_w = h * ratio
    dw = w - ar_w
    # narrow from two sides
    l_shift = -tf.minimum(dl, 0.)
    r_shift = -tf.minimum(dr, 0.)
    lr_shift = tf.minimum(tf.minimum(l_shift, r_shift), dw / 2.0)
    dl += lr_shift
    dr += lr_shift
    dw -= 2. * lr_shift
    l_shift = tf.cond(
        tf.logical_and(dl < 0., 0. < dw),
        true_fn=lambda: tf.minimum(dw, -dl),
        false_fn=lambda: tf.constant(0.)
    )
    dl += l_shift
    dw -= l_shift
    r_shift = tf.cond(
        tf.logical_and(dr < 0., 0. < dw),
        true_fn=lambda: tf.minimum(dw, -dr),
        false_fn=lambda: tf.constant(0.)
    )
    dr += r_shift
    dw -= r_shift
    # if doesn't work, increase new_height
    dh = tf.cond(
        dw > 0.,
        true_fn=lambda: dw / ratio,
        false_fn=lambda: tf.constant(0., dtype=tf.float32)
    )
    dt = tf.cond(
        dw > 0.,
        true_fn=lambda: dt - dh / 2.,
        false_fn=lambda: dt
    )
    db = tf.cond(
        dw > 0.,
        true_fn=lambda: db - dh / 2.,
        false_fn=lambda: db
    )
    return dl, dr, dt, db


def _update_dx_tall(w, h, ratio, dl, dr, dt, db):
    # first try to decrease new_height
    ar_h = w / ratio
    dh = h - ar_h
    # narrow from two sides
    t_shift = -tf.minimum(dt, 0.)
    b_shift = -tf.minimum(db, 0.)
    tb_shift = tf.minimum(tf.minimum(t_shift, b_shift), dh / 2.0)
    dt += tb_shift
    db += tb_shift
    dh -= 2 * tb_shift
    t_shift = tf.cond(
        tf.logical_and(dt < 0., 0. < dh),
        true_fn=lambda: tf.minimum(dh, -dt),
        false_fn=lambda: tf.constant(0.)
    )
    dt += t_shift
    dh -= t_shift
    b_shift = tf.cond(
        tf.logical_and(db < 0., 0. < dh),
        true_fn=lambda: tf.minimum(db, -dt),
        false_fn=lambda: tf.constant(0.)
    )
    db += b_shift
    dh -= b_shift
    # If doesn't work, increase new_width
    dw = tf.cond(
        dh > 0.,
        true_fn=lambda: dh * ratio,
        false_fn=lambda: tf.constant(0., dtype=tf.float32)
    )
    dl = tf.cond(
        dh > 0.,
        true_fn=lambda: dl - dw / 2.,
        false_fn=lambda: dl
    )
    dr = tf.cond(
        dh > 0.,
        true_fn=lambda: dr - dw / 2.,
        false_fn=lambda: dr
    )
    return dl, dr, dt, db


def _update_dx_combined(w, h, ratio, dl, dr, dt, db):
    dl, dr, dt, db = tf.cond(
        w / h > ratio,
        true_fn=lambda: _update_dx_wide(w, h, ratio, dl, dr, dt, db),
        false_fn=lambda: _update_dx_tall(w, h, ratio, dl, dr, dt, db)
    )
    return dl, dr, dt, db


def aug_jitter_single_image(img, boxes, jitter=0.3, resize_ar=None):
    """Apply YOLO style jitter.

    See https://stackoverflow.com/questions/55038726

    Args:
        img: HWC RGB image, 0-255
        boxes: (N, 4) numpy arrays (xmin, ymin, xmax, ymax) containing bboxes. {x,y}{min,max} is
            in [0, 1] range.
        jitter (0, 1): jitter value
        resize_ar (float): network input width / height. Jitter will try to mimic this
    Returns:
        aug_img: img after jitter
        aug_boxes: boxes after jitter
    """

    # -jitter ~ jitter rand
    jt = tf.minimum((tf.random.uniform([4]) - 0.5) * 2 * jitter, 0.8)
    dl, dt, dr, db = tf.unstack(jt, axis=0)
    # make sure the result image is not too small
    cond1 = dl + dr > 0.8
    dr = tf.cond(
        cond1,
        true_fn=lambda: tf.minimum(dr, 0.4),
        false_fn=lambda: dr
    )
    dl = tf.cond(
        cond1,
        true_fn=lambda: tf.minimum(dl, 0.4),
        false_fn=lambda: dl
    )
    cond2 = dt + db > 0.8
    dt = tf.cond(
        cond2,
        true_fn=lambda: tf.minimum(dt, 0.4),
        false_fn=lambda: dt
    )
    db = tf.cond(
        cond2,
        true_fn=lambda: tf.minimum(db, 0.4),
        false_fn=lambda: db
    )
    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)
    dl *= w
    dr *= w
    dt *= h
    db *= h
    new_width = w - dl - dr
    new_height = h - dt - db
    dl, dr, dt, db = _update_dx_combined(
        w, h, resize_ar, dl, dr, dt, db
    )
    new_width = w - dl - dr
    new_height = h - dt - db
    # new image left top corner [dl, dt], height / width [new_height, new_width]
    # old image left top corner [0, 0], height/width [h, w]
    dl = tf.cast(tf.math.round(dl), tf.int32)
    dt = tf.cast(tf.math.round(dt), tf.int32)
    new_height = tf.cast(tf.math.round(new_height), tf.int32)
    new_width = tf.cast(tf.math.round(new_width), tf.int32)

    joint_l_on_img = tf.maximum(dl, 0)
    joint_t_on_img = tf.maximum(dt, 0)
    joint_r_on_img = tf.minimum(new_width + dl, tf.cast(w, tf.int32))
    joint_b_on_img = tf.minimum(new_height + dt, tf.cast(h, tf.int32))

    h_idx = tf.range(joint_t_on_img - dt, joint_b_on_img - dt, delta=1)
    w_idx = tf.range(joint_l_on_img - dl, joint_r_on_img - dl, delta=1)
    h_idx, w_idx = tf.meshgrid(h_idx, w_idx)
    h_idx = tf.reshape(tf.transpose(h_idx), (-1,))
    w_idx = tf.reshape(tf.transpose(w_idx), (-1,))
    # (k, 2)
    indices = tf.stack([h_idx, w_idx], axis=1)
    # (k, 3)
    updates = tf.reshape(
        img[joint_t_on_img:joint_b_on_img, joint_l_on_img:joint_r_on_img, :],
        (-1, 3)
    )
    # (H, W, 3)
    shape = tf.stack([new_height, new_width, 3], axis=0)
    new_img = tf.scatter_nd(
        indices,
        updates,
        shape
    )
    # replace all other pixels with mean pixels
    mean_img = tf.reduce_mean(img, axis=(0, 1), keepdims=True)
    new_img += mean_img
    # (k, 3)
    neg_mean = -1 * tf.broadcast_to(
        tf.reshape(mean_img, (1, 3)),
        tf.stack([tf.shape(indices)[0], 3])
    )
    new_img_delta = tf.scatter_nd(
        indices,
        neg_mean,
        tf.shape(new_img)
    )
    new_img = new_img + new_img_delta

    xmin = (boxes[:, 0] * w - tf.cast(dl, tf.float32)) / tf.cast(new_width, tf.float32)
    xmax = (boxes[:, 2] * w - tf.cast(dl, tf.float32)) / tf.cast(new_width, tf.float32)
    ymin = (boxes[:, 1] * h - tf.cast(dt, tf.float32)) / tf.cast(new_height, tf.float32)
    ymax = (boxes[:, 3] * h - tf.cast(dt, tf.float32)) / tf.cast(new_height, tf.float32)
    augmented_boxes = tf.stack([xmin, ymin, xmax, ymax], axis=-1)
    augmented_boxes = tf.clip_by_value(augmented_boxes, 0., 1.)
    return new_img, augmented_boxes


def _aug_jitter(img, boxes, resize_ar, jitter):
    """Apply YOLO style jitter.

    See https://stackoverflow.com/questions/55038726

    Args:
        img: RGB image in numpy array
        boxes: (N, 4) numpy arrays (xmin, ymin, xmax, ymax) containing bboxes. {x,y}{min,max} is
            in [0, 1] range.
        jitter (0, 1): jitter value
        resize_ar (float): network input width / height. Jitter will try to mimic this
    Returns:
        aug_img: img after jitter
        aug_boxes: boxes after jitter
    """

    # -jitter ~ jitter rand
    img = np.array(img)
    boxes = np.array(boxes)
    dl, dt, dr, db = np.minimum((np.random.rand(4) - 0.5) * 2 * jitter, 0.8)

    # make sure the result image is not too small
    if dl + dr > 0.8:
        dr = min(dr, 0.4)
        dl = min(dl, 0.4)

    if dt + db > 0.8:
        dt = min(dt, 0.4)
        db = min(db, 0.4)

    h, w, _ = img.shape
    dl *= w
    dr *= w
    dt *= h
    db *= h

    new_width = w - dl - dr
    new_height = h - dt - db

    if resize_ar is not None:
        if w / float(h) > resize_ar:
            # first try to decrease new_width
            ar_w = h * resize_ar
            dw = w - ar_w

            # narrow from two sides
            l_shift = -min(dl, 0)
            r_shift = -min(dr, 0)
            lr_shift = min(l_shift, r_shift, dw / 2.0)

            dl += lr_shift
            dr += lr_shift
            dw -= 2 * lr_shift

            if dl < 0 < dw:
                l_shift = min(dw, -dl)
                dl += l_shift
                dw -= l_shift

            if dr < 0 < dw:
                r_shift = min(dw, -dr)
                dr += r_shift
                dw -= r_shift

            # if doesn't work, increase new_height
            if dw > 0:
                dh = dw / resize_ar
                dt -= dh / 2.0
                db -= dh / 2.0
        else:
            # first try to decrease new_height
            ar_h = w / resize_ar
            dh = h - ar_h

            # narrow from two sides
            t_shift = -min(dt, 0)
            b_shift = -min(db, 0)
            tb_shift = min(t_shift, b_shift, dh / 2.0)

            dt += tb_shift
            db += tb_shift
            dh -= 2 * tb_shift

            if dt < 0 < dh:
                t_shift = min(dh, -dt)
                dt += t_shift
                dh -= t_shift
            if db < 0 < dh:
                b_shift = min(db, -dt)
                db += b_shift
                dh -= b_shift

            # If doesn't work, increase new_width
            if dh > 0:
                dw = dh * resize_ar
                dl -= dw / 2.0
                dr -= dw / 2.0

        new_width = w - dl - dr
        new_height = h - dt - db

    # new image left top corner [dl, dt], height / width [new_height, new_width]
    # old image left top corner [0, 0], height/width [h, w]
    dl = np.round(dl).astype(np.int32)
    dt = np.round(dt).astype(np.int32)
    new_height = np.round(new_height).astype(np.int32)
    new_width = np.round(new_width).astype(np.int32)

    joint_l_on_img = np.maximum(dl, 0)
    joint_t_on_img = np.maximum(dt, 0)
    joint_r_on_img = np.minimum(new_width + dl, w)
    joint_b_on_img = np.minimum(new_height + dt, h)

    new_img = np.zeros((new_height, new_width, 3), dtype=np.float)
    new_img += np.mean(img, axis=(0, 1), keepdims=True)
    new_img[joint_t_on_img - dt:joint_b_on_img - dt,
            joint_l_on_img - dl:joint_r_on_img - dl, :] = \
        img[joint_t_on_img:joint_b_on_img, joint_l_on_img:joint_r_on_img, :].astype(np.float)

    xmin = (boxes[:, 0] * w - dl) / new_width
    xmax = (boxes[:, 2] * w - dl) / new_width
    ymin = (boxes[:, 1] * h - dt) / new_height
    ymax = (boxes[:, 3] * h - dt) / new_height

    return new_img, np.stack([xmin, ymin, xmax, ymax], axis=-1)


def aug_jitter(img, boxes, resize_ar):
    """aug jitter in numpy."""
    img, boxes = _aug_jitter(img, boxes, resize_ar, 0.3)
    boxes = np.clip(boxes, 0., 1.)
    return img, boxes


def aug_letterbox_resize(img, boxes, resize_shape):
    """Apply letter box. resize image to resize_shape, not changing aspect ratio.

    Args:
        img: RGB image in numpy array
        boxes: (N, 4) numpy arrays (xmin, ymin, xmax, ymax) containing bboxes. {x,y}{min,max} is
            in [0, 1] range.
        resize_shape (int, int): (w, h) of new image
    Returns:
        aug_img: img after resize
        aug_boxes: boxes after resize
    """
    resize_shape_f = tf.cast(resize_shape, tf.float32)
    new_img = tf.zeros((resize_shape[1], resize_shape[0], 3), dtype=tf.float32)
    mean_img = tf.reduce_mean(img, axis=(0, 1), keepdims=True)
    new_img += mean_img
    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)
    ratio = tf.reduce_min([resize_shape_f[1] / h, resize_shape_f[0] / w])
    new_h = tf.cast(tf.math.round(ratio * h), tf.int32)
    new_w = tf.cast(tf.math.round(ratio * w), tf.int32)
    l_shift = (resize_shape[0] - new_w) // 2
    t_shift = (resize_shape[1] - new_h) // 2
    img = tf.image.resize_images(img, [new_h, new_w])
    # copy-paste img to new_img
    h_idx = tf.range(t_shift, t_shift+new_h, delta=1)
    w_idx = tf.range(l_shift, l_shift+new_w, delta=1)
    h_idx, w_idx = tf.meshgrid(h_idx, w_idx)
    h_idx = tf.reshape(tf.transpose(h_idx), (-1,))
    w_idx = tf.reshape(tf.transpose(w_idx), (-1,))
    # (k, 2)
    indices = tf.stack([h_idx, w_idx], axis=1)
    # (k, 3)
    updates = tf.reshape(img, (-1, 3))
    new_img_scattered = tf.scatter_nd(
        indices,
        updates,
        tf.shape(new_img)
    )
    new_img += new_img_scattered
    neg_mean = -1 * tf.broadcast_to(
        tf.reshape(mean_img, (1, 3)),
        tf.stack([tf.shape(indices)[0], 3])
    )
    new_img_delta = tf.scatter_nd(
        indices,
        neg_mean,
        tf.shape(new_img)
    )
    new_img += new_img_delta
    new_w_f = tf.cast(new_w, tf.float32)
    new_h_f = tf.cast(new_h, tf.float32)
    l_shift_f = tf.cast(l_shift, tf.float32)
    t_shift_f = tf.cast(t_shift, tf.float32)
    xmin = (boxes[:, 0] * new_w_f + l_shift_f) / resize_shape_f[0]
    xmax = (boxes[:, 2] * new_w_f + l_shift_f) / resize_shape_f[0]
    ymin = (boxes[:, 1] * new_h_f + t_shift_f) / resize_shape_f[1]
    ymax = (boxes[:, 3] * new_h_f + t_shift_f) / resize_shape_f[1]
    return new_img, tf.stack([xmin, ymin, xmax, ymax], axis=-1)


def apply_letterbox_resize(image, gt_labels, target_shape):
    """apply letterbox resize."""
    return aug_letterbox_resize(image, gt_labels, target_shape)


def inner_augmentations(image, gt_labels, ratio, xmax, augmentation_config):
    """yolo v3 augmentations inside tf.data.

    Args:
        image: NCHW RGB images.
        gt_labels(list): list of groundtruth labels for each image: (#gt, 6).
        augmentation_config: YOLO v3 augmentation config.

    Returns:
        augmented images and gt_labels.
    """
    # augmentation pipelines, applied on HWC images
    image_depth = int(augmentation_config.output_depth) or 8
    if image_depth == 8:
        image = aug_hsv_api(
            image,
            augmentation_config.hue,
            augmentation_config.saturation,
            augmentation_config.exposure,
        )
    else:
        image = aug_hsv(
            image,
            augmentation_config.hue,
            augmentation_config.saturation,
            augmentation_config.exposure,
            image_depth
        )
    image, gt_labels = aug_hflip(
        image,
        gt_labels,
        prob=augmentation_config.horizontal_flip,
        xmax=xmax
    )
    return image, gt_labels


def outer_augmentations(image, gt_labels, ratio, augmentation_config):
    """yolo v3 augmentations outside of tf.data.

    Args:
        image: NCHW RGB images.
        gt_labels(list): list of groundtruth labels for each image: (#gt, 6).
        augmentation_config: YOLO v3 augmentation config.

    Returns:
        augmented images and gt_labels.
    """
    # augmentation pipelines, applied on HWC images
    image, gt_labels = aug_jitter_single_image(
        image,
        gt_labels,
        jitter=augmentation_config.jitter,
        resize_ar=ratio
    )
    return image, gt_labels
