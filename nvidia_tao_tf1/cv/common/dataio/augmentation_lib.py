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

"""TLT augmentation library."""

import cv2
import numpy as np


def aug_hsv(img, h=0.1, s=1.5, v=1.5, depth=8):
    """Apply HSV augmentation.

    Args:
        img: RGB image in numpy array
        h (float): Change hue at most h * 180
        s, v (float): change sv at most s, v, 1/s, 1/v times
        depth(int): Number of bits per pixel per channel of the image.
    Returns:
        aug_img: img after augmentation
    """

    def rand_inv(x):
        return x if np.random.rand() < 0.5 else 1.0 / x

    sv_mul = np.random.rand(2) * np.array([s - 1.0, v - 1.0]) + 1.0
    sv_mul = np.array(list(map(rand_inv, sv_mul))).reshape(1, 1, 2)
    if depth not in [8, 16]:
        raise ValueError(
            f"Unsupported image depth: {depth}, should be 8 or 16."
        )
    hsv = cv2.cvtColor(
        np.clip(img, 0, 2 ** depth - 1).astype(np.float32),
        cv2.COLOR_RGB2HSV
    )
    hsv[..., 1:] *= sv_mul
    hsv[..., 0] += (np.random.rand() * 2.0 - 1.0) * h * 180
    hsv = np.round(hsv).astype(np.int)
    hsv[..., 0] %= 180
    hsv[..., 1:] = np.clip(hsv[..., 1:], 0, (2.**depth - 1))
    return cv2.cvtColor(hsv.astype(np.float32), cv2.COLOR_HSV2RGB)


def aug_flip(img, boxes, ftype=0):
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


def aug_jitter(img, boxes, jitter=0.3, resize_ar=None):
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
    dl = int(round(dl))
    dt = int(round(dt))
    new_height = int(round(new_height))
    new_width = int(round(new_width))

    joint_l_on_img = max(dl, 0)
    joint_t_on_img = max(dt, 0)
    joint_r_on_img = min(new_width + dl, w)
    joint_b_on_img = min(new_height + dt, h)

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


def aug_letterbox_resize(img, boxes, resize_shape=(512, 512)):
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

    new_img = np.zeros((resize_shape[1], resize_shape[0], 3), dtype=np.float)
    new_img += np.mean(img, axis=(0, 1), keepdims=True)
    h, w, _ = img.shape
    ratio = min(float(resize_shape[1]) / h, float(resize_shape[0]) / w)
    new_h = int(round(ratio * h))
    new_w = int(round(ratio * w))
    l_shift = (resize_shape[0] - new_w) // 2
    t_shift = (resize_shape[1] - new_h) // 2
    img = cv2.resize(img, (new_w, new_h), cv2.INTER_LINEAR)
    new_img[t_shift: t_shift+new_h, l_shift: l_shift+new_w] = img.astype(np.float)

    xmin = (boxes[:, 0] * new_w + l_shift) / float(resize_shape[0])
    xmax = (boxes[:, 2] * new_w + l_shift) / float(resize_shape[0])
    ymin = (boxes[:, 1] * new_h + t_shift) / float(resize_shape[1])
    ymax = (boxes[:, 3] * new_h + t_shift) / float(resize_shape[1])

    return new_img, np.stack([xmin, ymin, xmax, ymax], axis=-1)


def aug_random_crop(img, boxes, crop_ar, min_box_ratio):
    """Apply random crop according to crop_ar.

    Args:
        img: RGB image in numpy array
        boxes: (N, 4) numpy arrays (xmin, ymin, xmax, ymax) containing bboxes. {x,y}{min,max} is
            in [0, 1] range.
        crop_ar: output aspect ratio
        min_box_ratio: the minimum ratio the crop bbox will be compared to original image
    Returns:
        aug_img: img after flip
        aug_boxes: boxes after flip
    """
    h, w, _ = img.shape

    # let's decide crop box size first
    crop_ratio = np.random.rand() * (1.0 - min_box_ratio) + min_box_ratio
    if w / float(h) > crop_ar:
        crop_h = int(round(h * crop_ratio))
        crop_w = int(round(crop_h * crop_ar))
    else:
        crop_w = int(round(w * crop_ratio))
        crop_h = int(round(crop_w / crop_ar))

    # get crop box location
    t_shift = np.random.randint(h - crop_h + 1)
    l_shift = np.random.randint(w - crop_w + 1)

    new_img = img[t_shift: t_shift+crop_h, l_shift: l_shift+crop_w].astype(np.float)

    xmin = (boxes[:, 0] * w - l_shift) / float(crop_w)
    xmax = (boxes[:, 2] * w - l_shift) / float(crop_w)
    ymin = (boxes[:, 1] * h - t_shift) / float(crop_h)
    ymax = (boxes[:, 3] * h - t_shift) / float(crop_h)

    return new_img, np.stack([xmin, ymin, xmax, ymax], axis=-1)
