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

"""BpNet Inference utils."""

from enum import Enum
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import scipy.stats as st


class KeepAspectRatioMode(str, Enum):
    """Enum class containing the different modes to keep aspect ratio."""

    PAD_IMAGE_INPUT = "pad_image_input"
    ADJUST_NETWORK_INPUT = "adjust_network_input"


def pad_image_input(input_shape, raw_frame):
    """Pad raw input to maintain aspect ratio for inference.

    Args:
        input_shape (tuple): desired (height, width) of the
            network input
        raw_frame (np.ndarray): Unprocessed frame in HWC format.

    Returns:
        res (np.ndarray): Padded frame.
        offset (list): (x, y) offsets used during padding
    """
    image_height = raw_frame.shape[0]
    image_width = raw_frame.shape[1]

    # Offset for width, height.
    offset = [0, 0]
    desired_aspect_ratio = input_shape[1] / input_shape[0]

    if image_width / image_height == desired_aspect_ratio:
        return raw_frame, offset

    # Need to pad height.
    if image_width / image_height > desired_aspect_ratio:
        pad_length = int(
            (image_width / desired_aspect_ratio - image_height)
        )
        pad_length_half = int(pad_length / 2.0)
        offset[1] = pad_length_half
        # Border order: top, bottom, left, right
        res = cv2.copyMakeBorder(
            raw_frame,
            pad_length_half,
            pad_length - pad_length_half,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(128, 128, 128),
        )

    # Need to pad width.
    else:
        pad_length = int(
            (image_height * desired_aspect_ratio - image_width)
        )
        pad_length_half = int(pad_length / 2.0)
        offset[0] = pad_length_half
        # Border order: top, bottom, left, right
        res = cv2.copyMakeBorder(
            raw_frame,
            0,
            0,
            pad_length_half,
            pad_length - pad_length_half,
            cv2.BORDER_CONSTANT,
            value=(128, 128, 128),
        )
    return res, offset


def adjust_network_input(input_shape, image_shape):
    """Pad raw input to maintain aspect ratio for inference.

    Args:
        input_shape (tuple): desired (height, width) of the
            network input
        image_shape (tuple): (height, width) of the image

    Returns:
        scale (tuple): tuple containing the scaling factors.
        offset (list): list containing the x and y offset values
    """

    image_height = image_shape[0]
    image_width = image_shape[1]

    offset = [0, 0]
    desired_aspect_ratio = input_shape[1] / input_shape[0]

    # If the image aspect ratio is greater than desiered aspect ratio
    # fix the scale as the ratio of the heights, else fix it as ratio
    # of the widths. The other side gets adjusted by the same amount.
    if image_width / image_height > desired_aspect_ratio:
        scale = (input_shape[0] / image_height, input_shape[0] / image_height)
    else:
        scale = (input_shape[1] / image_width, input_shape[1] / image_width)

    return scale, offset


def convert_color_format(image, input_color_format, desired_color_format):
    """Convert from one image color format, to another.

    Args:
        image (np.ndarray): input image
        input_color_format (str): color format of input
        desired_color_format (str): color format to convert to

    Returns:
        image (np.ndarray): procesed image
    """
    # Enforce BGR (currently, doesn't support other formats)
    assert (
        "B" in input_color_format
        and "G" in input_color_format
        and "R" in input_color_format
    ), "Color order must have B,G,R"

    if desired_color_format == "RGB":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def resize_image(image, input_shape, keep_aspect_ratio_mode=None):
    """Resize the input image based on the given mode.

    Args:
        image (np.ndarray): input image
        input_shape (tuple): (height, width) of the image
        keep_aspect_ratio_mode (str): determines how the image is
            resized. Choices include [`adjust_network_input`,
            `pad_image_input`, None]

    Returns:
        image (np.ndarray): procesed image
        scale (tuple): scale used to resize the image (fy, fx)
        offset (list): (x, y) offsets used during padding
    """
    image_shape = image.shape
    # Estimate the scale and offset needed
    # NOTE: scale is (fy, fx)
    # No need to retain aspect ratio
    if keep_aspect_ratio_mode is None:
        offset = [0, 0]
        scale = (
            input_shape[0] / image_shape[0],
            input_shape[1] / image_shape[1]
        )
    # Retain aspect ratio by padding the network input accordingly
    elif keep_aspect_ratio_mode == KeepAspectRatioMode.ADJUST_NETWORK_INPUT:
        scale, offset = adjust_network_input(input_shape, image_shape)
    # Retain aspect ratio by padding the image input accordingly
    elif keep_aspect_ratio_mode == KeepAspectRatioMode.PAD_IMAGE_INPUT:
        image, offset = pad_image_input(input_shape, image)
        padded_image_shape = image.shape
        scale = (
            input_shape[0] / padded_image_shape[0],
            input_shape[1] / padded_image_shape[1]
        )
    else:
        raise ValueError("keep aspect ratio mode: {} not supported. Please \
        choose in [`pad_image_input`, `adjust_network_input`, None]".format(
            keep_aspect_ratio_mode))
    # Resize image using the given scale
    image = cv2.resize(
        image,
        (0, 0),
        fx=scale[1],
        fy=scale[0],
        interpolation=cv2.INTER_CUBIC)

    return image, scale, offset


def normalize_image(image, scale, offset):
    """Normalize image.

    Args:
        image (np.ndarray): input image
        scale (list): normalization scale used in training
        offset (list): normalization offset used in training

    Returns:
        (np.ndarray): procesed image
    """
    return np.subtract(np.divide(image, scale), offset)


def preprocess(orig_image,
               input_shape,
               normalization_offset,
               normalization_scale,
               input_color_format="BGR",
               desired_color_format="RGB",
               keep_aspect_ratio_mode=None):
    """Preprocess image.

    TODO: Ideally should be using augmentation module
    Args:
        image (HWC): input image
        input_shape (tuple): (height, width) of the image
        keep_aspect_ratio_mode (str): determines how the image is
            resized. Choices include [`adjust_network_input`,
            `pad_image_input`, None]
        normalization_scale (list): normalization scale used in training
        normalization_offset (list): normalization offset used in training
        input_color_format (str): color format of input
        desired_color_format (str): color format to convert to

    Returns:
        preprocessed_image (np.ndarray): procesed image
        preprocess_params (dict): contains the params used for pre-processing
    """
    image = orig_image.copy()

    image, scale, offset = resize_image(
        image, input_shape, keep_aspect_ratio_mode)

    # Convert to desired color format
    # NOTE: currently supports only BGR as input
    image = convert_color_format(
        image,
        input_color_format,
        desired_color_format)

    # Normalize image
    preprocessed_image = normalize_image(
        image, normalization_scale, normalization_offset)
    # preprocessed_image = image
    preprocess_params = {
        'scale': scale,
        'offset': offset
    }
    return preprocessed_image, preprocess_params


def pad_bottom_right(image, stride, pad_value):
    """Pad image on the bottom right side.

    Args:
        image (HWC): input image
        stride (int): stride size of the model
        pad_value (tuple): pixel value to use for padded regions

    Returns:
        img_padded (np.ndarray): procesed image
        pad (dict): contains the padding values
    """
    h = image.shape[0]
    w = image.shape[1]

    # Pad ordering: [top, left, bottom, right]
    pad = 4 * [None]
    pad[0] = 0
    pad[1] = 0
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)

    img_padded = cv2.copyMakeBorder(image, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return img_padded, pad


def get_gaussian_kernel(kernlen, num_channels, sigma=3, dtype=np.float32):
    """Get gaussian kernel to use as weights.

    Args:
        kernlen (HWC): kernel size to use
        num_channels (int): number of channels to filter
        dtype (dtype): data type to use for the gaussian kernel
        sigma (float): sigma value to use for the gaussian kernel

    Returns:
        out_filter (np.ndarray): gaussian kernel of shape
            (kernlen, kernlen, num_channels, 1)
    """
    interval = (2 * sigma + 1.) / (kernlen)
    x = np.linspace(-sigma - interval / 2., sigma + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=dtype)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, num_channels, axis=2)
    return out_filter


def apply_gaussian_smoothing(input_array, kernel_size=5, sigma=3, backend="cv"):
    """Apply gaussian smoothing.

    Args:
        input_array (np.ndarray): input array to apply gaussian
            smoothing with shape (H, W, C)
        kernel_size (int): kernel size to use
        sigma (float): sigma value to use for the gaussian kernel

    Returns:
        output_array (np.ndarray): output after gaussian smoothing
            with same shape as `input_array`
    """
    output_array = np.zeros(shape=input_array.shape)
    num_channels = input_array.shape[-1]

    if backend == "cv":
        for channel_idx in range(0, num_channels):
            output_array[:, :, channel_idx] = cv2.GaussianBlur(
                input_array[:, :, channel_idx],
                ksize=(kernel_size, kernel_size),
                sigmaX=sigma,
                sigmaY=sigma
            )
    elif backend == "scipy":
        for channel_idx in range(0, num_channels):
            output_array[:, :, channel_idx] = gaussian_filter(
                input_array[:, :, channel_idx],
                sigma=sigma
            )
    else:
        raise ValueError("Unsupported backend for gaussian smoothing.")
    return output_array


def nms_np(input_array, threshold=0.0):
    """Apply non-max suppression.

    Args:
        input_array (np.ndarray): Input array to apply nms
            with shape (H, W, C)
        threshold (float): Optional thhreshold to suppress
            values below a threshold.

    Returns:
        output_array (np.ndarray): output after nms
            with same shape as `input_array` and retaining
            only the peaks.
    """
    shift_val = 1
    output_array = np.zeros(shape=input_array.shape)
    num_channels = input_array.shape[-1]
    zeros_arr = np.zeros(input_array.shape[:-1])
    for channel_idx in range(0, num_channels):
        center_arr = input_array[:, :, channel_idx]
        # init shifted array placeholders with zeros
        shift_left, shift_right, shift_up, shift_down = np.tile(zeros_arr, (4, 1, 1))
        # shift input down by shift value
        shift_down[shift_val:, :] = center_arr[:-shift_val, :]
        # shift input up by shift value
        shift_up[:-shift_val, :] = center_arr[shift_val:, :]
        # shift input to the right by shift value
        shift_right[:, shift_val:] = center_arr[:, :-shift_val]
        # shift input to the left by shift value
        shift_left[:, :-shift_val] = center_arr[:, shift_val:]
        # Check where pixels the center values are max in the given
        # local window size
        peaks_binary = np.logical_and.reduce(
            (center_arr >= shift_left,
             center_arr >= shift_right,
             center_arr >= shift_up,
             center_arr >= shift_down,
             center_arr > threshold))
        # Copy over the only the peaks to output array, rest are suppressed.
        output_array[:, :, channel_idx] = peaks_binary * center_arr

    return output_array


class Visualizer(object):
    """Visualizer class definitions."""

    # TODO: Move this to separate visualizer module
    def __init__(self, topology):
        """Init.

        Args:
            topology (np.ndarray): N x 4 array where N is the number of
                connections, and the columns are (start_paf_idx, end_paf_idx,
                start_conn_idx, end_conn_idx)
        """
        self.topology = topology

    def keypoints_viz(self, image, keypoints):
        """Function to visualize the given keypoints.

        Args:
            image (np.ndarray): Input image
            keypoints (list): List of lists containing keypoints per skeleton

        Returns:
            image (np.ndarray): image with result overlay
        """
        topology = self.topology

        peak_color = (255, 150, 0)
        edge_color = (254, 0, 190)
        stick_width = 2

        for i in range(topology.shape[0]):
            start_idx = topology[i][2]
            end_idx = topology[i][3]
            for n in range(len(keypoints)):
                start_joint = keypoints[n][start_idx]
                end_joint = keypoints[n][end_idx]
                if 0 in start_joint or 0 in end_joint:
                    continue
                cv2.circle(
                    image, (int(
                        start_joint[0]), int(
                        start_joint[1])), 4, peak_color, thickness=-1)
                cv2.circle(
                    image, (int(
                        end_joint[0]), int(
                        end_joint[1])), 4, peak_color, thickness=-1)
                cv2.line(
                    image, (int(
                        start_joint[0]), int(
                        start_joint[1])), (int(
                            end_joint[0]), int(
                            end_joint[1])), edge_color, thickness=stick_width)

        return image
