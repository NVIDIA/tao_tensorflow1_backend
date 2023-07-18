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

"""BpNet DataIO utils."""

from pycocotools import mask as mask_utils


def annotation_to_rle(segmentation, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.

        Reference: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py

        Args:
            segmentation: Can be polygons, uncompressed RLE, or RLE
            height (int): Height of the image
            width (int): Width of the image

        Returns:
            rle (list): Run length encoding
        """
        if type(segmentation) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = mask_utils.frPyObjects(segmentation, height, width)
            rle = mask_utils.merge(rles)
        elif type(segmentation['counts']) == list:
            # uncompressed RLE
            rle = mask_utils.frPyObjects(segmentation, height, width)
        else:
            # rle
            rle = segmentation
        return rle


def annotation_to_mask(segmentation, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.

    Reference: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py

    Args:
        segmentation: Can be polygons, uncompressed RLE, or RLE
        height (int): Height of the image
        width (int): Width of the image

    Returns:
        binary mask (np.ndarray): Binary mask generated using the given annotation
    """
    rle = annotation_to_rle(segmentation, height, width)
    binary_mask = mask_utils.decode(rle)
    return binary_mask
