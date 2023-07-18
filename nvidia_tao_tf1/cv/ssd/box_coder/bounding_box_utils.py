'''
Copyright (C) 2018 Pierluigi Ferrari.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import division
import numpy as np


def convert_coordinates(tensor, start_index, conversion, border_pixels='half'):
    '''
    Convert coordinates for axis-aligned 2D boxes between two coordinate formats.

    Creates a copy of `tensor`, i.e. does not operate in place. Currently there are
    three supported coordinate formats that can be converted from and to each other:
        1) (xmin, xmax, ymin, ymax) - the 'minmax' format
        2) (xmin, ymin, xmax, ymax) - the 'corners' format
        2) (cx, cy, w, h) - the 'centroids' format

    Arguments:
        tensor (array): A Numpy nD array containing the four consecutive coordinates
            to be converted somewhere in the last axis.
        start_index (int): The index of the first coordinate in the last axis of `tensor`.
        conversion (str, optional): The conversion direction. Can be 'minmax2centroids',
            'centroids2minmax', 'corners2centroids', 'centroids2corners', 'minmax2corners',
            or 'corners2minmax'.
        border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
            Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
            to the boxes. If 'exclude', the border pixels do not belong to the boxes.
            If 'half', then one of each of the two horizontal and vertical borders belong
            to the boxex, but not the other.

    Returns:
        A Numpy nD array, a copy of the input tensor with the converted coordinates
        in place of the original coordinates and the unaltered elements of the original
        tensor elsewhere.
    '''
    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1
    elif border_pixels == 'exclude':
        d = -1

    ind = start_index
    tensor1 = np.copy(tensor).astype(np.float)
    if conversion == 'minmax2centroids':
        tensor1[..., ind] = (tensor[..., ind] +
                             tensor[..., ind+1]) / 2.0  # Set cx
        tensor1[..., ind+1] = (tensor[..., ind+2] +
                               tensor[..., ind+3]) / 2.0  # Set cy
        tensor1[..., ind+2] = tensor[..., ind+1] - \
            tensor[..., ind] + d  # Set w
        tensor1[..., ind+3] = tensor[..., ind+3] - \
            tensor[..., ind+2] + d  # Set h
    elif conversion == 'centroids2minmax':
        tensor1[..., ind] = tensor[..., ind] - \
            tensor[..., ind+2] / 2.0  # Set xmin
        tensor1[..., ind+1] = tensor[..., ind] + \
            tensor[..., ind+2] / 2.0  # Set xmax
        tensor1[..., ind+2] = tensor[..., ind+1] - \
            tensor[..., ind+3] / 2.0  # Set ymin
        tensor1[..., ind+3] = tensor[..., ind+1] + \
            tensor[..., ind+3] / 2.0  # Set ymax
    elif conversion == 'corners2centroids':
        tensor1[..., ind] = (tensor[..., ind] +
                             tensor[..., ind+2]) / 2.0  # Set cx
        tensor1[..., ind+1] = (tensor[..., ind+1] +
                               tensor[..., ind+3]) / 2.0  # Set cy
        tensor1[..., ind+2] = tensor[..., ind+2] - \
            tensor[..., ind] + d  # Set w
        tensor1[..., ind+3] = tensor[..., ind+3] - \
            tensor[..., ind+1] + d  # Set h
    elif conversion == 'centroids2corners':
        tensor1[..., ind] = tensor[..., ind] - \
            tensor[..., ind+2] / 2.0  # Set xmin
        tensor1[..., ind+1] = tensor[..., ind+1] - \
            tensor[..., ind+3] / 2.0  # Set ymin
        tensor1[..., ind+2] = tensor[..., ind] + \
            tensor[..., ind+2] / 2.0  # Set xmax
        tensor1[..., ind+3] = tensor[..., ind+1] + \
            tensor[..., ind+3] / 2.0  # Set ymax
    elif conversion in ('minmax2corners', 'corners2minmax'):
        tensor1[..., ind+1] = tensor[..., ind+2]
        tensor1[..., ind+2] = tensor[..., ind+1]
    else:
        raise ValueError(
            "Unexpected conversion value.")

    return tensor1


def convert_coordinates2(tensor, start_index, conversion):
    '''
    A matrix multiplication implementation of `convert_coordinates()`.

    Supports only conversion between the 'centroids' and 'minmax' formats.

    This function is marginally slower on average than `convert_coordinates()`,
    probably because it involves more (unnecessary) arithmetic operations (unnecessary
    because the two matrices are sparse).

    For details please refer to the documentation of `convert_coordinates()`.
    '''
    ind = start_index
    tensor1 = np.copy(tensor).astype(np.float)
    if conversion == 'minmax2centroids':
        M = np.array([[0.5, 0., -1.,  0.],
                      [0.5, 0.,  1.,  0.],
                      [0., 0.5,  0., -1.],
                      [0., 0.5,  0.,  1.]])
        tensor1[..., ind:ind+4] = np.dot(tensor1[..., ind:ind+4], M)
    elif conversion == 'centroids2minmax':
        M = np.array([[1., 1.,  0., 0.],
                      [0., 0.,  1., 1.],
                      [-0.5, 0.5,  0., 0.],
                      [0., 0., -0.5, 0.5]])  # The multiplicative inverse of the matrix above
        tensor1[..., ind:ind+4] = np.dot(tensor1[..., ind:ind+4], M)
    else:
        raise ValueError(
            "Unexpected conversion value.")

    return tensor1


def intersection_area(boxes1, boxes2,
                      coords='centroids',
                      mode='outer_product',
                      border_pixels='half'):
    '''
    Computes the intersection areas of two sets of axis-aligned 2D rectangular boxes.

    Let `boxes1` and `boxes2` contain `m` and `n` boxes, respectively.

    In 'outer_product' mode, returns an `(m,n)` matrix with the intersection areas for all possible
    combinations of the boxes in `boxes1` and `boxes2`.

    In 'element-wise' mode, `m` and `n` must be broadcast-compatible. Refer to the explanation
    of the `mode` argument for details.
    '''

    # Make sure the boxes have the right shapes.
    if boxes1.ndim > 2:
        raise ValueError(
            "boxes1 must have rank either 1 or 2, but has rank {}.".format(boxes1.ndim))
    if boxes2.ndim > 2:
        raise ValueError(
            "boxes2 must have rank either 1 or 2, but has rank {}.".format(boxes2.ndim))

    if boxes1.ndim == 1:
        boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1:
        boxes2 = np.expand_dims(boxes2, axis=0)

    # Convert the coordinates if necessary.
    if coords == 'centroids':
        boxes1 = convert_coordinates(
            boxes1, start_index=0, conversion='centroids2corners')
        boxes2 = convert_coordinates(
            boxes2, start_index=0, conversion='centroids2corners')
        coords = 'corners'
    elif not (coords in {'minmax', 'corners'}):
        raise ValueError(
            "Unexpected value for `coords`.")

    m = boxes1.shape[0]  # The number of boxes in `boxes1`
    n = boxes2.shape[0]  # The number of boxes in `boxes2`

    # Set the correct coordinate indices for the respective formats.
    if coords == 'corners':
        xmin = 0
        ymin = 1
        xmax = 2
        ymax = 3
    elif coords == 'minmax':
        xmin = 0
        xmax = 1
        ymin = 2
        ymax = 3

    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1
    elif border_pixels == 'exclude':
        d = -1

    # Compute the intersection areas.

    if mode == 'outer_product':

        # For all possible box combinations, get the greater xmin and ymin values.
        # This is a tensor of shape (m,n,2).
        min_xy = np.maximum(np.tile(np.expand_dims(boxes1[:, [xmin, ymin]],
                                                   axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:, [xmin, ymin]],
                                                   axis=0), reps=(m, 1, 1)))

        # For all possible box combinations, get the smaller xmax and ymax values.
        # This is a tensor of shape (m,n,2).
        max_xy = np.minimum(np.tile(np.expand_dims(boxes1[:, [xmax, ymax]],
                                                   axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:, [xmax, ymax]],
                                                   axis=0), reps=(m, 1, 1)))

        # Compute the side lengths of the intersection rectangles.
        side_lengths = np.maximum(0, max_xy - min_xy + d)

        return side_lengths[:, :, 0] * side_lengths[:, :, 1]

    min_xy = np.maximum(boxes1[:, [xmin, ymin]], boxes2[:, [xmin, ymin]])
    max_xy = np.minimum(boxes1[:, [xmax, ymax]], boxes2[:, [xmax, ymax]])

    # Compute the side lengths of the intersection rectangles.
    side_lengths = np.maximum(0, max_xy - min_xy + d)

    return side_lengths[:, 0] * side_lengths[:, 1]


def intersection_area_(boxes1, boxes2, coords='corners',
                       mode='outer_product', border_pixels='half'):
    '''The same as 'intersection_area()' without all the safety checks.'''

    m = boxes1.shape[0]  # The number of boxes in `boxes1`
    n = boxes2.shape[0]  # The number of boxes in `boxes2`

    # Set the correct coordinate indices for the respective formats.
    if coords == 'corners':
        xmin = 0
        ymin = 1
        xmax = 2
        ymax = 3
    elif coords == 'minmax':
        xmin = 0
        xmax = 1
        ymin = 2
        ymax = 3

    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1
    elif border_pixels == 'exclude':
        d = -1

    # Compute the intersection areas.

    if mode == 'outer_product':

        # For all possible box combinations, get the greater xmin and ymin values.
        # This is a tensor of shape (m,n,2).
        min_xy = np.maximum(np.tile(np.expand_dims(boxes1[:, [xmin, ymin]],
                                                   axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:, [xmin, ymin]],
                                                   axis=0), reps=(m, 1, 1)))

        # For all possible box combinations, get the smaller xmax and ymax values.
        # This is a tensor of shape (m,n,2).
        max_xy = np.minimum(np.tile(np.expand_dims(boxes1[:, [xmax, ymax]],
                                                   axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:, [xmax, ymax]],
                                                   axis=0), reps=(m, 1, 1)))

        # Compute the side lengths of the intersection rectangles.
        side_lengths = np.maximum(0, max_xy - min_xy + d)

        return side_lengths[:, :, 0] * side_lengths[:, :, 1]

    min_xy = np.maximum(boxes1[:, [xmin, ymin]], boxes2[:, [xmin, ymin]])
    max_xy = np.minimum(boxes1[:, [xmax, ymax]], boxes2[:, [xmax, ymax]])

    # Compute the side lengths of the intersection rectangles.
    side_lengths = np.maximum(0, max_xy - min_xy + d)

    return side_lengths[:, 0] * side_lengths[:, 1]


def iou(boxes1, boxes2, coords='centroids', mode='outer_product', border_pixels='half'):
    '''
    Computes the intersection-over-union similarity.

    Let `boxes1` and `boxes2` contain `m` and `n` boxes, respectively.

    In 'outer_product' mode, returns an `(m,n)` matrix with the IoUs for all possible
    combinations of the boxes in `boxes1` and `boxes2`.

    In 'element-wise' mode, `m` and `n` must be broadcast-compatible. Refer to the explanation
    of the `mode` argument for details.
    '''

    # Make sure the boxes have the right shapes.
    if boxes1.ndim > 2:
        raise ValueError(
            "boxes1 must have rank either 1 or 2, but has rank {}.".format(boxes1.ndim))
    if boxes2.ndim > 2:
        raise ValueError(
            "boxes2 must have rank either 1 or 2, but has rank {}.".format(boxes2.ndim))

    if boxes1.ndim == 1:
        boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1:
        boxes2 = np.expand_dims(boxes2, axis=0)

    # Convert the coordinates if necessary.
    if coords == 'centroids':
        boxes1 = convert_coordinates(
            boxes1, start_index=0, conversion='centroids2corners')
        boxes2 = convert_coordinates(
            boxes2, start_index=0, conversion='centroids2corners')
        coords = 'corners'
    elif not (coords in {'minmax', 'corners'}):
        raise ValueError(
            "Unexpected value for `coords`.")

    # Compute the IoU.

    # Compute the interesection areas.

    intersection_areas = intersection_area_(
        boxes1, boxes2, coords=coords, mode=mode)

    m = boxes1.shape[0]  # The number of boxes in `boxes1`
    n = boxes2.shape[0]  # The number of boxes in `boxes2`

    # Compute the union areas.

    # Set the correct coordinate indices for the respective formats.
    if coords == 'corners':
        xmin = 0
        ymin = 1
        xmax = 2
        ymax = 3
    elif coords == 'minmax':
        xmin = 0
        xmax = 1
        ymin = 2
        ymax = 3

    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1
    elif border_pixels == 'exclude':
        d = -1

    if mode == 'outer_product':

        boxes1_areas = np.tile(np.expand_dims((boxes1[:, xmax] - boxes1[:, xmin] + d) * (
            boxes1[:, ymax] - boxes1[:, ymin] + d), axis=1), reps=(1, n))
        boxes2_areas = np.tile(np.expand_dims((boxes2[:, xmax] - boxes2[:, xmin] + d) * (
            boxes2[:, ymax] - boxes2[:, ymin] + d), axis=0), reps=(m, 1))

    elif mode == 'element-wise':

        boxes1_areas = (boxes1[:, xmax] - boxes1[:, xmin] + d) * \
            (boxes1[:, ymax] - boxes1[:, ymin] + d)
        boxes2_areas = (boxes2[:, xmax] - boxes2[:, xmin] + d) * \
            (boxes2[:, ymax] - boxes2[:, ymin] + d)

    union_areas = boxes1_areas + boxes2_areas - intersection_areas

    return intersection_areas / union_areas


def iou_clean(boxes1, boxes2):
    '''
    numpy version of vectorized iou.

    Let `boxes1` and `boxes2` contain `m` and `n` boxes, respectively.

    returns an `(m, n)` matrix with the IoUs for all pairwise boxes

    Arguments:
        boxes1 (array of shape (m, 4)): x_min, y_min, x_max, y_max
        boxes2 (array of shape (n, 4)): x_min, y_min, x_max, y_max

    Returns:
        IOU (array of shape (m, n)): IOU score
    '''

    # Compute the IoU.
    xmin1, ymin1, xmax1, ymax1 = np.split(boxes1, 4, axis=1)
    xmin2, ymin2, xmax2, ymax2 = np.split(boxes2, 4, axis=1)

    xmin = np.maximum(xmin1, xmin2.T)
    ymin = np.maximum(ymin1, ymin2.T)
    xmax = np.minimum(xmax1, xmax2.T)
    ymax = np.minimum(ymax1, ymax2.T)

    intersection = np.maximum(xmax - xmin, 0) * np.maximum(ymax - ymin, 0)
    boxes1_areas = (xmax1 - xmin1) * (ymax1 - ymin1)
    boxes2_areas = (xmax2 - xmin2) * (ymax2 - ymin2)
    union = boxes1_areas + boxes2_areas.T - intersection

    return intersection / union
