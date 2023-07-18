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

"""Bbox Rasterizer Processor."""

import tensorflow as tf

from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors.processors import load_custom_tf_op, Processor
from nvidia_tao_tf1.core.types import data_format as modulus_data_format, DataFormat


class BboxRasterizer(Processor):
    """Processor that rasterizes rectangles and ellipses into images.

    Output of the rasterization operation is a 5D tensor with shape (N, C, G, H, W),
    where N is the number of images (=batch size), C is the number of classes, G is the
    number of gradient buffers (described later), H is image height, and W is image width.

    The processor supports drawing both rectangles and ellipses. These shapes are
    collectively referred to as bboxes. This is because an ellipse can be thought of as
    being embedded within its bounding box.

    The bboxes are input as a number of tensors specifying their attributes. Each bbox
    is described by a 3x3 matrix giving its size and location, a class ID, a number
    of gradient coefficients, horizontal and vertical coverage radii, and flags bitfield.
    Bboxes are drawn in the order they are specified, ie. a later bbox is drawn on top of
    an earlier one.
    The reason for using a matrix to describe the geometry is that it is a compact way to
    specify all affine transformations of the unit square/circle in a plane, ie. it allows
    drawing arbitrarily rotated, scaled, and sheared boxes and ellipses. We're restricting
    ourselves to matrices where the third column is always [0,0,1]^T as this processor does
    not support perspective projections. The matrix maps an output image plane pixel
    location P onto a location P' in rasterization space, where the bbox is the unit square
    (and the embedded ellipse is the unit circle). In the rasterization space determining
    whether a pixel is covered by the bbox is simply a matter of checking whether P' is within
    the unit square (for rectangles) or the unit circle (for ellipses). One intuitive way of
    forming the matrix is to construct a mapping that transforms the unit square to the
    desired shape, then inverting the resulting matrix. Higher level classes are
    expected to provide user friendly helper functions for constructing the matrix depending
    on application's needs.

    For each pixel and bbox, the rasterizer computes a coverage value by supersampling, ie.
    checking at 4 by 4 grid of locations within the pixel whether the location is inside the
    bbox. Coverage is defined as the fraction of locations inside the bbox, thus 0.0 means no
    part of the pixel is covered, while 1.0 means the whole pixel is covered. Fractional
    coverage occurs at object edges.

    Sometimes we want to make a foreground object stand out better by masking out background
    objects around it. For this, the rasterizer supports deadzone, which is defined using
    the coverage radii inputs that define the actual object's size within the bbox, while
    the rest is used for the deadzone. A coverage radius of 1.0 means the actual object
    covers the whole bbox, and thus there's no deadzone. A coverage radius of 0.5 means
    the central half of the bbox is used for the actual object, while the surrounding half is
    used for the deadzone.

    The third dimension of the output tensor contains a number of buffers with user specified
    linearly interpolated values (=gradients) that are optionally multiplied by each pixel's
    computed coverage value. Note that the term gradient comes from computer graphics and
    does not have anything to do with network gradients. Use cases include an object coverage
    buffer where a constant gradient of 1.0 is multiplied by pixel coverage, and a set of bbox
    distance buffers where four interpolated gradients are used for computing each pixel's
    distance to each of the four bbox edges. Note that in the latter case multiplying the
    interpolated value by coverage does not make sense.
    A gradient is defined by the formula g = A*px + B*py + C, where A, B, and C are user
    defined coefficients and px and py are pixel coordinates. It's easy to see that A describes
    the change in gradient value when moving one pixel right, and B the same when moving one
    pixel down. C is a constant offset. For a constant gradient, simply set C to the desired
    value and set A = B = 0. Higher level code is expected to provide user friendly helper
    functions for constructing the gradient coefficients depending on application's needs.
    The reason for using gradients is that it allows specifying all linear 1D functions on a
    2D plane, including constant values. This is a general way for supporting variable number
    of output buffers with application defined meanings. If a nonlinearity is needed on top of
    a linear function, it should be done in a postprocessing pass. Note that nonlinearly
    interpolated values such as radial functions cannot be done in a postprocess and instead
    require modifications to this op.
    Unlike coverage, a gradient's value is computed once per pixel. This has the effect that
    if a gradient is chosen to be multiplied by coverage, the result has smooth edges. If a
    gradient is not multiplied by coverage, the edges might appear rough.

    For each pixel, the bboxes with the same class ID are composited in back to front order.
    Compositing differs from
    standard computer graphics compositing modes in that we keep track of maximum coverage
    value seen so far, and only replace a pixel if the new bbox fragment has larger
    coverage. This has been shown to improve detection performance with small objects.
    If the pixel falls within the deadzone of a bbox and the bbox's coverage is larger than the
    maximum seen so far, we clear the background and set maximum coverage to zero, which has
    the effect of masking out the background objects.
    """

    # Supported bbox_flags
    DRAW_MODE_RECTANGLE = 0
    DRAW_MODE_ELLIPSE = 1

    # Supported gradient flags
    GRADIENT_MODE_PASSTHROUGH = 0
    GRADIENT_MODE_MULTIPLY_BY_COVERAGE = 1

    @save_args
    def __init__(self, verbose=False, data_format=None, **kwargs):
        """__init__ method.

        Args:
            data_format (str): A string representing the dimension ordering of the input data.
                Must be one of 'channels_last' or 'channels_first'. If ``None`` (default), the
                modulus global default will be used.

        Raises:
            NotImplementedError: if ``data_format`` is not in ['channels_first', 'channels_last'].
        """
        super(BboxRasterizer, self).__init__(**kwargs)
        self.verbose = verbose
        self.data_format = (
            data_format if data_format is not None else modulus_data_format()
        )
        if self.data_format not in [
            DataFormat.CHANNELS_FIRST,
            DataFormat.CHANNELS_LAST,
        ]:
            raise NotImplementedError(
                "Data format not supported, must be 'channels_first' or "
                "'channels_last', given {}.".format(self.data_format)
            )

    def call(
        self,
        num_images,
        num_classes,
        num_gradients,
        image_height,
        image_width,
        bboxes_per_image,
        bbox_class_ids,
        bbox_matrices,
        bbox_gradients,
        bbox_coverage_radii,
        bbox_flags,
        gradient_flags,
        bbox_sort_values=None,
        force_cpu=False,
    ):
        """Generate image tensors by rasterization in native Tensorflow.

        Args:
            num_images: 1D tensor with length of 1 that describes the number of output images
                (= batch size N). The value must be >= 1.
            num_classes: integer constant that describes the number of output classes C. The value
                must be >= 1.
            num_gradients: 1D tensor with length of 1 that describes the number of output gradients
                G. The value must be >= 1.
            image_height: 1D tensor with length of 1 that describes the height H of output images.
                The value must be >= 1.
            image_width: 1D tensor with length of 1 that describes the width W of output images. The
                value must be >= 1.
            bboxes_per_image: 1D int32 tensor of length N. Specifies the number of bboxes in each
                image.
            bbox_class_ids: 1D int32 tensor of length B (=total number of bboxes to draw). Contains
                a class ID for each bbox. Class ID must be a monotonically increasing value within
                each image.
            bbox_matrices: 3D float32 tensor of shape (B,3,3). Contains a 3x3 row major matrix that
                specifies the shape of each bbox to be drawn. The third column of the matrix is
                implicitly taken to be [0,0,1] (ie. the actual values in that column are ignored).
                In rectangle drawing mode, pixel coordinates form a row vector P=[px,py,1] that is
                multiplied by the matrix M from the right: Q = P M. The resulting coordinates Q that
                end up within the unit square around the origin (ie. Q is within [-1,1] range) are
                considered to be inside deadzone, and a Q that satisfy |Q.x| < coverage_radii.x AND
                |Q.y| < coverage_radii.y are considered to be inside coverage zone. Pixels inside
                coverage zone receive coverage value 1.0, and pixels outside coverage zone but
                inside deadzone receive coverage value 0.0. Since coverage value is computed using
                supersampling, pixels that cross zone edge receive coverage value between 0 and 1.
                In ellipse mode, the unit square is replaced by the unit circle. Pixels inside
                coverage zone satisfy (Q.x/coverage_radii.x)^2 + (Q.y/coverage_radii.y)^2 < 1.
            bbox_gradients: 3D float32 tensor of shape (B,G,3). Contains three gradient coefficients
                A, B, and C for each bbox and gradient. Used for computing a gradient value based on
                pixel coordinates using the gradient function g = A*px+B*py+C.
                Gradient values are written as is to all pixels within the actual object (ie. not
                deadzone), to output tensor location [im, cl, g, px, py], optionally multiplied
                by pixel's coverage value.
            bbox_coverage_radii: 2D float32 tensor of shape (B, 2). Sensible coverage radius values
                are between 0.0 and 1.0.
            bbox_flags: 1D uint8 tensor of length B. Contains per bbox flags. Currently the only
                supported flag chooses between rectangle mode and ellipse mode.
            gradient_flags: 1D uint8 tensor of length G. Contains per gradient flags. Currently the
                only supported flag chooses whether a particular gradient value should be multiplied
                by coverage value or not.
            bbox_sort_values: 1D float32 tensor of length B. Contains optional bbox sort values that
                define bbox drawing order within each image and class (the order is ascending:
                the bbox with the smallest sort value is drawn first). This input can be None, in
                which case bboxes are drawn in the input order.

        Returns:
            output_image: 5D tensor with shape (N, C, G, H, W) or (N, H, W, C, G).
        """
        # TODO(xiangbok): many of the inputs here are probably attributes, should be moved
        # from call() to __init__().
        bbox_sort_values = (
            tf.zeros_like(bbox_class_ids, dtype=tf.float32)
            if bbox_sort_values is None
            else bbox_sort_values
        )

        op = load_custom_tf_op("op_rasterize_bbox.so")
        if force_cpu:
            with tf.device('CPU:0'):
                output_image = op.rasterize_bbox(
                    num_images=num_images,
                    num_classes=num_classes,
                    num_gradients=num_gradients,
                    image_height=image_height,
                    image_width=image_width,
                    bboxes_per_image=bboxes_per_image,
                    bbox_class_ids=bbox_class_ids,
                    bbox_matrices=bbox_matrices,
                    bbox_gradients=bbox_gradients,
                    bbox_coverage_radii=bbox_coverage_radii,
                    bbox_flags=bbox_flags,
                    bbox_sort_values=bbox_sort_values,
                    gradient_flags=gradient_flags,
                    verbose=self.verbose,
                )
        else:
            output_image = op.rasterize_bbox(
                num_images=num_images,
                num_classes=num_classes,
                num_gradients=num_gradients,
                image_height=image_height,
                image_width=image_width,
                bboxes_per_image=bboxes_per_image,
                bbox_class_ids=bbox_class_ids,
                bbox_matrices=bbox_matrices,
                bbox_gradients=bbox_gradients,
                bbox_coverage_radii=bbox_coverage_radii,
                bbox_flags=bbox_flags,
                bbox_sort_values=bbox_sort_values,
                gradient_flags=gradient_flags,
                verbose=self.verbose,
            )

        # Op returns NCGHW natively, need to do a transpose to get NHWCG.
        if self.data_format == DataFormat.CHANNELS_LAST:
            output_image = tf.transpose(a=output_image, perm=[0, 3, 4, 1, 2])

        return output_image
