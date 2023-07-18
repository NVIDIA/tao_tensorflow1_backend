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

"""Polygon Rasterier Processor."""

import tensorflow as tf
from nvidia_tao_tf1.core.coreobject import save_args
from nvidia_tao_tf1.core.processors.processors import is_sparse, load_custom_tf_op, Processor
from nvidia_tao_tf1.core.types import data_format as modulus_data_format, DataFormat


class PolygonRasterizer(Processor):
    """Processor that draws polygons from coordinate/class lists into (rasterized) maps.

    Regardless of settings, the background will always be rendered as zeros. Any class with a value
    of -1 will be ignored (and not drawn). The class index of each polygon has an impact on the
    value in the rasterized maps, depending on the ``one_hot`` and ``binarize`` arguments.

    Args:
        width (int): width of the output map.
        height (int): height of the output map.
        nclasses (int or None): The number of classes. This value must be specified if ``one_hot``
            is ``True``, because ``one_hot`` creates a fixed output map for each class, but can be
            ``None`` (the default) if ``one_hot`` is ``False``.
        binarize (bool): Defaults to ``True``, but can be set to ``False`` if ``one_hot`` is
            ``True``. When ``one_hot`` is ``True``, the polygons of each class will be rendered in
            their own map, so it's OK that output values in each map aren't binary. (For example,
            a polygon intersecting a pixel over its diagonal could result in a value of 0.5.).
            Binarize=True is more expensive than binarizing by setting num_samples=1, so it should
            be used only when more accurate pixel coverage is required.
        one_hot (bool): Defaults to ``True``, meaning that one output map is created for each class.
            The background class is the first map at index 0. This means that a class index '0' will
            be drawn at map (channel) index '1'.
            If set to ``False`` instead, the rasterizer will produce only a single output map
            containing the polygons from all classes, with each class represented by a different
            discrete integer value.  (The current implementation is that each class_id will be
            rendered with an integer value of ``class_id+1``.)
        verbose (bool): If ``True``, shows verbose output from the backend implementation.
        data_format (str): A string representing the dimension ordering of the input data.
            Must be one of 'channels_last' or 'channels_first'. If ``None`` (default), the
            modulus global default will be used.
        num_samples (int): number of samples per box filter dimension. For each pixel in the
            output image, polygon coverage is evaluated by sampling with a pixel sized box
            filter. The total number of samples taken is num_samples * num_samples. Must be
            between 1 and 5. Note that 1 gives a binary result. Also note that setting
            num_samples > 1 and binarize=True results in fatter polygons compared to num_samples=1
            since the former setting approximates a one pixel wide box filter while the latter
            uses point sampling.
        include_background (bool): If set to true, the rasterized output would also include
            the background channel at channel index=0. This parameter only takes effect when
            `one_hot` parameter is set to `true`. Default `true`.
        kwargs (dict): keyword arguments passed to parent class.

    Raises:
        NotImplementedError: if ``data_format`` is not in ['channels_first', 'channels_last'].
        ValueError: if ``one_hot`` is set with ``nclassses`` unspecified, or if ``one_hot`` and
            ``binarize`` are both set to False.
    """

    @save_args
    def __init__(
        self,
        width,
        height,
        nclasses=None,
        binarize=True,
        one_hot=True,
        verbose=False,
        data_format=None,
        num_samples=5,
        include_background=True,
        **kwargs
    ):
        """__init__ method."""
        self.width = width
        self.height = height
        self.nclasses = nclasses
        self.binarize = binarize
        self.one_hot = one_hot
        self.verbose = verbose
        self.include_background = include_background
        self.data_format = (
            data_format if data_format is not None else modulus_data_format()
        )
        self.num_samples = num_samples

        # TODO(xiangbok): add an attribute that sets the class_id and the value that class should be
        # rendered at, if ``one_hot`` is ``False``. Currently, we're rendering those values as
        # ``class_id+1``. This is done to avoid ``class_id`` 0 to be drawn as 0 (background).

        if self.data_format not in [
            DataFormat.CHANNELS_FIRST,
            DataFormat.CHANNELS_LAST,
        ]:
            raise NotImplementedError(
                "Data format not supported, must be 'channels_first' or "
                "'channels_last', given {}.".format(self.data_format)
            )

        if one_hot and nclasses is None:
            raise ValueError("Using `one_hot` requires `nclasses` to be defined.")

        if one_hot is False and binarize is False:
            raise ValueError(
                "Setting `one_hot` False is incompatible with `binarize` as False."
            )

        if nclasses is None:
            self.nclasses = 1

        super(PolygonRasterizer, self).__init__(**kwargs)

    def _post_process(self, cov):
        """Post process the output tensor.

        Args:
            cov (Tensor): Tensor output by the custom op.

        Returns:
            Tensor with post processing operations applied.
        """
        if self.one_hot and self.include_background:
            # Add the background as a separate map. With out current implementation this is not
            # straightforward to add in the kernel (because the kernel loops over classes
            # individually).
            cov_sum = tf.reduce_sum(
                input_tensor=cov, axis=-3, keepdims=True
            )  # NCHW->N1HW or CHW->1HW.
            cov_background = tf.cast(
                tf.equal(cov_sum, 0), dtype=tf.float32
            )  # N1HW or 1HW.
            cov = tf.concat(
                [cov_background, cov], axis=-3
            )  # (N, 1+C, H, W) or (1+C, H, W).

        # TODO(xiangbok): Add native channels_last support.
        if self.data_format == DataFormat.CHANNELS_LAST:
            cov = DataFormat.convert(
                cov,
                from_format=DataFormat.CHANNELS_FIRST,
                to_format=DataFormat.CHANNELS_LAST,
            )

        return cov

    def call(
        self,
        polygon_vertices,
        vertex_counts_per_polygon,
        class_ids_per_polygon,
        polygons_per_image=None,
        force_cpu=False
    ):
        """call method.

        Actually runs the polygon rasterizer op with given inputs, and returns the rasterized
        map(s).

        Args:
            polygon_vertices: a tensor in the form of a list of lists. The top-level list contains
                sub-lists with 2 elements each; each sub-list contains the x/y coordinates (in that
                order) of a single vertex of a single polygon for a single image (= raster map). The
                length of the top-level list is therefore equal to the total number of vertices over
                all polygons that we are drawing over all raster maps.
            vertex_counts_per_polygon: a tensor in the form of a flat list. The elements of the list
                are the vertex counts for each polygon that we will draw during rasterization. Thus,
                the length of this list is equal to the number of polygons we will draw, and if we
                were to sum all the values in this list, the sum should equal the length of the
                ``polygon_vertices`` list above.
            class_ids_per_polygon: a tensor in the form of a flat list having the same shape as the
                ``vertex_counts_per_polygon`` list above. Each list element is an ID representing
                the class to which each polygon belongs.
            polygons_per_image: if `None` (the default), we assume only one single image (i.e. this
                call will output only a single raster map). Otherwise, this should be a tensor in
                the form of a flat list, where each list element is the number of polygons to be
                drawn for that image (raster). In this case, the sum of the list values should equal
                the length of the ``vertex_counts_per_polygon`` list above.

        Returns:
            cov: a fp32 tensor (`NCHW`) containing the output map if 'data_format' is set to
                'channels_first', or a fp32 tensor of shape (NHWC) if 'data_format' is set to
                'channels_last'. When ``one_hot`` is used, the number of channels `C` is equal
                to ``nclasses``, and when it is not used, it is equal to 1.
        """
        polygon_vertices = tf.cast(polygon_vertices, dtype=tf.float32)
        vertex_counts_per_polygon = tf.cast(vertex_counts_per_polygon, dtype=tf.int32)
        class_ids_per_polygon = tf.cast(class_ids_per_polygon, dtype=tf.int32)

        # If polygons_per_image is None, use an empty tensor to signal that we want
        # 3D output. In that case the number of polygons is infered from
        # vertex_counter_per_polygon.
        polygons_per_image = (
            tf.constant([], dtype=tf.int32)
            if polygons_per_image is None
            else tf.cast(polygons_per_image, dtype=tf.int32)
        )

        op = load_custom_tf_op("op_rasterize_polygon.so")
        if force_cpu:
            with tf.device('CPU:0'):
                cov = op.rasterize_polygon(
                    polygon_vertices=polygon_vertices,
                    vertex_counts_per_polygon=vertex_counts_per_polygon,
                    class_ids_per_polygon=class_ids_per_polygon,
                    polygons_per_image=polygons_per_image,
                    width=self.width,
                    height=self.height,
                    num_samples=tf.cast(self.num_samples, dtype=tf.int32),
                    nclasses=self.nclasses,
                    binarize=self.binarize,
                    one_hot=self.one_hot,
                    verbose=self.verbose,
                )
        else:
            cov = op.rasterize_polygon(
                polygon_vertices=polygon_vertices,
                vertex_counts_per_polygon=vertex_counts_per_polygon,
                class_ids_per_polygon=class_ids_per_polygon,
                polygons_per_image=polygons_per_image,
                width=self.width,
                height=self.height,
                num_samples=tf.cast(self.num_samples, dtype=tf.int32),
                nclasses=self.nclasses,
                binarize=self.binarize,
                one_hot=self.one_hot,
                verbose=self.verbose,
            )
        return self._post_process(cov)


class SparsePolygonRasterizer(PolygonRasterizer):
    """Polygon rasterizer for sparse polygon input.

    See ``PolygonRasterizer`` documentation.
    """

    def call(self, polygons, class_ids_per_polygon, force_cpu=False):
        """call method.

        Args:
            polygons (``tf.SparseTensor``): the polygons and its vertices wrapped in a sparse
                tensor. Polygons.dense_shape must be either 3D (PVC) or 4D (NPVC), where N is batch
                dimension, P is polygons, V vertices, and C coordinate index (0 or 1). In the
                3D case the op returns a 3D tensor (CHW or HWC). In the 4D case the first
                dimension of dense_shape specifies batch size, and the op returns a 4D tensor
                (NCHW or NHWC). Polygons.values is a flat fp32 list of interleaved vertex x
                and y coordinates. Polygons.indices is a 2D tensor with dimension 0 the size of
                the polygons.values tensor, and dimension 1 either 3D (PVC) or 4D (NPVC).
            class_ids_per_polygon: the class ids wrapped in a sparse tensor with indices
                corresponding to those of the polygons. class_ids_per_polygon.dense_shape must be
                either 2D (SC) for the 3D polygon case or 2D (NSC) for the 4D polygon case, where N
                is the batch dimension, S is the shape dimension and C is the class dimension.

                Each value is an ID representing the class to which each polygon belongs. If a class
                id is associated with a polygon id that do not exist in the polygon sparse tensor,
                the class id will be skipped from processing. If there exists a polygon that does
                not have a corresponding class id, the operation will result in an error.

        Returns:
            cov: an fp32 tensor (CHW or NCHW) containing the output map if 'data_format' is
                set to 'channels_first', or a fp32 tensor of shape (HWC or NHWC) if 'data_format'
                is set to 'channels_last'. When ``one_hot`` is used, the number of channels `C` is
                equal to ``nclasses``, and when it is not used, it is equal to 1.
        """
        assert is_sparse(polygons)
        indices = tf.cast(polygons.indices, dtype=tf.int32)
        values = tf.cast(polygons.values, dtype=tf.float32)
        dense_shape = tf.cast(polygons.dense_shape, dtype=tf.int32)

        class_ids_per_polygon_indices = tf.cast(
            class_ids_per_polygon.indices, dtype=tf.int32
        )

        class_ids_per_polygon_values = tf.cast(
            class_ids_per_polygon.values, dtype=tf.int32
        )

        class_dense_shape = tf.convert_to_tensor(
            value=class_ids_per_polygon.dense_shape, dtype=tf.int64
        )
        class_ids_per_polygon_dense_shape = tf.cast(class_dense_shape, dtype=tf.int32)

        op = load_custom_tf_op("op_rasterize_polygon.so")
        if force_cpu:
            with tf.device('CPU:0'):
                cov = op.rasterize_sparse_polygon(
                    polygon_indices=indices,
                    polygon_dense_shape=dense_shape,
                    polygon_values=values,
                    class_ids_per_polygon_indices=class_ids_per_polygon_indices,
                    class_ids_per_polygon_values=class_ids_per_polygon_values,
                    class_ids_per_polygon_dense_shape=class_ids_per_polygon_dense_shape,
                    width=self.width,
                    height=self.height,
                    num_samples=tf.cast(self.num_samples, dtype=tf.int32),
                    nclasses=self.nclasses,
                    binarize=self.binarize,
                    one_hot=self.one_hot,
                    verbose=self.verbose,
                )
        else:
            cov = op.rasterize_sparse_polygon(
                polygon_indices=indices,
                polygon_dense_shape=dense_shape,
                polygon_values=values,
                class_ids_per_polygon_indices=class_ids_per_polygon_indices,
                class_ids_per_polygon_values=class_ids_per_polygon_values,
                class_ids_per_polygon_dense_shape=class_ids_per_polygon_dense_shape,
                width=self.width,
                height=self.height,
                num_samples=tf.cast(self.num_samples, dtype=tf.int32),
                nclasses=self.nclasses,
                binarize=self.binarize,
                one_hot=self.one_hot,
                verbose=self.verbose,
            )

        return self._post_process(cov)
