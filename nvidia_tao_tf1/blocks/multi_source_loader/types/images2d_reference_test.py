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

"""Unit tests for Images2DReference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import shutil
import tempfile

import numpy as np
from parameterized import parameterized
from PIL import Image
import tensorflow as tf

from nvidia_tao_tf1.blocks.multi_source_loader.types import Canvas2D
from nvidia_tao_tf1.blocks.multi_source_loader.types import test_fixtures
from nvidia_tao_tf1.blocks.multi_source_loader.types.images2d_reference import (
    Images2DReference,
)


def _generate_images(
    image_type, export_path, sequence_id, camera_name, frame_numbers, height, width
):
    """Create files filled with random numbers that look like exported fp16/jpeg/png images."""
    dir_path = os.path.join(export_path, sequence_id, camera_name)
    os.makedirs(dir_path)

    paths = []
    for frame_number in frame_numbers:
        path = os.path.join(dir_path, "{}.{}".format(frame_number, image_type))
        if image_type == "fp16":
            # For fp16, the value of each pixel lies between 0~1.
            image = np.random.rand(3, height, width).astype(np.float16)
            image.tofile(path)
        else:
            # For jpeg and png, the value of each pixel lies between 0~255.
            image = np.random.randint(255, size=(height, width, 3), dtype=np.uint8)
            image = np.ascontiguousarray(image)
            Image.fromarray(image, "RGB").save(path)
        paths.append(path)
    return paths


class Images2DReferenceTest(tf.test.TestCase):
    def setUp(self):
        self.export_path = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.export_path)

    @parameterized.expand(
        [
            ["jpeg", 1, 3, 240, 480, np.float16],
            ["jpeg", 2, 3, 604, 960, np.float32],
            ["jpeg", 3, 3, 240, 480, np.float16],
            ["jpeg", 3, 3, 240, 480, np.uint8],
            ["png", 1, 3, 240, 480, np.float16],
            ["png", 2, 3, 604, 960, np.float32],
            ["png", 3, 3, 240, 480, np.float16],
            ["png", 3, 3, 240, 480, np.uint8],
        ]
    )
    def test_loads_4d_examples(
        self, image_type, batch_size, channel_count, height, width, output_dtype
    ):
        paths = _generate_images(
            image_type=image_type,
            export_path=self.export_path,
            sequence_id="4d_batch",
            camera_name="FOV_60",
            frame_numbers=[number for number in range(batch_size)],
            height=height,
            width=width,
        )
        extensions = ["." + image_type for _ in range(batch_size)]

        # Sprinkle some extra pixels to test that images are padded to a common output shape.
        output_height = height + random.randint(4, 8)
        output_width = width + random.randint(2, 10)
        shapes = test_fixtures.make_canvas2d(batch_size, output_height, output_width)

        input_height = [height] * batch_size
        input_width = [width] * batch_size

        assets = Images2DReference(
            path=tf.constant(paths),
            extension=tf.constant(extensions),
            canvas_shape=shapes,
            input_height=input_height,
            input_width=input_width,
        )

        loaded = assets.load(output_dtype=output_dtype)
        assert loaded.images.dtype == output_dtype
        assert [
            batch_size,
            channel_count,
            output_height,
            output_width,
        ] == loaded.images.shape

        with self.cached_session() as session:
            loaded = session.run(loaded)
            assert loaded.images.dtype == output_dtype
            assert (
                batch_size,
                channel_count,
                output_height,
                output_width,
            ) == loaded.images.shape

        if output_dtype == np.uint8:
            # pixel of png and jpeg with dtype uint8 is normalized between 0.0 and 255.0.
            self.assertAllInRange(loaded.images, 0.0, 255.0)
        else:
            self.assertAllInRange(loaded.images, 0.0, 1.0)

    @parameterized.expand(
        [
            ["jpeg", 1, 1, 3, 240, 480, np.uint8],
            ["jpeg", 1, 2, 3, 604, 960, np.float16],
            ["jpeg", 1, 3, 3, 240, 480, np.float32],
            ["jpeg", 2, 1, 3, 240, 480, np.uint8],
            ["jpeg", 2, 2, 3, 604, 960, np.float16],
            ["jpeg", 2, 3, 3, 240, 480, np.float32],
            ["png", 1, 1, 3, 240, 480, np.uint8],
            ["png", 1, 2, 3, 604, 960, np.float16],
            ["png", 1, 3, 3, 240, 480, np.float32],
            ["png", 2, 1, 3, 240, 480, np.uint8],
            ["png", 2, 2, 3, 604, 960, np.float16],
            ["png", 2, 3, 3, 240, 480, np.float32],
        ]
    )
    def test_loads_5d_examples(
        self,
        image_type,
        batch_size,
        window_size,
        channel_count,
        height,
        width,
        output_dtype,
    ):
        paths = []
        extensions = []
        shapes = []
        # Sprinkle some extra pixels to test that images are padded to a common output shape.
        output_height = height + random.randint(4, 8)
        output_width = width + random.randint(2, 10)
        for batch in range(batch_size):
            paths.append(
                _generate_images(
                    image_type=image_type,
                    export_path=self.export_path,
                    sequence_id="5d_batch_{}".format(batch),
                    camera_name="FOV_60",
                    frame_numbers=[number for number in range(window_size)],
                    height=height,
                    width=width,
                )
            )
            extensions.append(["." + image_type for _ in range(window_size)])
            shapes.append(
                test_fixtures.make_canvas2d(window_size, output_height, output_width)
            )

        input_height = [[height] * window_size] * batch_size
        input_width = [[width] * window_size] * batch_size

        assets = Images2DReference(
            path=tf.constant(paths),
            extension=tf.constant(extensions),
            canvas_shape=Canvas2D(
                height=tf.stack([shape.height for shape in shapes]),
                width=tf.stack([shape.width for shape in shapes]),
            ),
            input_height=input_height,
            input_width=input_width,
        )

        loaded = assets.load(output_dtype=output_dtype)
        assert loaded.images.dtype == output_dtype
        assert [
            batch_size,
            window_size,
            channel_count,
            output_height,
            output_width,
        ] == loaded.images.shape

        with self.cached_session() as session:
            loaded = session.run(loaded)
            assert loaded.images.dtype == output_dtype
            assert (
                batch_size,
                window_size,
                channel_count,
                output_height,
                output_width,
            ) == loaded.images.shape

        if output_dtype == np.uint8:
            # pixel of png and jpeg with dtype uint8 is normalized between 0.0 and 255.0.
            self.assertAllInRange(loaded.images, 0.0, 255.0)
        else:
            self.assertAllInRange(loaded.images, 0.0, 1.0)
