import hashlib
import io
import os
import numpy as np
from PIL import Image

import tensorflow as tf
from nvidia_tao_tf1.cv.mask_rcnn.dataloader import dataloader
from nvidia_tao_tf1.cv.mask_rcnn.dataloader import dataloader_utils
from nvidia_tao_tf1.cv.mask_rcnn.utils import dataset_utils


def test_dummy_example(tmpdir, include_masks=True):
    image_height = 512
    image_width = 512
    filename = "dummy_example.jpg"
    image_id = 1

    full_path = os.path.join(tmpdir, filename)
    # save dummy image to file
    dummy_array = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    dummy_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    dummy_mask[1:501, 1:501] = np.ones((500, 500))
    Image.fromarray(dummy_array, 'RGB').save(full_path)
    with open(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
        # encoded_jpg_b = bytearray(encoded_jpg)
    key = hashlib.sha256(encoded_jpg).hexdigest()

    xmin = [0.25]
    xmax = [0.5]
    ymin = [0.25]
    ymax = [0.5]
    is_crowd = [False]
    category_names = [b'void']
    category_ids = [0]
    area = [16384]
    encoded_mask_png = []
    pil_image = Image.fromarray(dummy_mask, '1')
    output_io = io.BytesIO()
    pil_image.save(output_io, format='PNG')
    encoded_mask_png.append(output_io.getvalue())

    feature_dict = {
        'image/height':
            dataset_utils.int64_feature(image_height),
        'image/width':
            dataset_utils.int64_feature(image_width),
        'image/filename':
            dataset_utils.bytes_feature(filename.encode('utf8')),
        'image/source_id':
            dataset_utils.bytes_feature(str(image_id).encode('utf8')),
        'image/key/sha256':
            dataset_utils.bytes_feature(key.encode('utf8')),
        'image/encoded':
            dataset_utils.bytes_feature(encoded_jpg),
        'image/caption':
            dataset_utils.bytes_list_feature([]),
        'image/format':
            dataset_utils.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin':
            dataset_utils.float_list_feature(xmin),
        'image/object/bbox/xmax':
            dataset_utils.float_list_feature(xmax),
        'image/object/bbox/ymin':
            dataset_utils.float_list_feature(ymin),
        'image/object/bbox/ymax':
            dataset_utils.float_list_feature(ymax),
        'image/object/class/text':
            dataset_utils.bytes_list_feature(category_names),
        'image/object/class/label':
            dataset_utils.int64_list_feature(category_ids),
        'image/object/is_crowd':
            dataset_utils.int64_list_feature(is_crowd),
        'image/object/area':
            dataset_utils.float_list_feature(area),
    }
    if include_masks:
        feature_dict['image/object/mask'] = (
            dataset_utils.bytes_list_feature(encoded_mask_png))
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    # dump tfrecords
    tfrecords_dir = tmpdir.mkdir("tfrecords")
    dummy_tfrecords = str(tfrecords_dir.join('/dummy-001'))
    writer = tf.python_io.TFRecordWriter(str(dummy_tfrecords))
    writer.write(example.SerializeToString())
    writer.close()
    input_dataset = dataloader.InputReader(
        file_pattern=os.path.join(tfrecords_dir, "dummy*"),
        mode=tf.estimator.ModeKeys.TRAIN,
        use_fake_data=False,
        use_instance_mask=True,
        seed=123
    )

    dataset = input_dataset(
        params={
            "anchor_scale": 8.0,
            "aspect_ratios": [[1.0, 1.0], [1.4, 0.7], [0.7, 1.4]],
            "batch_size": 1,
            "gt_mask_size": 112,
            "image_size": [512, 512],
            "include_groundtruth_in_features": False,
            "augment_input_data": False,
            "max_level": 6,
            "min_level": 2,
            "num_classes": 1,
            "num_scales": 1,
            "rpn_batch_size_per_im": 256,
            "rpn_fg_fraction": 0.5,
            "rpn_min_size": 0.,
            "rpn_nms_threshold": 0.7,
            "rpn_negative_overlap": 0.3,
            "rpn_positive_overlap": 0.7,
            "rpn_post_nms_topn": 1000,
            "rpn_pre_nms_topn": 2000,
            "skip_crowd_during_training": True,
            "use_category": True,
            "visualize_images_summary": False,
            "n_workers": 1,
            "shuffle_buffer_size": 16,
            "prefetch_buffer_size": 1,
            "max_num_instances": 200
        }
    )
    dataset_iterator = dataset.make_initializable_iterator()
    X = dataset_iterator.get_next()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = False

    with tf.compat.v1.Session(config=config) as sess:
        sess.run(dataset_iterator.initializer)
        sess.run(tf.compat.v1.global_variables_initializer())
        x, y = sess.run(X)
        assert np.allclose(x['images'][0, 0, 0, :], [-2.1651785, -2.0357141, -1.8124998])
        assert np.allclose(y['gt_boxes'][0][0], [128, 128, 256, 256])
        assert len(y['gt_boxes'][0]) == dataloader_utils.MAX_NUM_INSTANCES
