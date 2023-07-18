r"""Convert raw COCO dataset to TFRecord for object_detection."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections

import hashlib
import io
import json
import multiprocessing
import os
import numpy as np
import PIL.Image
from pycocotools import mask
from skimage import measure

import tensorflow as tf

from nvidia_tao_tf1.cv.common.dataset import dataset_util
from nvidia_tao_tf1.cv.common.dataset import label_map_util
import nvidia_tao_tf1.cv.common.logging.logging as status_logging


def create_tf_example(image,
                      bbox_annotations,
                      image_dir,
                      category_index,
                      include_masks=False,
                      inspect_mask=True):
    """Converts image and annotations to a tf.Example proto.

    Args:
        image: dict with keys:
        [u'license', u'file_name', u'coco_url', u'height', u'width',
        u'date_captured', u'flickr_url', u'id']
        bbox_annotations:
        list of dicts with keys:
        [u'segmentation', u'area', u'iscrowd', u'image_id',
        u'bbox', u'category_id', u'id']
        Notice that bounding box coordinates in the official COCO dataset are
        given as [x, y, width, height] tuples using absolute coordinates where
        x, y represent the top-left (0-indexed) corner.  This function converts
        to the format expected by the Tensorflow Object Detection API (which is
        which is [ymin, xmin, ymax, xmax] with coordinates normalized relative
        to image size).
        image_dir: directory containing the image files.
        category_index: a dict containing COCO category information keyed
        by the 'id' field of each category.  See the
        label_map_util.create_category_index function.
        include_masks: Whether to include instance segmentations masks
        (PNG encoded) in the result. default: False.
    Returns:
        example: The converted tf.Example
        num_annotations_skipped: Number of (invalid) annotations that were ignored.

    Raises:
        ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    image_height = image['height']
    image_width = image['width']
    filename = image['file_name']
    image_id = image['id']

    full_path = os.path.join(image_dir, filename)
    with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    key = hashlib.sha256(encoded_jpg).hexdigest()

    xmin = []
    xmax = []
    ymin = []
    ymax = []
    is_crowd = []
    category_names = []
    category_ids = []
    area = []
    encoded_mask_png = []
    num_annotations_skipped = 0
    log_warnings = {}
    box_oob = []
    mask_oob = []
    for object_annotations in bbox_annotations:
        object_annotations_id = object_annotations['id']
        (x, y, width, height) = tuple(object_annotations['bbox'])
        if width <= 0 or height <= 0 or x + width > image_width or y + height > image_height:
            num_annotations_skipped += 1
            box_oob.append(object_annotations_id)
            continue

        if width <= 1 and height <= 1:
            raise ValueError('Please check the "annotations" dict in annotations json file, the "bbox" or "segmentation" \
                should use absolute coordinate instead of normalized coordinate.')
        xmin.append(float(x) / image_width)
        xmax.append(float(x + width) / image_width)
        ymin.append(float(y) / image_height)
        ymax.append(float(y + height) / image_height)
        is_crowd.append(object_annotations['iscrowd'])
        category_id = int(object_annotations['category_id'])
        if category_id == 0:
            raise ValueError('Please check the "annotations" dict in annotations json file, make sure "category_id" index \
                starts from 1 (not 0)')
        if 0 in category_index:
            raise ValueError('Please check the "categories" dict in annotations json file, make sure "id" index \
                starts from 1 (not 0)')
        category_ids.append(category_id)
        category_names.append(category_index[category_id]['name'].encode('utf8'))
        area.append(object_annotations['area'])

        if include_masks:
            if 'segmentation' not in object_annotations:
                raise ValueError(
                    f"segmentation groundtruth is missing in object: {object_annotations_id}.")
            # pylygon (e.g. [[289.74,443.39,302.29,445.32, ...], [1,2,3,4]])
            if isinstance(object_annotations['segmentation'], list):
                rles = mask.frPyObjects(object_annotations['segmentation'],
                                        image_height, image_width)
                rle = mask.merge(rles)
            elif 'counts' in object_annotations['segmentation']:
                # e.g. {'counts': [6, 1, 40, 4, 5, 4, 5, 4, 21], 'size': [9, 10]}
                if isinstance(object_annotations['segmentation']['counts'], list):
                    rle = mask.frPyObjects(object_annotations['segmentation'],
                                           image_height, image_width)
                else:
                    rle = object_annotations['segmentation']
            else:
                raise ValueError('Please check the segmentation format.')
            binary_mask = mask.decode(rle)
            contours = measure.find_contours(binary_mask, 0.5)
            if inspect_mask:
                # check if mask is out of bound compared to bbox
                min_x, max_x = image_width + 1, -1
                min_y, max_y = image_height + 1, -1
                for cont in contours:
                    c = np.array(cont)
                    min_x = min(min_x, np.amin(c, axis=0)[1])
                    max_x = max(max_x, np.amax(c, axis=0)[1])
                    min_y = min(min_y, np.amin(c, axis=0)[0])
                    max_y = max(max_y, np.amax(c, axis=0)[0])
                xxmin, xxmax, yymin, yymax = \
                    float(x) - 1, float(x + width) + 1, float(y) - 1, float(y + height) + 1
                if xxmin > min_x or yymin > min_y or xxmax < max_x or yymax < max_y:
                    mask_oob.append(object_annotations_id)

            # if not object_annotations['iscrowd']:
            #     binary_mask = np.amax(binary_mask, axis=2)
            pil_image = PIL.Image.fromarray(binary_mask)
            output_io = io.BytesIO()
            pil_image.save(output_io, format='PNG')
            encoded_mask_png.append(output_io.getvalue())

    feature_dict = {
        'image/height':
            dataset_util.int64_feature(image_height),
        'image/width':
            dataset_util.int64_feature(image_width),
        'image/filename':
            dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id':
            dataset_util.bytes_feature(str(image_id).encode('utf8')),
        'image/key/sha256':
            dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded':
            dataset_util.bytes_feature(encoded_jpg),
        'image/format':
            dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin':
            dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax':
            dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin':
            dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax':
            dataset_util.float_list_feature(ymax),
        'image/object/class/text':
            dataset_util.bytes_list_feature(category_names),
        'image/object/class/label':
            dataset_util.int64_list_feature(category_ids),
        'image/object/is_crowd':
            dataset_util.int64_list_feature(is_crowd),
        'image/object/area':
            dataset_util.float_list_feature(area),
    }
    if include_masks:
        feature_dict['image/object/mask'] = (
            dataset_util.bytes_list_feature(encoded_mask_png))
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    if mask_oob or box_oob:
        log_warnings[image_id] = {}
        log_warnings[image_id]['box'] = box_oob
        log_warnings[image_id]['mask'] = mask_oob
    return key, example, num_annotations_skipped, log_warnings


def _pool_create_tf_example(args):
    return create_tf_example(*args)


def _load_object_annotations(object_annotations_file):
    with tf.io.gfile.GFile(object_annotations_file, 'r') as fid:
        obj_annotations = json.load(fid)

    images = obj_annotations['images']
    category_index = label_map_util.create_category_index(
        obj_annotations['categories'])

    img_to_obj_annotation = collections.defaultdict(list)
    tf.compat.v1.logging.info('Building bounding box index.')
    for annotation in obj_annotations['annotations']:
        image_id = annotation['image_id']
        img_to_obj_annotation[image_id].append(annotation)

    missing_annotation_count = 0
    for image in images:
        image_id = image['id']
        if image_id not in img_to_obj_annotation:
            missing_annotation_count += 1

    tf.compat.v1.logging.info('%d images are missing bboxes.', missing_annotation_count)

    return images, img_to_obj_annotation, category_index


def _merge_log(log_a, log_b):
    log_ab = log_a.copy()
    for k, v in log_b.items():
        if k in log_ab:
            log_ab[k] += v
        else:
            log_ab[k] = v
    return log_ab


def _create_tf_record_from_coco_annotations(object_annotations_file,
                                            image_dir, output_path, include_masks, num_shards):
    """Loads COCO annotation json files and converts to tf.Record format.

    Args:
        object_annotations_file: JSON file containing bounding box annotations.
        image_dir: Directory containing the image files.
        output_path: Path to output tf.Record file.
        include_masks: Whether to include instance segmentations masks
        (PNG encoded) in the result. default: False.
        num_shards: Number of output files to create.
    """

    tf.compat.v1.logging.info('writing to output path: %s', output_path)
    writers = [
        tf.io.TFRecordWriter(
            output_path + '-%05d-of-%05d.tfrecord' %
            (i, num_shards)) for i in range(num_shards)
    ]

    images, img_to_obj_annotation, category_index = (
        _load_object_annotations(object_annotations_file))

    pool = multiprocessing.Pool()
    total_num_annotations_skipped = 0
    log_total = {}
    for idx, (_, tf_example, num_annotations_skipped, log_warnings) in enumerate(
        pool.imap(_pool_create_tf_example, [(
            image,
            img_to_obj_annotation[image['id']],
            image_dir,
            category_index,
            include_masks) for image in images])):
        if idx % 100 == 0:
            tf.compat.v1.logging.info('On image %d of %d', idx, len(images))

        total_num_annotations_skipped += num_annotations_skipped
        log_total = _merge_log(log_total, log_warnings)
        writers[idx % num_shards].write(tf_example.SerializeToString())

    pool.close()
    pool.join()

    for writer in writers:
        writer.close()

    tf.compat.v1.logging.info(
        'Finished writing, skipped %d annotations.', total_num_annotations_skipped)
    return log_total


def main(args=None):
    """Convert COCO format json and images into TFRecords."""
    args = parse_command_line_arguments(args)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    status_file = os.path.join(args.results_dir, "status.json")
    status_logging.set_status_logger(
        status_logging.StatusLogger(
            filename=status_file,
            is_master=True,
            verbosity=1,
            append=True
        )
    )
    s_logger = status_logging.get_status_logger()
    s_logger.write(
        status_level=status_logging.Status.STARTED,
        message="Starting tfrecords conversion."
    )
    tag = args.tag or os.path.splitext(os.path.basename(args.annotations_file))[0]
    output_path = os.path.join(args.output_dir, tag)

    log_total = _create_tf_record_from_coco_annotations(
        args.annotations_file,
        args.image_dir,
        output_path,
        args.include_masks,
        num_shards=args.num_shards)

    if log_total:
        with open(os.path.join(args.output_dir, f'{tag}_warnings.json'), "w") as f:
            json.dump(log_total, f)
    s_logger.write(
        status_level=status_logging.Status.SUCCESS,
        message="Conversion finished successfully."
    )


def build_command_line_parser(parser=None):
    """Build the command line parser using argparse.

    Args:
        parser (subparser): Provided from the wrapper script to build a chained
                parser mechanism.
    Returns:
        parser
    """
    if parser is None:
        parser = argparse.ArgumentParser(
            prog='dataset_convert', description='Convert COCO format dataset to TFRecords.')

    parser.add_argument(
        '-i',
        '--image_dir',
        type=str,
        required=True,
        help='Path to the image directory.')
    parser.add_argument(
        '-a',
        '--annotations_file',
        type=str,
        required=True,
        help='Path to the annotation JSON file.')
    parser.add_argument(
        '-o',
        '--output_dir',
        type=str,
        required=True,
        help='Output directory where TFRecords are saved.'
    )
    parser.add_argument(
        '-r',
        '--results_dir',
        type=str,
        default='/tmp',
        required=False,
        help='Output directory where the status log is saved.'
    )
    parser.add_argument(
        '-t',
        '--tag',
        type=str,
        required=False,
        default=None,
        help='Tag for the converted TFRecords (e.g. train, val, test). \
              Default to the name of annotation file.'
    )
    parser.add_argument(
        '-s',
        '--num_shards',
        type=int,
        required=False,
        default=256,
        help='Number of shards.'
    )
    parser.add_argument(
        "--include_masks",
        action="store_true",
        default=True,
        help="Whether to include instance segmentation masks.")
    return parser


def parse_command_line_arguments(args=None):
    """Simple function to parse command line arguments."""
    parser = build_command_line_parser(args)
    return parser.parse_args(args)


if __name__ == '__main__':
    try:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        main()
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Dataset convert was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e
