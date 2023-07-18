# Copyright 2019-2020 NVIDIA Corporation.  All rights reserved.
"""MaskRCNN TensorRT inference."""
import argparse
from functools import partial
import json
import os

import numpy as np
from PIL import Image, ImageDraw
import PIL.ImageColor as ImageColor
import pycocotools.mask as maskUtils
from skimage import measure
from tqdm import tqdm

from nvidia_tao_tf1.cv.makenet.utils.preprocess_input import preprocess_input
from nvidia_tao_tf1.cv.mask_rcnn.utils.coco_metric import generate_segmentation_from_masks
from nvidia_tao_tf1.cv.mask_rcnn.utils.spec_loader import load_experiment_spec


class Inferencer(object):
    """Manages TensorRT objects for model inference."""

    def __init__(self, keras_model=None, batch_size=None, trt_engine_path=None,
                 infer_process_fn=None, class_mapping=None, threshold=0.6):
        """Initializes Keras / TensorRT objects needed for model inference.

        Args:
            keras_model (keras model or None): Keras model object for inference
            batch_size (int or None): an int if keras_model is present
            trt_engine_path (str or None): TensorRT engine path.
            infer_process_fn (Python function): takes in the Inferencer object (self) and the
                model prediction, returns list of length batch_size. Each element is of size (n, 6)
                where n is the number of boxes and for each box:
                    class_id, confidence, xmin, ymin, xmax, ymax
            class_mapping (dict): a dict mapping class_id to class_name
            threshold (float): confidence threshold to draw/label a bbox.
        """
        self.dump_coco = True
        self.infer_process_fn = infer_process_fn
        self.class_mapping = class_mapping

        if trt_engine_path is not None:
            # use TensorRT for inference

            # Import TRTInferencer only if it's a TRT Engine.
            # Note: import TRTInferencer after fork() or in MPI might fail.
            from nvidia_tao_tf1.cv.common.inferencer.trt_inferencer import TRTInferencer

            self.trt_inf = TRTInferencer(trt_engine_path)
            self.batch_size = self.trt_inf.trt_engine.max_batch_size
            self.model_input_height = self.trt_inf._input_shape[1]
            self.model_input_width = self.trt_inf._input_shape[2]
            img_channel = self.trt_inf._input_shape[0]
            self.pred_fn = self.trt_inf.infer_batch
        else:
            raise ValueError("Need trt_engine_path.")

        self.model_img_mode = 'RGB' if img_channel == 3 else 'L'
        self.threshold = threshold
        assert self.threshold > 0, "Confidence threshold must be bigger than 0.0"
        assert self.threshold < 1, "Confidence threshold must be smaller than 1.0"
        self.supported_img_format = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']

    def _load_img(self, img_path):
        """load an image and returns the original image and a numpy array for model to consume.

        Args:
            img_path (str): path to an image
        Returns:
            img (PIL.Image): PIL image of original image.
            ratio (float): resize ratio of original image over processed image
            inference_input (array): numpy array for processed image
        """

        img = Image.open(img_path)
        orig_w, orig_h = img.size
        ratio = min(self.model_input_width/float(orig_w), self.model_input_height/float(orig_h))

        # do not change aspect ratio
        new_w = int(round(orig_w*ratio))
        new_h = int(round(orig_h*ratio))
        # resize and pad
        im = img.resize((new_w, new_h), Image.ANTIALIAS)
        inf_img = Image.new(
            self.model_img_mode,
            (self.model_input_width, self.model_input_height)
        )
        inf_img.paste(im, (0, 0))
        inf_img = np.array(inf_img).astype(np.float32)
        inference_input = preprocess_input(inf_img.transpose(2, 0, 1), mode="torch")

        return img, float(orig_w)/new_w, inference_input

    def generate_annotation_single_img(self, img_in_path, img, img_ratio,
                                       y_decoded, y_mask, is_draw_bbox,
                                       is_draw_mask, is_label_export):
        """helper function to draw bbox on original img and get kitti label on single image.

        Note: img will be modified in-place.
        """
        kitti_txt = ''
        json_txt = []
        draw = ImageDraw.Draw(img)
        ww, hh = img.size
        color_list = ['Black', 'Red', 'Blue', 'Gold', 'Purple']
        for idx, i in enumerate(y_decoded):
            if float(i[-1]) < self.threshold:
                continue
            i[0:4] *= img_ratio
            ii = i[:4].astype(np.int)
            ii[0] = min(max(0, ii[0]), hh)
            ii[1] = min(max(0, ii[1]), ww)
            ii[2] = max(min(hh, ii[2]), 0)
            ii[3] = max(min(ww, ii[3]), 0)

            if (ii[2] - ii[0]) <= 0 or (ii[3] - ii[1]) <= 0:
                continue

            if is_draw_bbox:
                draw.rectangle(
                    ((ii[1], ii[0]), (ii[3], ii[2])),
                    outline=color_list[int(i[-2]) % len(color_list)]
                )
                # txt pad
                draw.rectangle(((ii[1], ii[0]), (ii[1] + 100, ii[0]+10)),
                               fill=color_list[int(i[-2]) % len(color_list)])

                if self.class_mapping:
                    draw.text(
                        (ii[1], ii[0]), "{0}: {1:.2f}".format(
                            self.class_mapping[int(i[-2])], i[-1]))

            # Compute segms from masks
            detected_boxes = np.expand_dims(
                [i[1], i[0], i[3] - i[1], i[2] - i[0]],
                axis=0)
            masks = np.expand_dims(y_mask[idx, int(i[-2]), :, :], axis=0)
            segms = generate_segmentation_from_masks(
                masks, detected_boxes,
                image_height=hh,
                image_width=ww,
                is_image_mask=False)
            segms = segms[0, :, :]

            if is_draw_mask:
                img = self.draw_mask_on_image_array(
                    img, segms, color=color_list[int(i[-2] % len(color_list))], alpha=0.4)
                draw = ImageDraw.Draw(img)
            if is_label_export:
                # KITTI export is for INTERNAL only
                if self.dump_coco:
                    json_obj = {}
                    hhh, www = ii[3] - ii[1], ii[2] - ii[0]
                    json_obj['area'] = int(www * hhh)
                    json_obj['is_crowd'] = 0
                    json_obj['image_id'] = os.path.basename(img_in_path)
                    json_obj['bbox'] = [int(ii[1]), int(ii[0]), int(hhh), int(www)]
                    json_obj['id'] = idx
                    json_obj['category_id'] = int(i[-2])
                    json_obj['score'] = float(i[-1])
                    # convert mask to polygon
                    use_rle = True
                    if use_rle:
                        # use RLE
                        encoded_mask = maskUtils.encode(
                            np.asfortranarray(segms.astype(np.uint8)))
                        encoded_mask['counts'] = encoded_mask['counts'].decode('ascii')
                        json_obj["segmentation"] = encoded_mask
                    else:
                        # use polygon
                        json_obj["segmentation"] = []
                        contours = measure.find_contours(segms, 0.5)
                        for contour in contours:
                            contour = np.flip(contour, axis=1)
                            segmentation = contour.ravel().tolist()
                            json_obj["segmentation"].append(segmentation)
                    json_txt.append(json_obj)
                else:
                    if i[-1] >= self.threshold:
                        kitti_txt += self.class_mapping[int(i[-2])] + ' 0 0 0 ' + \
                            ' '.join(str(x) for x in [ii[1], ii[0], ii[3], ii[2]]) + \
                            ' 0 0 0 0 0 0 0 ' + \
                            str(i[-1])+'\n'
        return img, json_txt if self.dump_coco else kitti_txt

    def draw_mask_on_image_array(self, pil_image, mask, color='red', alpha=0.4):
        """Draws mask on an image.

        Args:
            image: PIL image (img_height, img_width, 3)
            mask: a uint8 numpy array of shape (img_height, img_width) with
                values of either 0 or 1.
            color: color to draw the keypoints with. Default is red.
            alpha: transparency value between 0 and 1. (default: 0.4)

        Raises:
            ValueError: On incorrect data type for image or masks.
        """
        rgb = ImageColor.getrgb(color)
        solid_color = np.expand_dims(np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
        pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
        pil_mask = Image.fromarray(np.uint8(255.0 * alpha * mask)).convert('L')
        pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
        return pil_image

    def _predict_batch(self, inf_inputs):
        '''function to predict a batch.'''
        y_pred = self.pred_fn(np.array(inf_inputs))
        if self.infer_process_fn:
            return self.infer_process_fn(y_pred)
        return y_pred

    def _inference_single_img(self, img_in_path, img_out_path, label_out_path, draw_mask):
        """inference for a single image.

        Args:
            img_in_path: the input path for an image
            img_out_path: the output path for the image
            label_out_path: the output path for the label
        """
        if os.path.splitext(img_in_path)[1] not in self.supported_img_format:
            raise NotImplementedError(
                "only "+' '.join(self.supported_img_format)+' are supported for input.')

        img, ratio, inf_input = self._load_img(img_in_path)
        y_pred_decoded = self._predict_batch([inf_input])

        img, json_txt = self.generate_annotation_single_img(
            img_in_path, img, ratio, y_pred_decoded[0][0, ...],  y_pred_decoded[1][0, ...],
            img_out_path, draw_mask, label_out_path
        )
        if img_out_path:
            if os.path.splitext(img_out_path)[1] not in self.supported_img_format:
                raise NotImplementedError(
                    "only "+' '.join(self.supported_img_format)+' are supported for image output.')
            try:
                img.save(img_out_path)
            except Exception:
                img.convert("RGB").save(img_out_path)

        if label_out_path:
            if os.path.splitext(label_out_path)[1].lower() != '.json':
                raise NotImplementedError("only .json is supported for label output.")
            with open(label_out_path, "w") as json_file:
                json.dump(json_txt, json_file, indent=4, sort_keys=True)

    def _inference_folder(self, img_in_path, img_out_path, label_out_path, draw_mask):
        """inference in a folder.

        Args:
            img_in_path: the input folder path for an image
            img_out_path: the output folder path for the image
            label_out_path: the output path for the label
        """

        # Create output directories
        if img_out_path and not os.path.exists(img_out_path):
            os.mkdir(img_out_path)
        if label_out_path and not os.path.exists(label_out_path):
            os.mkdir(label_out_path)

        image_path_basename = []
        for img_path in os.listdir(img_in_path):
            base_name, ext = os.path.splitext(img_path)
            if ext in self.supported_img_format:
                image_path_basename.append((os.path.join(img_in_path, img_path), base_name, ext))

        n_batches = (len(image_path_basename) + self.batch_size - 1) // self.batch_size

        for batch_idx in tqdm(range(n_batches)):
            imgs = []
            ratios = []
            inf_inputs = []
            base_names = []
            exts = []
            full_img_paths = []

            for img_path, base_name, ext in image_path_basename[
                batch_idx*self.batch_size:(batch_idx+1)*self.batch_size
            ]:
                base_names.append(base_name)
                img, ratio, inf_input = self._load_img(img_path)
                imgs.append(img)
                ratios.append(ratio)
                inf_inputs.append(inf_input)
                exts.append(ext)
                full_img_paths.append(img_path)
            y_pred_decoded = self._predict_batch(inf_inputs)

            for idx, base_name in enumerate(base_names):
                img, label_dump = self.generate_annotation_single_img(
                    full_img_paths[idx], imgs[idx], ratios[idx],
                    y_pred_decoded[0][idx, ...], y_pred_decoded[1][idx, ...],
                    img_out_path, draw_mask, label_out_path)

                if img_out_path:
                    try:
                        img.save(os.path.join(img_out_path, base_name+exts[idx]))
                    except Exception:
                        img.convert("RGB").save(os.path.join(img_out_path, base_name+exts[idx]))

                if label_out_path:
                    if self.dump_coco:
                        with open(os.path.join(label_out_path, base_name+'.json'), "w") as json_f:
                            json.dump(label_dump, json_f, indent=4, sort_keys=True)
                    else:
                        with open(os.path.join(label_out_path, base_name+'.txt'), 'w') as kitti_f:
                            kitti_f.write(label_dump)

    def infer(self, img_in_path, img_out_path, label_out_path, draw_mask=False):
        """Wrapper function."""

        if not os.path.exists(img_in_path):
            raise ValueError("Input path does not exist")
        if not (img_out_path or label_out_path):
            raise ValueError("At least one of image or label output path should be set")
        if os.path.isdir(img_in_path):
            self._inference_folder(img_in_path, img_out_path, label_out_path, draw_mask)
        else:
            self._inference_single_img(img_in_path, img_out_path, label_out_path, draw_mask)


def postprocess_fn(y_pred, nms_size, mask_size, n_classes):
    """Proccess raw output from TRT engine."""
    y_detection = y_pred[0].reshape((-1, nms_size, 6))
    y_mask = y_pred[1].reshape((-1, nms_size, n_classes, mask_size, mask_size))
    y_mask[y_mask < 0] = 0
    return [y_detection, y_mask]


def build_command_line_parser(parser=None):
    '''Parse command line arguments.'''
    if parser is None:
        parser = argparse.ArgumentParser(description='MaskRCNN TensorRT Inference Tool')
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        required=True,
                        help='Path to a TensorRT engine.')
    parser.add_argument('-i',
                        '--in_image_path',
                        required=True,
                        type=str,
                        help='The path to input image or directory.')
    parser.add_argument('-o',
                        '--out_image_path',
                        type=str,
                        default=None,
                        help='The path to output image or directory.')
    parser.add_argument('-e',
                        '--config_path',
                        required=True,
                        type=str,
                        help='Path to an experiment spec file for training.')
    parser.add_argument('-c',
                        '--class_label',
                        type=str,
                        default=None,
                        help='The path to the class label file.')
    parser.add_argument('-l',
                        '--out_label_path',
                        type=str,
                        default=None,
                        help='The path to the output txt file or directory.')
    parser.add_argument('-t',
                        '--threshold',
                        type=float,
                        default=0.6,
                        help='Class confidence threshold for inference.')
    parser.add_argument('--include_mask', action='store_true',
                        help="Whether to draw masks.")
    return parser


def parse_command_line(args):
    """Simple function to build and parser command line args."""
    parser = build_command_line_parser(parser=None)
    return parser.parse_args(args)


def get_label_dict(label_txt):
    """Create label dict from txt file."""
    with open(label_txt, 'r') as f:
        labels = f.readlines()
        return {i + 1: label[:-1] for i, label in enumerate(labels)}


def main(args=None):
    """Run TRT inference."""
    arguments = parse_command_line(args)
    spec = load_experiment_spec(arguments.config_path)
    mask_size = int(spec.maskrcnn_config.mrcnn_resolution)
    nms_size = int(spec.maskrcnn_config.test_detections_per_image)
    assert nms_size > 0, "test_detections_per_image must be greater than 0."
    assert mask_size > 1, "mask_size must be greater than 1."
    n_classes = int(spec.data_config.num_classes)
    assert n_classes > 1, "Please verify the num_classes in the spec file."
    trt_output_process_fn = partial(postprocess_fn, mask_size=mask_size,
                                    n_classes=n_classes, nms_size=nms_size)
    class_mapping = {}
    if arguments.class_label:
        class_mapping = get_label_dict(arguments.class_label)
    # Initialize inferencer
    inferencer = Inferencer(trt_engine_path=arguments.model,
                            infer_process_fn=trt_output_process_fn,
                            class_mapping=class_mapping,
                            threshold=arguments.threshold)

    inferencer.infer(arguments.in_image_path,
                     arguments.out_image_path,
                     arguments.out_label_path,
                     arguments.include_mask)


if __name__ == '__main__':
    main()
