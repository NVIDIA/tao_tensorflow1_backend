# Copyright 2019-2020 NVIDIA Corporation.  All rights reserved.
"""Utility class for performing TensorRT image inference."""

import os

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from nvidia_tao_tf1.cv.makenet.utils.preprocess_input import preprocess_input


class Inferencer(object):
    """Manages TensorRT objects for model inference."""

    def __init__(self, keras_model=None, batch_size=None, trt_engine_path=None,
                 infer_process_fn=None, class_mapping=None, threshold=0.3,
                 img_mean=None, keep_aspect_ratio=True, image_depth=8):
        """Initializes Keras / TensorRT objects needed for model inference.

        Args:
            keras_model (keras model or None): Keras model object for inference
            batch_size (int or None): an int if keras_model is present or using dynamic bs engine
            trt_engine_path (str or None): TensorRT engine path.
            infer_process_fn (Python function): takes in the Inferencer object (self) and the
                model prediction, returns list of length batch_size. Each element is of size (n, 6)
                where n is the number of boxes and for each box:
                    class_id, confidence, xmin, ymin, xmax, ymax
            class_mapping (dict): a dict mapping class_id to class_name
            threshold (float): confidence threshold to draw/label a bbox.
            image_depth(int): Bit depth of images(8 or 16).
        """

        self.infer_process_fn = infer_process_fn
        self.class_mapping = class_mapping

        if trt_engine_path is not None:
            # use TensorRT for inference

            # Import TRTInferencer only if it's a TRT Engine.
            # Note: import TRTInferencer after fork() or in MPI might fail.
            from nvidia_tao_tf1.cv.common.inferencer.trt_inferencer import TRTInferencer

            self.trt_inf = TRTInferencer(trt_engine_path, batch_size=batch_size)
            self.batch_size = self.trt_inf.max_batch_size
            self.model_input_height = self.trt_inf._input_shape[1]
            self.model_input_width = self.trt_inf._input_shape[2]
            img_channel = self.trt_inf._input_shape[0]
            self.pred_fn = self.trt_inf.infer_batch
        elif (keras_model is not None) and (batch_size is not None):
            # use keras model for inference
            self.keras_model = keras_model
            self.batch_size = batch_size
            img_channel = keras_model.layers[0].output_shape[-3]
            self.model_input_width = keras_model.layers[0].output_shape[-1]
            self.model_input_height = keras_model.layers[0].output_shape[-2]
            self.pred_fn = self.keras_model.predict
        else:
            raise ValueError("Need one of (keras_model, batch_size) and trt_engine_path.")
        if image_depth == 8:
            self.model_img_mode = 'RGB' if img_channel == 3 else 'L'
        elif image_depth == 16:
            # PIL int32 mode for 16-bit images
            self.model_img_mode = "I"
        else:
            raise ValueError(
                f"Unsupported image depth: {image_depth}, should be 8 or 16"
            )
        self.threshold = threshold
        assert self.threshold > 0, "Confidence threshold must be bigger than 0.0"
        assert self.threshold < 1, "Confidence threshold must be smaller than 1.0"
        if image_depth == 8:
            self.supported_img_format = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']
        else:
            # Only PNG can support 16-bit depth
            self.supported_img_format = ['.png', '.PNG']
        self.keep_aspect_ratio = keep_aspect_ratio
        self.img_mean = img_mean

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

        if self.keep_aspect_ratio:
            im = img.resize((new_w, new_h), Image.ANTIALIAS)
        else:
            im = img.resize((self.model_input_width, self.model_input_height), Image.ANTIALIAS)

        if im.mode in ('RGBA', 'LA') or \
                (im.mode == 'P' and 'transparency' in im.info) and \
                self.model_img_mode == 'L':

            # Need to convert to RGBA if LA format due to a bug in PIL
            im = im.convert('RGBA')
            inf_img = Image.new("RGBA", (self.model_input_width, self.model_input_height))
            inf_img.paste(im, (0, 0))
            inf_img = inf_img.convert(self.model_img_mode)
        else:
            inf_img = Image.new(
                self.model_img_mode,
                (self.model_input_width, self.model_input_height)
            )
            inf_img.paste(im, (0, 0))

        inf_img = np.array(inf_img).astype(np.float32)
        # Single channel image, either 8-bit or 16-bit
        if self.model_img_mode in ['L', 'I']:
            inf_img = np.expand_dims(inf_img, axis=2)
            inference_input = inf_img.transpose(2, 0, 1) - self.img_mean[0]
        else:
            inference_input = preprocess_input(inf_img.transpose(2, 0, 1),
                                               img_mean=self.img_mean)

        return img, float(orig_w)/new_w, inference_input

    def _get_bbox_and_kitti_label_single_img(
        self, img, img_ratio, y_decoded,
        is_draw_img, is_kitti_export
    ):
        """helper function to draw bbox on original img and get kitti label on single image.

        Note: img will be modified in-place.
        """
        kitti_txt = ""
        draw = ImageDraw.Draw(img)
        color_list = ['Black', 'Red', 'Blue', 'Gold', 'Purple']
        for i in y_decoded:
            if float(i[1]) < self.threshold:
                continue

            if self.keep_aspect_ratio:
                i[2:6] *= img_ratio
            else:
                orig_w, orig_h = img.size
                ratio_w = float(orig_w) / self.model_input_width
                ratio_h = float(orig_h) / self.model_input_height
                i[2] *= ratio_w
                i[3] *= ratio_h
                i[4] *= ratio_w
                i[5] *= ratio_h

            if is_kitti_export:
                kitti_txt += self.class_mapping[int(i[0])] + ' 0 0 0 ' + \
                    ' '.join([str(x) for x in i[2:6]])+' 0 0 0 0 0 0 0 ' + str(i[1])+'\n'
            if is_draw_img:
                draw.rectangle(
                    ((i[2], i[3]), (i[4], i[5])),
                    outline=color_list[int(i[0]) % len(color_list)]
                )
                # txt pad
                draw.rectangle(((i[2], i[3]), (i[2] + 100, i[3]+10)),
                               fill=color_list[int(i[0]) % len(color_list)])

                draw.text((i[2], i[3]), "{0}: {1:.2f}".format(self.class_mapping[int(i[0])], i[1]))

        return img, kitti_txt

    def _predict_batch(self, inf_inputs):
        '''function to predict a batch.'''

        y_pred = self.pred_fn(np.array(inf_inputs))
        y_pred_decoded = self.infer_process_fn(self, y_pred)
        return y_pred_decoded

    def _inference_single_img(self, img_in_path, img_out_path, label_out_path):
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
        img, kitti_txt = self._get_bbox_and_kitti_label_single_img(
            img, ratio, y_pred_decoded[0],
            img_out_path, label_out_path
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
            if os.path.splitext(label_out_path)[1].lower() != '.txt':
                raise NotImplementedError("only .txt is supported for label output.")
            open(label_out_path, 'w').write(kitti_txt)

    def _inference_folder(self, img_in_path, img_out_path, label_out_path):
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

            for img_path, base_name, ext in image_path_basename[
                batch_idx*self.batch_size:(batch_idx+1)*self.batch_size
            ]:
                base_names.append(base_name)
                img, ratio, inf_input = self._load_img(img_path)
                imgs.append(img)
                ratios.append(ratio)
                inf_inputs.append(inf_input)
                exts.append(ext)

            y_pred_decoded = self._predict_batch(inf_inputs)

            for idx, base_name in enumerate(base_names):
                img, kitti_txt = self._get_bbox_and_kitti_label_single_img(
                    imgs[idx], ratios[idx], y_pred_decoded[idx],
                    img_out_path, label_out_path)

                if img_out_path:
                    img.save(os.path.join(img_out_path, base_name+exts[idx]))
                if label_out_path:
                    open(os.path.join(label_out_path, base_name+'.txt'), 'w').write(kitti_txt)

    def infer(self, img_in_path, img_out_path, label_out_path):
        """Wrapper function."""

        if not os.path.exists(img_in_path):
            raise ValueError("Input path does not exist")
        if not (img_out_path or label_out_path):
            raise ValueError("At least one of image or label output path should be set")
        if os.path.isdir(img_in_path):
            self._inference_folder(img_in_path, img_out_path, label_out_path)
        else:
            self._inference_single_img(img_in_path, img_out_path, label_out_path)
