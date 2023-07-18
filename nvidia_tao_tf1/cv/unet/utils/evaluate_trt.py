# Copyright 2019-2020 NVIDIA Corporation.  All rights reserved.
"""UNet TensorRT Evaluation Derived from Inference."""

import os
import numpy as np
from tqdm import tqdm
from nvidia_tao_tf1.cv.unet.utils.inference_trt import Inferencer


class Evaluator(Inferencer):
    """Manages TensorRT objects for model evaluation."""

    def __init__(self, *args, **kwargs):
        """Init function."""

        super(Evaluator, self).__init__(*args, **kwargs)

    def _load_img_label(self, img_filename, mask_filename):
        """load an image and returns images and corresponding label numpy array.

        Args:
            img_filename (str): path to an image
            mask_filename (str): path to mask filename.
        Returns:
            inputs: Pre-processed numpy array
            labels: One hot encoded corresponding labels
        """

        inputs, labels, _ = self.dataset.read_image_and_label_tensors(img_filename,
                                                                      mask_filename)
        inputs, labels, _ = self.dataset.rgb_to_bgr_tf(inputs, labels)
        inputs, labels, _ = self.dataset.cast_img_lbl_dtype_tf(inputs, labels)
        inputs, labels, _ = self.dataset.resize_image_and_label_tf(inputs, labels)
        inputs, labels, _ = self.dataset.normalize_img_tf(inputs, labels)
        inputs, labels, _ = self.dataset.transpose_to_nchw(inputs, labels)
        inputs, labels, _ = self.dataset.prednn_categorize_label(inputs, labels)
        inputs = inputs.eval(session=self.session)
        labels = labels.eval(session=self.session)
        inputs = np.array(inputs)
        labels = np.array(labels)

        return inputs, labels

    def _predict_batch(self, inf_inputs, inf_labels):
        '''function to predict a batch and compute conf matrix.'''

        inf_inputs_np = np.array(inf_inputs)
        inf_labels_np = np.array(inf_labels)
        y_pred = self.pred_fn(inf_inputs_np)
        predictions_batch = self.eval_process_fn(y_pred, inf_labels_np)
        return predictions_batch

    def eval_process_fn(self, y_pred, inf_labels_np):
        '''Post process the TRT inference output by reshaping.'''

        predictions_batch = []

        for idx in range(y_pred[0].shape[0]):
            gt = inf_labels_np[idx, ...]
            pred = np.reshape(y_pred[0][idx, ...], (self.dataset.model_output_height,
                                                    self.dataset.model_output_width,
                                                    1))
            if self.activation == "sigmoid":
                pred = np.squeeze(pred, axis=-1)
                gt = np.squeeze(gt, axis=0)
                pred = np.where(pred > 0.5, 1, 0)
            else:
                gt = np.argmax(gt, axis=0)
            pred_flatten = pred.flatten()
            gt_flatten = gt.flatten()
            conf_matrix = self.compute_confusion_matrix(gt_flatten, pred_flatten)
            pred_dic = {"conf_matrix": conf_matrix}
            predictions_batch.append(pred_dic)
        return predictions_batch

    def compute_confusion_matrix(self, true, pred):
        '''Sklearn equivalent function that handles GT without 1 class.'''

        true = true.astype(np.int32)
        pred = pred.astype(np.int32)
        K = self.num_conf_mat_classes
        result = np.zeros((K, K))
        for i in range(len(true)):
            result[true[i]][pred[i]] += 1

        return result

    def _evaluate_folder(self, img_names_list, masks_names_list):
        """evaluate in a folder of images.

        Args:
            img_names_list: list of img names
            masks_names_list: list of mask names
        """

        predictions = []
        n_batches = (len(img_names_list) + self.batch_size - 1) // self.batch_size
        for batch_idx in tqdm(range(n_batches)):
            inf_inputs = []
            inf_labels = []
            for img_path, mask_path in zip(img_names_list[
                batch_idx*self.batch_size:(batch_idx+1)*self.batch_size
            ], masks_names_list[
                batch_idx*self.batch_size:(batch_idx+1)*self.batch_size
            ]):
                _, ext = os.path.splitext(img_path)
                if ext not in self.supported_img_format:
                    raise ValueError("Provided image format {} is not supported!".format(ext))
                inf_input, inf_label = self._load_img_label(img_path, mask_path)
                inf_labels.append(inf_label)
                inf_inputs.append(inf_input)
            y_pred_batch = self._predict_batch(inf_inputs, inf_labels)
            predictions += y_pred_batch

        return predictions

    def evaluate(self, image_names_list, masks_names_list):
        """Wrapper function for evaluation."""

        if not image_names_list or not masks_names_list:
            raise ValueError("Input images and Input masks should not"
                             "be empty for evaluation!")
        predictions = self._evaluate_folder(image_names_list,
                                            masks_names_list)
        self.session.close()
        return predictions
