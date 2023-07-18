# Copyright 2019-2020 NVIDIA Corporation.  All rights reserved.
"""UNet TensorRT inference."""
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm


class Inferencer(object):
    """Manages TensorRT objects for model inference."""

    def __init__(self, keras_model=None, batch_size=None, trt_engine_path=None,
                 dataset=None, activation="softmax", num_conf_mat_classes=None):
        """Initializes Keras / TensorRT objects needed for model inference.

        Args:
            keras_model (keras model or None): Keras model object for inference
            batch_size (int or None): an int if keras_model is present
            trt_engine_path (str or None): TensorRT engine path.
            dataset (class object): Dataset class object.
            activation (string): activation used in the model
        """

        if trt_engine_path is not None:
            # use TensorRT for inference
            # Import TRTInferencer only if it's a TRT Engine.
            # Note: import TRTInferencer after fork() or in MPI might fail.
            from nvidia_tao_tf1.cv.common.inferencer.trt_inferencer import TRTInferencer

            self.trt_inf = TRTInferencer(trt_engine_path, batch_size=batch_size)
            self.batch_size = self.trt_inf.max_batch_size
            self.trt_inf = TRTInferencer(trt_engine_path, batch_size=self.batch_size)
            self.model_input_height = self.trt_inf._input_shape[1]
            self.model_input_width = self.trt_inf._input_shape[2]
            self.pred_fn = self.trt_inf.infer_batch
        else:
            raise ValueError("Need trt_engine_path.")

        self.supported_img_format = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']
        self.dataset = dataset
        self.activation = activation
        self.num_conf_mat_classes = num_conf_mat_classes
        self.session = self.set_session()

    def set_session(self):
        '''Helper function to set TF operations to CPU.'''
        # Configuring tensorflow to use CPU so that is doesn't interfere
        # with tensorrt.
        device_count = {'GPU': 0, 'CPU': 1}
        session_config = tf.compat.v1.ConfigProto(
            device_count=device_count
        )
        session = tf.compat.v1.Session(
            config=session_config,
            graph=tf.get_default_graph()
        )

        return session

    def _load_img(self, img_filename):
        """load an image and returns the original image and a numpy array for model to consume.

        Args:
            img_filename (str): path to an image
        Returns:
            inputs: Pre-processed numpy array
        """

        inputs = self.dataset.read_image_and_label_tensors(img_filename)
        inputs = self.dataset.rgb_to_bgr_tf(inputs)
        inputs = self.dataset.cast_img_lbl_dtype_tf(inputs)
        inputs = self.dataset.resize_image_and_label_tf(inputs)
        inputs = self.dataset.normalize_img_tf(inputs)
        inputs = self.dataset.transpose_to_nchw(inputs)
        inputs = inputs.eval(session=self.session)
        inputs = np.array(inputs)
        return inputs

    def _predict_batch(self, inf_inputs):
        '''function to predict a batch.'''
        inf_inputs_np = np.array(inf_inputs)
        y_pred = self.pred_fn(inf_inputs_np)
        predictions_batch = self.infer_process_fn(y_pred)
        return predictions_batch

    def infer_process_fn(self, y_pred):
        '''Post process the TRT inference output by reshaping.'''

        predictions_batch = []
        for idx in range(y_pred[0].shape[0]):
            pred = np.reshape(y_pred[0][idx, ...], (self.dataset.model_output_height,
                                                    self.dataset.model_output_width,
                                                    1))
            pred = np.squeeze(pred, axis=-1)
            if self.activation == "sigmoid":
                pred = np.where(pred > 0.5, 1, 0)
            pred = pred.astype(np.uint8)
            pred_dic = {"logits": pred}
            predictions_batch.append(pred_dic)
        return predictions_batch

    def _inference_folder(self, img_names_list):
        """inference in a folder.

        Args:
            img_in_path: the input folder path for an image
        """

        predictions = []
        full_img_paths = []

        n_batches = (len(img_names_list) + self.batch_size - 1) // self.batch_size
        for batch_idx in tqdm(range(n_batches)):
            inf_inputs = []
            for img_path in img_names_list[
                batch_idx*self.batch_size:(batch_idx+1)*self.batch_size
            ]:
                _, ext = os.path.splitext(img_path)
                if ext not in self.supported_img_format:
                    raise ValueError("Provided image format {} is not supported!".format(ext))
                inf_input = self._load_img(img_path)
                inf_inputs.append(inf_input)
                full_img_paths.append(img_path)
            y_pred_batch = self._predict_batch(inf_inputs)
            predictions += y_pred_batch

        return predictions, full_img_paths

    def infer(self, image_names_list):
        """Wrapper function."""

        if not image_names_list:
            raise ValueError("Image input folder for inference should not be empty!")
        predictions, full_img_paths = self._inference_folder(image_names_list)
        self.session.close()
        return predictions, full_img_paths
