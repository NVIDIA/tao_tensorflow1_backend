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

"""BpNet Inference definition."""

import copy
import json
import logging
import os
import cv2

import numpy as np
import tqdm

from nvidia_tao_tf1.core.coreobject import TAOObject
from nvidia_tao_tf1.cv.bpnet.dataloaders.pose_config import BpNetPoseConfig
from nvidia_tao_tf1.cv.bpnet.inferencer.postprocessor import BpNetPostprocessor
import nvidia_tao_tf1.cv.bpnet.inferencer.utils as inferencer_utils
from nvidia_tao_tf1.cv.common.utilities.tlt_utils import model_io

# Setup logger.
formatter = logging.Formatter(
    '%(levelname)-8s%(asctime)s | %(name)s: %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logging.basicConfig(
    level='INFO'
)
logging.getLogger().handlers = []
logging.getLogger().addHandler(handler)

logger = logging.getLogger(__name__)


class BpNetInferencer(TAOObject):
    """BpNet Inferencer Class."""

    def __init__(self,
                 model_full_path,
                 inference_spec,
                 experiment_spec,
                 key=None,
                 **kwargs):
        """Init.

        Args:
            model_full_path (string): Full path to the model.
            inference_spec (dict): Inference specification.
            experiment_spec (dict): Training experiment specification.
        """
        # Load the model.
        if model_full_path.endswith('.engine'):
            # Use TensorRT for inference
            # import TRTInferencer only if it's a TRT Engine.
            from nvidia_tao_tf1.cv.core.inferencer.trt_inferencer import TRTInferencer
            self.model = TRTInferencer(model_full_path)
            self.is_trt_model = True
            # NOTE: There is bug when using tensorflow graph alongside tensorrt engine excution
            # Model inference gives completely wrong results. So disable tf section when using
            # tensorrt engine for inference.
            self.use_tf_postprocess = False
        else:
            if key is None:
                raise ValueError("Missing key argument needed to load model!")
            self.model = model_io(model_full_path, enc_key=key)
            self.is_trt_model = False
            self.use_tf_postprocess = True
            # logger.info(self.model.summary())

        logger.info(model_full_path)
        logger.info("Successfully loaded {}".format(model_full_path))

        self._experiment_spec = experiment_spec
        self._input_shape = inference_spec.get('input_shape', [256, 256])
        self._keep_aspect_ratio_mode = inference_spec.get(
            'keep_aspect_ratio_mode', 'adjust_network_input')
        # Output blob that should be used for evaluation. The network may consist
        # of multiple stages of refinement and this option lets us pick the stage]
        # to evaluate our results. If None, last stage is picked.
        self._output_stage_to_use = inference_spec.get(
            'output_stage_to_use', None)
        # Threshold value to use for filtering peaks after Non-max supression.
        self.heatmap_threshold = inference_spec.get('heatmap_threshold', 0.1)
        # Threshold value to use for suppressing connection in part affinity
        # fields.
        self.paf_threshold = inference_spec.get('paf_threshold', 0.05)

        # Read params from the experiment spec
        self._channel_format = experiment_spec['model']['data_format']
        # Normalization params
        normalization_params = experiment_spec['dataloader']['normalization_params']
        self.normalization_scale = normalization_params['image_scale']
        self.normalization_offset = normalization_params['image_offset']
        # Output shape and stride
        self._train_model_target_shape = \
            experiment_spec['dataloader']['pose_config']['target_shape']
        image_dims = experiment_spec['dataloader']['image_config']['image_dims']
        self._train_model_input_shape = [
            image_dims['height'], image_dims['width']]
        self._model_stride = self._train_model_input_shape[0] // self._train_model_target_shape[0]

        # (fy, fy) factors by which the output blobs need to upsampled before post-processing.
        # If None, this will be the same as the stride value of the model.
        self.output_upsampling_factor = inference_spec.get(
            'output_upsampling_factor', [self._model_stride, self._model_stride])

        # Get pose config to generate the topology
        pose_config_path = experiment_spec['dataloader']['pose_config']['pose_config_path']
        self.bpnet_pose_config = BpNetPoseConfig(
            self._train_model_target_shape,
            pose_config_path
        )
        self.pose_config = self.bpnet_pose_config.pose_config
        self.topology = self.bpnet_pose_config.topology
        # Initialize visualization object
        self.visualizer = inferencer_utils.Visualizer(self.topology)
        # List of scales to use for multi-scale evaluation. Only used
        # when `multi_scale_inference` is True.
        self.multi_scale_inference = inference_spec.get(
            'multi_scale_inference', False)
        self.scales = inference_spec.get('scales', None)
        if self.scales is None:
            self.scales = [1.0]

        # Intialize results dictionary with pose config and inference spec
        self.results = copy.deepcopy(self.pose_config)
        self.results['inference_spec'] = inference_spec

        # Initialize post-processor
        self.bpnet_postprocessor = BpNetPostprocessor(
            self.topology,
            self.bpnet_pose_config.num_parts,
            use_tf_postprocess=self.use_tf_postprocess
        )

        self.valid_image_ext = ['jpg', 'jpeg', 'png']

        # Check valid cases for trt inference
        if self.is_trt_model:
            if self.multi_scale_inference:
                logger.warning("Multi-scale inference not supported for trt inference. "
                               "Switching to single scale!!")
                self.multi_scale_inference = False
            if self._keep_aspect_ratio_mode == "adjust_network_input":
                logger.warning("Keep aspect ratio mode `adjust_network_input` not supported"
                               " for trt inference. Switching to `pad_image_input`!!")
                self._keep_aspect_ratio_mode = "pad_image_input"

    def infer(self, input_tensor):
        """Run model prediction.

        Args:
            input_tensor (numpy.ndarray): Model input

        Returns:
            heatmap (numpy.ndarray): heatmap tensor of shape (H, W, C1)
            paf (numpy.ndarray): part affinity field tensor of shape (H, W, C2)
        """

        # create input tensor (1 x H x W x C)
        # NOTE: Assumes channels_last.
        input_tensor = np.transpose(np.float32(
            input_tensor[:, :, :, np.newaxis]), (3, 0, 1, 2))

        if self.is_trt_model:
            # Run model prediction using trt engine
            try:
                output_blobs = self.model.predict(input_tensor)
            except Exception as error:
                logger.error("TRT execution failed. Please ensure that the `input_shape` "
                             "matches the model input dims")
                logger.error(error)
                raise error
            output_blobs = list(output_blobs.values())
            assert len(output_blobs) == 2, "Number of outputs more than 2. Please verify."
            heatmap_idx, paf_idx = (-1, -1)
            for idx in range(len(output_blobs)):
                if output_blobs[idx].shape[-1] == self.bpnet_pose_config.num_heatmap_channels:
                    heatmap_idx = idx
                if output_blobs[idx].shape[-1] == self.bpnet_pose_config.num_paf_channels:
                    paf_idx = idx

            if heatmap_idx == -1 or paf_idx == -1:
                raise Exception("Please verify model outputs!")
            heatmap = np.squeeze(output_blobs[heatmap_idx])
            paf = np.squeeze(output_blobs[paf_idx])
        else:
            # Run model prediction using keras model
            output_blobs = self.model.predict(input_tensor)
            total_stages = len(output_blobs) // 2
            if self._output_stage_to_use is None:
                self._output_stage_to_use = total_stages
            heatmap = np.squeeze(output_blobs[(self._output_stage_to_use - 1)][0])
            paf = np.squeeze(
                output_blobs[total_stages + (self._output_stage_to_use - 1)][0])

        try:
            assert heatmap.shape[:-1] == paf.shape[:-1]
        except AssertionError as error:
            logger.error("Model outputs are not as expected. "
                         "The heatmaps and part affinity maps have the following "
                         "dimensions: {} and {}, whereas, the height and width of "
                         "both should be same. Ensure the model has been exported "
                         "correctly. (Hint: use --sdk_compatible_model only for "
                         "deployment)".format(heatmap.shape[:-1], paf.shape[:-1]))
            raise error

        return heatmap, paf

    def postprocess(self,
                    heatmap,
                    paf,
                    image,
                    scale_factor,
                    offset_factor):
        """Postprocess function.

        Args:
            heatmap (numpy.ndarray): heatmap tensor of shape (H, W, C1)
            paf (numpy.ndarray): part affinity field tensor of shape (H, W, C2)
            image (numpy.ndarray): Input image
            scale_factor (list): scale factor with format (fx, fy)
            offset_factor (list): offset factor with format (oy, ox)

        Returns:
            keypoints (list): List of list consiting of keypoints of every
                detected person.
            viz_image (numpy.ndarray): Image with skeleton overlay
        """

        # Find peak candidates
        peaks, _ = self.bpnet_postprocessor.find_peaks(heatmap)

        # Find connection candidates
        connection_all = self.bpnet_postprocessor.find_connections(
            peaks, paf, image.shape[1])

        # Connect the parts
        humans, candidate_peaks = self.bpnet_postprocessor.connect_parts(
            connection_all, peaks, self.topology)

        # Get final keypoint list and scores
        keypoints, scores = self.bpnet_postprocessor.get_final_keypoints(
            humans, candidate_peaks, scale_factor, offset_factor)

        # Visualize the results on the image
        viz_image = self.visualizer.keypoints_viz(image.copy(), keypoints)

        return keypoints, scores, viz_image

    def run_pipeline(self, image):
        """Run bodypose infer pipeline.

        Args:
            image (np.ndarray): Input image to run inference on.
                It is in BGR and (H, W, C) format.

        Returns:
            heatmap (np.ndarray): heatmap tensor of shape (H, W, C1)
                where C1 corresponds to num_parts + 1 (for background)
            paf (np.ndarray): part affinity field tensor of shape
                (H, W, C2) where C2 corresponds to num_connections * 2
            scale_factor (list): scale factor with format (fx, fy)
            offset_factor (list): offset factor with format (oy, ox)
        """

        # Preprocess the input with desired input shape and aspect ratio mode
        # Normalize the image with coeffs from training
        preprocessed_image, preprocess_params = inferencer_utils.preprocess(
            image,
            self._input_shape,
            self.normalization_offset,
            self.normalization_scale,
            keep_aspect_ratio_mode=self._keep_aspect_ratio_mode,
        )

        # Infer on the preprocessed input tensor
        heatmap, paf = self.infer(preprocessed_image)

        fy, fx = (
            self.output_upsampling_factor[0], self.output_upsampling_factor[1])
        heatmap = cv2.resize(
            heatmap, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
        paf = cv2.resize(
            paf, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)

        # Compute the scale factor and offset factor to bring the
        # keypoints back to original image space.
        # NOTE: scale format is (fx, fy)
        scale_factor = [
            1. / preprocess_params['scale'][0],
            1. / preprocess_params['scale'][1]]
        if self.output_upsampling_factor is not None:
            scale_factor[0] = scale_factor[0] * \
                (self._model_stride / self.output_upsampling_factor[1])
            scale_factor[1] = scale_factor[1] * \
                (self._model_stride / self.output_upsampling_factor[0])
        offset_factor = [-preprocess_params['offset']
                         [0], -preprocess_params['offset'][1]]

        return heatmap, paf, scale_factor, offset_factor

    def run_multi_scale_pipeline(self, image):
        """Run bodypose multi-scale infer pipeline.

        Args:
            image (np.ndarray): Input image to run inference on.
                It is in BGR and (H, W, C) format.

        Returns:
            final_heatmaps (np.ndarray): heatmap tensor of shape (H, W, C1)
                where C1 corresponds to num_parts + 1 (for background)
            final_pafs (np.ndarray): part affinity field tensor of shape
                (H, W, C2) where C2 corresponds to num_connections * 2
            scale_factor (list): scale factor with format (fx, fy)
            offset_factor (list): offset factor with format (oy, ox)
        """

        # Sort the scales
        self.scales.sort()
        max_scale_idx = len(self.scales) - 1

        results = {}
        # Iterate over the scales
        for idx, scale in enumerate(self.scales):
            # Get the scaled input shape
            scaled_input_shape = [
                self._input_shape[0] * scale,
                self._input_shape[1] * scale]
            # Preprocess the input with desired input shape and aspect ratio mode
            # Normalize the image with coeffs from training
            preprocessed_image, preprocess_params = inferencer_utils.preprocess(
                image,
                scaled_input_shape,
                self.normalization_offset,
                self.normalization_scale,
                keep_aspect_ratio_mode=self._keep_aspect_ratio_mode,
            )

            # Pad the image to account for stride
            padded_image, padding = inferencer_utils.pad_bottom_right(
                preprocessed_image, self._model_stride, (0, 0, 0))

            # Infer on the preprocessed input tensor
            heatmap, paf = self.infer(padded_image)

            results[idx] = {
                'scale': scale,
                'preprocessed_image_shape': preprocessed_image.shape[:2],
                'padded_image_shape': padded_image.shape[:2],
                'padding': padding,
                'preprocess_params': preprocess_params.copy(),
                'heatmap': heatmap,
                'paf': paf
            }

        # Resize the output layers to the largest scale network input size
        output_blob_shapes = results[max_scale_idx]['preprocessed_image_shape']
        # NOTE: For multi-scale inference, the output_sampling_factor is fixed to model stride
        # of the largest scale.
        output_upsampling_factor = [self._model_stride, self._model_stride]
        # Initialize final heatmaps.pafs that will be computed as a combination
        # of heatmaps/pafs at different scales.
        final_heatmaps = np.zeros(
            (output_blob_shapes[0], output_blob_shapes[1], 19), dtype=np.float32)
        final_pafs = np.zeros(
            (output_blob_shapes[0],
             output_blob_shapes[1],
             38),
            dtype=np.float32)
        for idx in results.keys():

            scale = results[idx]['scale']
            padded_image_shape = results[idx]['padded_image_shape']
            padding = results[idx]['padding']

            # Resize the heatmap and paf x the model stride
            fy, fx = (output_upsampling_factor[0], output_upsampling_factor[1])
            heatmap = cv2.resize(
                results[idx]['heatmap'],
                (0, 0),
                fx=fx,
                fy=fy,
                interpolation=cv2.INTER_CUBIC)
            paf = cv2.resize(
                results[idx]['paf'],
                (0, 0),
                fx=fx,
                fy=fy,
                interpolation=cv2.INTER_CUBIC)
            # Remove the padding from the heatmaps and paf.
            # This is equivalent to what was added to the image
            heatmap = heatmap[:padded_image_shape[0] - padding[2],
                              :padded_image_shape[1] - padding[3], :]
            paf = paf[:padded_image_shape[0] - padding[2],
                      :padded_image_shape[1] - padding[3], :]
            # Resize the heatmap and paf to the shape of preprocessed input
            # for the largest scale
            heatmap = cv2.resize(
                heatmap,
                (output_blob_shapes[1], output_blob_shapes[0]),
                interpolation=cv2.INTER_CUBIC)
            paf = cv2.resize(
                paf,
                (output_blob_shapes[1], output_blob_shapes[0]),
                interpolation=cv2.INTER_CUBIC)
            # Compute the average heatmaps and pafs
            final_heatmaps = final_heatmaps + heatmap / len(self.scales)
            final_pafs = final_pafs + paf / len(self.scales)
        # Compute the scale factor and offset factor to bring the
        # keypoints back to original image space
        preprocess_params = results[max_scale_idx]['preprocess_params']
        scale_factor = [
            1. / preprocess_params['scale'][0],
            1. / preprocess_params['scale'][1]]
        offset_factor = [-preprocess_params['offset']
                         [0], -preprocess_params['offset'][1]]

        return final_heatmaps, final_pafs, scale_factor, offset_factor

    def dump_results(self, results_dir, results):
        """Save the results.

        Args:
            results_dir (str): Path to the directory to save results.
            results (dict): results that is to be saved as json.
        """
        results_path = os.path.join(results_dir, 'detections.json')
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        f.close()

    def run(self,
            data,
            results_dir=None,
            image_root_path='',
            visualize=False,
            dump_visualizations=False):
        """Run bodypose infer pipeline.

        Args:
            data: Input data to run inference on. This could be
                a list or a dictionary.
            results_dir (str): directory path to save the results
            image_root_path (str): Root path of the images. If specified,
                it is appended in front of the image paths.
             visualize (bool): Option to enable visualization.
             dump_visualizations (bool): If enabled, saves images with
                inference visualization to `results/images_annotated` directory

        Returns:
            results (dict): Dictionary containing image paths and results.
        """
        if isinstance(data, list):
            self.results['images'] = [
                dict(full_image_path=img_path) for img_path in data]
        elif isinstance(data, dict):
            self.results.update(data)
            if data.get('images') is None:
                raise Exception("Verify input json format!")
        # Check whether to dump image visualizations.
        if dump_visualizations and not results_dir:
            logger.warning("No results_dir provided. Ignoring visualization dumping!")
            dump_visualizations = False
        if dump_visualizations:
            visualization_dir = os.path.join(results_dir, "images_annotated")
            if not os.path.exists(visualization_dir):
                os.makedirs(visualization_dir)

        for idx, data_point in enumerate(tqdm.tqdm(self.results['images'])):
            # Get full path and verify extension
            image_path = data_point['full_image_path']
            full_path = os.path.join(image_root_path, image_path)
            if not full_path.split('.')[-1].lower() in self.valid_image_ext:
                continue
            # Read image
            image = cv2.imread(full_path)
            if image is None:
                logger.error("Error reading image: {}".format(full_path))
                continue
            # Check if to do multi-scale inference
            if self.multi_scale_inference:
                heatmap, paf, scale_factor, offset_factor = self.run_multi_scale_pipeline(
                    image)
            else:
                heatmap, paf, scale_factor, offset_factor = self.run_pipeline(
                    image)
            # Post-process the heatmap and the paf to obtain the final parsed
            # skeleton results
            keypoints, scores, viz_image = self.postprocess(
                heatmap,
                paf,
                image,
                scale_factor,
                offset_factor
            )
            # Visualize the image
            if visualize:
                cv2.imshow('output', viz_image)
                cv2.waitKey(0)
            # Add the results to the results dict
            self.results['images'][idx]['keypoints'] = keypoints
            self.results['images'][idx]['scores'] = scores

            if dump_visualizations:
                self.results['images'][idx]['viz_id'] = idx
                # Save annotated image
                cv2.imwrite(os.path.join(visualization_dir, "{}.png".format(idx)), viz_image)

        # Dump the results to results dir
        if results_dir:
            self.dump_results(results_dir, self.results)

        return self.results
