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

'''build input dataset for training or evaluation.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf1.cv.retinanet.dataio.coco_loader import RetinaCocoDataSequence
from nvidia_tao_tf1.cv.retinanet.dataio.kitti_loader import RetinaKittiDataSequence
from nvidia_tao_tf1.cv.ssd.builders.dalipipeline_builder import SSDDALIDataset


def build(experiment_spec,
          training=True,
          root_path=None,
          device_id=0,
          shard_id=0,
          num_shards=1,
          use_dali=False):
    '''
    Build a model for training with or without training tensors.

    For inference, this function can be used to build a base model, which can be passed into
    eval_builder to attach a decode layer.
    '''
    supported_data_loader = {'kitti': RetinaKittiDataSequence,
                             'coco': RetinaCocoDataSequence}
    # train/val batch size
    train_bs = experiment_spec.training_config.batch_size_per_gpu
    val_bs = experiment_spec.eval_config.batch_size
    dl_type = experiment_spec.dataset_config.type or 'kitti'
    assert dl_type in list(supported_data_loader.keys()), \
        "dataloader type is invalid. Only coco and kitti are supported."
    if use_dali:
        dataset = SSDDALIDataset(experiment_spec=experiment_spec,
                                 device_id=device_id,
                                 shard_id=shard_id,
                                 num_shards=num_shards)
    else:
        dataset = supported_data_loader[dl_type](
            dataset_config=experiment_spec.dataset_config,
            augmentation_config=experiment_spec.augmentation_config,
            batch_size=train_bs if training else val_bs,
            is_training=training,
            encode_fn=None,
            root_path=root_path)
    return dataset
