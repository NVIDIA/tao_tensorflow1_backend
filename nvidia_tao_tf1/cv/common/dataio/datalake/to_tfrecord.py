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
"""Converts a gaze table from the datalake to TFRecord."""

import argparse
import maglev
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def determine_tf_type(value):
    """Convert value to respective TF Feature type."""
    if type(value) == str:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))
    if type(value) == float:
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    if type(value) == bool:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    if type(value) == np.ndarray:
        assert value.dtype == np.float64
        return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))
    return None


def write_from_pandas(df, writer):
    """Writes a TFRecord from a Pandas DF."""
    columns = df.columns
    for _, row in df.iterrows():
        features_dict = {}
        for col in columns:
            features_dict[col] = determine_tf_type(row[col])
        example = tf.train.Example(features=tf.train.Features(feature=features_dict))
        writer.write(example.SerializeToString())


def write_to_tfrecord(tbl_name, tf_folder):
    "Converts a gaze table to a TFRecord."
    client = maglev.Client.default_service_client()
    tbl = client.get_table(table=tbl_name, database="driveix")
    tf_filepath = "{}/{}.tfrecords".format(tf_folder, tbl_name)
    writer = tf.python_io.TFRecordWriter(tf_filepath)

    for pq in tqdm(tbl._files):
        # convert every partition parquet into pandas df and write to TFRecord
        df = pq.to_pandas()
        write_from_pandas(df, writer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="/home/driveix.cosmos639/eddiew/tfrecords/",
        help="Folder path to save generated TFrecord",
    )
    parser.add_argument(
        "-t",
        "--tbl",
        type=str,
        default="gaze_kpi_1",
        help="Table from driveix to convert.",
    )
    args = parser.parse_args()

    tbl_name = args.tbl
    tf_folder = args.path

    write_to_tfrecord(tbl_name, tf_folder)
