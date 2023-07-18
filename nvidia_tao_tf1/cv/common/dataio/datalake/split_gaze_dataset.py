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
"""Splits driveix.gaze into train, validation and kpi tables in the datalake."""

import maglev
from maglev_sdk.spark.catalog import CatalogTable, SparkCatalog
from sklearn.model_selection import train_test_split
import yaml


def split_dataset(id_, train_size, kpi_users):
    """Splits driveix.gaze into train, validation, and kpi tables."""
    client = maglev.Client.default_service_client()
    spark = SparkCatalog.spark()

    gaze_tbl = CatalogTable(table="gaze", database="driveix")
    train_tbl = CatalogTable(table="gaze_train_{}".format(id_), database="driveix")
    validation_tbl = CatalogTable(table="gaze_validation_{}".format(id_), database="driveix")
    kpi_tbl = CatalogTable(table="gaze_kpi_{}".format(id_), database="driveix")

    catalog = SparkCatalog(client, spark)
    catalog.register(
        read_tables=[gaze_tbl], write_tables=[train_tbl, validation_tbl, kpi_tbl]
    )

    gaze_df = catalog.read(gaze_tbl)
    users = set(gaze_df.select("user_id").rdd.flatMap(lambda x: x).collect())
    kpi_users = set(kpi_users)

    # with remaining users, split into train and validation
    leftover_users = users - kpi_users
    train_users, validation_users = train_test_split(
        list(leftover_users), train_size=train_size, random_state=42
    )

    # get dfs for respective splits
    kpi_df = gaze_df[gaze_df.user_id.isin(kpi_users)]
    train_df = gaze_df[gaze_df.user_id.isin(train_users)]
    validation_df = gaze_df[gaze_df.user_id.isin(validation_users)]

    # Write tables
    catalog.write(kpi_tbl, kpi_df)
    catalog.write(train_tbl, train_df)
    catalog.write(validation_tbl, validation_df)


if __name__ == "__main__":
    yaml_path = "nvidia_tao_tf1/cv/common/dataio/datalake/config.yaml"
    with open(yaml_path, "r") as f:
        data = yaml.load(f)
        id_ = data["id_"]
        train_size = data["train_size"]
        kpi_users = data["kpi_users"]

    split_dataset(id_, train_size, kpi_users)
