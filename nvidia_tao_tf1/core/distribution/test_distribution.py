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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from imp import reload  # Python 2/3 support. pylint: disable=redefined-builtin, disable=W4901

import mock

from nvidia_tao_tf1.core import distribution

import pytest
import tensorflow as tf


def _has_hvd_support():
    try:
        distribution.hvd()
    except ImportError:
        return False
    return True


def test_get_and_set_distributor():
    """Test the distributor."""
    # Attempt to load Horovod at the beginning of the test
    # v.s. formerly during test collection. This is to avoid
    # Horovod getting in the way of our TensorFlow ungreedy
    # configuration in conftest.py.
    if not _has_hvd_support():
        raise pytest.skip("requires horovod")
    default_distributor = distribution.get_distributor()
    assert isinstance(default_distributor, distribution.Distributor)

    # Set the horovod distributor.
    hvd_distributor = distribution.HorovodDistributor()
    distribution.set_distributor(hvd_distributor)

    # Get the distributor we just set.
    distributor = distribution.get_distributor()
    assert hvd_distributor == distributor

    # Make sure to reload after this test, as the distributor is static.
    reload(distribution)


def test_master_rank():
    """Test the default and setting of the master rank."""
    distributor = distribution.Distributor()
    assert distributor._master_rank == 0
    assert distributor.size() == 1
    assert distributor.rank() == 0
    assert distributor.local_rank() == 0
    assert distributor.is_master()
    assert not distributor.is_multi_node()

    # Test that a master rank larger than the size throws an error.
    master_rank = 1
    with pytest.raises(ValueError):
        distributor = distribution.Distributor(master_rank=master_rank)

    # Check that that a current rank different from the master rank is not the master.
    distributor = distribution.Distributor()
    distributor._master_rank = 3
    assert not distributor.is_master()

    # Check the configuration returns the right object
    assert isinstance(distributor.get_config(), tf.compat.v1.ConfigProto)


# def test_shutdown_default_distributor():
#     """Test shutdown behavior for Distributor."""
#     distributor = distribution.Distributor()
#     with pytest.raises(SystemExit, match=r"*shutdown the distribution strategy.*"):
#         distributor.shutdown()


@pytest.mark.skipif(not _has_hvd_support(), reason="requires horovod")
def test_shutdown_horovod_distributor():
    """Test shutdown behavior for HorovodDistributor."""
    distributor = distribution.HorovodDistributor()
    with mock.patch.object(distribution.hvd(), "shutdown") as mocked_shutdown:
        distributor.shutdown()

    mocked_shutdown.assert_called_once()


# @mock.patch("nvidia_tao_tf1.core.distribution.distribution.tf.compat.v1.Session")
# def test_tensorflow_graph_mode_init(mocked_session):
#     # Graph mode based initialization requires tf.Session.
#     distribution.set_distributor(distribution.HorovodDistributor())
#     mocked_session.assert_called_once()
