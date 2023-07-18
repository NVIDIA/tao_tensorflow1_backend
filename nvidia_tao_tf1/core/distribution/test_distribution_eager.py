import mock
from nvidia_tao_tf1.core import distribution

import tensorflow as tf

tf.compat.v1.enable_eager_execution()


@mock.patch(
    "nvidia_tao_tf1.core.distribution.distribution.tf.config.experimental." "set_visible_devices"
)
def test_tensorflow_eager_mode_init(mocked_set_visible_devices):
    # Eager mode based initialization uses explicit device set function.
    distribution.set_distributor(distribution.HorovodDistributor())
    mocked_set_visible_devices.assert_called_once()
