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
"""Process-distribution functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf

try:
    from functools import lru_cache
except ImportError:
    from functools32 import lru_cache

if os.environ.get("TF_KERAS"):
    from tensorflow import keras  # pylint: disable=C0412
else:
    import keras


class Distributor(object):
    """Base distributor object.

    This base distributor object behaves as if only one process is running, without any actual
    distribution.
    """

    def __init__(self, master_rank=0, per_process_gpu_memory_fraction=None):
        """__init__ method.

        Args:
            master_rank (int): specifies the intended rank of the master.
            per_process_gpu_memory_fraction (float): fraction of GPU memory to reserve for
                                                     TensorFlow. Ignored if unset.
        """
        if master_rank >= self.size():
            raise ValueError(
                "Requested a master rank of {}, which should be smaller than "
                "the distribution size ({}).".format(master_rank, self.size())
            )
        self._master_rank = master_rank
        self._per_process_gpu_memory_fraction = per_process_gpu_memory_fraction

    def get_config(self):
        """Get configuration to pass to tf Session."""
        config = tf.compat.v1.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.visible_device_list = str(self.local_rank())
        config.gpu_options.allow_growth = True
        if self._per_process_gpu_memory_fraction is not None:
            config.gpu_options.per_process_gpu_memory_fraction = (
                self._per_process_gpu_memory_fraction
            )
        return config

    def size(self):
        """Get the size.

        Returns:
            Total amount of processes.
        """
        return 1

    def local_size(self):
        """Get the local size.

        NOTE: 'local' is defined as 'inside the current node'.

        Returns:
            Total amount of distributed processes locally (int).
        """
        return 1

    def rank(self):
        """Get the rank.

        Returns:
            Global rank (index).
        """
        return 0

    def local_rank(self):
        """Get the local rank.

        NOTE: 'local' is defined as 'inside the current node'.

        Returns:
            Local rank (int).
        """
        return 0

    def is_multi_node(self):
        """Check if we are running distribution over multiple nodes.

        Returns:
            A boolean indicating if we have processes running on (True) multiple nodes or
                a single node (False).
        """
        return self.size() != self.local_size()

    def is_master(self):
        """Check if the current process is the master process.

        Returns:
            A boolean indicating if the current process is the master process.
        """
        return self._master_rank == self.rank()

    def is_distributed(self):
        """Check if we're running in distributed mode.

        Returns:
            A boolean indicating if we in a distributed setting.
        """
        return self.size() > 1

    def distributed_seed(self, seed):
        """Get a distributed seed, to avoid the same seed per process.

        Args:
            seed (int): the current fixed seed.

        Returns:
            A perturbed seed depending on rank, as a function of the input seed.
        """
        if seed is None:
            return seed
        return seed + self.rank()

    def broadcast_global_variables(self):
        """Broadcast variables from master rank to all other processes."""
        pass

    def distribute_optimizer(self, optimizer):
        """Distribute the input optimizer."""
        return optimizer

    def allreduce(self, value):
        """Allreduce operation that sums value across GPUs.

        Args:
            value: value to be summed.
        Returns:
            Sum of value across all GPUs.
        """
        return value

    def shutdown(self):
        """Shut the distribution strategy down."""
        sys.exit("A request has been made to shutdown the distribution strategy.")

    def distributed_gradient_tape(self, tape):
        """Add distributed GradientTape for tensorflow eager mode.

        Args:
            tape (tf.GradientTape): Recorded operations of automatic differentiation.
        Returns:
            tape (tf.GradientTape): The input tape wrapped in a tape that takes
                care of the distribution.
        """
        return tape

    def broadcast_variables(self, variables, root_rank=0):
        """Broadcast variables from root_rank to other ranks.

        Args:
            variables (tf.Variable): Tensorflow variables that need to be broadcast.
            root_rank (int): From which rank the variables need to be broadcast.
        """
        pass


@lru_cache()
def hvd():
    """Lazily load and return the (cached) horovod module."""
    import horovod.tensorflow as hvd

    return hvd


class HorovodDistributor(Distributor):
    """Horovod distributor object.

    This object wraps several horovod functions and provides some utilities. Notice that the
    horovod module is lazily loaded so that it is only a dependency to maglev when you want to
    use the HorovodDistributor object.

    The documentation of horovod is hosted on `<https://github.com/uber/horovod>`_. Horovod's core
    principles are based on
    `MPI <https://github.com/uber/horovod/blob/master/docs/concepts.md>`_.

    This distributor parallelizes your training script by using custom Tensorflow operations
    and leveraging MPI. The parallelization of your script is done through launching your script
    using ``MPI``, for example using OpenMPI's `mpirun`. So instead of::

        python train.py

    One would launch 4 local processes using::

        mpirun -np 4 python train.py

    Where ``train.py`` should use the current distributor and its methods. Horovod will then
    map each of the 4 local processes on different GPUs. If you do not want to use the horovod
    distributor, but want to have your code distributor-enabled, you can just use the base
    :any:`Distributor` class, that is undistributed by default and acts as
    a passthrough.
    """

    def __init__(self, **kwargs):
        """__init__ method.

        Initializes horovod, and pins GPUs to the current session. This initialization should
        happen before any of the other distribution functions are imported, used or called.

        Args:
            **kwargs: arbitrary keyword arguments.
        """
        hvd().init()
        super(HorovodDistributor, self).__init__(**kwargs)
        if not tf.executing_eagerly():
            # Pin GPU to be used to process local rank (one GPU per process)
            session = tf.compat.v1.Session(config=self.get_config())
            keras.backend.set_session(session)
        else:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if self.local_rank() >= len(gpus):
                raise ValueError(
                    "Requesting a local rank {}, which should be"
                    "smaller than the gpu count {}.".format(
                        self.local_rank(), len(gpus)
                    )
                )
            gpu = gpus[self.local_rank()]
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpu, "GPU")

    def size(self):
        """Get the size.

        Returns:
            Total amount of distributed processes (int).
        """
        return hvd().size()

    def local_size(self):
        """Get the local size.

        NOTE: 'local' is defined as 'inside the current node'.

        Returns:
            Total amount of distributed processes locally (int).
        """
        return hvd().local_size()

    def rank(self):
        """Get the rank.

        The rank can be considered the current global unique rank (or index), where all nodes and
        all processes within a node are considered.

        Returns:
            Rank (int)
        """
        return hvd().rank()

    def local_rank(self):
        """Get the local rank.

        NOTE: 'local' is defined as 'inside the current node'.

        Returns:
            Local rank (int).
        """
        return hvd().local_rank()

    def broadcast_global_variables(self):
        """Broadcast variables from master rank to all other processes.

        This function should be called after all variables are created, but before evaluating any
        operations that require distribution, like allreduce or using the distributed optimizer.
        """
        broadcast_ops = hvd().broadcast_global_variables(self._master_rank)
        keras.backend.get_session().run(broadcast_ops)

    def broadcast_global_variables_hook(self):
        """Broadcast variables from master rank to all other processes.

        BroadcastGlobalVariablesHook broadcasts initial variable states from rank 0 to all other
        processes. This is necessary to ensure consistent initialization of all workers when
        training is started with random weights or restored from a checkpoint.

        Returns:
            A instance that inherits from a `tf.estimator.SessionRunHook` object that takes care of
                variable initialization across processes.
        """
        # Note that the class inherits from a lazy horovod import, which is why it is defined
        # inline.
        class _ScopedBroadcastGlobalVariablesHook(hvd().BroadcastGlobalVariablesHook):
            """Class that wraps the global variables broadcast hook into one with op scopes."""

            def begin(self, *args, **kwargs):
                """Call begin by forwarding the begin call within a tf name scope."""
                with tf.compat.v1.name_scope("horovod_broadcast"):
                    super(_ScopedBroadcastGlobalVariablesHook, self).begin(
                        *args, **kwargs
                    )

        return _ScopedBroadcastGlobalVariablesHook(0)

    def distribute_optimizer(self, optimizer):
        """Distribute the input optimizer.

        Args:
            optimizer: a tensorflow optimizer object to be distributed.
        Returns:
            The input optimizer wrapped in an optimizer that takes care of the distribution.
        """
        hvd_optimizer = hvd().DistributedOptimizer(optimizer)
        return hvd_optimizer

    def allreduce(self, value):
        """Allreduce operation that sums value across GPUs.

        Args:
            value: value to be summed.
        Returns:
            Sum of value across all GPUs.
        """
        return hvd().allreduce(value)

    def shutdown(self):
        """Shut horovod down.

        Note that while this does not exit the process, if, later down the line, another process
        sends tensors to be reduced / gathered through horovod, the latter will detect that it
        has been shutdown, and crash as (hopefully) appropriate.
        """
        hvd().shutdown()

    def distributed_gradient_tape(self, tape):
        """Add Horovod Distributed GradientTape for tensorflow eager mode.

        Args:
            tape (tf.GradientTape): Recorded operations of automatic differentiation.
        Returns:
            tape (tf.GradientTape): The input tape wrapped in a tape that takes
                care of the distribution.
        """
        return hvd().DistributedGradientTape(tape)

    def broadcast_variables(self, variables, root_rank=0):
        """Broadcast variables from root_rank to other ranks.

        Args:
            variables (tf.Variable): tensorflow variables that need to be broadcast.
            root_rank (int): From which rank the variables need to be broadcast.
        """
        hvd().broadcast_variables(variables, root_rank=root_rank)


# Define the distributor here so it's static.
_DISTRIBUTOR = Distributor()


def set_distributor(d):
    """Set the distributor.

    Args:
        d: an instance who's class derives from Distributor to serve as the new distribution object.
    """
    global _DISTRIBUTOR  # pylint: disable=W0603
    _DISTRIBUTOR = d


def get_distributor():
    """Get the distributor."""
    global _DISTRIBUTOR  # pylint: disable=W0602,W0603
    return _DISTRIBUTOR
