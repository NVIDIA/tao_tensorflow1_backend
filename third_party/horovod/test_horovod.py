"""Smoke tests for the built horovod wheel."""

import horovod


def test_built_horovod_version():
    """Test horovod available in ai-infra at the correct version."""
    assert horovod.__version__ == "0.22.1"


def test_lazy_horovod_init():
    """Test horovod with tensorflow lazy initialization."""
    import horovod.tensorflow as hvd

    hvd.init()
