"""Library to convert protobuf objects to python dicts."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nvidia_tao_tf1.core.coreobject.protobuf_to_dict.protobuf_to_dict import (
    dict_to_protobuf,
    protobuf_to_dict,
    protobuf_to_modulus_dict,
    REVERSE_TYPE_CALLABLE_MAP,
    TYPE_CALLABLE_MAP,
)

__all__ = [
    "dict_to_protobuf",
    "protobuf_to_modulus_dict",
    "protobuf_to_dict",
    "REVERSE_TYPE_CALLABLE_MAP",
    "TYPE_CALLABLE_MAP",
]
