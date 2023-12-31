# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nvidia_tao_tf1/cv/efficientdet/proto/model_config.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='nvidia_tao_tf1/cv/efficientdet/proto/model_config.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n7nvidia_tao_tf1/cv/efficientdet/proto/model_config.proto\"\xb2\x01\n\x0bModelConfig\x12\x12\n\nmodel_name\x18\x01 \x01(\t\x12\x11\n\tfreeze_bn\x18\x02 \x01(\x08\x12\x15\n\rfreeze_blocks\x18\x03 \x01(\t\x12\x15\n\raspect_ratios\x18\x04 \x01(\t\x12\x14\n\x0c\x61nchor_scale\x18\x05 \x01(\x02\x12\x11\n\tmin_level\x18\x06 \x01(\r\x12\x11\n\tmax_level\x18\x07 \x01(\r\x12\x12\n\nnum_scales\x18\x08 \x01(\rb\x06proto3')
)




_MODELCONFIG = _descriptor.Descriptor(
  name='ModelConfig',
  full_name='ModelConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='model_name', full_name='ModelConfig.model_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='freeze_bn', full_name='ModelConfig.freeze_bn', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='freeze_blocks', full_name='ModelConfig.freeze_blocks', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='aspect_ratios', full_name='ModelConfig.aspect_ratios', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='anchor_scale', full_name='ModelConfig.anchor_scale', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='min_level', full_name='ModelConfig.min_level', index=5,
      number=6, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_level', full_name='ModelConfig.max_level', index=6,
      number=7, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_scales', full_name='ModelConfig.num_scales', index=7,
      number=8, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=60,
  serialized_end=238,
)

DESCRIPTOR.message_types_by_name['ModelConfig'] = _MODELCONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ModelConfig = _reflection.GeneratedProtocolMessageType('ModelConfig', (_message.Message,), dict(
  DESCRIPTOR = _MODELCONFIG,
  __module__ = 'nvidia_tao_tf1.cv.efficientdet.proto.model_config_pb2'
  # @@protoc_insertion_point(class_scope:ModelConfig)
  ))
_sym_db.RegisterMessage(ModelConfig)


# @@protoc_insertion_point(module_scope)
