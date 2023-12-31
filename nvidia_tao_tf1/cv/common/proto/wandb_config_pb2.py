# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nvidia_tao_tf1/cv/common/proto/wandb_config.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='nvidia_tao_tf1/cv/common/proto/wandb_config.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n1nvidia_tao_tf1/cv/common/proto/wandb_config.proto\"\xea\x01\n\x0bWandBConfig\x12\x0f\n\x07\x65nabled\x18\x01 \x01(\x08\x12\x0b\n\x03key\x18\x02 \x01(\t\x12\x0f\n\x07project\x18\x03 \x01(\t\x12\x0e\n\x06\x65ntity\x18\x04 \x01(\t\x12\x0e\n\x06reinit\x18\x05 \x01(\x08\x12\x0c\n\x04name\x18\x06 \x01(\t\x12\x0c\n\x04tags\x18\x07 \x03(\t\x12\x11\n\twandb_dir\x18\x08 \x01(\t\x12\r\n\x05notes\x18\t \x01(\t\x12\x1f\n\x04mode\x18\n \x01(\x0e\x32\x11.WandBConfig.MODE\"-\n\x04MODE\x12\n\n\x06ONLINE\x10\x00\x12\x0b\n\x07OFFLINE\x10\x01\x12\x0c\n\x08\x44ISABLED\x10\x02\x62\x06proto3')
)



_WANDBCONFIG_MODE = _descriptor.EnumDescriptor(
  name='MODE',
  full_name='WandBConfig.MODE',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='ONLINE', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OFFLINE', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DISABLED', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=243,
  serialized_end=288,
)
_sym_db.RegisterEnumDescriptor(_WANDBCONFIG_MODE)


_WANDBCONFIG = _descriptor.Descriptor(
  name='WandBConfig',
  full_name='WandBConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='enabled', full_name='WandBConfig.enabled', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='key', full_name='WandBConfig.key', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='project', full_name='WandBConfig.project', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='entity', full_name='WandBConfig.entity', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='reinit', full_name='WandBConfig.reinit', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='name', full_name='WandBConfig.name', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tags', full_name='WandBConfig.tags', index=6,
      number=7, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='wandb_dir', full_name='WandBConfig.wandb_dir', index=7,
      number=8, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='notes', full_name='WandBConfig.notes', index=8,
      number=9, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mode', full_name='WandBConfig.mode', index=9,
      number=10, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _WANDBCONFIG_MODE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=54,
  serialized_end=288,
)

_WANDBCONFIG.fields_by_name['mode'].enum_type = _WANDBCONFIG_MODE
_WANDBCONFIG_MODE.containing_type = _WANDBCONFIG
DESCRIPTOR.message_types_by_name['WandBConfig'] = _WANDBCONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

WandBConfig = _reflection.GeneratedProtocolMessageType('WandBConfig', (_message.Message,), dict(
  DESCRIPTOR = _WANDBCONFIG,
  __module__ = 'nvidia_tao_tf1.cv.common.proto.wandb_config_pb2'
  # @@protoc_insertion_point(class_scope:WandBConfig)
  ))
_sym_db.RegisterMessage(WandBConfig)


# @@protoc_insertion_point(module_scope)
