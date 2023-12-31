# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nvidia_tao_tf1/cv/detectnet_v2/proto/kitti_config.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='nvidia_tao_tf1/cv/detectnet_v2/proto/kitti_config.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n7nvidia_tao_tf1/cv/detectnet_v2/proto/kitti_config.proto\"\xa5\x02\n\x0bKITTIConfig\x12\x1b\n\x13root_directory_path\x18\x01 \x01(\t\x12\x16\n\x0eimage_dir_name\x18\x02 \x01(\t\x12\x16\n\x0elabel_dir_name\x18\x03 \x01(\t\x12\x18\n\x10point_clouds_dir\x18\x04 \x01(\t\x12\x18\n\x10\x63\x61librations_dir\x18\x05 \x01(\t\x12%\n\x1dkitti_sequence_to_frames_file\x18\x06 \x01(\t\x12\x17\n\x0fimage_extension\x18\x07 \x01(\t\x12\x16\n\x0enum_partitions\x18\x08 \x01(\r\x12\x12\n\nnum_shards\x18\t \x01(\r\x12\x16\n\x0epartition_mode\x18\n \x01(\t\x12\x11\n\tval_split\x18\x0b \x01(\x02\x62\x06proto3')
)




_KITTICONFIG = _descriptor.Descriptor(
  name='KITTIConfig',
  full_name='KITTIConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='root_directory_path', full_name='KITTIConfig.root_directory_path', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='image_dir_name', full_name='KITTIConfig.image_dir_name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='label_dir_name', full_name='KITTIConfig.label_dir_name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='point_clouds_dir', full_name='KITTIConfig.point_clouds_dir', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='calibrations_dir', full_name='KITTIConfig.calibrations_dir', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='kitti_sequence_to_frames_file', full_name='KITTIConfig.kitti_sequence_to_frames_file', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='image_extension', full_name='KITTIConfig.image_extension', index=6,
      number=7, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_partitions', full_name='KITTIConfig.num_partitions', index=7,
      number=8, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_shards', full_name='KITTIConfig.num_shards', index=8,
      number=9, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='partition_mode', full_name='KITTIConfig.partition_mode', index=9,
      number=10, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='val_split', full_name='KITTIConfig.val_split', index=10,
      number=11, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
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
  serialized_end=353,
)

DESCRIPTOR.message_types_by_name['KITTIConfig'] = _KITTICONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

KITTIConfig = _reflection.GeneratedProtocolMessageType('KITTIConfig', (_message.Message,), dict(
  DESCRIPTOR = _KITTICONFIG,
  __module__ = 'nvidia_tao_tf1.cv.detectnet_v2.proto.kitti_config_pb2'
  # @@protoc_insertion_point(class_scope:KITTIConfig)
  ))
_sym_db.RegisterMessage(KITTIConfig)


# @@protoc_insertion_point(module_scope)
