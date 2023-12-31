# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nvidia_tao_tf1/cv/faster_rcnn/proto/input_image.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='nvidia_tao_tf1/cv/faster_rcnn/proto/input_image.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n5nvidia_tao_tf1/cv/faster_rcnn/proto/input_image.proto\"!\n\x12ImageSizeConfigMin\x12\x0b\n\x03min\x18\x01 \x01(\r\";\n\x1aImageSizeConfigHeightWidth\x12\x0e\n\x06height\x18\x01 \x01(\r\x12\r\n\x05width\x18\x02 \x01(\r\"\x86\x03\n\x10InputImageConfig\x12\x1e\n\nimage_type\x18\x06 \x01(\x0e\x32\n.ImageType\x12\'\n\x08size_min\x18\x01 \x01(\x0b\x32\x13.ImageSizeConfigMinH\x00\x12\x38\n\x11size_height_width\x18\x02 \x01(\x0b\x32\x1b.ImageSizeConfigHeightWidthH\x00\x12\x1b\n\x13image_channel_order\x18\x05 \x01(\t\x12\x43\n\x12image_channel_mean\x18\x03 \x03(\x0b\x32\'.InputImageConfig.ImageChannelMeanEntry\x12\x1c\n\x14image_scaling_factor\x18\x04 \x01(\x02\x12!\n\x19max_objects_num_per_image\x18\x07 \x01(\r\x1a\x37\n\x15ImageChannelMeanEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x02:\x02\x38\x01\x42\x13\n\x11image_size_config*$\n\tImageType\x12\x07\n\x03RGB\x10\x00\x12\x0e\n\nGRAY_SCALE\x10\x01\x62\x06proto3')
)

_IMAGETYPE = _descriptor.EnumDescriptor(
  name='ImageType',
  full_name='ImageType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='RGB', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GRAY_SCALE', index=1, number=1,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=546,
  serialized_end=582,
)
_sym_db.RegisterEnumDescriptor(_IMAGETYPE)

ImageType = enum_type_wrapper.EnumTypeWrapper(_IMAGETYPE)
RGB = 0
GRAY_SCALE = 1



_IMAGESIZECONFIGMIN = _descriptor.Descriptor(
  name='ImageSizeConfigMin',
  full_name='ImageSizeConfigMin',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='min', full_name='ImageSizeConfigMin.min', index=0,
      number=1, type=13, cpp_type=3, label=1,
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
  serialized_start=57,
  serialized_end=90,
)


_IMAGESIZECONFIGHEIGHTWIDTH = _descriptor.Descriptor(
  name='ImageSizeConfigHeightWidth',
  full_name='ImageSizeConfigHeightWidth',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='height', full_name='ImageSizeConfigHeightWidth.height', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='width', full_name='ImageSizeConfigHeightWidth.width', index=1,
      number=2, type=13, cpp_type=3, label=1,
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
  serialized_start=92,
  serialized_end=151,
)


_INPUTIMAGECONFIG_IMAGECHANNELMEANENTRY = _descriptor.Descriptor(
  name='ImageChannelMeanEntry',
  full_name='InputImageConfig.ImageChannelMeanEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='InputImageConfig.ImageChannelMeanEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='InputImageConfig.ImageChannelMeanEntry.value', index=1,
      number=2, type=2, cpp_type=6, label=1,
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
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=468,
  serialized_end=523,
)

_INPUTIMAGECONFIG = _descriptor.Descriptor(
  name='InputImageConfig',
  full_name='InputImageConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='image_type', full_name='InputImageConfig.image_type', index=0,
      number=6, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='size_min', full_name='InputImageConfig.size_min', index=1,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='size_height_width', full_name='InputImageConfig.size_height_width', index=2,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='image_channel_order', full_name='InputImageConfig.image_channel_order', index=3,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='image_channel_mean', full_name='InputImageConfig.image_channel_mean', index=4,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='image_scaling_factor', full_name='InputImageConfig.image_scaling_factor', index=5,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_objects_num_per_image', full_name='InputImageConfig.max_objects_num_per_image', index=6,
      number=7, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_INPUTIMAGECONFIG_IMAGECHANNELMEANENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='image_size_config', full_name='InputImageConfig.image_size_config',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=154,
  serialized_end=544,
)

_INPUTIMAGECONFIG_IMAGECHANNELMEANENTRY.containing_type = _INPUTIMAGECONFIG
_INPUTIMAGECONFIG.fields_by_name['image_type'].enum_type = _IMAGETYPE
_INPUTIMAGECONFIG.fields_by_name['size_min'].message_type = _IMAGESIZECONFIGMIN
_INPUTIMAGECONFIG.fields_by_name['size_height_width'].message_type = _IMAGESIZECONFIGHEIGHTWIDTH
_INPUTIMAGECONFIG.fields_by_name['image_channel_mean'].message_type = _INPUTIMAGECONFIG_IMAGECHANNELMEANENTRY
_INPUTIMAGECONFIG.oneofs_by_name['image_size_config'].fields.append(
  _INPUTIMAGECONFIG.fields_by_name['size_min'])
_INPUTIMAGECONFIG.fields_by_name['size_min'].containing_oneof = _INPUTIMAGECONFIG.oneofs_by_name['image_size_config']
_INPUTIMAGECONFIG.oneofs_by_name['image_size_config'].fields.append(
  _INPUTIMAGECONFIG.fields_by_name['size_height_width'])
_INPUTIMAGECONFIG.fields_by_name['size_height_width'].containing_oneof = _INPUTIMAGECONFIG.oneofs_by_name['image_size_config']
DESCRIPTOR.message_types_by_name['ImageSizeConfigMin'] = _IMAGESIZECONFIGMIN
DESCRIPTOR.message_types_by_name['ImageSizeConfigHeightWidth'] = _IMAGESIZECONFIGHEIGHTWIDTH
DESCRIPTOR.message_types_by_name['InputImageConfig'] = _INPUTIMAGECONFIG
DESCRIPTOR.enum_types_by_name['ImageType'] = _IMAGETYPE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ImageSizeConfigMin = _reflection.GeneratedProtocolMessageType('ImageSizeConfigMin', (_message.Message,), dict(
  DESCRIPTOR = _IMAGESIZECONFIGMIN,
  __module__ = 'nvidia_tao_tf1.cv.faster_rcnn.proto.input_image_pb2'
  # @@protoc_insertion_point(class_scope:ImageSizeConfigMin)
  ))
_sym_db.RegisterMessage(ImageSizeConfigMin)

ImageSizeConfigHeightWidth = _reflection.GeneratedProtocolMessageType('ImageSizeConfigHeightWidth', (_message.Message,), dict(
  DESCRIPTOR = _IMAGESIZECONFIGHEIGHTWIDTH,
  __module__ = 'nvidia_tao_tf1.cv.faster_rcnn.proto.input_image_pb2'
  # @@protoc_insertion_point(class_scope:ImageSizeConfigHeightWidth)
  ))
_sym_db.RegisterMessage(ImageSizeConfigHeightWidth)

InputImageConfig = _reflection.GeneratedProtocolMessageType('InputImageConfig', (_message.Message,), dict(

  ImageChannelMeanEntry = _reflection.GeneratedProtocolMessageType('ImageChannelMeanEntry', (_message.Message,), dict(
    DESCRIPTOR = _INPUTIMAGECONFIG_IMAGECHANNELMEANENTRY,
    __module__ = 'nvidia_tao_tf1.cv.faster_rcnn.proto.input_image_pb2'
    # @@protoc_insertion_point(class_scope:InputImageConfig.ImageChannelMeanEntry)
    ))
  ,
  DESCRIPTOR = _INPUTIMAGECONFIG,
  __module__ = 'nvidia_tao_tf1.cv.faster_rcnn.proto.input_image_pb2'
  # @@protoc_insertion_point(class_scope:InputImageConfig)
  ))
_sym_db.RegisterMessage(InputImageConfig)
_sym_db.RegisterMessage(InputImageConfig.ImageChannelMeanEntry)


_INPUTIMAGECONFIG_IMAGECHANNELMEANENTRY._options = None
# @@protoc_insertion_point(module_scope)
