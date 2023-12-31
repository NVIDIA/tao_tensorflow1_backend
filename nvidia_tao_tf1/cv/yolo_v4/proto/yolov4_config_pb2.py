# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nvidia_tao_tf1/cv/yolo_v4/proto/yolov4_config.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='nvidia_tao_tf1/cv/yolo_v4/proto/yolov4_config.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n3nvidia_tao_tf1/cv/yolo_v4/proto/yolov4_config.proto\"\x9b\x04\n\x0cYOLOv4Config\x12\x18\n\x10\x62ig_anchor_shape\x18\x01 \x01(\t\x12\x18\n\x10mid_anchor_shape\x18\x02 \x01(\t\x12\x1a\n\x12small_anchor_shape\x18\x03 \x01(\t\x12 \n\x18matching_neutral_box_iou\x18\x04 \x01(\x02\x12\x18\n\x10\x62ox_matching_iou\x18\x05 \x01(\x02\x12\x0c\n\x04\x61rch\x18\x06 \x01(\t\x12\x0f\n\x07nlayers\x18\x07 \x01(\r\x12\x18\n\x10\x61rch_conv_blocks\x18\x08 \x01(\r\x12\x17\n\x0floss_loc_weight\x18\t \x01(\x02\x12\x1c\n\x14loss_neg_obj_weights\x18\n \x01(\x02\x12\x1a\n\x12loss_class_weights\x18\x0b \x01(\x02\x12\x15\n\rfreeze_blocks\x18\x0c \x03(\x02\x12\x11\n\tfreeze_bn\x18\r \x01(\x08\x12\x12\n\nforce_relu\x18\x0e \x01(\x08\x12\x12\n\nactivation\x18\x15 \x01(\t\x12\x18\n\x10\x66ocal_loss_alpha\x18\x0f \x01(\x02\x12\x18\n\x10\x66ocal_loss_gamma\x18\x10 \x01(\x02\x12\x17\n\x0flabel_smoothing\x18\x11 \x01(\x02\x12\x1a\n\x12\x62ig_grid_xy_extend\x18\x12 \x01(\x02\x12\x1a\n\x12mid_grid_xy_extend\x18\x13 \x01(\x02\x12\x1c\n\x14small_grid_xy_extend\x18\x14 \x01(\x02\x62\x06proto3')
)




_YOLOV4CONFIG = _descriptor.Descriptor(
  name='YOLOv4Config',
  full_name='YOLOv4Config',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='big_anchor_shape', full_name='YOLOv4Config.big_anchor_shape', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mid_anchor_shape', full_name='YOLOv4Config.mid_anchor_shape', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='small_anchor_shape', full_name='YOLOv4Config.small_anchor_shape', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='matching_neutral_box_iou', full_name='YOLOv4Config.matching_neutral_box_iou', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='box_matching_iou', full_name='YOLOv4Config.box_matching_iou', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='arch', full_name='YOLOv4Config.arch', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nlayers', full_name='YOLOv4Config.nlayers', index=6,
      number=7, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='arch_conv_blocks', full_name='YOLOv4Config.arch_conv_blocks', index=7,
      number=8, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='loss_loc_weight', full_name='YOLOv4Config.loss_loc_weight', index=8,
      number=9, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='loss_neg_obj_weights', full_name='YOLOv4Config.loss_neg_obj_weights', index=9,
      number=10, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='loss_class_weights', full_name='YOLOv4Config.loss_class_weights', index=10,
      number=11, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='freeze_blocks', full_name='YOLOv4Config.freeze_blocks', index=11,
      number=12, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='freeze_bn', full_name='YOLOv4Config.freeze_bn', index=12,
      number=13, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='force_relu', full_name='YOLOv4Config.force_relu', index=13,
      number=14, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='activation', full_name='YOLOv4Config.activation', index=14,
      number=21, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='focal_loss_alpha', full_name='YOLOv4Config.focal_loss_alpha', index=15,
      number=15, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='focal_loss_gamma', full_name='YOLOv4Config.focal_loss_gamma', index=16,
      number=16, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='label_smoothing', full_name='YOLOv4Config.label_smoothing', index=17,
      number=17, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='big_grid_xy_extend', full_name='YOLOv4Config.big_grid_xy_extend', index=18,
      number=18, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mid_grid_xy_extend', full_name='YOLOv4Config.mid_grid_xy_extend', index=19,
      number=19, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='small_grid_xy_extend', full_name='YOLOv4Config.small_grid_xy_extend', index=20,
      number=20, type=2, cpp_type=6, label=1,
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
  serialized_start=56,
  serialized_end=595,
)

DESCRIPTOR.message_types_by_name['YOLOv4Config'] = _YOLOV4CONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

YOLOv4Config = _reflection.GeneratedProtocolMessageType('YOLOv4Config', (_message.Message,), dict(
  DESCRIPTOR = _YOLOV4CONFIG,
  __module__ = 'nvidia_tao_tf1.cv.yolo_v4.proto.yolov4_config_pb2'
  # @@protoc_insertion_point(class_scope:YOLOv4Config)
  ))
_sym_db.RegisterMessage(YOLOv4Config)


# @@protoc_insertion_point(module_scope)
