# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nvidia_tao_tf1/cv/faster_rcnn/proto/model.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from nvidia_tao_tf1.cv.faster_rcnn.proto import input_image_pb2 as nvidia__tao__tf1_dot_cv_dot_faster__rcnn_dot_proto_dot_input__image__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='nvidia_tao_tf1/cv/faster_rcnn/proto/model.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n/nvidia_tao_tf1/cv/faster_rcnn/proto/model.proto\x1a\x35nvidia_tao_tf1/cv/faster_rcnn/proto/input_image.proto\"/\n\x0f\x41nchorBoxConfig\x12\r\n\x05scale\x18\x01 \x03(\x02\x12\r\n\x05ratio\x18\x02 \x03(\x02\";\n\x10RoiPoolingConfig\x12\x11\n\tpool_size\x18\x01 \x01(\r\x12\x14\n\x0cpool_size_2x\x18\x02 \x01(\x08\"\xa8\x01\n\nActivation\x12\x17\n\x0f\x61\x63tivation_type\x18\x01 \x01(\t\x12\x44\n\x15\x61\x63tivation_parameters\x18\x02 \x03(\x0b\x32%.Activation.ActivationParametersEntry\x1a;\n\x19\x41\x63tivationParametersEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x02:\x02\x38\x01\"\xdd\x03\n\x0bModelConfig\x12-\n\x12input_image_config\x18\x01 \x01(\x0b\x32\x11.InputImageConfig\x12\x0c\n\x04\x61rch\x18\x02 \x01(\t\x12+\n\x11\x61nchor_box_config\x18\x03 \x01(\x0b\x32\x10.AnchorBoxConfig\x12\x16\n\x0eroi_mini_batch\x18\x04 \x01(\r\x12\x12\n\nrpn_stride\x18\x05 \x01(\r\x12\x11\n\tfreeze_bn\x18\x06 \x01(\x08\x12\x14\n\x0c\x64ropout_rate\x18\x11 \x01(\x02\x12\x19\n\x11\x64rop_connect_rate\x18\x12 \x01(\x02\x12\x1f\n\x17rpn_cls_activation_type\x18\x07 \x01(\t\x12\x15\n\rfreeze_blocks\x18\t \x03(\x02\x12\x10\n\x08use_bias\x18\n \x01(\x08\x12-\n\x12roi_pooling_config\x18\x0b \x01(\x0b\x32\x11.RoiPoolingConfig\x12\x11\n\trfcn_mode\x18\x0c \x01(\x08\x12\x19\n\x11tf_proposal_layer\x18\r \x01(\x08\x12\x17\n\x0f\x61ll_projections\x18\x0e \x01(\x08\x12\x13\n\x0buse_pooling\x18\x0f \x01(\x08\x12\x1f\n\nactivation\x18\x13 \x01(\x0b\x32\x0b.Activationb\x06proto3')
  ,
  dependencies=[nvidia__tao__tf1_dot_cv_dot_faster__rcnn_dot_proto_dot_input__image__pb2.DESCRIPTOR,])




_ANCHORBOXCONFIG = _descriptor.Descriptor(
  name='AnchorBoxConfig',
  full_name='AnchorBoxConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='scale', full_name='AnchorBoxConfig.scale', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ratio', full_name='AnchorBoxConfig.ratio', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=106,
  serialized_end=153,
)


_ROIPOOLINGCONFIG = _descriptor.Descriptor(
  name='RoiPoolingConfig',
  full_name='RoiPoolingConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='pool_size', full_name='RoiPoolingConfig.pool_size', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pool_size_2x', full_name='RoiPoolingConfig.pool_size_2x', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
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
  serialized_start=155,
  serialized_end=214,
)


_ACTIVATION_ACTIVATIONPARAMETERSENTRY = _descriptor.Descriptor(
  name='ActivationParametersEntry',
  full_name='Activation.ActivationParametersEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='Activation.ActivationParametersEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='Activation.ActivationParametersEntry.value', index=1,
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
  serialized_start=326,
  serialized_end=385,
)

_ACTIVATION = _descriptor.Descriptor(
  name='Activation',
  full_name='Activation',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='activation_type', full_name='Activation.activation_type', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='activation_parameters', full_name='Activation.activation_parameters', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_ACTIVATION_ACTIVATIONPARAMETERSENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=217,
  serialized_end=385,
)


_MODELCONFIG = _descriptor.Descriptor(
  name='ModelConfig',
  full_name='ModelConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='input_image_config', full_name='ModelConfig.input_image_config', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='arch', full_name='ModelConfig.arch', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='anchor_box_config', full_name='ModelConfig.anchor_box_config', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='roi_mini_batch', full_name='ModelConfig.roi_mini_batch', index=3,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rpn_stride', full_name='ModelConfig.rpn_stride', index=4,
      number=5, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='freeze_bn', full_name='ModelConfig.freeze_bn', index=5,
      number=6, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dropout_rate', full_name='ModelConfig.dropout_rate', index=6,
      number=17, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='drop_connect_rate', full_name='ModelConfig.drop_connect_rate', index=7,
      number=18, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rpn_cls_activation_type', full_name='ModelConfig.rpn_cls_activation_type', index=8,
      number=7, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='freeze_blocks', full_name='ModelConfig.freeze_blocks', index=9,
      number=9, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='use_bias', full_name='ModelConfig.use_bias', index=10,
      number=10, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='roi_pooling_config', full_name='ModelConfig.roi_pooling_config', index=11,
      number=11, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rfcn_mode', full_name='ModelConfig.rfcn_mode', index=12,
      number=12, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tf_proposal_layer', full_name='ModelConfig.tf_proposal_layer', index=13,
      number=13, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='all_projections', full_name='ModelConfig.all_projections', index=14,
      number=14, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='use_pooling', full_name='ModelConfig.use_pooling', index=15,
      number=15, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='activation', full_name='ModelConfig.activation', index=16,
      number=19, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
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
  serialized_start=388,
  serialized_end=865,
)

_ACTIVATION_ACTIVATIONPARAMETERSENTRY.containing_type = _ACTIVATION
_ACTIVATION.fields_by_name['activation_parameters'].message_type = _ACTIVATION_ACTIVATIONPARAMETERSENTRY
_MODELCONFIG.fields_by_name['input_image_config'].message_type = nvidia__tao__tf1_dot_cv_dot_faster__rcnn_dot_proto_dot_input__image__pb2._INPUTIMAGECONFIG
_MODELCONFIG.fields_by_name['anchor_box_config'].message_type = _ANCHORBOXCONFIG
_MODELCONFIG.fields_by_name['roi_pooling_config'].message_type = _ROIPOOLINGCONFIG
_MODELCONFIG.fields_by_name['activation'].message_type = _ACTIVATION
DESCRIPTOR.message_types_by_name['AnchorBoxConfig'] = _ANCHORBOXCONFIG
DESCRIPTOR.message_types_by_name['RoiPoolingConfig'] = _ROIPOOLINGCONFIG
DESCRIPTOR.message_types_by_name['Activation'] = _ACTIVATION
DESCRIPTOR.message_types_by_name['ModelConfig'] = _MODELCONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

AnchorBoxConfig = _reflection.GeneratedProtocolMessageType('AnchorBoxConfig', (_message.Message,), dict(
  DESCRIPTOR = _ANCHORBOXCONFIG,
  __module__ = 'nvidia_tao_tf1.cv.faster_rcnn.proto.model_pb2'
  # @@protoc_insertion_point(class_scope:AnchorBoxConfig)
  ))
_sym_db.RegisterMessage(AnchorBoxConfig)

RoiPoolingConfig = _reflection.GeneratedProtocolMessageType('RoiPoolingConfig', (_message.Message,), dict(
  DESCRIPTOR = _ROIPOOLINGCONFIG,
  __module__ = 'nvidia_tao_tf1.cv.faster_rcnn.proto.model_pb2'
  # @@protoc_insertion_point(class_scope:RoiPoolingConfig)
  ))
_sym_db.RegisterMessage(RoiPoolingConfig)

Activation = _reflection.GeneratedProtocolMessageType('Activation', (_message.Message,), dict(

  ActivationParametersEntry = _reflection.GeneratedProtocolMessageType('ActivationParametersEntry', (_message.Message,), dict(
    DESCRIPTOR = _ACTIVATION_ACTIVATIONPARAMETERSENTRY,
    __module__ = 'nvidia_tao_tf1.cv.faster_rcnn.proto.model_pb2'
    # @@protoc_insertion_point(class_scope:Activation.ActivationParametersEntry)
    ))
  ,
  DESCRIPTOR = _ACTIVATION,
  __module__ = 'nvidia_tao_tf1.cv.faster_rcnn.proto.model_pb2'
  # @@protoc_insertion_point(class_scope:Activation)
  ))
_sym_db.RegisterMessage(Activation)
_sym_db.RegisterMessage(Activation.ActivationParametersEntry)

ModelConfig = _reflection.GeneratedProtocolMessageType('ModelConfig', (_message.Message,), dict(
  DESCRIPTOR = _MODELCONFIG,
  __module__ = 'nvidia_tao_tf1.cv.faster_rcnn.proto.model_pb2'
  # @@protoc_insertion_point(class_scope:ModelConfig)
  ))
_sym_db.RegisterMessage(ModelConfig)


_ACTIVATION_ACTIVATIONPARAMETERSENTRY._options = None
# @@protoc_insertion_point(module_scope)
