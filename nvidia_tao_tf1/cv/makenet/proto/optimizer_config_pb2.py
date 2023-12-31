# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nvidia_tao_tf1/cv/makenet/proto/optimizer_config.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='nvidia_tao_tf1/cv/makenet/proto/optimizer_config.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n6nvidia_tao_tf1/cv/makenet/proto/optimizer_config.proto\"S\n\x12SgdOptimizerConfig\x12\n\n\x02lr\x18\x01 \x01(\x02\x12\r\n\x05\x64\x65\x63\x61y\x18\x02 \x01(\x02\x12\x10\n\x08momentum\x18\x03 \x01(\x02\x12\x10\n\x08nesterov\x18\x04 \x01(\x08\"a\n\x13\x41\x64\x61mOptimizerConfig\x12\n\n\x02lr\x18\x01 \x01(\x02\x12\x0e\n\x06\x62\x65ta_1\x18\x02 \x01(\x02\x12\x0e\n\x06\x62\x65ta_2\x18\x03 \x01(\x02\x12\x0f\n\x07\x65psilon\x18\x04 \x01(\x02\x12\r\n\x05\x64\x65\x63\x61y\x18\x05 \x01(\x02\"Q\n\x16RmspropOptimizerConfig\x12\n\n\x02lr\x18\x01 \x01(\x02\x12\x0b\n\x03rho\x18\x02 \x01(\x02\x12\x0f\n\x07\x65psilon\x18\x03 \x01(\x02\x12\r\n\x05\x64\x65\x63\x61y\x18\x04 \x01(\x02\"\x90\x01\n\x0fOptimizerConfig\x12\"\n\x03sgd\x18\x01 \x01(\x0b\x32\x13.SgdOptimizerConfigH\x00\x12$\n\x04\x61\x64\x61m\x18\x02 \x01(\x0b\x32\x14.AdamOptimizerConfigH\x00\x12*\n\x07rmsprop\x18\x03 \x01(\x0b\x32\x17.RmspropOptimizerConfigH\x00\x42\x07\n\x05optimb\x06proto3')
)




_SGDOPTIMIZERCONFIG = _descriptor.Descriptor(
  name='SgdOptimizerConfig',
  full_name='SgdOptimizerConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='lr', full_name='SgdOptimizerConfig.lr', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='decay', full_name='SgdOptimizerConfig.decay', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='momentum', full_name='SgdOptimizerConfig.momentum', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nesterov', full_name='SgdOptimizerConfig.nesterov', index=3,
      number=4, type=8, cpp_type=7, label=1,
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
  serialized_start=58,
  serialized_end=141,
)


_ADAMOPTIMIZERCONFIG = _descriptor.Descriptor(
  name='AdamOptimizerConfig',
  full_name='AdamOptimizerConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='lr', full_name='AdamOptimizerConfig.lr', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='beta_1', full_name='AdamOptimizerConfig.beta_1', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='beta_2', full_name='AdamOptimizerConfig.beta_2', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='epsilon', full_name='AdamOptimizerConfig.epsilon', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='decay', full_name='AdamOptimizerConfig.decay', index=4,
      number=5, type=2, cpp_type=6, label=1,
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
  serialized_start=143,
  serialized_end=240,
)


_RMSPROPOPTIMIZERCONFIG = _descriptor.Descriptor(
  name='RmspropOptimizerConfig',
  full_name='RmspropOptimizerConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='lr', full_name='RmspropOptimizerConfig.lr', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rho', full_name='RmspropOptimizerConfig.rho', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='epsilon', full_name='RmspropOptimizerConfig.epsilon', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='decay', full_name='RmspropOptimizerConfig.decay', index=3,
      number=4, type=2, cpp_type=6, label=1,
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
  serialized_start=242,
  serialized_end=323,
)


_OPTIMIZERCONFIG = _descriptor.Descriptor(
  name='OptimizerConfig',
  full_name='OptimizerConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='sgd', full_name='OptimizerConfig.sgd', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='adam', full_name='OptimizerConfig.adam', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rmsprop', full_name='OptimizerConfig.rmsprop', index=2,
      number=3, type=11, cpp_type=10, label=1,
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
    _descriptor.OneofDescriptor(
      name='optim', full_name='OptimizerConfig.optim',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=326,
  serialized_end=470,
)

_OPTIMIZERCONFIG.fields_by_name['sgd'].message_type = _SGDOPTIMIZERCONFIG
_OPTIMIZERCONFIG.fields_by_name['adam'].message_type = _ADAMOPTIMIZERCONFIG
_OPTIMIZERCONFIG.fields_by_name['rmsprop'].message_type = _RMSPROPOPTIMIZERCONFIG
_OPTIMIZERCONFIG.oneofs_by_name['optim'].fields.append(
  _OPTIMIZERCONFIG.fields_by_name['sgd'])
_OPTIMIZERCONFIG.fields_by_name['sgd'].containing_oneof = _OPTIMIZERCONFIG.oneofs_by_name['optim']
_OPTIMIZERCONFIG.oneofs_by_name['optim'].fields.append(
  _OPTIMIZERCONFIG.fields_by_name['adam'])
_OPTIMIZERCONFIG.fields_by_name['adam'].containing_oneof = _OPTIMIZERCONFIG.oneofs_by_name['optim']
_OPTIMIZERCONFIG.oneofs_by_name['optim'].fields.append(
  _OPTIMIZERCONFIG.fields_by_name['rmsprop'])
_OPTIMIZERCONFIG.fields_by_name['rmsprop'].containing_oneof = _OPTIMIZERCONFIG.oneofs_by_name['optim']
DESCRIPTOR.message_types_by_name['SgdOptimizerConfig'] = _SGDOPTIMIZERCONFIG
DESCRIPTOR.message_types_by_name['AdamOptimizerConfig'] = _ADAMOPTIMIZERCONFIG
DESCRIPTOR.message_types_by_name['RmspropOptimizerConfig'] = _RMSPROPOPTIMIZERCONFIG
DESCRIPTOR.message_types_by_name['OptimizerConfig'] = _OPTIMIZERCONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SgdOptimizerConfig = _reflection.GeneratedProtocolMessageType('SgdOptimizerConfig', (_message.Message,), dict(
  DESCRIPTOR = _SGDOPTIMIZERCONFIG,
  __module__ = 'nvidia_tao_tf1.cv.makenet.proto.optimizer_config_pb2'
  # @@protoc_insertion_point(class_scope:SgdOptimizerConfig)
  ))
_sym_db.RegisterMessage(SgdOptimizerConfig)

AdamOptimizerConfig = _reflection.GeneratedProtocolMessageType('AdamOptimizerConfig', (_message.Message,), dict(
  DESCRIPTOR = _ADAMOPTIMIZERCONFIG,
  __module__ = 'nvidia_tao_tf1.cv.makenet.proto.optimizer_config_pb2'
  # @@protoc_insertion_point(class_scope:AdamOptimizerConfig)
  ))
_sym_db.RegisterMessage(AdamOptimizerConfig)

RmspropOptimizerConfig = _reflection.GeneratedProtocolMessageType('RmspropOptimizerConfig', (_message.Message,), dict(
  DESCRIPTOR = _RMSPROPOPTIMIZERCONFIG,
  __module__ = 'nvidia_tao_tf1.cv.makenet.proto.optimizer_config_pb2'
  # @@protoc_insertion_point(class_scope:RmspropOptimizerConfig)
  ))
_sym_db.RegisterMessage(RmspropOptimizerConfig)

OptimizerConfig = _reflection.GeneratedProtocolMessageType('OptimizerConfig', (_message.Message,), dict(
  DESCRIPTOR = _OPTIMIZERCONFIG,
  __module__ = 'nvidia_tao_tf1.cv.makenet.proto.optimizer_config_pb2'
  # @@protoc_insertion_point(class_scope:OptimizerConfig)
  ))
_sym_db.RegisterMessage(OptimizerConfig)


# @@protoc_insertion_point(module_scope)
