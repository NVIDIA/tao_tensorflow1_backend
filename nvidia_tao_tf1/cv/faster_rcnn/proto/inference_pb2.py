# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nvidia_tao_tf1/cv/faster_rcnn/proto/inference.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from nvidia_tao_tf1.cv.faster_rcnn.proto import trt_config_pb2 as nvidia__tao__tf1_dot_cv_dot_faster__rcnn_dot_proto_dot_trt__config__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='nvidia_tao_tf1/cv/faster_rcnn/proto/inference.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n3nvidia_tao_tf1/cv/faster_rcnn/proto/inference.proto\x1a\x34nvidia_tao_tf1/cv/faster_rcnn/proto/trt_config.proto\"\xc4\x03\n\x0fInferenceConfig\x12\x12\n\nimages_dir\x18\x01 \x01(\t\x12\r\n\x05model\x18\x02 \x01(\t\x12\x12\n\nbatch_size\x18\x0f \x01(\r\x12\x19\n\x11rpn_pre_nms_top_N\x18\n \x01(\r\x12\x19\n\x11rpn_nms_max_boxes\x18\x07 \x01(\r\x12!\n\x19rpn_nms_overlap_threshold\x18\x08 \x01(\x02\x12 \n\x18\x62\x62ox_visualize_threshold\x18\x05 \x01(\x02\x12\x1f\n\x17object_confidence_thres\x18\x10 \x01(\x02\x12 \n\x18\x63lassifier_nms_max_boxes\x18\t \x01(\r\x12(\n classifier_nms_overlap_threshold\x18\x06 \x01(\x02\x12\"\n\x1a\x64\x65tection_image_output_dir\x18\x0b \x01(\t\x12\x17\n\x0f\x62\x62ox_caption_on\x18\x0c \x01(\x08\x12\x17\n\x0flabels_dump_dir\x18\r \x01(\t\x12$\n\rtrt_inference\x18\x0e \x01(\x0b\x32\r.TrtInference\x12\x16\n\x0enms_score_bits\x18\x11 \x01(\rb\x06proto3')
  ,
  dependencies=[nvidia__tao__tf1_dot_cv_dot_faster__rcnn_dot_proto_dot_trt__config__pb2.DESCRIPTOR,])




_INFERENCECONFIG = _descriptor.Descriptor(
  name='InferenceConfig',
  full_name='InferenceConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='images_dir', full_name='InferenceConfig.images_dir', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model', full_name='InferenceConfig.model', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='batch_size', full_name='InferenceConfig.batch_size', index=2,
      number=15, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rpn_pre_nms_top_N', full_name='InferenceConfig.rpn_pre_nms_top_N', index=3,
      number=10, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rpn_nms_max_boxes', full_name='InferenceConfig.rpn_nms_max_boxes', index=4,
      number=7, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rpn_nms_overlap_threshold', full_name='InferenceConfig.rpn_nms_overlap_threshold', index=5,
      number=8, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bbox_visualize_threshold', full_name='InferenceConfig.bbox_visualize_threshold', index=6,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='object_confidence_thres', full_name='InferenceConfig.object_confidence_thres', index=7,
      number=16, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='classifier_nms_max_boxes', full_name='InferenceConfig.classifier_nms_max_boxes', index=8,
      number=9, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='classifier_nms_overlap_threshold', full_name='InferenceConfig.classifier_nms_overlap_threshold', index=9,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='detection_image_output_dir', full_name='InferenceConfig.detection_image_output_dir', index=10,
      number=11, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bbox_caption_on', full_name='InferenceConfig.bbox_caption_on', index=11,
      number=12, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='labels_dump_dir', full_name='InferenceConfig.labels_dump_dir', index=12,
      number=13, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='trt_inference', full_name='InferenceConfig.trt_inference', index=13,
      number=14, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nms_score_bits', full_name='InferenceConfig.nms_score_bits', index=14,
      number=17, type=13, cpp_type=3, label=1,
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
  serialized_start=110,
  serialized_end=562,
)

_INFERENCECONFIG.fields_by_name['trt_inference'].message_type = nvidia__tao__tf1_dot_cv_dot_faster__rcnn_dot_proto_dot_trt__config__pb2._TRTINFERENCE
DESCRIPTOR.message_types_by_name['InferenceConfig'] = _INFERENCECONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

InferenceConfig = _reflection.GeneratedProtocolMessageType('InferenceConfig', (_message.Message,), dict(
  DESCRIPTOR = _INFERENCECONFIG,
  __module__ = 'nvidia_tao_tf1.cv.faster_rcnn.proto.inference_pb2'
  # @@protoc_insertion_point(class_scope:InferenceConfig)
  ))
_sym_db.RegisterMessage(InferenceConfig)


# @@protoc_insertion_point(module_scope)
