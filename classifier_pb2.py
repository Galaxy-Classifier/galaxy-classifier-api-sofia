# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: classifier.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='classifier.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x10\x63lassifier.proto\"2\n\x15\x43lassificationRequest\x12\n\n\x02id\x18\x01 \x01(\t\x12\r\n\x05image\x18\x02 \x01(\x0c\"&\n\x13\x43lassificationReply\x12\x0f\n\x07message\x18\x01 \x01(\t2Q\n\nClassifier\x12\x43\n\x11GetClassification\x12\x16.ClassificationRequest\x1a\x14.ClassificationReply\"\x00\x62\x06proto3'
)




_CLASSIFICATIONREQUEST = _descriptor.Descriptor(
  name='ClassificationRequest',
  full_name='ClassificationRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='ClassificationRequest.id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='image', full_name='ClassificationRequest.image', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
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
  serialized_start=20,
  serialized_end=70,
)


_CLASSIFICATIONREPLY = _descriptor.Descriptor(
  name='ClassificationReply',
  full_name='ClassificationReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='message', full_name='ClassificationReply.message', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
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
  serialized_start=72,
  serialized_end=110,
)

DESCRIPTOR.message_types_by_name['ClassificationRequest'] = _CLASSIFICATIONREQUEST
DESCRIPTOR.message_types_by_name['ClassificationReply'] = _CLASSIFICATIONREPLY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ClassificationRequest = _reflection.GeneratedProtocolMessageType('ClassificationRequest', (_message.Message,), {
  'DESCRIPTOR' : _CLASSIFICATIONREQUEST,
  '__module__' : 'classifier_pb2'
  # @@protoc_insertion_point(class_scope:ClassificationRequest)
  })
_sym_db.RegisterMessage(ClassificationRequest)

ClassificationReply = _reflection.GeneratedProtocolMessageType('ClassificationReply', (_message.Message,), {
  'DESCRIPTOR' : _CLASSIFICATIONREPLY,
  '__module__' : 'classifier_pb2'
  # @@protoc_insertion_point(class_scope:ClassificationReply)
  })
_sym_db.RegisterMessage(ClassificationReply)



_CLASSIFIER = _descriptor.ServiceDescriptor(
  name='Classifier',
  full_name='Classifier',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=112,
  serialized_end=193,
  methods=[
  _descriptor.MethodDescriptor(
    name='GetClassification',
    full_name='Classifier.GetClassification',
    index=0,
    containing_service=None,
    input_type=_CLASSIFICATIONREQUEST,
    output_type=_CLASSIFICATIONREPLY,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_CLASSIFIER)

DESCRIPTOR.services_by_name['Classifier'] = _CLASSIFIER

# @@protoc_insertion_point(module_scope)