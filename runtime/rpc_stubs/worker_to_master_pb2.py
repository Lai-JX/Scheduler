# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: worker_to_master.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x16worker_to_master.proto\"Q\n\x15RegisterWorkerRequest\x12\x11\n\tworker_ip\x18\x01 \x01(\t\x12\x13\n\x0bworker_port\x18\x02 \x01(\r\x12\x10\n\x08num_gpus\x18\x03 \x01(\r\"<\n\x16RegisterWorkerResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\x11\n\tworker_id\x18\x02 \x01(\r\"g\n\x0b\x44oneRequest\x12\x0e\n\x06job_id\x18\x01 \x01(\x05\x12\x13\n\x0bjob_counter\x18\x02 \x01(\r\x12\x11\n\tworker_id\x18\x03 \x01(\r\x12\x0c\n\x04gpus\x18\x04 \x01(\t\x12\x12\n\nreturncode\x18\x05 \x01(\x05\"\x1f\n\x0c\x44oneResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\x32z\n\x0eWorkerToMaster\x12\x43\n\x0eRegisterWorker\x12\x16.RegisterWorkerRequest\x1a\x17.RegisterWorkerResponse\"\x00\x12#\n\x04\x44one\x12\x0c.DoneRequest\x1a\r.DoneResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'worker_to_master_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_REGISTERWORKERREQUEST']._serialized_start=26
  _globals['_REGISTERWORKERREQUEST']._serialized_end=107
  _globals['_REGISTERWORKERRESPONSE']._serialized_start=109
  _globals['_REGISTERWORKERRESPONSE']._serialized_end=169
  _globals['_DONEREQUEST']._serialized_start=171
  _globals['_DONEREQUEST']._serialized_end=274
  _globals['_DONERESPONSE']._serialized_start=276
  _globals['_DONERESPONSE']._serialized_end=307
  _globals['_WORKERTOMASTER']._serialized_start=309
  _globals['_WORKERTOMASTER']._serialized_end=431
# @@protoc_insertion_point(module_scope)
