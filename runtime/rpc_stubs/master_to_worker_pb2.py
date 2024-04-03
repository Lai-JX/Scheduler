# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: master_to_worker.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x16master_to_worker.proto\"\xb9\x01\n\x07JobInfo\x12\x0b\n\x03num\x18\x01 \x01(\r\x12\x0f\n\x07node_id\x18\x02 \x03(\r\x12\x0e\n\x06job_id\x18\x03 \x03(\x05\x12\x10\n\x08job_name\x18\x04 \x03(\t\x12\x12\n\nbatch_size\x18\x05 \x03(\r\x12\x12\n\niterations\x18\x06 \x03(\r\x12\x0c\n\x04gpus\x18\x07 \x01(\t\x12\x13\n\x0bjob_counter\x18\x08 \x03(\r\x12\x0f\n\x07num_gpu\x18\t \x01(\r\x12\x12\n\nis_resumed\x18\n \x03(\x08\",\n\x0e\x45xecuteRequest\x12\x1a\n\x08job_info\x18\x01 \x01(\x0b\x32\x08.JobInfo\"\"\n\x0f\x45xecuteResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\")\n\x0bKillRequest\x12\x1a\n\x08job_info\x18\x01 \x01(\x0b\x32\x08.JobInfo\"\x1f\n\x0cKillResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\"\x14\n\x12\x45xitCommandRequest\"&\n\x13\x45xitCommandResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\"\x1e\n\x0eGetUtilRequest\x12\x0c\n\x04secs\x18\x01 \x01(\r\"F\n\x0fGetUtilResponse\x12\x10\n\x08gpu_util\x18\x01 \x01(\x01\x12\x10\n\x08\x63pu_util\x18\x02 \x01(\x01\x12\x0f\n\x07io_read\x18\x03 \x01(\x01\x32\xd3\x01\n\x0eMasterToWorker\x12.\n\x07\x45xecute\x12\x0f.ExecuteRequest\x1a\x10.ExecuteResponse\"\x00\x12%\n\x04Kill\x12\x0c.KillRequest\x1a\r.KillResponse\"\x00\x12:\n\x0b\x45xitCommand\x12\x13.ExitCommandRequest\x1a\x14.ExitCommandResponse\"\x00\x12.\n\x07GetUtil\x12\x0f.GetUtilRequest\x1a\x10.GetUtilResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'master_to_worker_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_JOBINFO']._serialized_start=27
  _globals['_JOBINFO']._serialized_end=212
  _globals['_EXECUTEREQUEST']._serialized_start=214
  _globals['_EXECUTEREQUEST']._serialized_end=258
  _globals['_EXECUTERESPONSE']._serialized_start=260
  _globals['_EXECUTERESPONSE']._serialized_end=294
  _globals['_KILLREQUEST']._serialized_start=296
  _globals['_KILLREQUEST']._serialized_end=337
  _globals['_KILLRESPONSE']._serialized_start=339
  _globals['_KILLRESPONSE']._serialized_end=370
  _globals['_EXITCOMMANDREQUEST']._serialized_start=372
  _globals['_EXITCOMMANDREQUEST']._serialized_end=392
  _globals['_EXITCOMMANDRESPONSE']._serialized_start=394
  _globals['_EXITCOMMANDRESPONSE']._serialized_end=432
  _globals['_GETUTILREQUEST']._serialized_start=434
  _globals['_GETUTILREQUEST']._serialized_end=464
  _globals['_GETUTILRESPONSE']._serialized_start=466
  _globals['_GETUTILRESPONSE']._serialized_end=536
  _globals['_MASTERTOWORKER']._serialized_start=539
  _globals['_MASTERTOWORKER']._serialized_end=750
# @@protoc_insertion_point(module_scope)