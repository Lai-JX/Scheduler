# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: trainer_to_scheduler.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1atrainer_to_scheduler.proto\"R\n\x16RegisterTrainerRequest\x12\x12\n\ntrainer_ip\x18\x01 \x01(\t\x12\x14\n\x0ctrainer_port\x18\x02 \x01(\r\x12\x0e\n\x06job_id\x18\x03 \x03(\x05\"*\n\x17RegisterTrainerResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\"M\n\x15ReportIterTimeRequest\x12\x0e\n\x06job_id\x18\x01 \x03(\x05\x12\x11\n\titer_time\x18\x02 \x03(\x01\x12\x11\n\tsrc_utils\x18\x03 \x03(\x01\")\n\x16ReportIterTimeResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\x32\xa1\x01\n\x12TrainerToScheduler\x12\x46\n\x0fRegisterTrainer\x12\x17.RegisterTrainerRequest\x1a\x18.RegisterTrainerResponse\"\x00\x12\x43\n\x0eReportIterTime\x12\x16.ReportIterTimeRequest\x1a\x17.ReportIterTimeResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'trainer_to_scheduler_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_REGISTERTRAINERREQUEST']._serialized_start=30
  _globals['_REGISTERTRAINERREQUEST']._serialized_end=112
  _globals['_REGISTERTRAINERRESPONSE']._serialized_start=114
  _globals['_REGISTERTRAINERRESPONSE']._serialized_end=156
  _globals['_REPORTITERTIMEREQUEST']._serialized_start=158
  _globals['_REPORTITERTIMEREQUEST']._serialized_end=235
  _globals['_REPORTITERTIMERESPONSE']._serialized_start=237
  _globals['_REPORTITERTIMERESPONSE']._serialized_end=278
  _globals['_TRAINERTOSCHEDULER']._serialized_start=281
  _globals['_TRAINERTOSCHEDULER']._serialized_end=442
# @@protoc_insertion_point(module_scope)
