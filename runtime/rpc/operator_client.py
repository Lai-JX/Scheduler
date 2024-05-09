import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))

from runtime.rpc_stubs.operator_pb2 import AddJobsRequest, DeleteJobsRequest
import runtime.rpc_stubs.operator_pb2_grpc as t2s_rpc

import grpc


class OperatorClient(object):
    def __init__(self, server_ip, server_port) -> None:
        super().__init__()

        # self._logger = logger

        self._scheduler_ip = server_ip
        self._scheduler_port = server_port
        # self._logger.info(f'{self.addr}')
        channel = grpc.insecure_channel(self.addr)
        self._stub = t2s_rpc.OperatorStub(channel)
    

    @property
    def addr(self):
        return f'{self._scheduler_ip}:{self._scheduler_port}'
    

    def add_jobs(self, file_path):
        request = AddJobsRequest(trace_file=file_path)
        # self._logger.info(f'job {job_id} {request}')
        try:
            response = self._stub.AddJobs(request)
            # self._logger.info(f'job {job_id}, register, {response.success}')
            return response.success
        except Exception as e:
            # self._logger.info(f'job {job_id}, register, fail, {e}')
            print("AddJobs fail",e)
            return False, None

    def delete_jobs(self, jobs_id):
        request = DeleteJobsRequest()
        request.job_id.extend(jobs_id)
        response = self._stub.DeleteJobs(request)
        assert response.success == True
        return response.success

