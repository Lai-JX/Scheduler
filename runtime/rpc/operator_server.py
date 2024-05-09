import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))


from runtime.rpc_stubs.operator_pb2 import AddJobsRequest, DeleteJobsRequest,AddJobsResponse, DeleteJobsResponse
from runtime.rpc_stubs.operator_pb2_grpc import OperatorServicer
import runtime.rpc_stubs.operator_pb2_grpc as t2s_rpc

import grpc
from concurrent import futures


class OperatorServer(OperatorServicer):
    def __init__(self, logger, callbacks) -> None:
        super().__init__()

        self._logger = logger
        self._callbacks = callbacks
    

    def AddJobs(self, request: AddJobsRequest, context):
        # return super().RegisterTrainer(request, context)
        assert 'AddJobs' in self._callbacks
        add_jobs_impl = self._callbacks['AddJobs']

        success = add_jobs_impl(request.trace_file)
        response = AddJobsResponse(success=success)

        return response

    def DeleteJobs(self, request: DeleteJobsRequest, context) -> DeleteJobsResponse:
        assert 'DeleteJobs' in self._callbacks
        delete_jobs_impl = self._callbacks['DeleteJobs']

        success = delete_jobs_impl(request.job_id)
        response = DeleteJobsResponse(success=success)

        return response


def serve(port, logger, callbacks):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    t2s_rpc.add_OperatorServicer_to_server(OperatorServer(logger, callbacks), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    # logger.info(f'scheduler, rpc, start, server @ {port}')
    
    server.wait_for_termination()