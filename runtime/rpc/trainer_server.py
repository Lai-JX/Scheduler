import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))

from runtime.rpc_stubs.scheduler_to_trainer_pb2 import QueryStatsResponse, SaveModelResponse
from runtime.rpc_stubs.scheduler_to_trainer_pb2_grpc import SchedulerToTrainerServicer
import runtime.rpc_stubs.scheduler_to_trainer_pb2_grpc as s2t_rpc

import grpc
from concurrent import futures


class TrainerServerForScheduler(SchedulerToTrainerServicer):
    def __init__(self, logger, callbacks) -> None:
        super().__init__()
        self._logger = logger
        self._callbacks = callbacks
    

    def QueryStats(self, request, context):
        # return super().QueryStats(request, context)
        assert 'QueryStats' in self._callbacks
        query_stats_impl = self._callbacks['QueryStats']

        finished_iterations, iteration_time = query_stats_impl()
        response = QueryStatsResponse(finished_iterations=finished_iterations, iteration_time=iteration_time)

        return response
    
    def SaveModel(self, request, context):
        # return super().QueryStats(request, context)
        assert 'SaveModel' in self._callbacks
        save_model_impl = self._callbacks['SaveModel']

        success = save_model_impl()
        response = SaveModelResponse(success=success)

        return response


def serve(port, logger, callbacks):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    s2t_rpc.add_SchedulerToTrainerServicer_to_server(TrainerServerForScheduler(logger, callbacks), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    logger.info(f'trainer, rpc, start, server @ {port}')
    
    server.wait_for_termination()