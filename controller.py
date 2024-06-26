import argparse
import time
import threading
import utils
import queue
import flags

from runtime.rpc import master_server, master_client
import log
FLAGS = flags.FLAGS

class Controller(object):
    def __init__(self, port: int, num_workers: int) -> None:
        super().__init__()
        print("\nljx: Controller init!")
        self._logger = utils.make_logger(__name__, FLAGS.log_path+'/master.log')

        self._num_workers = num_workers
        self._workers = []

        self.done_queue = queue.Queue()
        self.done_queue_lock = threading.Lock()

        self._server_for_worker = self.make_server_for_worker(port)
        
        self.wait_for_workers()
        
        self._jump_time = 0
        self._start_time = time.time()
        print("ljx: Controller wait finish!\n")
    
    def set_start_time(self):
        self._start_time = time.time()

    def get_time(self):
        return time.time()-self._start_time

    
    def make_server_for_worker(self, port: int):
        callbacks = {
            'RegisterWorker' : self._register_worker_impl,
            'Done' : self._done_impl,
        }

        server_thread = threading.Thread(
            target=master_server.serve,
            args=(port, self._logger, callbacks))
        server_thread.setDaemon(True)
        server_thread.start()

        return server_thread
    

    def execute(self, job_info):
        self._logger.info(f'controller execute job {list(job_info.job_id)}, use node: {list(job_info.node_id)}, gpu_num: {job_info.num_gpu}, resume: {job_info.is_resumed}')
        self._workers[min(list(job_info.node_id))].execute(job_info)        # 这里按理说应该向所有node_id对应的worker发送execute命令 → 不用，交给hovorod和mpi去管
    

    def kill(self, job_info):
        self._workers[min(list(job_info.node_id))].kill(job_info)

    def get_util(self, secs=20):
        num_workers = len(self._workers)
        avg_gpu_util_all, avg_cpu_util_all, avg_sm_util_all = 0, 0, 0
        def worker_get_util(worker):
            nonlocal avg_gpu_util_all, avg_cpu_util_all, avg_sm_util_all
            avg_gpu_util, avg_cpu_util, avg_sm_util = worker.get_util(secs)
            avg_gpu_util_all += avg_gpu_util
            avg_cpu_util_all += avg_cpu_util
            avg_sm_util_all += avg_sm_util
            
        threads = []
        for worker in self._workers:
            self._logger.info(f'controller get util of {num_workers} worker(s) of {worker._worker_id}: {secs}s')
            # print(f'controller get util of {num_workers} worker(s) of {worker}: {secs}s')
            thread = threading.Thread(target=worker_get_util, args=(worker,))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        
        avg_gpu_util_all /= num_workers
        avg_cpu_util_all /= num_workers
        avg_sm_util_all /= num_workers
        return avg_gpu_util_all, avg_cpu_util_all, avg_sm_util_all
    

    def _register_worker_impl(self, worker_ip, worker_port, num_gpus):
        success = True
        worker_id = len(self._workers)
        self._workers.append(master_client.MasterClientForWorker(self._logger, worker_id, worker_ip, worker_port))

        self._logger.info(f'controller, register, {worker_id}, {worker_ip}:{worker_port}')

        return success, worker_id
    

    def _done_impl(self, job_id, job_counter, worker_id, gpus, returncode) -> bool:
        self.done_queue_lock.acquire()
        success = True
        self.done_queue.put((self.get_time(), job_id, worker_id, gpus, returncode))
        self._logger.info(f'controller, done, {worker_id}, {job_id} - {job_counter} @ {worker_id}, {gpus}, return code: {returncode}')
        self.done_queue_lock.release()
        return success

    
    def wait_for_workers(self):
        while len(self._workers) < self._num_workers:
            time.sleep(5)

    def kill_workers(self):
        for workers in self._workers:
            workers.exit_command()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9012)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    controller = Controller(args.port, args.num_workers)