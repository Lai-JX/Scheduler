import csv
from runtime.rpc import scheduler_server, scheduler_client, operator_server
from controller import Controller
from cluster.cluster import CLUSTER

import argparse
import threading
import utils
import copy
from jobs import JOBS
import flags
FLAGS = flags.FLAGS

def parse_job_file(trace_file):
    fd = open(trace_file, 'r')
    deli = ','
    # 根据文件类型确定分隔符
    if ((trace_file.find('.csv') == (len(trace_file) - 4))):
        deli = ','
    elif ((trace_file.find('.txt') == (len(trace_file) - 4))):
        deli = ' '

    reader = csv.DictReader(fd, delimiter = deli) 
    ''' Add job from job trace file'''
    keys = reader.fieldnames
    utils.print_fn('--------------------------------- Read TF jobs from: %s ---------------------------------' % trace_file) 
    utils.print_fn('    we get the following fields:\n        %s' % keys)
    job_idx = 0
    print("num_gpu_all:",CLUSTER.num_gpu)
    for row in reader: 
        #add job into JOBS,  JOBS = _TFJobs()
        if int(row['num_gpu']) <= CLUSTER.num_gpu:     # ljx (row['model_name'] != 'bert' and row['model_name'] != 'gpt2') and 
            JOBS.add_job(row, True)
            job_idx += 1

    assert JOBS.num_job == len(JOBS.job_list) 
    JOBS.sort_all_jobs()
    utils.print_fn('---------------------------------- Get %d TF jobs in total ----------------------------------' % job_idx)
    # JOBS.read_all_jobs()
    fd.close()

class Scheduler(object):
    def __init__(self, scheduler_port: int, controller_port: int, operator_port: int,) -> None:
        super().__init__()
        utils.print_fn("\nljx: Scheduler init!")
        self._logger = utils.make_logger(__name__, FLAGS.log_path+'/master.log')

        self._trainers = dict()                                                     # 根据job_id存对应的client（to trainer）
        self._server_for_trainer = self.make_server_for_trainer(scheduler_port)     # Scheduler._server_for_worker 运行scheduler_server.server
        self._server_for_operator = self.make_server_for_operator(operator_port)     # Scheduler._server_for_worker 运行scheduler_server.server

        self._num_workers = CLUSTER.num_node_p_switch                               # worker=node
        self._controller = Controller(controller_port, self._num_workers)           # Controller._server_for_worker 运行master_server.serve, 会等待所有的worker都准备好了
        self._src_num = 4           # ljx
        self._src_utils = [0 for _ in range(self._src_num)]
        self.finished_job_cnt = 0
        utils.print_fn('--------------------------------- End of Log and Scheduler ---------------------------------')

        # self._start_time = self._controller.get_time()

    def get_time(self):
        return self._controller.get_time()

    def make_server_for_trainer(self, port):
        callbacks = {
            'RegisterTrainer': self._register_trainer_impl,
            'ReportIterTime': self._report_itertime_impl, 
        }

        server_thread = threading.Thread(
            target=scheduler_server.serve,
            args=(port, self._logger, callbacks))
        server_thread.setDaemon(True)
        server_thread.start()

        return server_thread
    
    def make_server_for_operator(self, port):
        callbacks = {
            'AddJobs': self._add_jobs_impl,
            'DeleteJobs': self._delete_jobs_impl, 
        }

        server_thread = threading.Thread(
            target=operator_server.serve,
            args=(port, self._logger, callbacks))
        server_thread.setDaemon(True)
        server_thread.start()

        return server_thread
    

    def _register_trainer_impl(self, trainer_ip, trainer_port, job_id_list):
        success = True
        job_id = max(job_id_list)
        self._logger.info(f'scheduler, before register, {job_id} {trainer_ip}:{trainer_port} {self._trainers.keys()}')
        
        # assert job_id not in self._trainers
        tmp_client = scheduler_client.SchedulerClientForTrainer(self._logger, job_id_list, trainer_ip, trainer_port)
        self._trainers[job_id] = tmp_client
        self._logger.info(f'scheduler, register, {job_id}-{job_id_list}, {trainer_ip}:{trainer_port}')

        return success
    
    def _report_itertime_impl(self, job_id, iter_time, src_utils):
        success = True
        num_gpu = 0
        for rjob_id in job_id:
            if rjob_id>=0:
                rjob = JOBS.find_runnable_job(rjob_id)
                rjob['real_itertime'] = copy.deepcopy(list(iter_time))
                num_gpu = rjob['num_gpu']
        for i in range(self._src_num): # cpu util is approximate
            self._src_utils[i] += src_utils[i]*num_gpu
        self._logger.info(f'scheduler, update job {job_id} iter_time {list(iter_time)}; src_utils {src_utils} -> {self._src_utils}')
        return success

    def _add_jobs_impl(self, file_path):
        success = True
        JOBS.job_lock.acquire()

        JOBS.print_job_events()
        parse_job_file(file_path)
        JOBS.prepare_job_start_events()

        JOBS.job_lock.release()
        self._logger.info(f'scheduler, add jobs, from path:{file_path}')
        return success
    
    def _delete_jobs_impl(self, job_id_list):
        success = True
        for job_id in job_id_list:
            JOBS.delete_queue.put(job_id)
        self._logger.info(f'scheduler, delete jobs, job_id_list:{job_id_list}')
        return success
    

    def query_stats(self, job_id_list):
        job_id = max(job_id_list)
        self._logger.info(f'scheduler, query_stats, job_id: {job_id} ')
        assert job_id in self._trainers
        finished_iterations, iteration_time = self._trainers[job_id].query_stats()
        return finished_iterations, iteration_time

    def save_model(self, job_id_list):
        job_id = max(job_id_list)
        assert job_id in self._trainers
        success = self._trainers[job_id].save_model()
        return success

    def has_ready_jobs(self, tmp_time):
        if len(JOBS.job_events)>0 and JOBS.job_events[0]['time']<=tmp_time:
            return True 
        else:
            return False

    def has_running_trainers(self, running_jobs):                   # done_queue中存储的是，running_jobs中已完成的job
        if running_jobs>self._controller.done_queue.qsize():
            return True
        else:
            return False
    
    def clear_src_utils(self):
        self._src_utils = [0 for _ in range(self._src_num)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scheduler_port', type=int, default=9011)
    parser.add_argument('--controller_port', type=int, default=9012)
    args = parser.parse_args()

    scheduler = Scheduler(args.scheduler_port, args.controller_port)