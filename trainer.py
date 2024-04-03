from runtime.rpc import trainer_client, trainer_server

import argparse
import utils
import time
import threading


class Trainer(object):
    def __init__(self, scheduler_ip, scheduler_port, trainer_ip, trainer_port, job_id) -> None:
        super().__init__()

        self._trainer_ip = trainer_ip
        self._trainer_port = trainer_port
        self._job_id = job_id
        # self._batch_size = batch_size
        # self._demotion_threshold = demotion_threshold

        self._logger = utils.make_logger(__name__)
        self._start_time = time.time()
        self._finished_iteraions = 0
        self._reported_iteraions = 0        # 已经上报的迭代次数
        self._save_flag = False             # 是否开始保存模型
        self._save_finished = False         # 模型是否保存完成
        self._iteration_time = 0            # 迭代时间（实时更新）
        self._iteration_time_list = []      # 保存最近10次迭代的时间

        self._client_for_scheduler = trainer_client.TrainerClientForScheduler(self._logger, scheduler_ip, scheduler_port)
        self.init_stats()

        self._server_for_scheduler = self.make_server_for_scheduler(self._trainer_port)

        self.register()

        self._logger.info(f'job {self._job_id}, trainer, start, {self._start_time}')

    def register(self):
        success = False
        while success == False:
            success = self._client_for_scheduler.register_trainer(self._trainer_ip, self._trainer_port, self._job_id)
            # utils.print_ljx("register:",success)    # success=(Flase,None)    /   True
        
    def report_itertime(self, iter_time, src_utils):
        success = self._client_for_scheduler.report_itertime(self._job_id, iter_time, src_utils)
        utils.print_ljx("report_itertime:", success)
        self._logger.info(f'job {self._job_id} reported iteration time {iter_time} and resource utils {src_utils}')

    def init_stats(self):
        pass
    

    def update_stats(self, iteration_time):
        self._finished_iteraions += 1
        self._logger.info(f'trainer update_stats: {self._finished_iteraions}, {iteration_time}')


    def record(self, iteration_time):
        n = len(self._iteration_time_list)
        if n >= 10:
            self._iteration_time_list.pop(0)
            n -= 1
        self._iteration_time_list.append(iteration_time)

        self._iteration_time = sum(self._iteration_time_list) / (n+1)
        self.update_stats(iteration_time)                           # 记录时用的是单次的迭代时间，上报用的是过去10次的平均迭代时间
        if self._save_flag:                                         # 通知保存模型
            self._save_flag = False                                 # 避免多次保存，导致最后保存了一半
            return 'save'

        # if self.demotion() == True:
        #     self._client_for_scheduler.report_stats(self._job_id, self._finished_iteraions, True)



    def make_server_for_scheduler(self, port: int):
        callbacks = {
            'QueryStats' : self._query_stats_impl,
            'SaveModel' : self._save_model_impl,
        }

        server_thread = threading.Thread(
            target=trainer_server.serve,
            args=(port, self._logger, callbacks))
        server_thread.setDaemon(True)
        server_thread.start()

        return server_thread

    def _cal_report_iteration(self):
        tmp = self._finished_iteraions
        report_iteraions = tmp - self._reported_iteraions       # 两次查询的间隔执行了多少次迭代
        self._reported_iteraions = tmp
        return report_iteraions

    def _query_stats_impl(self):
        report_iteraions = self._cal_report_iteration()
        self._logger.info(f'trainer query stats, {report_iteraions}')
        return report_iteraions, self._iteration_time
    
    def _save_model_impl(self):
        print("\n\n\n\n\n model save, ", self._iteration_time)
        self._save_flag = True
        self._save_finished = False
        self._logger.info(f'trainer model save, {self._iteration_time}')
        time.sleep(self._iteration_time)
        while not self._save_finished:      # 等待完成，避免导致模型保存一半等问题(为了在两个进程间同步，不得已采用这种方法)
            time.sleep(1)
        return True
    
    def save_finish(self):
        self._save_finished = True



    # def demotion(self) -> bool:
    #     if self._demotion_threshold == None:
    #         return False
        
    #     return (time.time() - self._start_time >= self._demotion_threshold)


if __name__  == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scheduler_ip', type=str, required=True)
    parser.add_argument('--scheduler_port', type=int, default=9011)
    parser.add_argument('--trainer_port', type=int)
    parser.add_argument('--job_id', type=int, default=-1)
    # parser.add_argument('--batch_size', type=int, default=8)
    # parser.add_argument('--demotion_threshold', type=float, default=None)
    args = parser.parse_args()

    trainer = Trainer(args.scheduler_ip, args.scheduler_port, utils.get_host_ip(), args.trainer_port, args.job_id)