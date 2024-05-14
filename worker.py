import argparse
from logging import root
from statistics import mean
import utils
import threading
import time
import subprocess
import os
import signal
import math

from runtime.rpc import worker_server, worker_client
from task import Task


class Worker(object):
    def __init__(self, master_ip, master_port, worker_ip, worker_port, gpus: str, trace_name, this_dir, log_path) -> None:
        super().__init__()

        self._logger = utils.make_logger(__name__, log_path)
        self._model_path = log_path[:-12].replace("results", "model")

        self._master_ip = master_ip
        self._master_port = master_port
        self._work_ip = worker_ip
        self._worker_port = worker_port
        self._worker_id = None
        self._trace_name = trace_name       # job log dir
        self._this_dir = this_dir
        self._check_task_flag = True
        
        self._gpus = gpus.split(',')
        self._num_gpus = len(self._gpus)

        self._client_for_master = worker_client.WorkerClientForMaster(self._logger, self._master_ip, self._master_port)

        self._server_for_master = self.make_server_for_master(self._worker_port)
        # time.sleep(60)
        self._tasks = dict()
        self._tasks_lock = threading.Lock()

        self.register()
        
    

    def register(self):
        while True:
            success, worker_id = self._client_for_master.register_worker(self._work_ip, self._worker_port, self._num_gpus)

            if success == True:
                self._worker_id = worker_id
                break
            
            time.sleep(5)
    

    def check_tasks(self):
        while self._check_task_flag:
            finished_tasks = []
            error_tasks = []
            self._tasks_lock.acquire()
            for (job_id, job_counter), (task, job_info) in self._tasks.items():
                if task.return_code == None: # the pid of task is the pid of mpirun
                    continue
                
                self._client_for_master.done(job_id, job_counter, self._worker_id, task._gpus, task.return_code)
                finished_tasks.append((job_id, job_counter))
                if task.return_code != 0:
                    print("job_id:",job_id)
                    with open(f'{self._this_dir}/workloads/test_{job_id}.txt', 'r') as f:
                        error_text = f.read()
                        self._logger.info(f'error info: {job_id} {job_counter} \n'+error_text)
            
            for (job_id, job_counter) in finished_tasks: 
                self._tasks.pop((job_id, job_counter))
            self._tasks_lock.release()
            time.sleep(1)
    

    def make_server_for_master(self, port: int):
        callbacks = {
            'Execute' : self._execute_impl,
            'Kill' : self._kill_impl,
            'ExitCommand' : self._exit_command_impl,
            'GetUtil' : self._get_util_impl,
        }
        # GetUtil is deprecated

        server_thread = threading.Thread(
            target=worker_server.serve,
            args=(port, self._logger, callbacks))
        server_thread.setDaemon(True)
        server_thread.start()

        return server_thread
    

    def _execute_impl(self, job_info) -> bool:
        # self._logger.info(f'{"ljx: worker execute!", "schedule ip:", self._master_ip}')
        success = True

        self._logger.info(f'job_info.node_id: {job_info.node_id}')
        task = Task(job_info, self._master_ip, self._trace_name, self._this_dir, self._model_path)
        cmd = task.run()
        self._tasks_lock.acquire()
        self._tasks[(max(task._job_id), max(task._job_counter))] = (task, job_info)
        self._tasks_lock.release()

        self._logger.info(f'{self._worker_id}, execute, {task._job_id} - {task._job_counter}, {task._gpus}, {" ".join(cmd)}')

        # print(success)

        return success
    

    def _kill_impl(self, job_info) -> bool:
        job_id = max(job_info.job_id)
        job_counter = max(job_info.job_counter)
        self._tasks_lock.acquire()
        if (job_id, job_counter) not in self._tasks:
            self._tasks_lock.release()
            return False

        task, _ = self._tasks.pop((job_id, job_counter))
        self._tasks_lock.release()
        task.terminate()
        task.wait()
        self._logger.info(f'{self._worker_id}, kill, {job_id} - {job_counter}, {job_info.gpus}')
        
        with open(f'{self._this_dir}/workloads/test_{job_id}.txt', 'r') as f:
            kill_text = f.read()
            self._logger.info(f'kill info: {job_id} {job_counter} \n'+kill_text)

        return True

    def _exit_command_impl(self):
        self._logger.info(f'{self._worker_id} exit')
        self._check_task_flag = False
        return True

    def _get_util_impl(self, secs):
        # prepare
        device_list = range(self._num_gpus)
        process_list = []
        os.system("rm -rf /root/tmp/profiling_*_*.xml")
        os.system("rm -rf /root/tmp/profiling_*_*.out")
        self._logger.info(f'current path: {os.getcwd()}')
        # start subprocess
        # gpu
        for device in device_list:
            filename = "/root/tmp/profiling_" + str(self._worker_id) + "_" +str(device) + ".xml"
            command = "exec nvidia-smi -q -i " + str(device) + " -x -l 1 -f " + filename        # -x 表示输出格式为xml
            process_list.append(subprocess.Popen(command, shell=True))
        # cpu
        cpu_command = "exec top -d 1 -bn " + str(secs) + " | grep Cpu > /root/tmp/profiling_" + str(self._worker_id) + "_cpu.out"   # -d刷新间隔时间，-b: 表示以批处理模式运行，这种模式下 top 不会占用终端，允许其他命令的输出， -n刷新次数
        cpu_process = subprocess.Popen(cpu_command, shell=True)
        # io
        sm_command = "exec dcgmi dmon -g 2 -e 1002,1003,1005,1004  > /root/tmp/profiling_" + str(self._worker_id) + "_sm.out"
        sm_process = subprocess.Popen(sm_command, shell=True)

        # wait
        count = 0
        time.sleep(secs)
        for process in process_list:
            process.send_signal(signal.SIGINT)
            process.terminate()
            count += 1
            process.wait()
        cpu_process.send_signal(signal.SIGINT)
        sm_process.send_signal(signal.SIGINT)
        cpu_process.wait()
        sm_process.wait()

        # handle results
        useful_ratio = 0
        gpu_utils = []
        for device in device_list:
            filename = "/root/tmp/profiling_" + str(self._worker_id) + "_" +str(device) + ".xml"
            memory_usage, utilization = utils.parse_xml(filename)
            for i in range(len(memory_usage)):
                memory_usage[i] = int(memory_usage[i].split(' ')[0])
                utilization[i] = int(utilization[i].split(' ')[0])
            # self._logger.info(f'{memory_usage}, {utilization}')
            sorted_memory_usages = sorted(memory_usage)
            gpu_util_device = 0
            gpu_util_cnt = 0
            # print(memory_usage, sorted_memory_usages)
            for i in range(len(memory_usage)):
                if math.isclose(memory_usage[i], sorted_memory_usages[-2], rel_tol=1e-1):
                    gpu_util_device += utilization[i]
                    gpu_util_cnt += 1
            gpu_utils.append(gpu_util_device/gpu_util_cnt)
            self._logger.info(f'gpu util of device {device}: {memory_usage} {utilization} {gpu_utils[-1]}')
        gpu_util = sum(gpu_utils)/len(gpu_utils)
        self._logger.info(f'gpu util: {gpu_utils} {gpu_util}')

        
        cpu_util_list = []
        util_str_list = open("/root/tmp/profiling_" + str(self._worker_id) + "_cpu.out", "r").read().split('\n')
        for i in range(secs):
            # print(util_str_list[i])
            if util_str_list[i].split()[7] == 'id,':
                print(util_str_list[i])
                idle = 100.0
            else:
                idle = float(util_str_list[i].split()[7])
            cpu_util_list.append(round(100.0 -idle, 3)) # 由于之前记录的是空闲的cpu时间占比，所以这里要用100来减
        cpu_util = sum(cpu_util_list)/len(cpu_util_list)
        self._logger.info(f'cpu util: {cpu_util_list}, {cpu_util}')
        
        sm_util_list, sm_util = [], 0
        # util_str_list = open("/root/tmp/profiling_" + str(self._worker_id) + "_sm.out", "r").read().split('\n')
        # if len(util_str_list) > 1:
        #     for i in range(secs):
        #         print(util_str_list,util_str_list[i])
        #         kB_read = float(util_str_list[i].split()[2])
        #         io_util_list.append(kB_read)
        #     io_read = sum(io_util_list[1:])/(len(io_util_list)-1)
        # self._logger.info(f'io read: {io_util_list} {io_read}')
        # res = {'SMACT':[], 'SMOCC':[], 'TENSO':[], 'DRAMA':[]}
        
        # sm_util = 0
        with open("/root/tmp/profiling_" + str(self._worker_id) + "_sm.out", 'r') as file:
            lines = file.readlines()
            n, i = len(lines), 1
            while i < n:
                line = lines[i].split()
                if len(line) > 0 and line[0]=='GPU':
                    sm_util_list.append(float(line[2]))
                i += 1
        sm_util = mean(sm_util_list)
        self._logger.info(f'sm_util: {sm_util}')

        return gpu_util, cpu_util, sm_util

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--master_ip', type=str, required=True)
    parser.add_argument('--master_port', type=int, default=9012)
    parser.add_argument('--worker_port', type=int, default=9001)
    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7')  # ljx 4049上只有两个gpu，记得更改
    parser.add_argument('--trace_name', type=str, default='test')
    parser.add_argument('--this-dir', type=str, default='./')
    parser.add_argument('--log_path',type=str, default='./my_log.log')  # ljx

    args = parser.parse_args()

    worker_ip = utils.get_host_ip()
    worker = Worker(args.master_ip, args.master_port, worker_ip, args.worker_port, args.gpus, args.trace_name, args.this_dir, args.log_path)   # ljx
    worker.check_tasks()