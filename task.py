from runtime.rpc_stubs.master_to_worker_pb2 import JobInfo

import subprocess
import os
import utils

port_count = -1

class Task(object):
    def __init__(self, job_info: JobInfo, scheduler_ip, trace_name, this_dir, model_path) -> None:
        super().__init__()

        self._job_num = job_info.num                # job数
        self._node_id = list(job_info.node_id)      # 需要用到的node
        self._job_id = job_info.job_id              # list
        self._job_name = job_info.job_name          # list
        self._batch_size = job_info.batch_size      # list
        self._iterations = job_info.iterations      # list
        self._gpus = job_info.gpus                  # 第一个node的gpu_list
        self._scheduler_ip = scheduler_ip           # 即master_ip
        self._num_gpu = job_info.num_gpu            # 第一个job的gpu
        self._this_dir = this_dir
        self._job_counter = job_info.job_counter    # list
        self._trace_name = trace_name
        self._is_resumed = job_info.is_resumed      # 是否运行过
        self._model_path = model_path
    

    def get_idle_port(self):
        # return 9013 + 2*min(self._node_id) + int(self._gpus.split(',')[0])      # ljx 每个节点只有两个2个gpu，所以这里先改为2
        global port_count
        port_count += 1
        return 9013 + (port_count % 90)                                           # ljx
    
        # return utils.find_free_port()
    def set_hostfile(self):
        # 配置
        hosts = {0:'job-ljx-70bbc-5pd66'}   # node_id:hostname
        ssh_ports = {0:22}                  # node_id:ssh_port
        gpu_per_node = 8

        hostfile_list = []
        count = 0                           # 已分配的gpu数
        for node_id in self._node_id:
            gpu_to_allocate = self._num_gpu - count
            if gpu_to_allocate < gpu_per_node:
                hostfile_list.append(f'{hosts[node_id]} slots={gpu_to_allocate} port={ssh_ports[node_id]}')
                count += gpu_to_allocate
            else:
                hostfile_list.append(f'{hosts[node_id]} slots={gpu_per_node} port={ssh_ports[node_id]}')
                count += gpu_per_node
        return hostfile_list



    @staticmethod
    def test_kill_restart():
        bash_cmd = 'nvidia-smi; sleep 2m; date'
        return bash_cmd


    def real_job(self):
        bash_cmd = f'bash {self._this_dir}/workloads/run.sh'
        print("job_num:", self._job_num, self._is_resumed)
        for i in range(self._job_num):
            # resumed = ['--resume' if is_resumed else '' for is_resumed in self._is_resumed]
            # 设置job的参数，依次为model    batch-size    num-worker    prefetch-factor    train-dir    iters    job-id     resume         iters为剩余迭代次数
            bash_cmd += f' {self._job_name[i]} {self._batch_size[i]} 0 2 -1 {self._iterations[i]} {self._job_id[i]} {self._job_counter[i]} {self._is_resumed[i]}' # 0、2、-1分别代表num_worker、prefetch_factor和train_dir

        bash_cmd += f' {self._num_gpu}'
        bash_cmd += f' {self._model_path}'      # 模型保存的位置
        bash_cmd += f' --scheduler-ip {self._scheduler_ip}'
        bash_cmd += f' --trainer-port {self.get_idle_port()} --this-dir {self._this_dir}/workloads'
        return bash_cmd

    def run(self):
        bash_cmd = ''
        # if self._job_name == 'test_kill_restart':
        #     bash_cmd = self.test_kill_restart()
        # else:
        bash_cmd = self.real_job()

        cmd = bash_cmd.split()

        hostfile_dir = self._this_dir+'/workloads/hostfiles'
        assert os.path.exists(hostfile_dir)
        print('self._node_id:',self._node_id)
        hostfile_list = self.set_hostfile()         # 设置hostfile
        

        ch = '-'
        job_id_str = ch.join([str(x) for x in list(self._job_id)])
        job_counter_str = ch.join([str(x) for x in list(self._job_counter)])
        # print(self._iterations)
        with open(hostfile_dir+f'/hostfile-[{job_id_str}]-[{job_counter_str}]', 'w') as f:
            f.writelines(hostfile_list)
        utils.print_ljx("task.run:hostfile_list:", hostfile_list)
        utils.print_ljx("log path after here:",self.log_path, '\n')
        utils.print_ljx('environ_dict["CUDA_VISIBLE_DEVICES"]',self._gpus)
        # exit(0)
        environ_dict = dict(os.environ)
        environ_dict['CUDA_VISIBLE_DEVICES'] = self._gpus
        # print(environ_dict)
        with open(self.log_path, 'w+') as f:
            self._handler = subprocess.Popen(
                cmd, 
                stdout=f,
                stderr=f,               # 之后再改为f
                env=environ_dict,
            )

        return cmd
    

    def terminate(self):
        self._handler.terminate()
    
    def wait(self):
        self._handler.wait()
    

    @property
    def return_code(self):              # 检查进程是否终止，如果终止返回 returncode，否则返回 None
        return self._handler.poll()

    @property
    def pid(self):
        return self._handler.pid


    @property
    def log_path(self):
        # print('self._trace_name:',self._trace_name,os.path.exists(f'{self._trace_name}/'))
        # if not os.path.exists(f'{self._trace_name}/'):
        #     print(not os.path.exists(f'{self._trace_name}/'))
        #     os.makedirs(f'{self._trace_name}/')
        path = ''
        for i in range(self._job_num):
            if i==0:
                path = f'{self._trace_name}/{self._job_id[i]}-{self._job_counter[i]}-{self._job_name[i]}'
            else:
                path += f'_{self._job_id[i]}-{self._job_counter[i]}-{self._job_name[i]}'
        return path + '.txt'
