from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess
import csv
import math

import utils
import flags 
from cluster import cluster 
import jobs
import time

FLAGS = flags.FLAGS
CLUSTER = cluster.CLUSTER
JOBS = jobs.JOBS



class _Log(object):

    def __init__(self):
        self.log_path = ''
        self.log_file = ''
        self.log_cpu = ''
        self.log_gpu = ''
        self.log_network = ''
        self.log_mem = ''
        self.log_job = ''
        self.log_list = list()
        self.cpu_list = list()
        self.gpu_list = list()
        self.job_list = list()
        self.mem_list = list()
        self._start_time = 0
        self._gpu_util = 0
        self._cpu_util = 0
        self._io_speed = 0

    def init_log(self):
        self.log_path = FLAGS.log_path
        if self.log_path[-1] == '/':
            self.log_path = self.log_path[:-1]
        # utils.print_fn(self.log_path)
        # utils.print_fn(' ')

        #prepare folder
        cmd = 'mkdir -p ' + self.log_path
        ''' python 2.7
        status, output = commands.getstatusoutput(cmd)
        '''
        #python 2.7 & 3
        ret = subprocess.check_output(cmd, shell=True)

        self.log_file = self.log_path + '/cluster.csv'
        self.log_job = self.log_path + '/job.csv'


        fd = open(self.log_file, 'w+')
        log_writer = csv.writer(fd)  
        # if FLAGS.scheme == 'gandiva':
        #     log_writer.writerow(['time', 'idle_node', 'busy_node', 'full_node', 'fra_gpu', 'busy_gpu', 'pending_job', 'running_job', 'completed_job', 'len_g1', 'len_g2', 'len_g4', 'len_g8', 'len_g16', 'len_g32', 'len_g64'])
        # else:
        log_writer.writerow(['time', 'queue_length', 'blocking_index', 'gpu_util', 'cpu_util', 'sm_util'])
        fd.close()

            
        fd = open(self.log_job, 'w+')
        log_writer = csv.writer(fd)  
        if FLAGS.schedule == 'gpu-demands':
            log_writer.writerow(['time', '1-GPU', '2-GPU', '4-GPU', '8-GPU', '12-GPU', '16-GPU', '24-GPU', '32-GPU'])
        else:
            # if FLAGS.scheme == 'count':
            log_writer.writerow(['time', 'job_id', 'num_gpu', 'submit_time', 'start_time', 'end_time', 'executed_time', 'real_executed_time', 'JCT', 'pending_time', 'preempt', 'resume', 'promote'])   # 'duration',
            # else:
            #     log_writer.writerow(['time', 'job_id', 'num_gpu', 'submit_time', 'start_time', 'end_time', 'executed_time', 'real_executed_time', 'JCT', 'duration', 'pending_time', 'job_counter', 'promote'])
        fd.close()

        self._start_time = time.time()


    def dump_all_logs(self):
        fd = open(self.log_file, 'a+')
        log_writer = csv.writer(fd)  
        for log in self.log_list:
            log_writer.writerow(log)
        fd.close()
        del self.log_list[:]



    def checkpoint(self, event_time, scheduler, new_util=False, secs=20):
        '''
        Record cluster, and job information, including:
        time
        queue length 
        blocking index
        gpu util
        cpu util
        SMACT
        '''
        queue_length = 0
        blocking_index = 0
        for rjob in JOBS.runnable_jobs:
            if rjob['status'] == 'PENDING':
                queue_length += 1
                # print(rjob['job_idx'])
                blocking_index += rjob['pending_time']/(rjob['remaining_iterations']*rjob['iteration_time'])
        if queue_length>0:
            blocking_index /= queue_length
        if new_util:
            self._gpu_util, self._cpu_util, self._io_speed = scheduler._controller.get_util(secs)
        self.log_list.append([event_time, queue_length, blocking_index, self._gpu_util, self._cpu_util, self._io_speed])
        if len(self.log_list) >= 1:
            self.dump_all_logs()

    def checkpoint_utils(self, event_time, scheduler):          # ljx:Deprecated
        '''
        Record cluster, and job information, including:
        time
        queue length 
        blocking index
        gpu util
        cpu util
        io read speed
        '''
        queue_length = 0
        blocking_index = 0
        for rjob in JOBS.runnable_jobs:
            if rjob['status'] == 'PENDING':
                queue_length += 1
                blocking_index += rjob['pending_time']/(rjob['remaining_iterations']*rjob['iteration_time'])
        if queue_length>0:
            blocking_index /= queue_length
        self.log_list.append([event_time, queue_length, blocking_index, scheduler._src_utils[0]/CLUSTER.num_gpu, scheduler._src_utils[1]/CLUSTER.num_node, scheduler._src_utils[2]/CLUSTER.num_node])
        if len(self.log_list) >= 1:
            self.dump_all_logs()


    def dump_job_logs(self):
        fd = open(self.log_job, 'a+')
        log_writer = csv.writer(fd)  
        for log in self.job_list:
            log_writer.writerow(log)
        fd.close()
        del self.job_list[:]

    def job_complete(self, job, event_time):
        '''
        ['even_time', 'job_id', 'num_gpu', 'submit_time', 'start_time', 'end_time', 'executed time', 'real executed time', 'jct', 'duration', 'pending_time', 'job_counter', 'promote']
        '''
        job['end_time'] = event_time
        executed_time = job['end_time'] - job['start_time']
        real_executed_time = job['total_executed_time']
        jct = job['end_time'] - job['submit_time']
        # if FLAGS.scheme == 'count':
        self.job_list.append([event_time, job['job_id'], job['num_gpu'], job['submit_time'], job['start_time'], job['end_time'], executed_time, real_executed_time, jct, job['pending_time'], job['preempt'], job['resume'], job['promote']])   # job['duration']
        # else:
        #     self.job_list.append([event_time, job['job_id'], job['num_gpu'], job['submit_time'], job['start_time'], job['end_time'], executed_time, real_executed_time, jct, job['duration'], job['pending_time'], job['job_counter'], job['promote']])


        if len(self.job_list) >= 1:
            self.dump_job_logs()



LOG = _Log()

_allowed_symbols = [
    'LOG'
]