from __future__ import print_function
import csv
import re
import sys
import types
import time
import math
import subprocess
#parse args
import argparse
import copy
import os
import cvxpy as cp
import numpy as np

import utils
import flags
import jobs
from cluster import cluster
import log
# import lp     # 暂时注释，'balance' 对应的放置策略
# from matching import Blossom_Same, _Packing
from scheduler import Scheduler
# import hosts
# import placement_scheme as scheme
# import cmd

from scheduler_impl.schedulers import *    # ljx
from scheduler_impl.schedulers_pre import *    # ljx

#parse input arguments
flags.DEFINE_string('trace_file', 'tf_job.csv',
                '''Provide TF job trace file (*.csv, *.txt).
                    *.csv file, use \',\' as delimiter; *.txt file, user \' \' as deliminter. 
                    Default file is tf_job.csv ''')
flags.DEFINE_string('log_path', 'result-' + time.strftime("%Y%m%d-%H-%M-%S", time.localtime()),
                '''Simulation output folder, including cluster/node/gpu usage trace, pending job_queue info.
                Default folder is result-[time]''')
flags.DEFINE_string('scheme', 'yarn',
                '''
                Job placement scheme:
                0.count, just resource counting, without assignment (which gpu, which cpu)
                1.yarn, ms yarn
                2.random
                3.crandom (consolidate + random)
                4.greedy
                5.balance
                6.cbalance (consolidate + balance)
                Default is yarn''')
flags.DEFINE_string('schedule', 'fifo',
                '''
                Job schedule scheme:
                1.fifo
                2.shortest, shortest-remaining-time job first
                3.shortest-gpu, shortest-remaining-gputime job first 
                4.dlas, discretized las 
                5.dlas-gpu, dlas using gpu time
                6.antman, AntMan
                7.themis, Themis
                8.multi-resource-blossom-same-gpu(-unaware), match jobs with same #GPU using blossom algorithm using gputime (unaware job duration)
                Default is fifo''')
flags.DEFINE_integer('num_switch', 1, 
                '''Part of cluster spec: the number of switches in this cluster, default is 1''')
flags.DEFINE_integer('num_node_p_switch', 32, 
                '''Part of cluster spec: the number of nodes under a single switch, default is 32''')
flags.DEFINE_integer('num_gpu_p_node', 8, 
                '''Part of cluster spec: the number of gpus on each node, default is 8''')
flags.DEFINE_integer('num_cpu_p_node', 64,
                '''Part of cluster spec: the number of cpus on each node, default is 64''')
flags.DEFINE_integer('mem_p_gpu', 49140,
                '''Part of cluster spec: memory capacity on each gpu(M), default is 49140''')
flags.DEFINE_string('cluster_spec', None,
                '''Part of cluster spec: cluster infra spec file, 
                this file will overwrite the specs from num_switch, num_node_p_switch, and num_gpu_p_node
                Spec format:
                    num_switch,num_node_p_switch,num_gpu_p_node
                    int,int,int''')
# # for multi_resource sharing
# flags.DEFINE_integer('multi_resource', 4, 
#                 '''Part of job spec: the num of resources used for each job, default is 4''')
# flags.DEFINE_float('weight_lbd', 0.0, '''The factor of the lower bound of expected weight (i jobs packing of n resources: i/n)''')

flags.DEFINE_boolean('print', True,         # ljx: 原本默认为Flase 
                '''Enable print out information, default is False''')
flags.DEFINE_boolean('flush_stdout', True, 
                '''Flush stdout, default is True''')
flags.DEFINE_integer('scheduler_port', 9011, '''The port of scheduler''')
flags.DEFINE_integer('controller_port', 9012, '''The port of controler''')  # trainer_port: begin with 9013; worker_port: 9001
flags.DEFINE_integer('schedule_interval', 10, '''The scheduling interval of scheduler, default is 10s''')
# flags.DEFINE_integer('fast_forwarding', 0, 
#                 '''Running iters when fast-forwarding for cluster experiment, default is 0, i.e., no fast-forwarding''')
# flags.DEFINE_integer('packing_num', 4, 
#                 '''maximum number of jobs in one packing''')
flags.DEFINE_version('0.1')


FLAGS = flags.FLAGS

#prepare JOBS list
JOBS = jobs.JOBS

#get host info
CLUSTER = cluster.CLUSTER
# CLUSTER_TMP = cluster.CLUSTER_TMP

#get LOG object
LOG = log.LOG


def parse_job_file(trace_file):
    #check trace_file is *.csv， 如cluster_exp/trace-data/cluster_trace.csv
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
    for row in reader: 
        #add job into JOBS,  JOBS = _TFJobs()
        # if int(row['num_gpu']) <= 16:     # ljx (row['model_name'] != 'bert' and row['model_name'] != 'gpt2') and 
        JOBS.add_job(row)
        job_idx += 1
        # if job_idx == 20:   # ljx:先只采用20个job
        #     break
        # JOBS.read_job_info(job_idx, 'num_gpu')
        # job_idx += 1    

    assert job_idx == len(JOBS.job_list) 
    assert JOBS.num_job == len(JOBS.job_list) 
    # JOBS.print_all_job_size_info()
    JOBS.sort_all_jobs()
    # print(lp.prepare_job_info(JOBS.job_list[0]))
    utils.print_fn('---------------------------------- Get %d TF jobs in total ----------------------------------' % job_idx)
    # JOBS.read_all_jobs()
    fd.close()

def parse_cluster_spec():
    if FLAGS.cluster_spec:
        # print(FLAGS.cluster_spec)
        spec_file = FLAGS.cluster_spec
        fd = open(spec_file, 'r')
        deli = ','
        if ((spec_file.find('.csv') == (len(spec_file) - 4))):
            deli = ','
        elif ((spec_file.find('.txt') == (len(spec_file) - 4))):
            deli = ' '
        reader = csv.DictReader(fd, delimiter = deli) 
        keys = reader.fieldnames
        utils.print_fn(keys)
        if 'num_switch' not in keys:
            return
        if 'num_node_p_switch' not in keys:
            return
        if 'num_gpu_p_node' not in keys:
            return
        if 'num_cpu_p_node' not in keys:
            return
        if 'mem_p_gpu' not in keys:
            return
        
        ''' there should be only one line remaining'''
        assert reader.line_num == 1

        ''' get cluster spec '''
        for row in reader:
            # utils.print_fn('num_switch %s' % row['num_switch'])
            FLAGS.num_switch = int(row['num_switch'])
            FLAGS.num_node_p_switch = int(row['num_node_p_switch'])
            FLAGS.num_gpu_p_node = int(row['num_gpu_p_node'])
            FLAGS.num_cpu_p_node = int(row['num_cpu_p_node'])
            FLAGS.mem_p_gpu = int(row['mem_p_gpu'])
        fd.close()

    utils.print_fn("num_switch: %d" % FLAGS.num_switch)
    utils.print_fn("num_node_p_switch: %d" % FLAGS.num_node_p_switch)
    utils.print_fn("num_gpu_p_node: %d" % FLAGS.num_gpu_p_node)
    utils.print_fn("num_cpu_p_node: %d" % FLAGS.num_cpu_p_node)
    utils.print_fn("mem_p_gpu: %d" % FLAGS.mem_p_gpu)

    '''init infra'''
    CLUSTER.init_infra()
    # CLUSTER_TMP.init_infra()
    # utils.print_fn(lp.prepare_cluster_info())
    utils.print_fn('--------------------------------- End of cluster spec ---------------------------------')
    return 





def main():

    # if FLAGS.schedule == 'multi-dlas-gpu': 
    #     if FLAGS.scheme != 'count':
    #         utils.print_fn("In Main, multi-dlas-gpu without count")
    #         exit()
    ''' Parse input'''
    parse_job_file(FLAGS.trace_file)    # 读取工作如cluster_exp/trace-data/cluster_trace.csv，并按提交时间排序
    parse_cluster_spec()                # 如cluster_exp/cluster_specs/n1g8.csv

    ''' prepare logging '''
    LOG.init_log()

    ''' scheduler '''
    scheduler = Scheduler(FLAGS.scheduler_port, FLAGS.controller_port)  # 启动master（Controller）服务端和Schedule服务端，会等待所有的worker都准备好了

    # lp.placement(JOBS.job_list[0])
    ''' Prepare jobs'''
    JOBS.prepare_job_start_events()                                     # 遍历所有job，将所有time添加到JOBS.job_events(sort events based on their submit time), 并将job加到对应的start_jobs列表中 every job has been in EVENT status

    # sim_job_events()
    if FLAGS.schedule == 'fifo':                                                                # fifo
        fifo_sim_jobs(scheduler)
    elif FLAGS.schedule == 'sjf':                                                               # shortestf first
        sjf_sim_jobs(scheduler)
    elif FLAGS.schedule == 'sjf-test':                                                          # shortestf first
        sjf_sim_jobs(scheduler)
    elif FLAGS.schedule == 'Tiresias':                                                          # shortestf first
        Tiresias_jobs(scheduler)
    elif FLAGS.schedule == 'sjf-ffs' or FLAGS.schedule == 'sjf-ffss':                           # sjf-ffs
        sjf_ffs_jobs(scheduler)
    elif FLAGS.schedule == 'sjf-ffs-m' or FLAGS.schedule == 'sjf-ffss-m':                           # sjf-ffs
        sjf_ffs_jobs(scheduler,is_preempt=True,with_mps=True)
    elif FLAGS.schedule == 'sjf-ffs-no-preempt' or FLAGS.schedule == 'sjf-ffss-no-preempt':                                               # sjf-ffs
        sjf_ffs_jobs(scheduler,is_preempt=False)
    elif FLAGS.schedule == 'sjf-ffs-no-preempt-m' or FLAGS.schedule == 'sjf-ffss-no-preempt-m':                                               # sjf-ffs
        sjf_ffs_jobs(scheduler,is_preempt=False,with_mps=True)
    elif FLAGS.schedule == 'sjf-bsbf' or FLAGS.schedule == 'sjf-bsbfs':                                                          # sjf-bsbf
        sjf_bsbf_jobs(scheduler)
    elif FLAGS.schedule == 'sjf-bsbf-m' or FLAGS.schedule == 'sjf-bsbfs-m':                                                        # sjf-bsbf-m
        sjf_bsbf_jobs(scheduler,is_preempt=True,with_mps=True)
    elif FLAGS.schedule == 'sjf-bsbf-no-preempt' or FLAGS.schedule == 'sjf-bsbfs-no-preempt':                                               # sjf-bsbf-no-preempt
        sjf_bsbf_jobs(scheduler,is_preempt=False)
    elif FLAGS.schedule == 'sjf-bsbf-no-preempt-m' or FLAGS.schedule == 'sjf-bsbfs-no-preempt-m':                                             # sjf-bsbf-no-preempt-m
        sjf_bsbf_jobs(scheduler,is_preempt=False,with_mps=True)
    # elif FLAGS.schedule == 'shortest':                                                        # SRTF
    #     shortest_first_sim_jobs(scheduler)
    # elif FLAGS.schedule == 'shortest-gpu':                                                  # SRSF
    #     shortest_first_sim_jobs(scheduler, True)
    # elif FLAGS.schedule == 'dlas-gpu':                                                      # Tiresias
    #     dlas_sim_jobs(scheduler, True)
    # elif FLAGS.schedule == 'multi-resource-blossom-same-gpu':                               # Muri-S
    #     multi_resource_blossom_same_sim_jobs(scheduler, True)
    # elif FLAGS.schedule == 'multi-resource-blossom-same-gpu-unaware':                       # Muri-L
    #     multi_resource_blossom_same_sim_jobs(scheduler, True, know_duration=False)
    # elif FLAGS.schedule == 'multi-resource-blossom-same-gpu-unaware-worstordering':
    #     multi_resource_blossom_same_sim_jobs(scheduler, True, know_duration=False, ordering=2)
    # elif FLAGS.schedule == 'multi-resource-gpu-unaware':
    #     multi_resource_blossom_same_sim_jobs(scheduler, True, know_duration=False, blossom=False)
    # elif FLAGS.schedule == 'themis':
    #     themis_sim_jobs(scheduler, )
    # elif FLAGS.schedule == 'nps':                                                           # nps  (shortestf first)
    #     nps_sim_jobs(scheduler)
    
    else:
        print('not support scheduler') 

    scheduler._controller.kill_workers()

if __name__ == '__main__':
    # print('Hello world %d' % 2)
    # mps_sim_jobs()
    main()
    