from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import queue
import threading


'''
JOB status:
ADDED: add job into JOBS
EVENT: init job events into event list
PENDING:
RUNNING: running job
END: completed
ERROR
'''
# import numpy
import math
import utils
from workloads import models_msg
import csv
import time
import sys
import random
import copy
# import cluster
# from switch import _Switch
# from node import _Node
# from cluster import _Cluster

# #get host info
# CLUSTER = cluster.CLUSTER
import flags
FLAGS = flags.FLAGS

from runtime.rpc_stubs.master_to_worker_pb2 import JobInfo

class _TFJobs(object):

    def __init__(self):
        self.num_job = 0        
        self.job_list = list()
        ''' job events is a list of tuple
            (time, dict)
        dict:
            'start_jobs': [xxx,xxx,xxx]
            'end_jobs': [xxx,xxx,xxx]
        '''
        self.job_events = list()        # 记录每个时间的start_job和end_job
        #holding pending jobs, add job_idx
        self.pending_jobs = list() # [{job_dict}, {job_dict}]
        self.runnable_jobs = list() # pending + running
        self.running_jobs = list() # running
        self.completed_jobs = list()

        self.job_name_map = {'resnet18':'ResNet-18', 'shufflenet_v2_x1_0':'ShuffleNet-v2', 'vgg19':'VGG19', 'vgg16':'VGG16', 'dqn':'DQN', 'a2c':'A2C', 'bert':'BERT', 'gpt2':'GPT2'}

        # job itertime and interference
        self.itertime = utils.json_to_dict('./trace-data/itertime.json')
        self.interference = utils.json_to_dict('./trace-data/ratio.json')

        # self.migratable_jobs = list()
        self.num_queue = 2
        self.queues = [list() for i in range(self.num_queue)]     # dlas
        self.queue_limit = [700,]
        self.job_lock = threading.Lock()
        self.cur_time = 0                   # 动态 add jobs 时使用
        self.delete_queue = queue.Queue()



    def get_job_model(self, job_dict):
        # if job_dict.has_key('model_name') and job_dict.has_key('model_scale'):
        # if ('model_name' in job_dict) and ('model_scale' in job_dict):
        #     job_dict['model'] = models.get_model_with_scale(job_dict['model_name'], job_dict['model_scale'])
        # else:
        #     utils.print_fn('Not enough model information to get the details')
        job_dict['model'] = models_msg.get_model(job_dict['model_name'])



    def add_job(self, job_dict, add_later=False):
        ''' Add job (job_dict) into job_list'''
        for key, value in job_dict.items():
            if (value is None) or ('resource_time' == key):
                continue
            # if 'resource_time' in key:
            #     job_dict['resource_time'].append(float(value))
            elif value.isdigit():
                job_dict[key] = int(value)

        # job_dict['rank'] = sys.maxsize


        if add_later:           # 后续由用户动态添加的job
            job_dict['submit_time'] = self.cur_time + FLAGS.schedule_interval
        else:
            job_dict['submit_time'] /= 1000

        self.set_itertime(job_dict)
        job_dict['tput']  = 1/job_dict['iteration_time']
    
        # if 'batch_size' not in job_dict:
        #     job_dict['batch_size'] = 16

        if 'start_time' not in job_dict:
            job_dict['start_time'] = 0
        if 'end_time' not in job_dict:
            job_dict['end_time'] = 0
        if 'pending_time' not in job_dict:
            job_dict['pending_time'] = 0


        if 'submit_time' in job_dict:
            job_dict['r_submit_time'] = int(-1 * job_dict['submit_time'])
            # job_dict['submit_time'] -= 275400                                 # ljx: 减少等待时间
        # if 'antman' in FLAGS.schedule or FLAGS.scheme == 'merge':             # ljx: Deprecated
        #     if 'priority' not in job_dict:
        #         job_dict['priority'] = random.randint(0,1)
        #     if 'gpu_util' not in job_dict:
        #         if job_dict['priority']==0:
        #             job_dict['gpu_util'] = 0.1 # not real                       # ljx
        #         else:
        #             job_dict['gpu_util'] = 0.9

        job_dict['start_time'] = sys.maxsize
        job_dict['end_time'] = 0
        job_dict['pending_time'] = 0

        job_dict['packing_used'] = 0 # 0 - not used; 1 - prepare for packing; 2 - used

        # How much time this job has been executed? For preemption algorithms, this should be accumulated
        # job_dict['execution_time'] = 0
        # job_dict['last_start_time'] = 0
        # job_dict['last_check_time'] = 0
        job_dict['executed_time'] = 0
        job_dict['remaining_iterations'] = job_dict['iterations']

        job_dict['preempt'] = 0
        job_dict['resume'] = 0
        job_dict['promote'] = 0
        job_dict['job_counter'] = 0     # job运行次数
        job_dict['packing'] = None

        job_dict['status'] = 'ADDED'
        job_dict['job_idx'] = len(self.job_list)

        job_dict['gpus'] = list()
        job_dict['placements'] = list() #prepare an empty job_placement 
        # job_dict['ps_placements'] = list()
        # job_dict['w_placements'] = list()
        job_dict['remaining_gpu'] = job_dict['num_gpu']     # 还需要多少gpu
        job_dict['last_node_id'] = None
        '''
        MS_YARN: only one switch is allowed
        template:
        [{'switch': xx, 'nodes': [{'id':xx, 'num_gpu':xxx}]},
         {'switch': xx, 'nodes': [{'id':xx, 'num_gpu':xxx}, {'id':xx, 'num_gpu':xxx}]},
         {'switch': xx, 'nodes': [{'id':xx, 'num_gpu':xxx}]},
        ]
        '''

        #get detailed model inforamtion
        self.get_job_model(job_dict)    # job_dict['model']（是一个dict）

        #add job ps/worker information
        # self.get_network_load(job_dict)

        self.job_list.append(job_dict)
        self.num_job += 1


    def set_itertime(self, job_dict):
        tmp_model_name = self.job_name_map[job_dict['model_name']]
        tmp_num_gpu = str(job_dict['num_gpu'])
        if tmp_model_name in self.itertime and tmp_num_gpu in self.itertime[tmp_model_name]:
            job_dict['iteration_time'] = self.itertime[tmp_model_name][tmp_num_gpu]
        else:
            job_dict['iteration_time'] /= 1000

    # def print_all_job_size_info(self):
    #     '''        
    #     print job tensor info
    #     '''

    #     ps_max_ave_fd = open('ps_max_ave.csv', 'w+')
    #     ps_max_ave_writer = csv.writer(ps_max_ave_fd)  
    #     ps_max_ave_writer.writerow(['ps_max_ave'])

    #     ps_max99_ave_fd = open('ps_max99_ave.csv', 'w+')
    #     ps_max99_ave_writer = csv.writer(ps_max99_ave_fd)  
    #     ps_max99_ave_writer.writerow(['ps_max99_ave'])

    #     w_fd = open('w.csv', 'w+')
    #     w_writer = csv.writer(w_fd)  
    #     w_writer.writerow(['w'])

    #     ps_fd = open('ps.csv', 'w+')
    #     ps_writer = csv.writer(ps_fd)  
    #     ps_writer.writerow(['ps'])

    #     ps_w_fd = open('ps_w.csv', 'w+')
    #     ps_w_writer = csv.writer(ps_w_fd)  
    #     ps_w_writer.writerow(['ps_w'])

    #     utils.print_fn("Start to dump job information")
    #     for job in self.job_list:
    #         if job['ps_ave'] != 0:
    #             ps_max_ave_writer.writerow(list([job['ps_max_ave']]))
    #             ps_max99_ave_writer.writerow(list([job['ps_max99_ave']]))
    #             w_writer.writerow(list([job['w_network'][0]]))
    #             # ps_w_writer.writerow(job['w_network'][0])
    #             # for ps in job['ps_network']:
    #             #     ps_writer.writerow(ps)
    #             #     ps_w_writer.writerow(ps)
                
    #     ps_max_ave_fd.close()
    #     ps_max99_ave_fd.close()
    #     w_fd.close()
    #     ps_fd.close()
    #     ps_w_fd.close()
        
    def find_runnable_job(self, job_idx):
        for job in self.runnable_jobs:
            if job['job_idx'] == job_idx:
                return job
        print(f'Not found {job_idx} in runnable_jobs.')
        print([job['job_idx'] for job in self.runnable_jobs])
        assert 1==0

    def find_job_by_id(self, job_id):
        for job in self.job_list:
            if job['job_id'] == job_id:
                return job
        print(f'Not found {job_id} in runnable_jobs.')
        print([job['job_id'] for job in self.runnable_jobs])
        assert 1==0

    def read_job_info(self, job_idx, field=None):
        ''' Read  job information, if field == NONE, show all job info'''
        ''' job_id,num_gpu,submit_time,start_time,duration,model_size '''
        print('  Job[%d]: ' % job_idx)

        for job in self.job_list:
            if job['job_idx'] == job_idx:
                #find the job
                if field:
                    if isinstance(job[field], int):
                        print('%s :  %d' % (field, job[field]))
                    else:
                        print('%s :  %s' % (field, job[field]))
                else:
                    print(job)
                print('')

    def read_all_jobs(self, field=None):
        for j in self.job_list:
            print('  Job[%d]: ' % j['job_idx'])
            if field:
                if isinstance(j[field], int):
                    print('%s :  %d' % (field, j[field]))
                else:
                    print('%s :  %s' % (field, j[field]))
            else:
                print(j)
            print('')

    def sort_all_jobs(self, mode=None):
        '''
        Sort jobs based on their sumbit_time
        '''
        
        # ljx：缩放时间
        # max_submit_time = 0.0
        # for job in self.job_list:
        #     max_submit_time = job['submit_time'] if job['submit_time'] > max_submit_time else max_submit_time
        # for job in self.job_list:
        #     job['submit_time'] = job['submit_time'] / max_submit_time * 300 -81     # 将时间缩放到半小时内 - 600

        self.job_list.sort(key = lambda e:e.__getitem__('submit_time'))
        utils.print_fn('   Jobs are sorted with their start time')
        # self.read_all_jobs()
        # if FLAGS.schedule == 'multi-dlas-gpu' and FLAGS.scheme == 'count':
        #     for num_gpu, gjob in self.gpu_job.items():
        #         utils.print_fn('%d-GPU jobs have %d ' % (num_gpu, gjob.total_job))

    def create_multi_nodes_placement(self, job, switch_id, node_list):
        tmp_dict = dict() 
        tmp_dict['switch'] = switch_id
        tmp_dict['nodes'] = node_list
        job['placements'].append(tmp_dict)

    def create_multi_nodes_placement_same_switch(self, job, switch_id, node_list):
        if len(job['placements'])==0:
            self.create_multi_nodes_placement(job, switch_id, node_list)       
        else:
            for placement in job['placements']:
                if placement['switch'] == switch_id:
                    placement['nodes'].extend(node_list)


    def create_single_node_placement(self, job, switch_id, node_id, num_gpu, num_cpu, mem=0, gpu_list=[], not_first=False):
        '''
        under this switch, there is only one need used
        {'switch': xx, 'nodes': [{'id':xx, 'num_gpu':xxx, 'num_cpu': xxx, 'tasks': [w0, w1, ps1]}]}
        '''
        if not_first:
            node_dict = job['placements'][0]['nodes'][0]
            node_dict['num_gpu']+=num_gpu
            node_dict['num_cpu']+=num_cpu
            node_dict['job_per_gpu_mem']+=mem
            node_dict['gpu_list'].extend(gpu_list)
            # print(job['job_idx'], job['placements'][0]['nodes'][0])
        else:
            tmp_dict = dict() 
            tmp_dict['switch'] = switch_id
            node_dict = dict()
            node_dict['id'] = node_id
            node_dict['num_gpu'] = num_gpu
            node_dict['num_cpu'] = num_cpu
            node_dict['job_per_gpu_mem'] = mem
            node_dict['tasks'] = list()

            node_dict['gpu_list'] = gpu_list     

            tmp_dict['nodes'] = list()
            tmp_dict['nodes'].append(node_dict)
            job['placements'].append(tmp_dict)


    def remove_from_pending(self, job, event_time):             # ljx:Deprecated
        job['status'] = 'RUNNING'
        job['start_time'] = event_time
        job['end_time'] = job['start_time'] + job['duration']
        job['pending_time'] = job['start_time'] - job['submit_time']

        self.pending_jobs.remove(job)

    def move_to_pending(self, job):
        job['status'] = 'PENDING'
        self.pending_jobs.append(job)


    def update_pending_time(self, event_time):
        for job in self.pending_jobs:
            if 'sumbit_time' in job:
                job['pending_time'] = int(event_time - job['submit_time'])

    def add_to_runnable(self, job):
        job['status'] = 'PENDING'
        self.runnable_jobs.append(job)

    def push_job_to_running(self, job, event_time):
        if job['status'] != 'PENDING':
            return
        job['status'] = 'RUNNING'
        if job['start_time'] == 0:
            job['start_time'] = event_time
        job['last_start_time'] = event_time


    # def sort_shortest_runnable_jobs(self, event_time):
    #     for job in self.runnable_jobs:
    #         if job['status'] == 'RUNNING':
    #             new_execution_time = int(event_time - job['last_check_time'])
    #             job['execution_time'] = int(job['execution_time'] + new_execution_time)
    #             job['remaining_time'] = int(job['duration'] - job['execution_time'])

    #         elif job['status'] == 'PENDING':
    #             job['execution_time'] = 0
    #             job['remaining_time'] = int(job['duration'])

    #         job['last_check_time'] = int(event_time)

    #     JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('remaining_time'))

    def move_to_runnable(self, job):
        ''' job gets into the system: pending or running, and finally END'''
        #job not started yet
        job['status'] = 'PENDING'
        job['start_time'] = sys.maxsize
        job['last_start_time'] = 0
        job['last_check_time'] = job['submit_time']
        job['total_executed_time'] = 0 # total
        job['total_executed_gputime'] = 0
        job['calc_executed_time'] = 0
        job['executed_time'] = 0 # used for deciding priority queue, may be zeroed by last_pending_time
        job['pending_time'] = 0
        job['last_pending_time'] = 0 # how much pending_time the job has since last entering the highest priority queue
        self.runnable_jobs.append(job)

    def update_priority_queues(self, gputime=False):                # ljx:Deprecated
        for queue in self.queues:
            del queue[:]
        for job in self.runnable_jobs:
            if gputime:
                j_gt = int(job['executed_time'] * job['num_gpu'])
            else:
                j_gt = int(job['executed_time'])

            if j_gt < self.queue_limit[0]:
                self.queues[0].append(job)
                job['q_id'] = 0
            else:
                self.queues[1].append(job)
                job['q_id'] = 1

            # elif j_gt < self.queue_limit[1]:
            #     self.queues[1].append(job)
            #     job['q_id'] = 1
            # elif j_gt < self.queue_limit[2]:
            #     self.queues[2].append(job)
            #     job['q_id'] = 2
            # else:
            #     self.queues[3].append(job)
            #     job['q_id'] = 3

   
    def print_job_events(self):
        utils.print_fn('    Print all job events ')
        for event in self.job_events:
            # utils.print_fn('      event.time[%d], with %d start_jobs, and %d end_jobs' % 
                            # (event['time'], len(event['start_jobs']), len(event['end_jobs'])))
            print('      event.time, with start_jobs, and end_jobs', 
                            event['time'], event['start_jobs'], event['end_jobs'])

        utils.print_fn(' ')

    def remove_job_in_events(self, job):
        event = None
        tmp_job = None
        flag = False
        for event in self.job_events:
            for tmp_job in event['start_jobs']:
                if tmp_job['job_idx'] == job['job_idx']:
                    flag = True
                    break
            if flag:
                break
        if flag:
            event['start_jobs'].remove(tmp_job)
            if len(event['start_jobs']) == 0:
                self.job_events.remove(event)


    def add_job_end_event(self, job):
        #for job end 
        tmp_dict = utils.search_dict_list(self.job_events, 'time', job['end_time'])
        if tmp_dict == None:
            #not found, add the time into to job_events
            tmp_dict = dict()
            tmp_dict['time'] = job['end_time']
            tmp_dict['start_jobs'] = list()
            tmp_dict['end_jobs'] = list()
            tmp_dict['end_jobs'].append(job)
            self.job_events.append(tmp_dict)
        else:
            tmp_dict['end_jobs'].append(job)

        # ''' sort events based on their time'''
        # self.job_events.sort(key = lambda e:e.__getitem__('time'))



    def prepare_job_start_events(self):
        '''
        add job start events into job_events list
        end events should be added when they are starting
        '''
        for job in self.job_list:
            if job['status'] != 'ADDED':
                continue
            start_t = job['submit_time']
            # utils.print_fn('%d, %d' % (start_t, end_t))

            #for job start
            tmp_dict = utils.search_dict_list(self.job_events, 'time', start_t)
            if tmp_dict == None:
                #not found, add the time into to job_events
                tmp_dict = dict()
                tmp_dict['time'] = start_t
                tmp_dict['start_jobs'] = list()
                tmp_dict['end_jobs'] = list()
                tmp_dict['start_jobs'].append(job)
                self.job_events.append(tmp_dict)
            else:
                tmp_dict['start_jobs'].append(job)


            job['status'] = 'EVENT' #job has been in EVENT status

        ''' sort events based on their time'''
        self.job_events.sort(key = lambda e:e.__getitem__('time'))
        utils.print_fn('Init, add job start events')
        self.print_job_events()
        utils.print_fn('--------------------------------- End of job events ---------------------------------')

    def print_placement(self, ejob):
        print("placement of job ", ejob['job_idx'])
        print(ejob['placements'])

    def to_jobinfo(self, tmp_ejob, is_packing=False):  
        '''
        根据job生成RPC接口参数，用于执行或kill任务
        '''

        
        jobinfo = None

        gpu_list = {}
        if not is_packing:
            ejob = [tmp_ejob]
            # while len(ejob)<FLAGS.multi_resource:
            while len(ejob)<4:
                ejob.append(None)
        else:
            ejob = tmp_ejob
        
        if len(ejob[0]['placements'])!=1:
            print(len(ejob[0]['placements']),ejob[0])
        assert len(ejob[0]['placements'])==1    # 重新调度前job的placement会被清除，所以基本只会有一个placement
        placement = ejob[0]['placements'][0]
        job_id_list = [rjob['job_idx'] if rjob!=None else -1 for rjob in ejob]
        job_name_list = [rjob['model_name'] if rjob!=None else '-1' for rjob in ejob]    # ljx: 根据workloads/run.sh，这里 '0' 应该改为 '-1'(其实关系不大，因为batch_size为0，模型不会被运行)
        batch_size_list = [rjob['batch_size'] if rjob!=None else 0 for rjob in ejob]

        iters_list = [rjob['remaining_iterations'] if rjob!=None else 0 for rjob in ejob]
        
        job_counter_list = [rjob['job_counter'] if rjob!=None else 0 for rjob in ejob]
        job_num = len(ejob)
        node_id_list = []
        for node in placement['nodes']:
            node_id = node['id']
            node_id_list.append(node_id)
            assert node_id not in gpu_list
            # assert len(placement['nodes'])==1 or (len(placement['nodes'])>1 and len(node['gpu_list'])==8)       # ljx: 将8改为2，每个节点只有4个gpu
            for gpu in node['gpu_list']:
                if node_id not in gpu_list:
                    gpu_list[node_id] = str(gpu.idx)
                else:
                    gpu_list[node_id] += f',{gpu.idx}'          # gpu_list={'node_id':gpu_list}
        gpu_list_str=[]
        for k,v in gpu_list.items():
            gpu_list_str.append(str(k)+'-'+v)
        gpu_list_str = "/".join(gpu_list_str)
        resumed_list = [True if rjob and rjob['resume'] > 0 else False for rjob in ejob]                                   # 是否为之前执行过的job
        # jobinfo = JobInfo(num=job_num, gpus=gpu_list[node_id_list[0]], num_gpu=ejob[0]['num_gpu'])
        jobinfo = JobInfo(num=job_num, gpus=gpu_list_str, num_gpu=ejob[0]['num_gpu'])                   # id0-gpu_list0/id1-gpu_list1/...
        jobinfo.node_id.extend(node_id_list)
        jobinfo.job_id.extend(job_id_list)
        jobinfo.job_name.extend(job_name_list)
        jobinfo.batch_size.extend(batch_size_list)
        jobinfo.iterations.extend(iters_list)
        jobinfo.job_counter.extend(job_counter_list)
        jobinfo.is_resumed.extend(resumed_list)
        # exit(0)
        return jobinfo
    
    # below is deprecated

    def calc_packing_finished_info(self, rjob, tmp_time, last_check_time):
        iter_list = list()
        # print('in calc_packing_info: ', rjob['job_idx'], [tjob.job_idx for tjob in rjob['packing'].packing_jobs])
        if rjob['packing']==None:
            iter_list.append(rjob['remaining_iterations'])
        else:
            for pjob_mini in rjob['packing'].packing_jobs:
                pjob=self.find_runnable_job(pjob_mini.job_idx)
                iter_list.append(pjob['remaining_iterations'])
            sim_itertime = rjob['packing'].calc_iteration_time()
            real_itertime = rjob['real_itertime'][0]
            overhead_error = (real_itertime-sim_itertime)/sim_itertime
            self.overhead_list[len(rjob['packing'].packing_jobs)].append(overhead_error)
                # print(pjob['job_idx'], pjob['real_itertime'], pjob['remaining_iterations'])
        iter_list = list(set(iter_list))
        iter_list.sort()
        if iter_list[0]==0:
            del iter_list[0]
        # print('calc_packing_finished_info, real_itertime vs iter_list: ', iter_list)
        assert len(rjob['real_itertime']) == len(iter_list)
        finished_iteration = 0
        if rjob['last_finish_time']>last_check_time:
            overhead_time = copy.deepcopy(rjob['last_finish_time'])
            # print('calc jobs: ', rjob['job_idx'], rjob['last_iters'])
            last_iter = 0
            assert len(rjob['last_iters']) == len(rjob['real_itertime'])
            for idx, itertime in enumerate(rjob['real_itertime']):
                overhead_time -= (rjob['last_iters'][idx]-last_iter) * itertime
                last_iter = rjob['last_iters'][idx]
        else:
            overhead_time = last_check_time
        remaining_time = tmp_time - overhead_time
        finished_time = overhead_time
        done_idx = 0
        last_iter = 0
        # print('calc jobs: ', rjob['job_idx'], overhead_time-last_check_time)
        for idx, iters in enumerate(iter_list):
            if iters > rjob['remaining_iterations']:
                break
            time0 = (iters-last_iter)*rjob['real_itertime'][idx]
            if remaining_time - time0>=0:
                remaining_time -= time0
                finished_time += time0
                finished_iteration += (iters-last_iter)
                done_idx += 1
                last_iter = iters
            else:
                finished_time = tmp_time
                finished_iteration += int(remaining_time / rjob['real_itertime'][idx])
                break
        
        assert finished_iteration<=rjob['remaining_iterations']
        return finished_time, finished_iteration, done_idx
    
    def add_gpu_job(self, job):
        '''
        only used in sim-gpu-demands
        '''
        num_gpu = job['num_gpu']
        if num_gpu not in self.gpu_job:
            self.gpu_job[num_gpu] = 0
        self.gpu_job[num_gpu] = self.gpu_job[num_gpu] + 1

    def delete_gpu_job(self, job):
        num_gpu = job['num_gpu']
        if num_gpu not in self.gpu_job:
            print("Error in release_gpu_job")

        self.gpu_job[num_gpu] = self.gpu_job[num_gpu] - 1

    def end_job(self, e_job):
        if FLAGS.schedule != 'multi-dlas-gpu':
            utils.print_fn("Not multi-dlas-gpu")
            exit()
        
        num_gpu = e_job['num_gpu']
        gjob = self.gpu_job[num_gpu]
        gjob.release_job_gpu(1)
        gjob.runnable_jobs.remove(e_job)
        # gjob.running_jobs.remove(e_job)
        gjob.queues[e_job['q_id']].remove(e_job)       
        gjob.end_job += 1


    def init_reserve_gpus(self, total_num):
        num_group = len(self.gpu_job)
        ave_gpu = math.floor(total_num / num_group)
        for num_gpu, gjob in self.gpu_job.items():
            gjob.get_gpu_reservation(ave_gpu)

    def reserve_gpus(self, total_num):
        '''
        GPU cluster reserve gpus for gpu_job groups
        '''
        num_group = len(self.gpu_job)
        ave_gpu = math.floor(total_num / num_group)

        job_list = list()
        for num_gpu, gjob in self.gpu_job.items():
            tmp_dict = dict()
            tmp_dict['num_gpu'] = num_gpu
            tmp_dict['used_gpu'] = gjob.total_gpu - gjob.free_gpu
            tmp_dict['demands'] = gjob.get_gpu_demands()
            tmp_dict['cur_gpu'] = gjob.total_gpu
            tmp_dict['cur_free_gpu'] = gjob.free_gpu
            tmp_dict['reserve'] = 0
            job_list.append(tmp_dict)

        total_free_gpu = total_num - sum(k['used_gpu'] for k in job_list) 
        total_demands = sum(k['demands'] for k in job_list)
        # print('total_free %d, total_demands %d' % (total_free_gpu, total_demands))
        if total_demands == 0: 
            return
        
        '''demand-based, keep current used_gpu'''
        remain_free_gpu = total_free_gpu
        job_list.sort(key = lambda e:e.__getitem__('demands'))
        for job_dict in job_list:
            if job_dict['demands'] == 0:
                continue

            ratio = round((job_dict['demands'] * 1.0) / total_demands, 2)
            cal_gpu = int(math.floor((ratio * total_num) / job_dict['num_gpu']) * job_dict['num_gpu'])
            cal_gpu = job_dict['demands'] if job_dict['demands'] <= cal_gpu else cal_gpu
            extra_gpu = cal_gpu - job_dict['used_gpu']
            if extra_gpu <= 0:
                extra_gpu = 0
            elif extra_gpu > remain_free_gpu:
                extra_gpu = int(math.floor(remain_free_gpu / job_dict['num_gpu']) * job_dict['num_gpu'])

            # print('%d-GPU, u%d, cal_gpu %d, extra_g %d' %(job_dict['num_gpu'], job_dict['used_gpu'], cal_gpu, extra_gpu))
            job_dict['reserve'] = job_dict['used_gpu'] + extra_gpu
            remain_free_gpu -= extra_gpu
            # if remain_free_gpu <= 0:
            #     break

        ''' still remaining, give to the right job group'''
        job_list.sort(key = lambda e:e.__getitem__('num_gpu'))
        num_full = 0
        while remain_free_gpu > 0:
            # if all are satisfied
            if num_full >= len(job_list):
                break
            else:
                num_full = 0

            for job_dict in job_list:
                if job_dict['demands'] <= job_dict['reserve']:
                    num_full += 1
                    continue
                if remain_free_gpu >= job_dict['num_gpu']:                
                    remain_free_gpu -= job_dict['num_gpu']
                    job_dict['reserve'] += job_dict['num_gpu']
                else:
                    num_full += 1

                if remain_free_gpu <= 0: 
                    break

        #execute reservation
        for job_dict in job_list:
            num_gpu = job_dict['num_gpu']
            self.gpu_job[num_gpu].get_gpu_reservation(job_dict['reserve'])
            print("%d-j, T%d, F%d, U%d, N%d, R%d; " % (job_dict['num_gpu'], job_dict['cur_gpu'], job_dict['cur_free_gpu'], job_dict['used_gpu'], job_dict['demands'], job_dict['reserve']), end=' ')

        for num_gpu, gjob in self.gpu_job.items():
            if gjob.free_gpu < 0:
                print("Error free gpu, %d" % num_gpu)
                exit()


        utils.print_fn(' %s is done' % sys._getframe().f_code.co_name)

    def completion_check(self):
        for num_gpu, gjob in self.gpu_job.items():
            if gjob.end_job != gjob.total_job:
                utils.print_fn('!!!! Miss-match %d completed jobs with %d total jobs in %d-GPU jobs' % (gjob.end_job, gjob.total_job, num_gpu))

    def test_reserve_gpus(self, total_num):
        for num_gpu, gjob in self.gpu_job.items():
            gjob.total_gpu = 0
            gjob.free_gpu = 0
            gjob.runnable_jobs = []

        self.gpu_job[8].total_gpu = 32
        self.gpu_job[8].free_gpu = 0 
        self.gpu_job[8].runnable_jobs.extend([4,5,6,7,8])

        self.gpu_job[16].total_gpu = 32 
        self.gpu_job[16].free_gpu = 16
        self.gpu_job[16].runnable_jobs.extend([5,6,7,8,9])

        self.reserve_gpus(total_num)

JOBS = _TFJobs()

_allowed_symbols = [
    'JOBS'
]

# tmp_job = {'job_id': 35, 'num_gpu': 1, 'submit_time': 5803.37621700266, 'iterations': 7447, 'model_name': 'shufflenet_v2_x1_0', 'batch_size': 128, 'duration': 5794.0, 'interval': 535000, 'iteration_time': 0.798, 'resource_time_0': '706', 'resource_time_1': '59', 'resource_time_2': '0', 'priority': 0, 'resource_time': [706.0, 59.0, 0.0], 'rank': 9223372036854775807, 'tput': 1.2531328320802004, 'start_time': 6120.00066113472, 'end_time': 0, 'pending_time': 316.6244441320596, 'r_submit_time': -107845, 'packing_used': 0, 'execution_time': 0, 'last_start_time': 0, 'last_check_time': 6120.00066113472, 'executed_time': 0, 'remaining_iterations': 7447, 'preempt': 0, 'resume': 0, 'promote': 0, 'job_counter': 1, 'packing': None, 'status': 'PENDING', 'job_idx': 0, 'gpus': [], 'placements': [{'switch': 0, 'nodes': [{'id': 0, 'num_gpu': 1, 'num_cpu': 2, 'mem': 5, 'tasks': [], 'network': 0, 'gpu_list': [0]}]}], 'ps_placements': [], 'w_placements': [], 'remaining_gpu': 1, 'last_node_id': None, 'model_scale': 1, 'model': {'name': 'inception3', 'ind': 8, 'tensors': [3.8, 2.1, 1.3, 1.6, 1.9, 1.7, 1.7, 2.2, 5.9, 1.7, 1.7, 2.5, 3.0, 1.7, 1.7, 3.5, 5.9, 1.7, 1.7, 1.5, 7.8], 'mem_util': 1, 'total_size': 56.6}, 'ps_network': [], 'w_network': [0], 'ps_ave': 0, 'total_executed_time': 0, 'total_executed_gputime': 0, 'calc_executed_time': 0, 'last_pending_time': 0, 'q_id': 0, 'remaining_time': 5942.706, 'remaining_gputime': 5942.706}
# JOBS.to_jobinfo(tmp_job)
