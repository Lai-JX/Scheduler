from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import utils
from cluster.gpu import _GPU

'''
TODO: add cpu and network load support in class _Node
'''
class _Node(object):
    def __init__(self, id, num_gpu=0, num_cpu=0, mem_p_gpu=0):
        self.id = id
        self.num_cpu = num_cpu
        self.free_cpus = num_cpu
        self.num_gpu = num_gpu       
        # self.free_gpus = num_gpu
        self.mem_p_gpu = mem_p_gpu

        self.gpu_list = []

        self.job_gpu = 0
        self.num_jobs = 0
        # self.gpu_job_list = [{0:[], 1:[]} for i in range(self.num_gpu)]
        # self.gpu_util_list = [0.0 for i in range(self.num_gpu)]         # deprecated, add GPU class
        self.gpu_job_list = None
        self.gpu_util_list = None         # 各个GPU的已使用的显存

        utils.print_fn('    Node[%d] has %d gpus, %d cpus, %d M memory per gpu' % (id, num_gpu, num_cpu, mem_p_gpu))
    
    def init_node(self, num_gpu=0, num_cpu=0, mem_p_gpu=0):
        if num_gpu != 0:
            self.num_gpu = num_gpu
        if num_cpu != 0:
            self.num_cpu = num_cpu
            self.free_cpus = num_cpu
        if mem_p_gpu != 0:
            self.mem_p_gpu = mem_p_gpu
        # self.gpu_job_list = [{0:[], 1:[]} for i in range(self.num_gpu)]
        # self.gpu_util_list = [0.0 for i in range(self.num_gpu)]

        self.set_gpus(self.num_gpu)        
        # self.add_cpus(self.num_gpu)    

    @property
    def workload(self):
        job_set = set()
        for gpu in self.gpu_list:
            for job in gpu.get_job_list():
                job_set.add(job['job_idx'])
        return len(job_set)   
    @property
    def free_gpus_num(self):
        tmp = 0
        for gpu in self.gpu_list:
            if len(gpu.get_job_list()) == 0:
                tmp += 1
        return tmp

    ''' GPU  '''
    def set_gpus(self):
        self.gpu_list = []
        for i in range(0, self.num_gpu):
            tmp_g = _GPU(self.id, '%s-%s'%(str(self.id),str(i)),i,self.mem_p_gpu)
            self.gpu_list.append(tmp_g)
    ''' GPU  '''
    def add_gpus(self):
        for i in range(0, self.num_gpu):
            tmp_g = _GPU(self.id, '%s-%s'%(str(self.id),str(i)),i,self.mem_p_gpu)
            self.gpu_list.append(tmp_g)
        # self.gpu_job_list = [{i.gpu_id:[]} for i in self.gpu_list]
        # self.gpu_util_list = [{i.gpu_id:[]} for i in self.gpu_list.keys()]

    # def check_free_gpus(self):
    #     return self.free_gpus

    def get_free_gpus(self, priority):
        avail_gpu_list = []
        if priority==0:
            for i in range(self.num_gpu):
                if len(self.gpu_job_list[i][0])==0:
                    avail_gpu_list.append(i)
        else:
            for i in range(self.num_gpu):
                if len(self.gpu_job_list[i][1])<=1:
                    avail_gpu_list.append(i)
        return avail_gpu_list
            


    def alloc_gpus(self,job_mem=0, avail_gpu_list=None, job=None):
        '''
        If enough free gpus, allocate gpus
        Return: True, for success;
                False, for failure
        '''
        # if num_gpu > self.free_gpus:
        #     return False
        # else:
        #     self.free_gpus -= num_gpu
        #     for avail_gpu in avail_gpu_list:
        #         avail_gpu.job_list.append(job_idx)
        #         avail_gpu.free_mem -= job_mem
        #     return True
        for avail_gpu in avail_gpu_list:
            avail_gpu.job_list.append(job)
            avail_gpu.free_mem -= job_mem
        return True

    def release_gpus(self,job_mem=0, avail_gpu_list=None, job=None):
        '''
        release using gpus back to free list
        '''
        # if priority>=0:
        #     for avail_gpu in avail_gpu_list:
        #         if job_idx in self.gpu_job_list[avail_gpu][priority]:
        #             assert job_idx in self.gpu_job_list[avail_gpu][priority]
        #             self.gpu_job_list[avail_gpu][priority].remove(job_idx)
        #             self.gpu_util_list[avail_gpu] -= gpu_util
        # if priority!=1:
        #     if self.free_gpus + num_gpu > self.num_gpu:
        #         self.free_gpus = self.num_gpu
        #         return False
        #     else:
        #         self.free_gpus += num_gpu
        #         return True
        # else:
        #     return True
        for avail_gpu in avail_gpu_list:
            avail_gpu.job_list.remove(job)
            avail_gpu.free_mem += job_mem


    ''' CPU '''

    def add_cpus(self, num_cpu=0):
        pass

    def check_free_cpus(self):
        return self.free_cpus

    def alloc_cpus(self, num_cpu=0):
        '''
        If enough free cpus, allocate gpus
        Return: True, for success;
                False, for failure
        '''
        if num_cpu > self.free_cpus:
            return False
        else:
            self.free_cpus -= num_cpu
            return True

    def release_cpus(self, num_cpu=0):
        '''
        release using cpus back to free list
        '''
        if self.free_cpus + num_cpu > self.num_cpu:
            self.free_cpus = self.num_cpu
            return False
        else:
            self.free_cpus += num_cpu
            return True 


    '''network'''

    def add_network_load(self, in_load=0, out_load=0):
        self.network_in += in_load
        self.network_out += out_load
        self.network_in = round(self.network_in, 1)
        self.network_out = round(self.network_in, 1)


    def release_network_load(self, in_load=0, out_load=0):
        self.network_in -= in_load
        self.network_out -= out_load
        self.network_in = round(self.network_in, 1)
        self.network_out = round(self.network_in, 1)

    def set_network_load(self, in_load=0, out_load=0):
        self.network_in = in_load
        self.network_out = out_load
        self.network_in = round(self.network_in, 1)
        self.network_out = round(self.network_in, 1)


    def alloc_job_res(self, num_cpu=0, job_mem=0, avail_gpu_list=None, job=None):
        '''
        alloc job resource
        '''
        gpu = self.alloc_gpus(job_mem, avail_gpu_list, job)
        cpu = self.alloc_cpus(num_cpu)

        # print(job_idx, gpu, cpu)

        if cpu == False or gpu == False:
            self.release_gpus(job_mem, avail_gpu_list, job)
            self.release_cpus(num_cpu)
            return False

        return True 

    def find_gpu_util(self, gpu_util_upper):
        gpu_list = []
        print('gpu_util_list and upper', self.gpu_util_list, gpu_util_upper)
        for i in range(self.num_gpu):
            if self.gpu_util_list[i]<gpu_util_upper:       # ljx < → <=
                gpu_list.append({'node':self.id, 'gpu':i})
                # print(self.id, i, self.gpu_util_list[i])
        return gpu_list

    def release_job_res(self, node_dict,job=None):
        '''
        input is node_dict from placement
        {'id':xx, 'num_gpu':xxx, 'num_cpu': xxx, 'network': xxxx, 'tasks': [w2, ps2]}
        '''
        # self.release_network_load(node_dict['network'], node_dict['network'])
        # cpu = True
        cpu = self.release_cpus(node_dict['num_cpu'])
        gpu = self.release_gpus(node_dict['job_per_gpu_mem'], node_dict['gpu_list'], job)

        # self.free_mem = self.free_mem + node_dict['mem']

        # print(job_idx, cpu, gpu)

        return (cpu and gpu)

    def release_job_gpu_cpu(self, num_gpu, num_cpu):
        '''
        input is gpu and cpu
        '''
        cpu = self.release_cpus(num_cpu)
        gpu = self.release_gpus(num_gpu)

        return (cpu and gpu)
