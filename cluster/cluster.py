from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import random
from cluster.switch import _Switch
from cluster.node import _Node
import utils
import flags 
# import jobs
# import log

# JOBS = jobs.JOBS
# LOG = log.LOG
FLAGS = flags.FLAGS

class _Cluster(object):

    def __init__(self, num_switch=0, num_node_p_switch=0, num_gpu_p_node=0, num_cpu_p_node=0, mem_p_node=0):
        ''' Init GPU cluster with basic switch, node, gpu information'''
        self.num_switch =  num_switch
        self.num_node_p_switch = num_node_p_switch
        self.num_gpu_p_node = num_gpu_p_node
        self.num_cpu_p_node = num_cpu_p_node
        self.mem_p_node = mem_p_node
        self.num_node = num_switch * num_node_p_switch
        self.num_gpu = self.num_node * num_gpu_p_node
        self.num_cpu = self.num_node * num_cpu_p_node
        self.mem = self.num_node * mem_p_node

        self.switch_list = list()

        #for non-placement
        self.free_gpu = self.num_gpu
        self.gpu_list = list()

        self.node_msg = utils.json_to_dict('./cluster/node_msg.json')


    def set_spec(self, num_switch=0, num_node_p_switch=0, num_gpu_p_node=0, num_cpu_p_node=0, mem_p_gpu=0):
        self.num_switch =  num_switch
        self.num_node_p_switch = num_node_p_switch
        self.num_gpu_p_node = num_gpu_p_node
        self.num_cpu_p_node = num_cpu_p_node
        self.mem_p_gpu = mem_p_gpu
        self.num_node = num_switch * num_node_p_switch
        self.num_gpu = self.num_node * num_gpu_p_node
        self.num_cpu = self.num_node * num_cpu_p_node
        self.free_gpu = self.num_gpu
        self.mem = self.num_node * mem_p_gpu


    def print_cluster_spec(self):
        print('Custer Spec')
        print('#ofswitch: %d, #ofnode: %d, #ofgpu: %d, #ofcpu: %d, #ofmem: %d'%(self.num_switch, self.num_node, self.num_gpu, self.num_cpu, self.mem))
        print('#ofnode/switch: %d, #ofgpu/node: %d, #ofcpu/node: %d, #ofmem/node: %d' % (self.num_node_p_switch, self.num_gpu_p_node, self.num_cpu_p_node, self.mem_p_node))


    def init_infra(self, num_switch=0, num_node_p_switch=0, num_gpu_p_node=0, num_cpu_p_node=0, mem_p_gpu=0):
        '''
        Init and create cluster infration entities (switches, nodes) by using class _Switch, _Node
        '''
        if num_switch == 0 and num_node_p_switch == 0 and num_gpu_p_node == 0 and num_cpu_p_node == 0 and mem_p_gpu == 0:
            #no new spec, apply FLAGS spec info
            self.set_spec(FLAGS.num_switch, FLAGS.num_node_p_switch, FLAGS.num_gpu_p_node, FLAGS.num_cpu_p_node, FLAGS.mem_p_gpu)
        else:
            self.set_spec(num_switch, num_node_p_switch, num_gpu_p_node, num_cpu_p_node, mem_p_gpu)

        '''create/init switch and node objects'''        
        for s in range(0, self.num_switch):
            tmp_s = _Switch(s, self.num_node_p_switch, self.num_gpu_p_node, self.num_cpu_p_node, self.mem_p_gpu) 
            tmp_s.add_nodes(self.num_node_p_switch, self.num_gpu_p_node, self.num_cpu_p_node, self.mem_p_gpu)
            self.switch_list.append(tmp_s)

        utils.print_fn('Cluster is ready to use')
        self.print_cluster_spec()

    def empty_infra(self):
        self.free_gpu = self.num_gpu
        for switch in self.switch_list:
            for node in switch.node_list:
                # if 'multi-resource' in FLAGS.schedule or 'shortest' in FLAGS.schedule:
                node.init_node(self.num_gpu_p_node, self.num_cpu_p_node, self.mem_p_node)
                # else:
                #     node.init_node(self.num_gpu_p_node, self.num_cpu_p_node)


    def ms_yarn_placement(self, job):
        '''
        MS_YARN, all gpus should come from the same switch
        '''
        for switch in self.switch_list:
            ret = switch.ms_yarn_alloc_res(job)
            if ret == True:
                return True
            else:
                continue
        return False

    def antman_placement(self, job):
        for switch in self.switch_list:
            ret = switch.antman_alloc_res(job, gpu_util_upper=1.0)    # ljx  gpu_util_upper=1.0
            if ret == True:
                return True
            else:
                continue
        return False
    def merge_placement(self, job):
        for switch in self.switch_list:
            ret = switch.merge_alloc_res(job) 
            if ret == True:
                return True
            else:
                continue
        return False
    
    def merge_placement_consolidate(self, job):
        for switch in self.switch_list:
            ret = switch.merge_alloc_res_consolidate(job) 
            if ret == True:
                return True
            else:
                continue
        return False
    
    def bsbf_placement(self, job=None,with_mps=False):
        for switch in self.switch_list:
            ret = switch.bsbf_alloc_res(job,with_mps)  
            if ret == True:
                return True
            else:
                continue
        return False
    
    def bsbfs_placement(self, job=None,with_mps=False):
        for switch in self.switch_list:
            ret = switch.bsbf_alloc_res_consolidate(job,with_mps)    
            if ret == True:
                return True
            else:
                continue
        return False


    def none_placement(self, job):
        num_w = job['num_gpu']

        if self.free_gpu >= num_w:
            self.free_gpu = int(self.free_gpu - num_w)
            return True
        else:
            return False

    def check_free_gpu(self):
        if FLAGS.scheme == 'count':
            return self.free_gpu
        else:
            free_gpu = 0
            for switch in self.switch_list:
                for node in switch.node_list:
                    free_gpu = int(free_gpu + node.check_free_gpus())
            return free_gpu
        


    def get_node_with_gid(self, gid):
        s_id = int(math.floor(gid / self.num_node_p_switch))
        n_id = int(gid % self.num_node_p_switch)

        switch = self.switch_list[s_id]
        node = switch.node_list[n_id]
        ret = dict()
        ret['switch'] = switch
        ret['node'] = node
        return ret


    def alloc_gpus(self, job):
        '''
        allocate gpus to job
        '''
        ret = self.ms_yarn_placement(job)
        if ret == True:
            job['status'] = 'RUNNING'
        return ret

    # not used
    def release_gpus(self, job):
        for placement in job['placements']:
            if ('switch' not in placement) or ('nodes' not in placement):
                job['status'] = 'ERROR'
                return False

            switch = self.switch_list[placement['switch']]
            ret = switch.release_gpus(placement['nodes'])
            if ret == False:
                job['status'] = 'ERROR'
                return False

        job['status'] = 'END'
        utils.print_fn('**** job[%d] completed' % job['job_idx'])
        return True

    '''
    release job res
    '''
    def release_job_res(self, job):
        '''
        release gpu/cpu/mem
        placements:
        [{'switch': xx, 'nodes': [{'id':xx, 'num_gpu':xxx, 'num_cpu': xxx, 'mem': xxxx, 'gpu_list': [], }]},]
        '''

        for placement in job['placements']:
            if ('switch' not in placement) or ('nodes' not in placement):
                job['status'] = 'ERROR'
                print("release error, no switch or nodes", job['job_idx'])
                return False

            switch = self.switch_list[placement['switch']]
            if 'antman' in FLAGS.schedule:
                ret = switch.release_job_res(placement['nodes'], job['priority'], job['job_idx'], job['gpu_util'])
            else:
                ret = switch.release_job_res(placement['nodes'], job)
            if ret == False:
                print("release error, switch release error", job['job_idx'])
                job['status'] = 'ERROR'
                return False

        job['status'] = 'END'
        utils.print_fn('  **** job[%d] completed or preempted' % job['job_idx'])
        return True


CLUSTER = _Cluster()
# CLUSTER_TMP = _Cluster()


_allowed_symbols = [
    'CLUSTER',
    # 'CLUSTER_TMP'
]