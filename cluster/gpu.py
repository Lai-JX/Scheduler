from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import utils


class _GPU(object):
    def __init__(self, node_id, gpu_id, idx, gpu_mem):
        # self.id = id
        self.node_id = node_id
        self.gpu_id = gpu_id
        self.idx = idx       
        self.gpu_mem = gpu_mem

        self.free_mem = self.gpu_mem
        self.job_list = []
        # self.free_mem = gpu_mem

        #node class for gandiva

        utils.print_fn('      GPU[%s] has %d M memory' % (gpu_id, gpu_mem))
    
    def init_gpu(self, node_id, gpu_id, idx, gpu_mem):
        self.node_id = node_id
        self.gpu_id = gpu_id
        self.idx = idx       
        self.gpu_mem = gpu_mem

        self.free_mem = self.gpu_mem     
        self.job_list = []   
    
    def get_job_list(self):
        return self.job_list



