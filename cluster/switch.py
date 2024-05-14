from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from cluster.node import _Node
import flags 
import utils
import jobs
import math
import copy

FLAGS = flags.FLAGS
JOBS = jobs.JOBS


class _Switch(object):

    def __init__(self, id, num_node=0, num_gpu_p_node=0, num_cpu_p_node=0, mem_p_gpu=0):
        self.num_node = num_node
        self.num_gpu_p_node = num_gpu_p_node
        self.num_cpu_p_node = num_cpu_p_node
        self.mem_p_gpu = mem_p_gpu
        self.id = id
        self.node_list = list()
        self.node_map = {}
        self.con_times = 0      # used by bsbf
        self.seq_times = 0      # used by bsbf 
        utils.print_fn('  Switch[%d] has %d nodes' % (id, num_node))

    def add_nodes(self, num_node=0, num_gpu_p_node=0, num_cpu_p_node=0, mem_p_gpu=0):
        if num_node != 0 and num_gpu_p_node != 0 and num_cpu_p_node != 0 and mem_p_gpu != 0:
            self.num_node = num_node
            self.num_gpu_p_node = num_gpu_p_node
            self.num_cpu_p_node = num_cpu_p_node
            self.mem_p_gpu = mem_p_gpu
        
        for n in range(0, self.num_node):
            tmp_n = _Node(n, self.num_gpu_p_node, self.num_cpu_p_node, self.mem_p_gpu)
            tmp_n.add_gpus()
            self.node_list.append(tmp_n)
            self.node_map[n] = tmp_n



    def alloc_gpus(self, job):
        '''
        alloc gpus to job
        '''
        pass 

    def try_cross_node_alloc(self, job, not_place=False):
        '''
        used in MS_YARN placement
        try get gpus from multiple nodes
            [need_gpu / gpu_p_node] nodes, and one node with [need_gpu % gpu_p_node]
        if can't find , give up, and return False
        '''
        need_gpu = job['num_gpu']
        num_full_nodes = math.floor(need_gpu / self.num_gpu_p_node)
        idle_node_cpu = int(self.num_gpu_p_node * 4) 
        
        last_node_gpu =  need_gpu % self.num_gpu_p_node
        last_node_cpu = int(last_node_gpu * 4)
        

        full_node_list = list()
        for node in self.node_list:
            # if node.check_free_gpus() == node.num_gpu and node.check_free_cpus() >= idle_node_cpu and node.free_mem >= (ps_w_mem * node.num_gpu):
            if node.check_free_cpus() >= idle_node_cpu:
                gpu_selected = [gpu for gpu in node.gpu_list if len(gpu.job_list) == 0 and gpu.free_mem >= job['model']['mem']]
                if len(gpu_selected) < node.num_gpu:
                    continue
                #get idle node
                full_node_list.append(node)
                if len(full_node_list) == num_full_nodes:
                    #enough full nodes
                    break
        if len(full_node_list) < num_full_nodes:
            return False 
        
        last_node = None
        last_node_gpu_list = None
        if last_node_gpu != 0:
            for node in self.node_list: 
                if node not in full_node_list and node.check_free_cpus() >= last_node_cpu:
                    gpu_selected = [gpu for gpu in node.gpu_list if len(gpu.job_list) == 0 and gpu.free_mem >= job['model']['mem']]
                    if gpu_selected >= last_node_gpu:
                    # if node.check_free_gpus() >= last_node_gpu and node.check_free_cpus() >= last_node_cpu and node.free_mem >= (ps_w_mem * last_node_gpu):
                        #get last node
                        last_node = node
                        last_node_gpu_list = gpu_selected[:last_node_gpu]
                        break
            if last_node == None:
                return False


        ''' can allocate, do resource counting and record job placement '''
        node_list = list()
        idx = 0
        for node in full_node_list:
            gpu_list = node.gpu_list
            node.alloc_job_res(idle_node_cpu, job['model']['mem'], gpu_list, job)  # 分配gpu和cpu，将job加入gpu的job_list，利用率也加上去

            node_dict = dict()
            node_dict['id'] = node.id
            node_dict['num_gpu'] = node.num_gpu
            node_dict['num_cpu'] = idle_node_cpu
            node_dict['job_per_gpu_mem'] = job['model']['mem']
            node_dict['gpu_list'] = gpu_list

            node_dict['tasks'] = list()
            node_list.append(node_dict)

        if last_node_gpu != 0:      # 如果最后一个结点需要的gpu数不为0
            gpu_list = last_node_gpu_list
            last_node.alloc_job_res(last_node_cpu, job['model']['mem'], gpu_list, job)

            node_dict = dict()
            node_dict['id'] = last_node.id
            node_dict['num_gpu'] = last_node_gpu
            node_dict['num_cpu'] = last_node_cpu
            node_dict['job_per_gpu_mem'] = job['model']['mem']
            node_dict['gpu_list'] = gpu_list

            node_dict['tasks'] = list()
            node_list.append(node_dict)

        JOBS.create_multi_nodes_placement(job, self.id, node_list)
        return True


    def try_single_node_alloc(self, job):
        '''
        used in MS_YARN placement
        try get gpus from a single node
        if can't find a node, give up, and return False
        '''
        
        '''
        如果没有ps，则需要2个cpu，有ps则需要6个gpu
        '''
        need_gpu = job['num_gpu']
        need_cpu = 4 * need_gpu
        # if len(job['ps_network']) == 0 and job['num_gpu'] == 1:
        #     need_cpu = int(need_gpu * 2) # worker:2
        # else:
        #     need_cpu = int(need_gpu * 6) # worker:2, ps:4

        # print("try_single_node_alloc: ", need_gpu, need_cpu, JOBS.worker_mem)

        for node in self.node_list:
            # print(node.id, node.check_free_gpus(), node.check_free_cpus(), node.free_mem)
            gpu_selected = []
            if (node.check_free_cpus() >= need_cpu):
                # if node.alloc_gpus(need_gpu) == False:
                gpu_selected = [gpu for gpu in node.gpu_list if len(gpu.job_list) == 0 and gpu.free_mem >= job['model']['mem']]
                if len(gpu_selected) >= need_gpu:
                    gpu_list = gpu_selected[:need_gpu] 
                    node.alloc_job_res(need_cpu, job['model']['mem'], gpu_list, job)

                    # node.free_mem = node.free_mem - JOBS.worker_mem
                    JOBS.create_single_node_placement(job, self.id, node.id, need_gpu, need_cpu, job['model']['mem'], gpu_list)   # 创建一个node_dict, 加入job['placement']
                    return True
            else:
                continue

        return False


    def ms_yarn_alloc_gpus(self, job):
        '''
        ms_yarn allocates gpus from a single switch, 
        if no enough gpus, give up, return False (all-or-nothing)

        if need_gpu > gpu_p_node
            then get [need_gpu / gpu_p_node] nodes, and one node with [need_gpu % gpu_p_node]
        if need_gpu <= gpu_p_node
            then get one node with enough gpus
        '''
        need_gpu = job['num_gpu']
        ret = False
        if need_gpu > self.num_gpu_p_node:
            ret = self.try_cross_node_alloc(job)
        else:
            ret = self.try_single_node_alloc(job)

        return ret

    def ms_yarn_alloc_res(self, job):
        '''
        ms_yarn allocates res from a single switch, 
        if no enough gpus, give up, return False (all-or-nothing)

        if need_gpu > gpu_p_node
            then get [need_gpu / gpu_p_node] nodes, and one node with [need_gpu % gpu_p_node]
        if need_gpu <= gpu_p_node
            then get one node with enough gpus
        '''
        
        # utils.print_ljx("job:\n",job)
        need_gpu = job['num_gpu']
        ret = False
        if need_gpu > self.num_gpu_p_node:
            ret = self.try_cross_node_alloc(job)
        else:
            ret = self.try_single_node_alloc(job)

        return ret
    
    def merge_alloc_res(self, job):            
        '''
        一定会有placement
        # cpu is enough
        '''
        need_gpu = job['num_gpu']
        selected_gpus = []
        # 节点 排序
        sorted_nodes = sorted(self.node_list, key=lambda x: (x.workload))       # 按照节点上job数量进行排序
        # 筛选gpu
        gpus = list()
        for node in sorted_nodes:
            gpus.extend([gpu for gpu in node.gpu_list if gpu.free_mem >= job['model']['mem'] and len(gpu.get_job_list()) <= 1])

        gpus_hp, gpus_lp = list(), list()
        for gpu in gpus:
            if len(gpu.get_job_list()) == 0:
                gpus_hp.append(gpu)
            else:
                gpus_lp.append(gpu)

        if len(gpus_hp) >= need_gpu:
            selected_gpus = gpus_hp[:need_gpu]
        elif len(gpus_hp) + len(gpus_lp) >= need_gpu:
            selected_gpus = gpus_hp + gpus_lp[:need_gpu-len(gpus_hp)]
        if selected_gpus == []:
            return False
        
        selected_gpus = sorted(selected_gpus, key=lambda x: (x.node_id, x.idx))
        print('selected_gpus:',[gpu.gpu_id for gpu in selected_gpus])
        node_gpu = {}               # {node_id:[gpu,gpu...]}
        for gpu in selected_gpus:
            if gpu.node_id not in node_gpu:
                node_gpu[gpu.node_id] = []
            node_gpu[gpu.node_id].append(gpu)

        # 分配
        node_list = []
        for node_id,gpu_list in node_gpu.items():                  # 获取各个节点分配的资源
            need_gpu = len(gpu_list)
            need_cpu = int(need_gpu * 4) # worker:2, ps:4
            node = self.node_map[node_id]
            node.alloc_job_res(need_cpu, job['model']['mem'], gpu_list, job)
            node_dict = dict()
            node_dict['id'] = node_id
            node_dict['num_gpu'] = need_gpu
            node_dict['num_cpu'] = need_cpu
            node_dict['job_per_gpu_mem'] = job['model']['mem']
            node_dict['gpu_list'] = gpu_list

            node_dict['tasks'] = list()
            node_list.append(node_dict)

        JOBS.create_multi_nodes_placement(job, self.id, node_list)
            # JOBS.create_single_node_placement(job, self.id, node_id, need_gpu, need_cpu, job['model']['mem'], gpu_list)
            # JOBS.create_single_node_placement(job, self.id, node_key, need_gpu, need_cpu, JOBS.worker_mem, tmp_node_dict[node_key]) # 不考虑network？
            # self.node_list[node_key].free_mem -= JOBS.worker_mem
        # job['remaining_gpu'] = 0
        self.print_switch_status()
        return True

    def merge_alloc_res_consolidate(self, job):        
        def get_sort_key(node):
            need_gpu = job['num_gpu']
            keys = (node.free_gpus_num-need_gpu if node.free_gpus_num>=need_gpu else self.num_gpu_p_node, -node.free_gpus_num, node.workload)
            print(keys)
            return keys
        need_gpu = job['num_gpu']
        selected_gpus = []
        # 节点 排序
        # sorted_nodes = sorted(self.node_list, key=lambda x: (abs(x.free_gpus_num-need_gpu), -x.free_gpus_num, x.workload))       # 按照节点上job数量进行排序
        sorted_nodes = sorted(self.node_list, key=get_sort_key)
        print('sorted_nodes',[node.id for node in sorted_nodes])
        # 筛选gpu
        gpus = list()
        for node in sorted_nodes:
            gpus = [gpu for gpu in node.gpu_list if gpu.free_mem >= job['model']['mem'] and len(gpu.get_job_list()) <= 1]

            gpus_hp, gpus_lp = list(), list()
            for gpu in gpus:
                if len(gpu.get_job_list()) == 0:
                    gpus_hp.append(gpu)
                else:
                    gpus_lp.append(gpu)

            if len(gpus_hp) >= need_gpu:
                selected_gpus = gpus_hp[:need_gpu]
                break
            elif len(gpus_hp) + len(gpus_lp) >= need_gpu:
                selected_gpus = gpus_hp + gpus_lp[:need_gpu-len(gpus_hp)]
                break
        if selected_gpus == []:             # 需要跨机器
            print('try multi nodes ')
            return self.merge_alloc_res(job)  
        else:
            selected_gpus = sorted(selected_gpus, key=lambda x: (x.node_id, x.idx))
            print('single node selected_gpus:',[gpu.gpu_id for gpu in selected_gpus])
            node_gpu = {}               # {node_id:[gpu,gpu...]}
            for gpu in selected_gpus:
                if gpu.node_id not in node_gpu:
                    node_gpu[gpu.node_id] = []
                node_gpu[gpu.node_id].append(gpu)

            # 分配
            node_list = []
            for node_id,gpu_list in node_gpu.items():                  # 获取各个节点分配的资源
                need_gpu = len(gpu_list)
                need_cpu = int(need_gpu * 4) # worker:2, ps:4
                node = self.node_map[node_id]
                node.alloc_job_res(need_cpu, job['model']['mem'], gpu_list, job)
                node_dict = dict()
                node_dict['id'] = node_id
                node_dict['num_gpu'] = need_gpu
                node_dict['num_cpu'] = need_cpu
                node_dict['job_per_gpu_mem'] = job['model']['mem']
                node_dict['gpu_list'] = gpu_list

                node_dict['tasks'] = list()
                node_list.append(node_dict)

            JOBS.create_multi_nodes_placement(job, self.id, node_list)
        self.print_switch_status()
        return True   

    def bsbf_get_interf(self, job1, job2, with_mps=False):
        job1_name = JOBS.job_name_map[job1['model_name']]
        job2_name = JOBS.job_name_map[job2['model_name']]
        job1_ngpu, job2_ngpu = str(job1['num_gpu']), str(job2['num_gpu'])
        res = JOBS.interference[job1_name][job2_name][job1_ngpu]       # batchsize
        # print(job1_name,job2_name,job2_ngpu)
        if not with_mps:
            return res[0]
        else:
            print("with mps")
            return res[1]

    def share_check(self, job_a, job_b, with_mps=False):        # @ljx job_a:new_job     job_b:exist_job
        interference_a = self.bsbf_get_interf(job_a, job_b, with_mps)
        interference_b = self.bsbf_get_interf(job_b, job_a, with_mps)

        iter_b = job_b["remaining_iterations"]
        iter_time_b = job_b["iteration_time"]
        etc_b = iter_b * iter_time_b
        # _xi_b = 1.0 / (1.0 - interference_b)
        _xi_b = interference_b
        etc_b_hat = _xi_b * (iter_b * iter_time_b)
        
        iter_a = job_a["remaining_iterations"]
        iter_time_a = job_a["iteration_time"]
        etc_a = iter_a * iter_time_a
        # _xi_a = 1.0 / (1.0 - interference_a)
        _xi_a = interference_a
        etc_a_hat = _xi_a * (iter_a * iter_time_a)
        
        def compute_t_con():
            if etc_a_hat >= etc_b_hat:
                return 0.5 * etc_a + (1 - 1 / (2 * _xi_a)) * etc_b_hat
            else:
                return 0.5 * etc_b + (1 - 1 / (2 * _xi_b)) * etc_a_hat

        t_con = compute_t_con()
        t_seq = etc_b + 0.5 * etc_a
        flag = False
        if t_con < t_seq:
            self.con_times += 1
            flag = True
        else:
            self.seq_times += 1
            flag = False
        return flag, t_con
    
    def bsbf_alloc_res(self, job=None,with_mps=False):            
        '''
        一定会有placement
        # cpu is enough
        '''
        need_gpu = job['num_gpu']
        selected_gpus = []
        # 节点 排序
        sorted_nodes = sorted(self.node_list, key=lambda x: (x.workload))       # 按照节点上job数量进行排序
        # 筛选gpu
        gpus = list()
        for node in sorted_nodes:
            gpus.extend([gpu for gpu in node.gpu_list if gpu.free_mem >= job['model']['mem'] and len(gpu.get_job_list()) <= 1])

        gpus_hp, gpus_lp = list(), list()
        for gpu in gpus:
            if len(gpu.get_job_list()) == 0:
                gpus_hp.append(gpu)
            else:
                gpus_lp.append(gpu)

        if len(gpus_hp) >= need_gpu:
            selected_gpus = gpus_hp[:need_gpu]
        elif len(gpus_hp) + len(gpus_lp) >= need_gpu:
            share_job_time = {}     # {job_idx:t_con}
            no_share_job = []
            tmp_selected_gpus = []
            flag = False
            for gpu in gpus_lp:
                for j in gpu.get_job_list():
                    if j["job_idx"] in share_job_time or j in no_share_job:
                        break
                    share_flag, t_con = self.share_check(job,j,with_mps)
                    if share_flag:
                        share_job_time[j["job_idx"]] = t_con
                    else:
                        no_share_job.append(j)
            share_job_time = dict(sorted(share_job_time.items(), key=lambda x: x[1]))
            for j in share_job_time.keys():
                for gpu in gpus_lp:
                    if gpu in tmp_selected_gpus:
                        continue
                    job_idx_list = [job['job_idx'] for job in gpu.get_job_list()]
                    if j in job_idx_list:
                        tmp_selected_gpus.append(gpu)
                    if len(tmp_selected_gpus) + len(gpus_hp) >= need_gpu:
                        flag = True
                        break
                if flag:
                    break
            if flag:
                selected_gpus = gpus_hp + tmp_selected_gpus
        if selected_gpus == []:
            return False
        
        selected_gpus = sorted(selected_gpus, key=lambda x: (x.node_id, x.idx))
        print('selected_gpus:',[gpu.gpu_id for gpu in selected_gpus])
        node_gpu = {}               # {node_id:[gpu,gpu...]}
        for gpu in selected_gpus:
            if gpu.node_id not in node_gpu:
                node_gpu[gpu.node_id] = []
            node_gpu[gpu.node_id].append(gpu)

        # 分配
        node_list = []
        for node_id,gpu_list in node_gpu.items():                  # 获取各个节点分配的资源
            need_gpu = len(gpu_list)
            need_cpu = int(need_gpu * 4) # worker:2, ps:4
            node = self.node_map[node_id]
            node.alloc_job_res(need_cpu, job['model']['mem'], gpu_list, job)
            node_dict = dict()
            node_dict['id'] = node_id
            node_dict['num_gpu'] = need_gpu
            node_dict['num_cpu'] = need_cpu
            node_dict['job_per_gpu_mem'] = job['model']['mem']
            node_dict['gpu_list'] = gpu_list

            node_dict['tasks'] = list()
            node_list.append(node_dict)

        JOBS.create_multi_nodes_placement(job, self.id, node_list)
            # JOBS.create_single_node_placement(job, self.id, node_id, need_gpu, need_cpu, job['model']['mem'], gpu_list)
            # JOBS.create_single_node_placement(job, self.id, node_key, need_gpu, need_cpu, JOBS.worker_mem, tmp_node_dict[node_key]) # 不考虑network？
            # self.node_list[node_key].free_mem -= JOBS.worker_mem
        # job['remaining_gpu'] = 0
        self.print_switch_status()
        return True
    def bsbf_alloc_res_consolidate(self, job=None,with_mps=False):         
        '''
        一定会有placement
        '''
        def get_sort_key(node):
            need_gpu = job['num_gpu']
            return (node.free_gpus_num-need_gpu if node.free_gpus_num>=need_gpu else self.num_gpu_p_node, -node.free_gpus_num, node.workload)
        need_gpu = job['num_gpu']
        selected_gpus = []
        # 节点 排序
        # sorted_nodes = sorted(self.node_list, key=lambda x: (abs(x.free_gpus_num-need_gpu), -x.free_gpus_num, x.workload))       # 按照节点上job数量进行排序
        sorted_nodes = sorted(self.node_list, key=get_sort_key)
        # 筛选gpu
        gpus = list()
        for node in sorted_nodes:
            gpus = [gpu for gpu in node.gpu_list if gpu.free_mem >= job['model']['mem'] and len(gpu.get_job_list()) <= 1]

            gpus_hp, gpus_lp = list(), list()
            for gpu in gpus:
                if len(gpu.get_job_list()) == 0:
                    gpus_hp.append(gpu)
                else:
                    gpus_lp.append(gpu)

            if len(gpus_hp) >= need_gpu:
                selected_gpus = gpus_hp[:need_gpu]
            elif len(gpus_hp) + len(gpus_lp) >= need_gpu:
                share_job_time = {}     # {job_idx:t_con}
                no_share_job = []
                tmp_selected_gpus = []
                flag = False
                for gpu in gpus_lp:
                    for j in gpu.get_job_list():
                        if j["job_idx"] in share_job_time or j in no_share_job:
                            break
                        share_flag, t_con = self.share_check(job,j,with_mps)
                        if share_flag:
                            share_job_time[j["job_idx"]] = t_con
                        else:
                            no_share_job.append(j)
                print('no_share_job:',no_share_job)
                share_job_time = dict(sorted(share_job_time.items(), key=lambda x: x[1]))
                for j in share_job_time.keys():
                    for gpu in gpus_lp:
                        if gpu in tmp_selected_gpus:
                            continue
                        job_idx_list = [job['job_idx'] for job in gpu.get_job_list()]
                        if j in job_idx_list:
                            tmp_selected_gpus.append(gpu)
                        if len(tmp_selected_gpus) + len(gpus_hp) >= need_gpu:
                            flag = True
                            break
                    if flag:
                        break
                if flag:
                    selected_gpus = gpus_hp + tmp_selected_gpus
            if not (selected_gpus == []):
                break
        if selected_gpus == []:             # 需要跨机器
            print('try multi nodes ')
            return self.bsbf_alloc_res(job, with_mps)
        else:
            selected_gpus = sorted(selected_gpus, key=lambda x: (x.node_id, x.idx))
            print('single node selected_gpus:',[gpu.gpu_id for gpu in selected_gpus])
            node_gpu = {}               # {node_id:[gpu,gpu...]}
            for gpu in selected_gpus:
                if gpu.node_id not in node_gpu:
                    node_gpu[gpu.node_id] = []
                node_gpu[gpu.node_id].append(gpu)

            # 分配
            node_list = []
            for node_id,gpu_list in node_gpu.items():                  # 获取各个节点分配的资源
                need_gpu = len(gpu_list)
                need_cpu = int(need_gpu * 4) # worker:2, ps:4
                node = self.node_map[node_id]
                node.alloc_job_res(need_cpu, job['model']['mem'], gpu_list, job)
                node_dict = dict()
                node_dict['id'] = node_id
                node_dict['num_gpu'] = need_gpu
                node_dict['num_cpu'] = need_cpu
                node_dict['job_per_gpu_mem'] = job['model']['mem']
                node_dict['gpu_list'] = gpu_list

                node_dict['tasks'] = list()
                node_list.append(node_dict)

            JOBS.create_multi_nodes_placement(job, self.id, node_list)
        self.print_switch_status()
        return True

    def add_job_gpu_util(self, job):
        for placement in job['placements']:
            for node_pl in placement['nodes']:
                print("node_pl: ", node_pl)
                for gpu_id in node_pl['gpu_list']:
                    self.node_list[node_pl['id']].gpu_util_list[gpu_id] += 0.01


    def find_gpu_util(self, gpu_util_upper=0.8):
        '''
        find gpus whose gpu util < gpu_util_upper
        '''
        gpu_list = []
        for node in self.node_list:
            gpu_list_node = node.find_gpu_util(gpu_util_upper)
            gpu_list.extend(gpu_list_node)
        return gpu_list
    
    def sortGPUutil(self, elem):
        return self.node_list[elem['node']].gpu_util_list[elem['gpu']]

    def min_load_nodes(self, gpus1, need_gpu):
        '''
        return need_gpu gpus whose gpu util are minimum
        '''
        gpus1.sort(key=self.sortGPUutil)
        return gpus1[:need_gpu]


 

    # not used
    def release_gpus(self, nodes):
        '''
        release gpus from nodes
        nodes:
        [{'id':xx, 'num_gpu':xxx}, {'id':xx, 'num_gpu':xxx}]
        '''
        for node_dict in nodes:
            if ('id' not in node_dict) or ('num_gpu' not in node_dict):
                return False
            node = self.node_list[node_dict['id']]
            ret = node.release_gpus(node_dict['num_gpu'])
            if ret == False:
                return False

        return True

    def release_job_res(self, nodes, job=None):
        '''
        release job resources from nodes
        nodes:
        [{'id':xx, 'num_gpu':xxx, 'num_cpu': xxx, 'network': xxxx, 'tasks': [w2, ps2]}, 
        {'id':xx, 'num_gpu':xxx, 'num_cpu': xxx, 'network': xxxx, 'tasks': [ps0]}]
        '''
        for node_dict in nodes:
            if ('id' not in node_dict) or ('num_gpu' not in node_dict) or ('num_cpu' not in node_dict) or ('tasks' not in node_dict):
                print("switch release error, no info", job['job_idx'])
                return False
            node = self.node_list[node_dict['id']]
            # ret = node.release_gpus(node_dict['num_gpu'])
            ret = node.release_job_res(node_dict, job)
            if ret == False:
                print("switch release error, node release error", job['job_idx'])
                return False
        print('\n  after release_job_res:')
        self.print_switch_status()
        return True
    
    def print_switch_status(self):
        for node in self.node_list:
            print(f"  node id:{node.id}")
            for gpu in node.gpu_list:
                print(f'    gpu {gpu.gpu_id}: {[job["job_idx"] for job in gpu.job_list]}')