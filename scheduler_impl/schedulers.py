from __future__ import print_function
import copy
import json
import sys
import time

import utils
import flags
import jobs
from cluster import cluster
import log
import grpc

# from run import try_get_job_res     # ljx:tmp


FLAGS = flags.FLAGS

#prepare JOBS list
JOBS = jobs.JOBS

#get host info
CLUSTER = cluster.CLUSTER
# CLUSTER_TMP = cluster.CLUSTER_TMP

#get LOG object
LOG = log.LOG


'''
Allocate job resource
'''
def try_get_job_res(job=None,with_mps=False):
    '''
    select placement scheme
    '''
    utils.print_fn(f'{job["job_idx"]} need {job["num_gpu"]}')
    if FLAGS.scheme == 'yarn':
        ret = CLUSTER.ms_yarn_placement(job)
    elif FLAGS.scheme == 'merge':     
        # print("merge")
        ret = CLUSTER.merge_placement(job)                  # sjf-ffs
    elif FLAGS.scheme == 'merge-s':     
        # print("merge-s")
        ret = CLUSTER.merge_placement_consolidate(job)      # sjf-ffs-consolidate
    elif FLAGS.scheme == 'bsbf':     
        # print("bsbf")
        ret = CLUSTER.bsbf_placement(job,with_mps)          # sjf-bsbf
    elif FLAGS.scheme == 'bsbfs':           
        # print("bsbfs")
        ret = CLUSTER.bsbfs_placement(job,with_mps)         # sjf-bsbf-consolidate
    else:
        ret = CLUSTER.ms_yarn_placement(job)
    if ret == True:
        # job['status'] = 'RUNNING'
        pass
    utils.print_fn(f'  ret:{ret},placement len:{len(job["placements"])}')
    if len(job["placements"]) > 0:
        print(f'  node len:{len(job["placements"][0]["nodes"])}\n')
    return ret

def delete_jobs(scheduler):
    '''
    按照用户请求删除现有的job
    '''
    while not JOBS.delete_queue.empty():
        job_id = JOBS.delete_queue.get()
        job = JOBS.find_job_by_id(job_id)
        scheduler._logger.info(f'delete_jobs, {job["status"]}')
        if 'RUNNING' == job['status']:
            scheduler._logger.info(f'before delete; check: running  {job["job_idx"]} {job["job_id"]} {job["remaining_iterations"]} {job["pending_time"]} {job["total_executed_time"]}')
            time_save_begin = time.time()
            try:   
                scheduler.save_model([job['job_idx']])      
                scheduler._logger.info(f'scheduler, {job["model_name"]}, model save time: {time.time()-time_save_begin}') 
                if job['job_idx'] in scheduler._trainers:
                    scheduler._trainers.pop(job['job_idx'])
                jobinfo = JOBS.to_jobinfo(job)
                scheduler._controller.kill(jobinfo)
                CLUSTER.release_job_res(job)
                JOBS.runnable_jobs.remove(job)
            except Exception as e:
                error_message = str(e)
                print('delete job', error_message)  
        elif 'PENDING' == job['status']:
            scheduler._logger.info(f'before delete; check: pending  {job["job_idx"]} {job["job_id"]} {job["remaining_iterations"]} {job["pending_time"]} {job["total_executed_time"]}')

            JOBS.runnable_jobs.remove(job)
            
        elif 'EVENT' == job['status']:  
            scheduler._logger.info(f'before delete; check: event  {job["job_idx"]} {job["job_id"]} {job["remaining_iterations"]} {job["pending_time"]}')
            JOBS.remove_job_in_events(job)
            # JOBS.job_list.remove(job)
        elif 'END' == job['status']: 
            # JOBS.runnable_jobs.remove(job)
            scheduler._logger.info(f'before delete; check: ending  {job["job_idx"]} {job["job_id"]} {job["remaining_iterations"]} {job["pending_time"]} {job["total_executed_time"]}')
            pass
        scheduler._logger.info(f'scheduler, {job["job_idx"]}, {job["job_id"]}, delete ')
        
# 调度流程 1：处理完成任务
def deal_done_jobs(scheduler):
    done_flag = False
    scheduler._controller.done_queue_lock.acquire()                     # 加锁
    while not scheduler._controller.done_queue.empty():
        finished_time, job_id, worker_id, gpus, returncode = scheduler._controller.done_queue.get()
        e_job = JOBS.find_runnable_job(job_id)
        tmp = finished_time-e_job['last_check_time']
        e_job['total_executed_time'] += tmp
        e_job['last_check_time'] = finished_time
        if returncode==0:
            CLUSTER.release_job_res(e_job)
            # print('returncode==0',e_job['job_idx'])
            scheduler._trainers.pop(e_job['job_idx'])
            LOG.job_complete(e_job, finished_time)
            scheduler.finished_job_cnt += 1
            scheduler._logger.info(f'**** job[{e_job["job_idx"]}] completed')
            scheduler._logger.info(f'scheduler finishes {scheduler.finished_job_cnt} jobs in all!')
            JOBS.runnable_jobs.remove(e_job)
        else:    
            scheduler._trainers.pop(e_job['job_idx'])
            CLUSTER.release_job_res(e_job)      # ljx 出错的job也应释放资源
            e_job['status'] = 'PENDING'
            del e_job['placements'][:]
            e_job['resume'] -= 1                # 由于采用resume来判断是否有保存的模型，为避免第一次运行失败后第二次去加载模型，这里减1
        done_flag = True
        utils.print_fn(f'time:{scheduler.get_time()}, check: finished , {e_job["job_idx"]}, {finished_time}')
    scheduler._controller.done_queue_lock.release() 
    return done_flag

# 调度流程 2：添加新任务
def add_new_jobs(scheduler):
    new_flag = False
    while scheduler.has_ready_jobs(scheduler.get_time()):
        event = JOBS.job_events.pop(0)
        assert 'start_jobs' in event
        for s_job in event['start_jobs']:
            JOBS.move_to_runnable(s_job)
            s_job['remaining_time'] = s_job['iteration_time'] * s_job['remaining_iterations']
            s_job['remaining_gputime'] = s_job['remaining_time'] * s_job['num_gpu']
            scheduler._logger.info(f'---- job[{s_job["job_idx"]}] is added')
        new_flag = True
    return new_flag

# 调度流程 3：状态更新
def update_jobs_status(scheduler, gputime):
    tmp_time = scheduler.get_time()
    JOBS.cur_time = tmp_time

    for rjob in JOBS.runnable_jobs:
        if 'RUNNING' == rjob['status']:

            tmp = tmp_time - rjob['last_check_time']
            rjob['total_executed_time'] = rjob['total_executed_time'] + tmp
            rjob['last_check_time'] = tmp_time
            utils.print_fn(f'query_stats: {rjob["job_idx"]}')

            if rjob['job_idx'] not in scheduler._trainers:
                utils.print_fn(f'{rjob["job_idx"]} not in scheduler._trainers; maybe pending in mpirun...')
                continue
            assert rjob['job_idx'] in scheduler._trainers

            try:
                finished_iter, iteration_time = scheduler.query_stats([rjob['job_idx']])    # 更新迭代时间,和剩余迭代次数
                rjob['remaining_iterations'] -= finished_iter
                rjob['iteration_time'] = iteration_time
            except Exception as e:
                scheduler._logger.info(f'job {rjob["job_idx"]} {rjob["model_name"]}: done but not in done queue')  
                rjob['remaining_iterations'] = 0

            rjob['remaining_time'] = rjob['iteration_time'] * rjob['remaining_iterations']  # 更新剩余时间
            if gputime:
                rjob['remaining_gputime'] = rjob['remaining_time'] * rjob['num_gpu']
            scheduler._logger.info(f'{tmp_time} check: running  {rjob["job_idx"]} {rjob["job_id"]} {rjob["remaining_iterations"]} {rjob["total_executed_time"]}')  

        elif 'PENDING' == rjob['status']:
            tmp = tmp_time - rjob['last_check_time']
            rjob['pending_time'] += tmp
            rjob['last_check_time'] = tmp_time
            scheduler._logger.info(f'{tmp_time} check: pending  {rjob["job_idx"]} {rjob["job_id"]} {rjob["remaining_iterations"]} {rjob["pending_time"]}')

        elif 'END' == rjob['status']: #almost impossible
            JOBS.runnable_jobs.remove(rjob)
            scheduler._logger.info(f'{tmp_time} check: ending  {rjob["job_idx"]} {rjob["job_id"]} {rjob["remaining_iterations"]}')
            pass

# 调度流程 4：任务调度
def sort_jobs(scheduler,gputime):

    delete_jobs(scheduler)
    #sort jobs with shortest first
    if gputime:
        JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('remaining_gputime'))
    else:
        JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('remaining_time'))

# 调度流程 5：资源分配
def jobs_placement(scheduler, is_preempt=True, with_mps = False):
    run_jobs = []
    preempt_jobs,preempt_jobs_set = [], set()
    utils.print_fn(f'before jobs_placement:{[(job["job_idx"],job["status"]) for job in JOBS.runnable_jobs]}')
    # 情况1：不抢占
    if not is_preempt:
        for idx, rjob in enumerate(JOBS.runnable_jobs):
            ret = False
            if rjob in run_jobs or rjob in preempt_jobs:
                continue

            # 1. 之前运行，现在仍然运行
            if 'RUNNING' == rjob['status'] or rjob['remaining_iterations'] == 0:                                   
                scheduler._logger.info(f'scheduler, {rjob["job_idx"]}, still running. remaining_iterations:{rjob["remaining_iterations"]} ')
                continue

            # 2. 新运行
            scheduler._logger.info(f'try_get_job_res, {rjob["job_idx"]}')
            ret = try_get_job_res(rjob,with_mps)                                                                    
            if ret:
                run_jobs.append(rjob)
        return run_jobs, preempt_jobs
    
    # 情况2：抢占   
    for idx, rjob in enumerate(JOBS.runnable_jobs):

        ret = False
        # 1. 之前运行，现在仍然运行 
        if 'RUNNING' == rjob['status']  or rjob['remaining_iterations'] == 0:                                            
            scheduler._logger.info(f'scheduler, {rjob["job_idx"]}, still running. remaining_iterations:{rjob["remaining_iterations"]} ')
            continue
        
        # 2. 新运行
        ret = try_get_job_res(rjob,with_mps)                                                                            
        if not ret:              # 资源不够
            # 2.1 寻找要抢占的job 
            for rev_idx in range(1, len(JOBS.runnable_jobs) - idx):           
                potential_job_to_preempt = JOBS.runnable_jobs[-rev_idx]
                if potential_job_to_preempt['status'] == 'RUNNING' and potential_job_to_preempt['remaining_iterations'] > 100:     # 抢占正在运行的剩余迭代次数大于100的job 

                    tmp = copy.deepcopy(potential_job_to_preempt)

                    preempt_jobs.append(tmp)                            # 抢占，将被抢占的job置为pending
                    CLUSTER.release_job_res(potential_job_to_preempt)   # 释放资源
                    potential_job_to_preempt['status'] = 'PENDING'
                    potential_job_to_preempt['preempt'] = int(potential_job_to_preempt['preempt'] + 1)

                    print('job to preempt',[(job['job_idx'],job['status']) for job in [potential_job_to_preempt]])
                    assert 'placements' in potential_job_to_preempt
                    del potential_job_to_preempt['placements'][:]

                    # 2.2 重新判断资源是否满足
                    ret = try_get_job_res(rjob,with_mps)                
                    if ret:
                        break
        if ret:
            run_jobs.append(rjob)

    utils.print_fn(f'jobs_placement run_jobs: {[(job["job_idx"],job["status"]) for job in run_jobs]}')
    utils.print_fn(f'jobs_placement preempt_jobs: {[(job["job_idx"],job["status"]) for job in preempt_jobs]}')
    return run_jobs, preempt_jobs

def check_placement_is_same(placements1, placements2):
    res = True
    for p1, p2 in zip(placements1,placements2):
        if p1['switch'] != p2['switch']:
            return False
        for n1, n2 in zip(p1['nodes'], p2['nodes']):
            if n1['id'] != n2['id']:
                return False
            for g1, g2 in zip(n1['gpu_list'],n2['gpu_list']):
                # print(g1,g2)
                if g1.gpu_id != g2.gpu_id:
                    return False
    return True

# 调度流程 6：任务执行
def jobs_execute(scheduler, run_jobs, preempt_jobs):
    count = 0
    # 6.1 去除分配资源不变得任务，减少切换开销
    for rj in run_jobs:
        for pj in preempt_jobs:
            if rj['job_idx'] == pj['job_idx']:
                # print(rj["job_idx"], rj['placements'])
                # print(pj["job_idx"], pj['placements'])
                # print([gpu.gpu_id for gpu in rj['placements'][0]["nodes"][0]["gpu_list"]])
                # print([gpu.gpu_id for gpu in pj['placements'][0]["nodes"][0]["gpu_list"]])
                # if rj['placements'] == pj['placements']:
                if check_placement_is_same(rj['placements'], pj['placements']):
                    rj['status'] = "RUNNING"
                    rj['preempt'] -= 1
                    run_jobs.remove(rj)
                    preempt_jobs.remove(pj)
                    count += 1
    # print("pre preempt count:",f'{len(preempt_jobs)} + {count}')
    # print('  jobs_placement run_jobs:', [(job['job_idx'],job['status']) for job in run_jobs])
    # print('  jobs_placement preempt_jobs:', [(job['job_idx'],job['status']) for job in preempt_jobs])
                    
    job_not_need_run = set()
    # 6.2 要抢占的job
    for job in preempt_jobs:
        time_save_begin = time.time()
        assert job['job_idx'] in scheduler._trainers
        try:   
            scheduler.save_model([job['job_idx']])      
            scheduler._logger.info(f'scheduler, {job["model_name"]}, model save time: {time.time()-time_save_begin}') 
            if job['job_idx'] in scheduler._trainers:
                scheduler._trainers.pop(job['job_idx'])

            jobinfo = JOBS.to_jobinfo(job)
            scheduler._controller.kill(jobinfo)
            scheduler._logger.info(f'scheduler, {job["job_idx"]}, preempt')
        except Exception as e:
            error_message = str(e)
            print(error_message)
            scheduler._logger.info(f'job {job["job_idx"]} {job["model_name"]}: done but not in done queue, not need to save and kill')  
            job_not_need_run.add(job['job_idx'])

    tmp_time = scheduler.get_time()
    # 6.3 要运行的job
    for job in run_jobs:
        if job['job_idx'] in job_not_need_run:
            continue
        job['job_counter'] += 1
        if sys.maxsize == job['start_time']:
            job['start_time'] = tmp_time

        jobinfo = JOBS.to_jobinfo(job)
        scheduler._controller.execute(jobinfo)
        
        job['status'] = 'RUNNING'
        job['resume'] = int(job['resume'] + 1)     

        scheduler._logger.info(f'scheduler, {job["job_idx"]}, run')
    utils.print_fn('--------------------------------- End of Current Scheduling ---------------------------------\n')
    

def fifo_sim_jobs(scheduler=None, gputime=False, place=False):

    scheduler._controller.set_start_time()

    while (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
        done_flag = False
        new_flag = False
        scheduler._logger.info(f'before: {scheduler.get_time()}')
        JOBS.job_lock.acquire()

        # 调度流程 1：处理完成任务
        done_flag = deal_done_jobs(scheduler)

        # 调度流程 2：添加新任务
        new_flag = add_new_jobs(scheduler)
        
        time00 = time.time()

        # 调度流程 3：状态更新
        update_jobs_status(scheduler,gputime)

        # 调度流程 4：任务调度
        JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('submit_time'))
        
        # 调度流程 5：资源分配
        run_jobs, preempt_jobs = jobs_placement(scheduler)
        JOBS.job_lock.release()

        # 调度流程 6：任务执行
        jobs_execute(scheduler, run_jobs, preempt_jobs)
        
        if (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
            time.sleep(30)  # 等待30s，让任务启动
            LOG.checkpoint(scheduler.get_time(), scheduler, done_flag or new_flag or len(run_jobs) > 0 or len(preempt_jobs) > 0)   
            time01 = time.time()
            # print('checkpoint and save model time', time01-time00)
            count = 0

            while count < FLAGS.schedule_interval-(time01-time00):      #  and scheduler._controller.done_queue.empty()
                time.sleep(2)
                count += 2
            time.sleep(1)

        scheduler._logger.info(f'at end: {scheduler.get_time()}\n\n')  

def sjf_sim_jobs(scheduler=None, gputime=False, place=False):

    scheduler._controller.set_start_time()

    while (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
        done_flag = False
        new_flag = False
        scheduler._logger.info(f'before: {scheduler.get_time()}')
        JOBS.job_lock.acquire()

        # 调度流程 1：处理完成任务
        done_flag = deal_done_jobs(scheduler,)

        # 调度流程 2：添加新任务
        new_flag = add_new_jobs(scheduler)
        
        time00 = time.time()

        # 调度流程 3：状态更新
        update_jobs_status(scheduler,gputime)

        # 调度流程 4：任务调度
        sort_jobs(scheduler,gputime)
        
        # 调度流程 5：资源分配
        run_jobs, preempt_jobs = jobs_placement(scheduler)
        JOBS.job_lock.release()

        # 调度流程 6：任务执行
        jobs_execute(scheduler, run_jobs, preempt_jobs)

        if (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
            time.sleep(30)  # 等待30s，让任务启动
            LOG.checkpoint(scheduler.get_time(), scheduler, done_flag or new_flag or len(run_jobs) > 0 or len(preempt_jobs) > 0)   
            time01 = time.time()
            # print('checkpoint and save model time', time01-time00)
            count = 0
            while count < FLAGS.schedule_interval-(time01-time00):      #  and scheduler._controller.done_queue.empty()
                time.sleep(1)
                count += 1

        scheduler._logger.info(f'at end: {scheduler.get_time()}\n\n')  

def Tiresias_jobs(scheduler=None, gputime=False, place=False):
    solve_starvation = 1
    # end_events = list()
    scheduler._controller.set_start_time()

    while (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
        done_flag = False
        new_flag = False
        scheduler._logger.info(f'before: {scheduler.get_time()}')
        JOBS.job_lock.acquire()
        # 1. deal with done queue
        scheduler._controller.done_queue_lock.acquire()                     # 加锁
        while not scheduler._controller.done_queue.empty():
            finished_time, job_id, worker_id, gpus, returncode = scheduler._controller.done_queue.get()
            e_job = JOBS.find_runnable_job(job_id)
            tmp = finished_time-e_job['last_check_time']
            e_job['total_executed_time'] += tmp
            e_job['last_check_time'] = finished_time
            if returncode==0:
                CLUSTER.release_job_res(e_job)
                # print('returncode==0',e_job['job_idx'])
                scheduler._trainers.pop(e_job['job_idx'])
                LOG.job_complete(e_job, finished_time)
                scheduler.finished_job_cnt += 1
                scheduler._logger.info(f'**** job[{e_job["job_idx"]}] completed')
                scheduler._logger.info(f'scheduler finishes {scheduler.finished_job_cnt} jobs in all!')
                JOBS.runnable_jobs.remove(e_job)
                JOBS.queues[e_job['q_id']].remove(e_job)
            else:    
                scheduler._trainers.pop(e_job['job_idx'])
                CLUSTER.release_job_res(e_job)      # ljx 出错的job也应释放资源 release_job_res会将job的status置为END，但这里应该时PENDING
                e_job['status'] = 'PENDING'
                del e_job['placements'][:]
                e_job['resume'] -= 1                # 由于采用resume来判断是否有保存的模型，为避免第一次运行失败后第二次去加载模型，这里减1
            done_flag = True
            utils.print_fn(f'time:{scheduler.get_time()}, check: finished , {e_job["job_idx"]}, {finished_time}')
        scheduler._controller.done_queue_lock.release() 

        # 2. add new jobs
        
        while scheduler.has_ready_jobs(scheduler.get_time()):
            event = JOBS.job_events.pop(0)
            assert 'start_jobs' in event
            for s_job in event['start_jobs']:
                JOBS.move_to_runnable(s_job)
                s_job['q_id'] = 0
                JOBS.queues[0].append(s_job)
                s_job['remaining_time'] = s_job['iteration_time'] * s_job['remaining_iterations']
                s_job['remaining_gputime'] = s_job['remaining_time'] * s_job['num_gpu']
                scheduler._logger.info(f'---- job[{s_job["job_idx"]}] is added')
            new_flag = True
        
        # 3. update_jobs_status
        demote_flag = False
        time00 = time.time()
        tmp_time = scheduler.get_time()
        JOBS.cur_time = tmp_time
        for rjob in JOBS.runnable_jobs:
            if 'RUNNING' == rjob['status']:
                tmp = tmp_time - rjob['last_check_time']
                rjob['total_executed_time'] = rjob['total_executed_time'] + tmp
                rjob['executed_time'] += tmp
                rjob['last_check_time'] = tmp_time
                assert rjob['job_idx'] in scheduler._trainers
                try:
                    utils.print_fn(f'query_stats: {rjob["job_idx"]}')
                    finished_iter, iteration_time = scheduler.query_stats([rjob['job_idx']])    # 同时更新迭代时间
                    rjob['remaining_iterations'] -= finished_iter
                    rjob['iteration_time'] = iteration_time
                except Exception as e:
                    scheduler._logger.info(f'job {rjob["job_idx"]} {rjob["model_name"]}: done but not in done queue')  
                    rjob['remaining_iterations'] = 0
                    # rjob['iteration_time'] = iteration_time

                # rjob['remaining_time'] = rjob['iteration_time'] * rjob['remaining_iterations']
                # if gputime:
                #     rjob['remaining_gputime'] = rjob['remaining_time'] * rjob['num_gpu']
                scheduler._logger.info(f'{tmp_time} check: running  {rjob["job_idx"]} {rjob["remaining_iterations"]} {rjob["total_executed_time"]}')  
                j_gt = 0
                if gputime:         # gputime=True
                    j_gt = rjob['executed_time'] * rjob['num_gpu']
                else:
                    j_gt = rjob['executed_time']
                cur_qid = rjob['q_id']
                while cur_qid<int(JOBS.num_queue - 1) and j_gt >=JOBS.queue_limit[cur_qid]: # 优先级队列的变化(降级)
                    rjob['q_id'] = cur_qid+1
                    JOBS.queues[rjob['q_id']].append(rjob)
                    JOBS.queues[cur_qid].remove(rjob)
                    # print(f'job {rjob["job_idx"]} demote to Q{rjob["q_id"]}')
                    demote_flag = True
                    cur_qid = rjob['q_id']
            elif 'PENDING' == rjob['status']:
                tmp = tmp_time - rjob['last_check_time']
                rjob['pending_time'] += tmp
                rjob['last_check_time'] = tmp_time
                if rjob['executed_time'] >0:
                    rjob['last_pending_time'] += tmp                                            # 本次pending持续的总时间
                if solve_starvation>0 and rjob['q_id']>0 and rjob['total_executed_time']>0 and rjob['executed_time']>0: # 饥饿太久 升级
                    if rjob['last_pending_time']>= rjob['executed_time'] * solve_starvation:
                        rjob['executed_time'] = 0
                        rjob['last_pending_time'] = 0
                        JOBS.queues[0].append(rjob)
                        JOBS.queues[rjob['q_id']].remove(rjob)
                        rjob['q_id'] = 0
                        rjob['promote'] = int(rjob['promote'] + 1)
                        demote_flag = True
                scheduler._logger.info(f'{tmp_time} check: pending  {rjob["job_idx"]} {rjob["remaining_iterations"]} {rjob["pending_time"]}')
            elif 'END' == rjob['status']: #almost impossible
                JOBS.runnable_jobs.remove(rjob)
                scheduler._logger.info(f'{tmp_time} check: ending  {rjob["job_idx"]} {rjob["remaining_iterations"]}')
                pass
        assert len(JOBS.runnable_jobs) == sum([len(queue) for queue in JOBS.queues])
        # 4. sort
        tmp_runnable_jobs = list()
        for queue in JOBS.queues:
            for rjob in queue:
                tmp_runnable_jobs.append(rjob)

        # 5. 分配资源
        run_jobs, preempt_jobs = jobs_placement(scheduler)
        JOBS.job_lock.release()

        # 6. 任务执行
        jobs_execute(scheduler, run_jobs, preempt_jobs)

        for idx,queue in enumerate(JOBS.queues):
            pending_job = list()
            for job in queue:
                if job['status'] == 'PENDING':
                    pending_job.append(job)
            for job in pending_job:
                queue.remove(job)
            queue.extend(pending_job)       # 将pending job放最后
            utils.print_fn(f'queue {idx}:{[(job["job_idx"],job["status"]) for job in queue]}')


        if (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
            time.sleep(30)  # 等待10s，让任务启动
            LOG.checkpoint(scheduler.get_time(), scheduler, done_flag or new_flag or len(run_jobs) > 0 or len(preempt_jobs) > 0)    # ljx 暂时注释
            time01 = time.time()
            # print('checkpoint and save model time', time01-time00)
            count = 0
            while count < FLAGS.schedule_interval-(time01-time00):      #  and scheduler._controller.done_queue.empty()
                time.sleep(1)
                count += 1

        scheduler._logger.info(f'at end: {scheduler.get_time()}\n\n')  

def sjf_ffs_jobs(scheduler=None, is_preempt=True, with_mps=False, gputime=False, place=False):
    scheduler._controller.set_start_time()

    while (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:

        scheduler._logger.info(f'before: {scheduler.get_time()}')
        JOBS.job_lock.acquire()

        # 调度流程 1：处理完成任务
        done_flag = deal_done_jobs(scheduler)

        # 调度流程 2：添加新任务
        new_flag = add_new_jobs(scheduler)

        time00 = time.time()

        # 调度流程 3：状态更新
        update_jobs_status(scheduler,gputime)

        # 调度流程 4：任务调度
        sort_jobs(scheduler,gputime)
        
        # 调度流程 5：资源分配
        run_jobs, preempt_jobs = jobs_placement(scheduler,is_preempt,with_mps)
        JOBS.job_lock.release()

        # 调度流程 6：任务执行
        jobs_execute(scheduler, run_jobs, preempt_jobs)

        if (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
            time.sleep(30)  # 等待30s，让任务启动
            LOG.checkpoint(scheduler.get_time(), scheduler, done_flag or new_flag or len(run_jobs) > 0 or len(preempt_jobs) > 0)    # ljx 暂时注释
            time01 = time.time()
            # print(f'at end: {scheduler.get_time()}\n\n', 'checkpoint and save model time', time01-time00, "\n\n")
            count = 0

            while count < FLAGS.schedule_interval-(time01-time00):      #  and scheduler._controller.done_queue.empty()
                time.sleep(1)
                count += 1

        scheduler._logger.info(f'at end: {scheduler.get_time()}\n\n')  



def sjf_bsbf_jobs(scheduler=None, is_preempt=True, with_mps=False, gputime=False, place=False):

    scheduler._controller.set_start_time()

    while (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
        scheduler._logger.info(f'before: {scheduler.get_time()}')
        JOBS.job_lock.acquire()

        # 调度流程 1：处理完成任务
        done_flag = deal_done_jobs(scheduler)

        # 调度流程 2：添加新任务
        new_flag = add_new_jobs(scheduler)

        time00 = time.time()

        # 调度流程 3：状态更新
        update_jobs_status(scheduler,gputime)

        # 调度流程 4：任务调度
        sort_jobs(scheduler,gputime)

        # 调度流程 5：资源分配
        run_jobs, preempt_jobs = jobs_placement(scheduler,is_preempt, with_mps)
        JOBS.job_lock.release()

        # 调度流程 6：任务执行
        jobs_execute(scheduler, run_jobs, preempt_jobs)

        if (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
            time.sleep(30)  # 等待30s，让任务启动
            LOG.checkpoint(scheduler.get_time(), scheduler, done_flag or new_flag or len(run_jobs) > 0 or len(preempt_jobs) > 0)   
            time01 = time.time()
            # print(f'at end: {scheduler.get_time()}\n\n', 'checkpoint and save model time', time01-time00, "\n\n")
            count = 0

            while count < FLAGS.schedule_interval-(time01-time00) :     # and scheduler._controller.done_queue.empty()
                time.sleep(1)
                count += 1

        scheduler._logger.info(f'at end: {scheduler.get_time()}\n\n')  


def sjf_bsbf_no_preempt(scheduler=None, with_mps=False, gputime=False, place=False):

    scheduler._controller.set_start_time()
    finished_job_cnt = 0
    while (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
        done_flag = False
        new_flag = False
        scheduler._logger.info(f'before: {scheduler.get_time()}')
        scheduler._controller.done_queue_lock.acquire()                         # 加锁
        while not scheduler._controller.done_queue.empty():
            finished_time, job_id, worker_id, gpus, returncode = scheduler._controller.done_queue.get()
            e_job = JOBS.find_runnable_job(job_id)
            tmp = finished_time-e_job['last_check_time']
            e_job['total_executed_time'] += tmp
            e_job['last_check_time'] = finished_time
            if returncode==0:
                CLUSTER.release_job_res(e_job)
                scheduler._trainers.pop(e_job['job_idx'])
                LOG.job_complete(e_job, finished_time)
                finished_job_cnt += 1
                scheduler._logger.info(f'**** job[{e_job["job_idx"]}] completed')
                scheduler._logger.info(f'scheduler finishes {finished_job_cnt} jobs in all!')
                JOBS.runnable_jobs.remove(e_job)
            else:    
                scheduler._trainers.pop(e_job['job_idx'])
                CLUSTER.release_job_res(e_job)      # ljx 出错的job也应释放资源 release_job_res会将job的status置为END，但这里应该时PENDING
                e_job['status'] = 'PENDING'
                del e_job['placements'][:]
                e_job['resume'] -= 1                # 由于采用resume来判断是否有保存的模型，为避免第一次运行失败后第二次去加载模型，这里减1
            done_flag = True
            print(scheduler.get_time(), 'check: done ', e_job['job_idx'], finished_time)
        scheduler._controller.done_queue_lock.release()                         # 解锁
        while scheduler.has_ready_jobs(scheduler.get_time()):
            event = JOBS.job_events.pop(0)
            assert 'start_jobs' in event
            for s_job in event['start_jobs']:
                JOBS.move_to_runnable(s_job)
                s_job['remaining_time'] = s_job['iteration_time'] * s_job['remaining_iterations']
                s_job['remaining_gputime'] = s_job['remaining_time'] * s_job['num_gpu']
                scheduler._logger.info(f'---- job[{s_job["job_idx"]}] is added')
            new_flag = True
        time00 = time.time()
        tmp_time = scheduler.get_time()
        if done_flag or new_flag:
            # assert tmp_time - last_check_time > FLAGS.schedule_interval or tmp_time < FLAGS.schedule_interval # 取消特定周期调度
            for rjob in JOBS.runnable_jobs:
                if 'RUNNING' == rjob['status']:
                    tmp = tmp_time - rjob['last_check_time']
                    rjob['total_executed_time'] = rjob['total_executed_time'] + tmp
                    rjob['last_check_time'] = tmp_time
                    try:                                                                                    # 用try，因为job可能在之前done检查之后完成了，此时连接断开，无法查询到job状态
                        finished_iter, iteration_time = scheduler.query_stats([rjob['job_idx']])    # 同时更新迭代时间
                        rjob['remaining_iterations'] -= finished_iter
                        rjob['iteration_time'] = iteration_time
                    except Exception as e:
                        scheduler._logger.info(f'job {rjob["job_idx"]} {rjob["model_name"]}: done but not in done queue')  
                        rjob['remaining_iterations'] = 0
                        rjob['iteration_time'] = iteration_time
                    # print(rjob['job_idx'], rjob['remaining_iterations'], finished_iter)
                    rjob['remaining_time'] = rjob['iteration_time'] * rjob['remaining_iterations']
                    if gputime:
                        rjob['remaining_gputime'] = rjob['remaining_time'] * rjob['num_gpu']
                    scheduler._logger.info(f'{tmp_time} check: running  {rjob["job_idx"]} {rjob["remaining_iterations"]} {rjob["total_executed_time"]}')  
                elif 'PENDING' == rjob['status']:
                    tmp = tmp_time - rjob['last_check_time']
                    rjob['pending_time'] += tmp
                    rjob['last_check_time'] = tmp_time
                    scheduler._logger.info(f'{tmp_time} check: pending  {rjob["job_idx"]} {rjob["remaining_iterations"]} {rjob["pending_time"]}')
                elif 'END' == rjob['status']: #almost impossible
                    JOBS.runnable_jobs.remove(rjob)
                    scheduler._logger.info(f'{tmp_time} check: ending  {rjob["job_idx"]} {rjob["remaining_iterations"]}')
                    pass
            #sort jobs with shortest first
            if gputime:
                JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('remaining_gputime'))
            else:
                JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('remaining_time'))

            run_jobs = list()
            preempt_jobs = list()

            for idx, rjob in enumerate(JOBS.runnable_jobs):
                ret = False
                if rjob in run_jobs or rjob in preempt_jobs:
                    continue
                if 'RUNNING' == rjob['status']:                                     # ① 之前运行，现在仍然运行
                    scheduler._logger.info(f'scheduler, {rjob["job_idx"]}, still running. remaining_iterations:{rjob["remaining_iterations"]} ')
                    continue

                scheduler._logger.info(f'try_get_job_res, {rjob["job_idx"]}')
                ret = try_get_job_res(rjob,with_mps)                                         # ② 新运行
                # if not ret:                                                         # 资源不够
                #     for rev_idx in range(1, len(JOBS.runnable_jobs) - idx):         # ③ 寻找要抢占的job   
                #         potential_job_to_preempt = JOBS.runnable_jobs[-rev_idx]
                #         if potential_job_to_preempt['status'] == 'RUNNING':
                #             scheduler._logger.info(f'release_job_res, {potential_job_to_preempt["job_idx"]}')
                #             CLUSTER.release_job_res(potential_job_to_preempt)   # 释放资源
                #             preempt_jobs.append(potential_job_to_preempt)       # 抢占，将被抢占的job置为pending

                #             ret = try_get_job_res(rjob)                         # 重新判断资源是否满足
                #             if ret:
                #                 break
                if ret:
                    run_jobs.append(rjob)
                # else:
                #     break

            # 要抢占的job
            for job in preempt_jobs:
                time_save_begin = time.time()   
                try:   
                    scheduler.save_model([job['job_idx']])      
                    scheduler._logger.info(f'scheduler, {job["model_name"]}, model save time: {time.time()-time_save_begin}') 
                except Exception as e:
                    scheduler._logger.info(f'job {rjob["job_idx"]} {rjob["model_name"]}: done but not in done queue, not need to save')  

                # assert scheduler.save_model([job['job_idx']])                   # 如果模型保存不成功会影响后续job重新运行
                # scheduler._logger.info(f'scheduler, {job["model_name"]}, model save time: {time.time()-time_save_begin}') 

                if job['job_idx'] in scheduler._trainers:
                    scheduler._trainers.pop(job['job_idx'])

                scheduler._logger.info(f'placements len, {len(job["placements"])}') 
                jobinfo = JOBS.to_jobinfo(job)
                scheduler._controller.kill(jobinfo)

                assert 'placements' in job
                del job['placements'][:]

                job['status'] = 'PENDING'
                job['preempt'] = int(job['preempt'] + 1)
                scheduler._logger.info(f'scheduler, {job["job_idx"]}, preempt')
            

            # 要运行的job
            for job in run_jobs:
                job['job_counter'] += 1
                if sys.maxsize == job['start_time']:
                    job['start_time'] = tmp_time

                jobinfo = JOBS.to_jobinfo(job)
                scheduler._controller.execute(jobinfo)
                
                job['status'] = 'RUNNING'
                job['resume'] = int(job['resume'] + 1)     
                scheduler._logger.info(f'scheduler, {job["job_idx"]}, run')
    
        time.sleep(10)  # 等待10s，让任务启动
        LOG.checkpoint(tmp_time, scheduler, done_flag or new_flag)    # ljx 暂时注释
        # LOG.checkpoint(tmp_time, scheduler)
        time01 = time.time()
        print(f'at end: {scheduler.get_time()}\n\n', 'checkpoint and save model time', time01-time00, "\n\n")
        count = 0
        interval = 120 if FLAGS.scheme == 'mps3' else 60
        while count < interval and scheduler._controller.done_queue.empty():
            time.sleep(1)
            count += 1
            # print("sleep", count)
        scheduler._logger.info(f'at end: {scheduler.get_time()}\n\n')  
        # time.sleep(FLAGS.schedule_interval-(time01-time00))       # 取消特定周期调度