from __future__ import print_function
import sys
import time

import utils
import flags
import jobs
import cluster
import log
import grpc

# from run import try_get_job_res     # ljx:tmp


FLAGS = flags.FLAGS

#prepare JOBS list
JOBS = jobs.JOBS

#get host info
CLUSTER = cluster.CLUSTER
CLUSTER_TMP = cluster.CLUSTER_TMP

#get LOG object
LOG = log.LOG


'''
Allocate job resource
'''
def try_get_job_res(job, not_place=False, alloc_flag=True):
    '''
    select placement scheme
    '''
    if 'antman' in FLAGS.schedule or FLAGS.scheme == 'merge':
        print("antman")
        ret = CLUSTER.antman_placement(job) if alloc_flag else CLUSTER_TMP.antman_placement(job)
    elif FLAGS.scheme == 'yarn':
        ret = CLUSTER.ms_yarn_placement(job, not_place) if alloc_flag else CLUSTER_TMP.ms_yarn_placement(job, not_place=True)
    elif FLAGS.scheme == 'balance':
        ret = lp.placement(job)
        # ret = lp.min_new_job(job)
    elif FLAGS.scheme == 'random':
        ret = CLUSTER.random_placement(job) if alloc_flag else CLUSTER_TMP.random_placement(job)
    elif FLAGS.scheme == 'crandom':
        ret = CLUSTER.consolidate_random_placement(job) if alloc_flag else CLUSTER_TMP.consolidate_random_placement(job)
    elif FLAGS.scheme == 'greedy':
        ret = CLUSTER.greedy_placement(job) if alloc_flag else CLUSTER_TMP.greedy_placement(job)
    elif FLAGS.scheme == 'gandiva':
        ret = CLUSTER.gandiva_placement(job) if alloc_flag else CLUSTER_TMP.gandiva_placement(job)
    elif FLAGS.scheme == 'count':
        ret = CLUSTER.none_placement(job) if alloc_flag else CLUSTER_TMP.none_placement(job)
    else:
        ret = CLUSTER.ms_yarn_placement(job) if alloc_flag else CLUSTER_TMP.ms_yarn_placement(job)
    if ret == True:
        # job['status'] = 'RUNNING'
        pass
    return ret

def sjf_sim_jobs(scheduler=None, gputime=False, place=False):

    # end_events = list()
    scheduler._controller.set_start_time()
    last_check_time = 0
    finished_job_cnt = 0
    if FLAGS.fast_forwarding == 0:
        while (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
            done_flag = False
            new_flag = False
            scheduler._logger.info(f'before: {scheduler.get_time()}')
            # scheduler._logger.info(f'{JOBS.job_events}\n{JOBS.runnable_jobs}')
            while not scheduler._controller.done_queue.empty():
                finished_time, job_id, worker_id, gpus, returncode = scheduler._controller.done_queue.get()
                e_job = JOBS.find_runnable_job(job_id)
                if returncode==0:
                    tmp = finished_time-e_job['last_check_time']
                    e_job['total_executed_time'] += tmp
                    e_job['last_check_time'] = finished_time
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
                        # pass    # 如果是抢占式，这里需要查询并更新job迭代次数
                        tmp = tmp_time - rjob['last_check_time']
                        rjob['total_executed_time'] = rjob['total_executed_time'] + tmp
                        rjob['last_check_time'] = tmp_time
                        finished_iter, iteration_time = scheduler.query_stats([rjob['job_idx']])    # 同时更新迭代时间
                        rjob['remaining_iterations'] -= finished_iter
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
                    if 'RUNNING' == rjob['status']:                                     # ① 之前运行，现在仍然运行
                        scheduler._logger.info(f'scheduler, {rjob["job_idx"]}, still running. remaining_iterations:{rjob["remaining_iterations"]} ')
                        continue
                    ret = try_get_job_res(rjob)                                         # ② 新运行
                    if not ret:                                                         # 资源不够
                        for rev_idx in range(1, len(JOBS.runnable_jobs) - idx):         # ③ 寻找要抢占的job   
                            potential_job_to_preempt = JOBS.runnable_jobs[-rev_idx]
                            if potential_job_to_preempt['status'] == 'RUNNING':
                                CLUSTER.release_job_res(potential_job_to_preempt)   # 释放资源
                                preempt_jobs.append(potential_job_to_preempt)       # 抢占，将被抢占的job置为pending

                                ret = try_get_job_res(rjob)                         # 重新判断资源是否满足
                                if ret:
                                    break
                    if ret:
                        run_jobs.append(rjob)
                    else:
                        break

                # 要抢占的job
                for job in preempt_jobs:
                    time_save_begin = time.time()   
                    scheduler.save_model([job['job_idx']])      
                    scheduler._logger.info(f'scheduler, {job["model_name"]}, model save time: {time.time()-time_save_begin}') 

                    if job['job_idx'] in scheduler._trainers:
                        scheduler._trainers.pop(job['job_idx'])

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

                last_check_time = tmp_time
        
            time.sleep(10)  # 等待10s，让任务启动
            LOG.checkpoint(tmp_time, scheduler, done_flag or new_flag)    # ljx 暂时注释
            # LOG.checkpoint(tmp_time, scheduler)
            time01 = time.time()
            print('checkpoint and save model time', time01-time00)
            count = 0
            while count < 60 and scheduler._controller.done_queue.empty():
                time.sleep(1)
                count += 1
                # print("sleep", count)
            scheduler._logger.info(f'at end: {scheduler.get_time()}\n\n')  
            # time.sleep(FLAGS.schedule_interval-(time01-time00))       # 取消特定周期调度




def mps_sim_jobs(scheduler=None, gputime=False, place=False):

    # end_events = list()
    scheduler._controller.set_start_time()
    last_check_time = 0
    finished_job_cnt = 0
    if FLAGS.fast_forwarding == 0:
        while (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
            done_flag = False
            new_flag = False
            scheduler._logger.info(f'before: {scheduler.get_time()}')
            # scheduler._logger.info(f'{JOBS.job_events}\n{JOBS.runnable_jobs}')
            while not scheduler._controller.done_queue.empty():
                finished_time, job_id, worker_id, gpus, returncode = scheduler._controller.done_queue.get()
                e_job = JOBS.find_runnable_job(job_id)
                if returncode==0:
                    tmp = finished_time-e_job['last_check_time']
                    e_job['total_executed_time'] += tmp
                    e_job['last_check_time'] = finished_time
                    CLUSTER.release_job_res(e_job)
                    scheduler._trainers.pop(e_job['job_idx'])
                    LOG.job_complete(e_job, finished_time)
                    finished_job_cnt += 1
                    scheduler._logger.info(f'**** job[{e_job["job_idx"]}] completed')
                    scheduler._logger.info(f'scheduler finishes {finished_job_cnt} jobs in all!')
                    JOBS.runnable_jobs.remove(e_job)
                else:
                    scheduler._trainers.pop(e_job['job_idx'])
                    CLUSTER.release_job_res(e_job)      # ljx 出错的job也应释放资源
                    e_job['status'] = 'PENDING'
                    del e_job['placements'][:]
                done_flag = True
                print(scheduler.get_time(), 'check: done ', e_job['job_idx'], finished_time)
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
                        # pass    # 如果是抢占式，这里需要查询并更新job迭代次数
                        tmp = tmp_time - rjob['last_check_time']
                        rjob['total_executed_time'] = rjob['total_executed_time'] + tmp
                        rjob['last_check_time'] = tmp_time
                        finished_iter, iteration_time = scheduler.query_stats([rjob['job_idx']])    # 同时更新迭代时间
                        rjob['remaining_iterations'] -= finished_iter
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
                
                # 基于空集群和当前job的顺序，判断哪些工作需要运行
                CLUSTER_TMP.empty_infra()
                # runnable_jobs_tmp = copy.deepcopy(JOBS.runnable_jobs)
                for rjob in JOBS.runnable_jobs:
                    ret = try_get_job_res(rjob, alloc_flag=False) 
                    scheduler._logger.info(f'{rjob["job_idx"]} need {rjob["num_gpu"]} gpu, ret:{ret}')
                    if ret:
                        run_jobs.append(rjob)
                        
                # 获取需要抢占的job
                for rjob in JOBS.runnable_jobs:
                    if 'RUNNING' == rjob['status']:
                        if rjob not in run_jobs:                                                    # 之前运行，现在不运行    
                            # 更新迭代次数等
                            # tmp = tmp_time - rjob['last_check_time']
                            # rjob['total_executed_time'] = rjob['total_executed_time'] + tmp
                            # rjob['last_check_time'] = tmp_time
                            # finished_iter = scheduler.query_stats([rjob['job_idx']])
                            # rjob['remaining_iterations'] -= finished_iter
                            # # print(rjob['job_idx'], rjob['remaining_iterations'], finished_iter)
                            # rjob['remaining_time'] = rjob['iteration_time'] * rjob['remaining_iterations']
                            # if gputime:
                            #     rjob['remaining_gputime'] = rjob['remaining_time'] * rjob['num_gpu']
                            # scheduler._logger.info(f'{tmp_time} check: running  {rjob["job_idx"]} {rjob["remaining_iterations"]} {rjob["total_executed_time"]}')  
                            # kill        
                            time_save_begin = time.time()   
                            scheduler.save_model([rjob['job_idx']])      
                            scheduler._logger.info(f'scheduler, {rjob["model_name"]}, model save time: {time.time()-time_save_begin}')  
                            # time.sleep(rjob['iteration_time']*1.2)               
                            if rjob['job_idx'] in scheduler._trainers:
                                scheduler._trainers.pop(rjob['job_idx'])
                            jobinfo = JOBS.to_jobinfo(rjob)
                            scheduler._controller.kill(jobinfo)
                            CLUSTER.release_job_res(rjob)   # 释放资源
                            assert 'placements' in rjob
                            del rjob['placements'][:]
                            preempt_jobs.append(rjob)       # 抢占
                # 执行新job
                for rjob in run_jobs:
                    if 'RUNNING' == rjob['status'] and rjob not in preempt_jobs:                     # 之前运行，现在仍然运行
                        scheduler._logger.info(f'scheduler, {rjob["job_idx"]}, still running. remaining_iterations:{rjob["remaining_iterations"]} ')
                    if 'PENDING' == rjob['status']:                     # 这次被调度执行的job
                        ret = try_get_job_res(rjob) 
                        scheduler._logger.info(f'scheduler, {rjob["job_idx"]}, remaining_iterations:{rjob["remaining_iterations"]}, real ret:{ret}')
                        assert ret == True
                        # test
                        # rjob['resume'] = 1
                        rjob['job_counter'] += 1
                        if sys.maxsize == rjob['start_time']:
                            rjob['start_time'] = tmp_time
                        jobinfo = JOBS.to_jobinfo(rjob)
                        scheduler._controller.execute(jobinfo) 


                for job in preempt_jobs:
                    job['status'] = 'PENDING'
                    job['preempt'] = int(job['preempt'] + 1)
                    scheduler._logger.info(f'scheduler, {job["job_idx"]}, preempt')
                for job in run_jobs:
                    job['status'] = 'RUNNING'
                    job['resume'] = int(job['resume'] + 1)      # 这个应该是不准的，因为running的job可能是一直在运行的
                    scheduler._logger.info(f'scheduler, {job["job_idx"]}, run')

                last_check_time = tmp_time
        
            time.sleep(10)  # 等待10s，让任务启动
            LOG.checkpoint(tmp_time, scheduler, done_flag or new_flag)    # ljx 暂时注释
            # LOG.checkpoint(tmp_time, scheduler)
            time01 = time.time()
            print('checkpoint and save model time', time01-time00)
            count = 0
            while count < 15 and scheduler._controller.done_queue.empty():
                time.sleep(1)
                count += 1
                # print("sleep", count)
            scheduler._logger.info(f'at end: {scheduler.get_time()}\n\n')  
            # time.sleep(FLAGS.schedule_interval-(time01-time00))       # 取消特定周期调度


def sjf_ffs_jobs(scheduler=None, gputime=False, place=False):

    # end_events = list()
    scheduler._controller.set_start_time()
    last_check_time = 0
    finished_job_cnt = 0
    if FLAGS.fast_forwarding == 0:
        while (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
            done_flag = False
            new_flag = False
            scheduler._logger.info(f'before: {scheduler.get_time()}')
            # scheduler._logger.info(f'{JOBS.job_events}\n{JOBS.runnable_jobs}')
            while not scheduler._controller.done_queue.empty():
                finished_time, job_id, worker_id, gpus, returncode = scheduler._controller.done_queue.get()
                e_job = JOBS.find_runnable_job(job_id)
                if returncode==0:
                    tmp = finished_time-e_job['last_check_time']
                    e_job['total_executed_time'] += tmp
                    e_job['last_check_time'] = finished_time
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
                        # pass    # 如果是抢占式，这里需要查询并更新job迭代次数
                        tmp = tmp_time - rjob['last_check_time']
                        rjob['total_executed_time'] = rjob['total_executed_time'] + tmp
                        rjob['last_check_time'] = tmp_time
                        try:                                                                                    # 用try，因为job可能在之前done检查之后完成了，此时连接断开，无法查询到job状态
                            finished_iter, iteration_time = scheduler.query_stats([rjob['job_idx']])            # 同时更新迭代时间
                            scheduler._logger.info(f'try, {finished_iter}, {iteration_time}')  
                        except:
                            finished_iter, iteration_time = rjob['remaining_iterations'], rjob['iteration_time']
                            scheduler._logger.info(f'except, {finished_iter}, {iteration_time}')  
                        rjob['remaining_iterations'] -= finished_iter
                        rjob['iteration_time'] = iteration_time
                        # print(rjob['job_idx'], rjob['remaining_iterations'], finished_iter)
                        rjob['remaining_time'] = rjob['iteration_time'] * rjob['remaining_iterations']
                        if gputime:
                            rjob['remaining_gputime'] = rjob['remaining_time'] * rjob['num_gpu']
                        scheduler._logger.info(f'{tmp_time} check: running  {rjob["job_idx"]} {rjob["remaining_iterations"]} {rjob["total_executed_time"]}')  

                        # test
                        # time_save_begin = time.time()   
                        # assert scheduler.save_model([rjob['job_idx']])                   # 如果模型保存不成功会影响后续job重新运行
                        # scheduler._logger.info(f'scheduler, {rjob["model_name"]}, model save time: {time.time()-time_save_begin}') 




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
                    if 'RUNNING' == rjob['status']:                                     # ① 之前运行，现在仍然运行
                        scheduler._logger.info(f'scheduler, {rjob["job_idx"]}, still running. remaining_iterations:{rjob["remaining_iterations"]} ')
                        continue
                    rjob['priority'] = 1        # ljx 多个job在一起
                    # if FLAGS.scheme == 'mps3': 
                    #     rjob['gpu_util'] = 0.4      # ljx 
                    # else:
                    #     rjob['gpu_util'] = 0.5
                    rjob['gpu_util'] = 0.5
                    scheduler._logger.info(f'try_get_job_res, {rjob["job_idx"]}')
                    ret = try_get_job_res(rjob)                                         # ② 新运行
                    if not ret:                                                         # 资源不够
                        for rev_idx in range(1, len(JOBS.runnable_jobs) - idx):         # ③ 寻找要抢占的job   
                            potential_job_to_preempt = JOBS.runnable_jobs[-rev_idx]
                            if potential_job_to_preempt['status'] == 'RUNNING':
                                scheduler._logger.info(f'release_job_res, {potential_job_to_preempt["job_idx"]}')
                                CLUSTER.release_job_res(potential_job_to_preempt)   # 释放资源
                                preempt_jobs.append(potential_job_to_preempt)       # 抢占，将被抢占的job置为pending

                                ret = try_get_job_res(rjob)                         # 重新判断资源是否满足
                                if ret:
                                    break
                    if ret:
                        run_jobs.append(rjob)
                    else:
                        break

                # 要抢占的job
                for job in preempt_jobs:
                    time_save_begin = time.time()   
                    assert scheduler.save_model([job['job_idx']])                   # 如果模型保存不成功会影响后续job重新运行
                    scheduler._logger.info(f'scheduler, {job["model_name"]}, model save time: {time.time()-time_save_begin}') 

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

                last_check_time = tmp_time
        
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

def fifo_sim_jobs(scheduler=None, gputime=False, place=False):

    # end_events = list()
    scheduler._controller.set_start_time()
    last_check_time = 0
    finished_job_cnt = 0

    while (len(JOBS.job_events) + len(JOBS.runnable_jobs))> 0:
        done_flag = False
        new_flag = False
        scheduler._logger.info(f'before: {scheduler.get_time()}')
        # scheduler._logger.info(f'{JOBS.job_events}\n{JOBS.runnable_jobs}')
        while not scheduler._controller.done_queue.empty():
            finished_time, job_id, worker_id, gpus, returncode = scheduler._controller.done_queue.get()
            e_job = JOBS.find_runnable_job(job_id)
            if returncode==0:
                tmp = finished_time-e_job['last_check_time']
                e_job['total_executed_time'] += tmp
                e_job['last_check_time'] = finished_time
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
            print('current time:',scheduler.get_time(), 'check: done ', 'job_id', e_job['job_idx'], 'finished_time:', finished_time)
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
                    # pass    # 如果是抢占式，这里需要查询并更新job迭代次数
                    tmp = tmp_time - rjob['last_check_time']
                    rjob['total_executed_time'] = rjob['total_executed_time'] + tmp
                    rjob['last_check_time'] = tmp_time
                    finished_iter, iteration_time = scheduler.query_stats([rjob['job_idx']])    # 同时更新迭代时间
                    rjob['remaining_iterations'] -= finished_iter
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
            # if gputime:
            #     JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('remaining_gputime'))
            # else:
            #     JOBS.runnable_jobs.sort(key = lambda e:e.__getitem__('remaining_time'))

            run_jobs = list()
            preempt_jobs = list()

            for idx, rjob in enumerate(JOBS.runnable_jobs):
                ret = False
                if 'RUNNING' == rjob['status']:                                     # ① 之前运行，现在仍然运行
                    scheduler._logger.info(f'scheduler, {rjob["job_idx"]}, still running. remaining_iterations:{rjob["remaining_iterations"]} ')
                    continue
                ret = try_get_job_res(rjob)                                         # ② 新运行
                if not ret:                                                         # 资源不够
                    for rev_idx in range(1, len(JOBS.runnable_jobs) - idx):         # ③ 寻找要抢占的job   
                        potential_job_to_preempt = JOBS.runnable_jobs[-rev_idx]
                        if potential_job_to_preempt['status'] == 'RUNNING':
                            CLUSTER.release_job_res(potential_job_to_preempt)   # 释放资源
                            preempt_jobs.append(potential_job_to_preempt)       # 抢占，将被抢占的job置为pending

                            ret = try_get_job_res(rjob)                         # 重新判断资源是否满足
                            if ret:
                                break
                if ret:
                    run_jobs.append(rjob)
                else:
                    break

            # 要抢占的job
            for job in preempt_jobs:
                time_save_begin = time.time()   
                scheduler.save_model([job['job_idx']])      
                scheduler._logger.info(f'scheduler, {job["model_name"]}, model save time: {time.time()-time_save_begin}') 

                if job['job_idx'] in scheduler._trainers:
                    scheduler._trainers.pop(job['job_idx'])

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

            last_check_time = tmp_time
    
        time.sleep(10)  # 等待10s，让任务启动
        LOG.checkpoint(tmp_time, scheduler, done_flag or new_flag)    # ljx 暂时注释
        # LOG.checkpoint(tmp_time, scheduler)
        time01 = time.time()
        # print('checkpoint and save model time', time01-time00)
        count = 0
        while count < 60 and scheduler._controller.done_queue.empty():
            time.sleep(1)
            count += 1
            # print("sleep", count)
        scheduler._logger.info(f'at end: {scheduler.get_time()}\n\n')  
        # time.sleep(FLAGS.schedule_interval-(time01-time00))       # 取消特定周期调度