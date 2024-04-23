#!/bin/bash
# Note: Due to the scripts are highly related to intracompany platform, 
#we only demonstrate the functionality and show the pseudocode of the 
#related scripts (e.g., run.sh, prepare_env.sh). Please adjust to your 
#platform if you would like to execute the testbed experiment.
echo -e "ljx:run.sh\n"

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"    # 当前目录的绝对路径

# set the worker ip and port
SCHEDULER_IP=$1
shift
WORKER_PORT=$1
shift
TRAINER_PORT=$1
shift
WORKER_ID=$1
shift

# prepare environment before start, takes several minutes
cd $THIS_DIR
# bash $THIS_DIR/prepare_env.sh $SCHEDULER_IP $WORKER_PORT $TRAINER_PORT $WORKER_ID  # 先 deprecated

service ssh restart     # 使ssh配置生效，方便使用mpirun


export schedules_all=$@         # ljx
shift
jobs=('trace_data_new_32')
# jobs=('trace_data_new1')
# jobs=('cluster_trace_12')
setups=("n4g8")
schedule_intervals=("150")          # 调度间隔

IFS=','
read -ra schedulers <<< "$schedules_all"

# set the scheduling policy and related parameters
select_placement()
{
    schedule=$1
    if [ $schedule == 'sjf'  ] || [ $schedule == 'fifo' ] || [ $schedule == 'sjf-test' ] || [ $schedule == 'Tiresias' ]; then
        placement=('yarn')
    elif [ $schedule == 'sjf-ffs'  ] || [ $schedule == 'sjf-ffs-m' ] || [ $schedule == 'sjf-ffs-no-preempt' ] || [ $schedule == 'sjf-ffs-no-preempt-m' ]; then
        placement=('merge')
    elif [ $schedule == 'sjf-ffss'  ] || [ $schedule == 'sjf-ffss-m' ] || [ $schedule == 'sjf-ffss-no-preempt' ] || [ $schedule == 'sjf-ffss-no-preempt-m' ]; then
        placement=('merge-s')
    elif [ $schedule == 'sjf-bsbf'  ] || [ $schedule == 'sjf-bsbf-m'  ] || [ $schedule == 'sjf-bsbf-no-preempt'  ] || [ $schedule == 'sjf-bsbf-no-preempt-m'  ]; then
        placement=('bsbf')   
    elif [ $schedule == 'sjf-bsbfs'  ] || [ $schedule == 'sjf-bsbfs-m'  ] || [ $schedule == 'sjf-bsbfs-no-preempt'  ] || [ $schedule == 'sjf-bsbfs-no-preempt-m'  ]; then
        placement=('bsbfs')    
    fi
    p=$placement
}
# placement=('mps' 'mps3')
echo 'schedulers:'${schedulers[@]}
# echo 'placement:'${placement[@]}

mkdir $THIS_DIR/results
for setup in ${setups[@]};do                                                                            # 集群配置
    cluster_spec="cluster_specs/${setup}.csv"
    for job in ${jobs[@]};do                                                                            # job集
        job_file="trace-data/${job}.csv"
        for schedule_interval in ${schedule_intervals[@]};do                                        # 调度的时间间隔
            trace_name="${setup}j${job}si${schedule_interval}"
            log_folder="results/${trace_name}"                                                      # results/n4g4jcluster_tracep4si360ff60
            mkdir $THIS_DIR/${log_folder}
            for s in ${schedulers[@]};do
                echo 'scheduler:'${s}
                p='yarn'
                select_placement ${s}
                echo 'placement:'${p}
                log_name="${log_folder}/${s}-${p}"                               # run.py的log path: results/n4g4jcluster_tracep4si360ff60/dlas-gpu-yarn-4
                mkdir $THIS_DIR/$log_name
                # rm $THIS_DIR/$log_name/*.log
                job_log="$THIS_DIR/job_logs/${trace_name}/${s}-${p}"             # worker.py的log path: job_logs/n4g4jcluster_tracep4si360ff60/dlas-gpu-yarn-4
                model_path="$THIS_DIR/model/${trace_name}/${s}-${p}"
                rm -rf $job_log
                rm -rf $model_path
                # rm $THIS_DIR/${log_name}/my_log.log
                mkdir -p $job_log      
                mkdir -p $model_path
                echo "running..." $setup $job $s 'worker-id:'$WORKER_ID $SCHEDULER_IP
                if [ $WORKER_ID -eq 0 ]; then
                    # start scheduler for the main node
                    echo -e '\nstart scheduler for the main node\n'
                    rm $THIS_DIR/$log_name/*.log
                    python -u $THIS_DIR/run.py --cluster_spec=$THIS_DIR/${cluster_spec} --print --scheme=${p} --trace_file=$THIS_DIR/${job_file} --schedule=${s} --log_path=$THIS_DIR/${log_name} --schedule_interval ${schedule_interval} >$THIS_DIR/${log_name}/scheduler.out &   # ljx >$THIS_DIR/${log_name}/scheduler.out
                    sleep 10s
                else
                    sleep 2m    # ljx 
                    # sleep 10s
                fi
                echo -e '\nstart worker\n'
                echo "python $THIS_DIR/worker.py --master_ip $SCHEDULER_IP --worker_port $WORKER_PORT --trace_name ${job_log} --this-dir ${THIS_DIR} $arg &"
                python -u $THIS_DIR/worker.py --master_ip $SCHEDULER_IP --worker_port $WORKER_PORT --trace_name ${job_log} --this-dir ${THIS_DIR} $arg --log_path=$THIS_DIR/${log_name}/worker${WORKER_ID}.log --gpus='0,1,2,3,4,5,6,7' >$THIS_DIR/${log_name}/worker${WORKER_ID}.out &     # ljx 由于家目录共享，所以加个WORKER_ID区分一下不同worker.out 添加--log_path=$THIS_DIR/${log_name} --gpus='0,1'
                # python /home/jxlai/project/Muri_exp/worker.py --master_ip 10.0.0.11 --worker_port 9001 --trace_name /home/jxlai/project/Muri_exp/job_logs/n4g4jcluster_tracep4si60ff60/dlas-gpu-yarn-4 --this-dir /home/jxlai/project/Muri_exp
                wait

                # get the results after execution
                if [ $WORKER_ID -eq 0 ]; then
                    echo "calcing..." $setup $job $s
                    python $THIS_DIR/calc.py $THIS_DIR/${log_name} >$THIS_DIR/${log_name}/result.out
                else
                    sleep 2m
                fi
            done
        done
    done
done

                