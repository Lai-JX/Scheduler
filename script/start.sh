#! /bin/bash

# cd /workspace/graduation_project/Scheduler
# SCHEDULER_IP=$1
# shift
# WORKER_NUM=$1
# shift
# schedules_all=$@

# for s in ${schedules_all[@]};do
#     cur_worker=0
#     while :; do
#         echo "./clean.sh; ./kill.sh; ./run.sh $SCHEDULER_IP 9001 9013 $cur_worker $s"

#         ((cur_worker++))
#         if [ $cur_worker -ge $WORKER_NUM ]; then
#             break
#         fi
#     done
# done

select_placement()
{
    schedule=$1
    if [ $schedule == 'sjf'  ] || [ $schedule == 'fifo' ] || [ $schedule == 'sjf-test' ] || [ $schedule == 'Tiresias' ]; then
        placement=('yarn')
    elif [ $schedule == 'sjf-ffs'  ] || [ $schedule == 'sjf-ffs-m' ]; then
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
s='sjf-ffs'
p='yarn'
select_placement $s
echo $p