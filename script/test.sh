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
# 输入字符串
get_gpu_list()
{
    input_str=$1

    # 使用 Bash 的内置字符串操作功能来找到 "g" 后面的数字
    num="${input_str#*g}"

    # 检查找到的数字是否为纯数字
    if ! [[ "$num" =~ ^[0-9]+$ ]]; then
        echo "Error: No数字 found after 'g' in the string."
        exit 1
    fi

    # 使用循环生成以逗号隔开的字符串
    gpu_list=""
    for (( i=0; i<num; i++ )); do
        if [ "$i" -ne "0" ]; then
            gpu_list+=","
        fi
        gpu_list+="$i"
    done

    # 输出结果
    echo "$gpu_list"
}
get_gpu_list "n1g4"

exit 

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