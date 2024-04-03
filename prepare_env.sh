#!/bin/bash
# Note: Due to the scripts are highly related to intracompany platform, 
#we only demonstrate the functionality and show the pseudocode of the 
#related scripts (e.g., run.sh, prepare_env.sh). Please adjust to your 
#platform if you would like to execute the testbed experiment.
echo -e "ljx:prepare_env.sh\n"

FA_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
THIS_DIR=$FA_DIR/workloads

# Prepare datasets and make sure that the used nodes can connect with
# each other.


# set worker ip and port
SCHEDULER_IP=$1
shift
WORKER_PORT=$1
shift
TRAINER_PORT=$1
shift
WORKER_ID=$1
shift

mkdir $THIS_DIR/hostfiles
hostfile=$THIS_DIR/hostfiles/hostfile-[0-0-0-0]-[0-0-0-0]
rm -f $hostfile
# echo "worker-${WORKER_ID}" >>${hostfile}    # 将"worker-${WORKER_ID}"写入文件hostfile   我们需要改为gpu1
echo "gpu${WORKER_ID}" >>${hostfile}    

# ljx
# CUDA_VISIBLE_DEVICES=0 bash $THIS_DIR/run_preenv.sh gpt2 4 0 2 -1 10 0 0 gpt2 4 0 2 -1 0 0 0 gpt2 4 0 2 -1 0 0 0 gpt2 4 0 2 -1 0 0 0 1 --scheduler-ip $SCHEDULER_IP --trainer-port $TRAINER_PORT
# CUDA_VISIBLE_DEVICES=0 bash $THIS_DIR/run_preenv.sh bert 4 0 2 -1 10 0 0 gpt2 4 0 2 -1 0 0 0 gpt2 4 0 2 -1 0 0 0 gpt2 4 0 2 -1 0 0 0 1 --scheduler-ip $SCHEDULER_IP --trainer-port $TRAINER_PORT

CUDA_VISIBLE_DEVICES=0 bash $THIS_DIR/run_preenv.sh dqn 4 0 2 -1 10 0 0 dqn 4 0 2 -1 0 0 0 dqn 4 0 2 -1 0 0 0 dqn 4 0 2 -1 0 0 0 1 --scheduler-ip $SCHEDULER_IP --trainer-port $TRAINER_PORT
# CUDA_VISIBLE_DEVICES=0 bash $THIS_DIR/run_preenv.sh a2c 4 0 2 -1 10 0 0 a2c 4 0 2 -1 0 0 0 a2c 4 0 2 -1 0 0 0 a2c 4 0 2 -1 0 0 0 1 --scheduler-ip $SCHEDULER_IP --trainer-port $TRAINER_PORT

