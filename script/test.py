

import subprocess
import sys
import time
import utils

if __name__ == '__main__':
    SCHEDULER_IP = sys.argv[1]
    WORKER_NUM = int(sys.argv[2])
    SCHEDULER_ALL = sys.argv[3:]
    node_msg = utils.json_to_dict('./cluster/node_msg.json')

    for s in SCHEDULER_ALL:
        for cur_worker in range(WORKER_NUM):
            if cur_worker == 0:
                command = f"exec ./clean.sh; ./kill.sh; ./run.sh {SCHEDULER_IP} 9001 9013 {cur_worker} {s}"
                process = subprocess.Popen(command, shell=True)
            else:
                command = f"exec ssh root@{node_msg['hosts'][str(cur_worker)]} -p {node_msg['ssh_ports'][str(cur_worker)]} cd /workspace/graduation_project/Scheduler && ./kill.sh && ./run.sh {SCHEDULER_IP} 9001 9013 {cur_worker} {s}"
                print(command)
                process = subprocess.Popen(command, shell=True)
            time.sleep(2)
