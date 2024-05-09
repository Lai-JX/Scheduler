

import argparse
import time

from run import parse_job_file
#prepare JOBS list
import jobs
JOBS = jobs.JOBS
from runtime.rpc import operator_client

class Operator(object):
    def __init__(self, schedule_ip, schedule_port) -> None:
        super().__init__()

        self._schedule_ip = schedule_ip
        self._schedule_port = schedule_port

        self._client_for_operator = operator_client.OperatorClient(self._schedule_ip, self._schedule_port)

        
    

    def add_jobs(self,file_path):
        while True:
            success = self._client_for_operator.add_jobs(file_path)

            if success == True:
                break
            
            time.sleep(5)
    def delete_jobs(self,job_id_list):
        while True:
            success = self._client_for_operator.delete_jobs(job_id_list)

            if success == True:
                break
            
            time.sleep(5)
    

    

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--schedule_ip', type=str, default='127.0.0.1')
    parser.add_argument('--schedule_port', type=str, default='9008')
    parser.add_argument('--command', type=str, required=True)
    parser.add_argument('--trace_file', type=str, default='./trace-data/add_job.csv')
    parser.add_argument('--job_id', type=str, default=None)

    args = parser.parse_args()

    operator = Operator(args.schedule_ip, args.schedule_port)

    cmd = args.command
    if cmd == 'add_jobs':
        file_path = args.trace_file
        operator.add_jobs(file_path)
    elif cmd == 'delete_jobs':
        if args.job_id is None:
            print("Please enter job id in the form of job_id1,job_id2...")
        else:
            job_id_list = args.job_id.split(",")
            job_id_list = [int(i) for i in job_id_list]
            print(job_id_list)
            operator.delete_jobs(job_id_list)
    else:
        print("Error Command!")
