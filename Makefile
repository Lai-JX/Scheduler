rpc:
	python3 -m grpc_tools.protoc -Iruntime/proto --python_out=runtime/rpc_stubs --grpc_python_out=runtime/rpc_stubs runtime/proto/worker_to_master.proto
	python3 -m grpc_tools.protoc -Iruntime/proto --python_out=runtime/rpc_stubs --grpc_python_out=runtime/rpc_stubs runtime/proto/master_to_worker.proto
	python3 -m grpc_tools.protoc -Iruntime/proto --python_out=runtime/rpc_stubs --grpc_python_out=runtime/rpc_stubs runtime/proto/trainer_to_scheduler.proto
	python3 -m grpc_tools.protoc -Iruntime/proto --python_out=runtime/rpc_stubs --grpc_python_out=runtime/rpc_stubs runtime/proto/scheduler_to_trainer.proto
	python3 -m grpc_tools.protoc -Iruntime/proto --python_out=runtime/rpc_stubs --grpc_python_out=runtime/rpc_stubs runtime/proto/operator.proto
clean:
	rm -rf runtime/rpc_stubs/*_pb2.py runtime/rpc_stubs/*_pb2_grpc.py

push:
	git add .; git commit -m update; git push origin dev;
pull:
	git pull origin dev;

reset:
	git fetch; git reset --hard HEAD; git merge origin/dev

help:
	python run.py --help
	
rm:
	./script/clean.sh;
kill:
	./script/kill.sh;

######################### run scheduler #########################
IP=192.158.0.12
WORKER_PORT=9001
TRAINER_PORT=9013
# fifo,sjf,Tiresias,sjf-ffs,sjf-bsbf,sjf-ffs-no-preempt,sjf-bsbf-no-preempt,sjf-ffss,sjf-bsbfs,sjf-ffss-no-preempt,sjf-bsbfs-no-preempt,sjf-ffss-no-preempt-m,sjf-bsbfs-no-preempt-m,
SCHEDULERS=sjf-ffs-no-preempt,sjf-bsbf-no-preempt

fifo:
	./script/clean.sh; ./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 0 fifo
fifo1:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 1 fifo

sjf:
	./script/clean.sh; ./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 0 sjf
sjf1:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 1 sjf

Tiresias:
	./script/clean.sh; ./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 0 Tiresias
Tiresias1:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 1 Tiresias

# merge 2
sjf-ffs:
	./script/clean.sh; ./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 0 sjf-ffs
sjf-ffs1:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 1 sjf-ffs

sjf-ffss:
	./script/clean.sh; ./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 0 sjf-ffss
sjf-ffss1:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 1 sjf-ffss

sjf-ffss-no-preempt:
	./script/clean.sh; ./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 0 sjf-ffss-no-preempt
sjf-ffss-no-preempt1:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 1 sjf-ffss-no-preempt

sjf-ffs-m:
	./script/clean.sh; ./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 0 sjf-ffs-m

sjf-bsbf:
	./script/clean.sh; ./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 0 sjf-bsbf
sjf-bsbf1:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 1 sjf-bsbf

sjf-bsbfs:
	./script/clean.sh; ./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 0 sjf-bsbfs
sjf-bsbfs1:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 1 sjf-bsbfs

sjf-bsbf-m:
	./script/clean.sh; ./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 0 sjf-bsbf-m

sjf-bsbfs-no-preempt:
	./script/clean.sh; ./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 0 sjf-bsbfs-no-preempt
sjf-bsbfs-no-preempt1:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 1 sjf-bsbfs-no-preempt

sjf-bsbf-no-preempt:
	./script/clean.sh; ./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 0 sjf-bsbf-no-preempt
sjf-bsbf-no-preempt-m:
	./script/clean.sh; ./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 0 sjf-bsbf-no-preempt-m

sjf-test:
	./script/clean.sh; ./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 0 sjf-test
sjf-test1:
	./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 1 sjf-test

ffs-test:
	./script/clean.sh; ./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 0 sjf-ffs-no-preempt
ffs-test1:
	./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 1 sjf-ffs-no-preempt

bsbf-test:
	./script/clean.sh; ./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 0 sjf-bsbf-no-preempt
bsbf-test1:
	./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 1 sjf-bsbf-no-preempt



run:
	./script/clean.sh; ./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 0 sjf-ffs-no-preempt,sjf-bsbf-no-preempt
run1:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 1 sjf-ffs-no-preempt,sjf-bsbf-no-preempt
run2:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 2 sjf-ffs-no-preempt,sjf-bsbf-no-preempt
run3:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./script/kill.sh; ./run.sh ${IP} ${WORKER_PORT} ${TRAINER_PORT} 3 sjf-ffs-no-preempt,sjf-bsbf-no-preempt
