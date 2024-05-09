rpc:
	python3 -m grpc_tools.protoc -Iruntime/proto --python_out=runtime/rpc_stubs --grpc_python_out=runtime/rpc_stubs runtime/proto/worker_to_master.proto
	python3 -m grpc_tools.protoc -Iruntime/proto --python_out=runtime/rpc_stubs --grpc_python_out=runtime/rpc_stubs runtime/proto/master_to_worker.proto
	python3 -m grpc_tools.protoc -Iruntime/proto --python_out=runtime/rpc_stubs --grpc_python_out=runtime/rpc_stubs runtime/proto/trainer_to_scheduler.proto
	python3 -m grpc_tools.protoc -Iruntime/proto --python_out=runtime/rpc_stubs --grpc_python_out=runtime/rpc_stubs runtime/proto/scheduler_to_trainer.proto
	python3 -m grpc_tools.protoc -Iruntime/proto --python_out=runtime/rpc_stubs --grpc_python_out=runtime/rpc_stubs runtime/proto/operator.proto
clean:
	rm -rf runtime/rpc_stubs/*_pb2.py runtime/rpc_stubs/*_pb2_grpc.py

push:
	git add .; git commit -m update; git push -u origin main;
pull:
	git pull origin main;

reset:
	git fetch; git reset --hard HEAD; git merge origin/master

rm:
	./script/clean.sh;
kill:
	./script/kill.sh;

######################### run scheduler #########################

fifo:
	./script/clean.sh; ./script/kill.sh; ./run.sh 192.158.0.11 9001 9013 0 fifo
fifo1:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./script/kill.sh; ./run.sh 192.158.0.11 9001 9013 1 fifo

sjf:
	./script/clean.sh; ./script/kill.sh; ./run.sh 192.158.0.11 9001 9013 0 sjf
sjf1:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./script/kill.sh; ./run.sh 192.158.0.11 9001 9013 1 sjf

Tiresias:
	./script/clean.sh; ./script/kill.sh; ./run.sh 192.158.0.11 9001 9013 0 Tiresias
Tiresias1:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./script/kill.sh; ./run.sh 192.158.0.11 9001 9013 1 Tiresias

# merge 2
sjf-ffs:
	./script/clean.sh; ./script/kill.sh; ./run.sh 192.158.0.11 9001 9013 0 sjf-ffs
sjf-ffs1:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./script/kill.sh; ./run.sh 192.158.0.11 9001 9013 1 sjf-ffs

sjf-ffss:
	./script/clean.sh; ./script/kill.sh; ./run.sh 192.158.0.11 9001 9013 0 sjf-ffss
sjf-ffss1:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./script/kill.sh; ./run.sh 192.158.0.11 9001 9013 1 sjf-ffss

sjf-ffss-no-preempt:
	./script/clean.sh; ./script/kill.sh; ./run.sh 192.158.0.11 9001 9013 0 sjf-ffss-no-preempt
sjf-ffss-no-preempt1:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./script/kill.sh; ./run.sh 192.158.0.11 9001 9013 1 sjf-ffss-no-preempt

sjf-ffs-m:
	./script/clean.sh; ./script/kill.sh; ./run.sh 192.158.0.11 9002 9013 0 sjf-ffs-m

sjf-bsbf:
	./script/clean.sh; ./script/kill.sh; ./run.sh 192.158.0.11 9002 9013 0 sjf-bsbf
sjf-bsbf1:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./script/kill.sh; ./run.sh 192.158.0.11 9002 9013 1 sjf-bsbf

sjf-bsbfs:
	./script/clean.sh; ./script/kill.sh; ./run.sh 192.158.0.11 9001 9013 0 sjf-bsbfs
sjf-bsbfs1:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./script/kill.sh; ./run.sh 192.158.0.11 9001 9013 1 sjf-bsbfs

sjf-bsbf-m:
	./script/clean.sh; ./script/kill.sh; ./run.sh 127.0.0.1 9002 9013 0 sjf-bsbf-m

sjf-bsbfs-no-preempt:
	./script/clean.sh; ./script/kill.sh; ./run.sh 192.158.0.11 9001 9013 0 sjf-bsbfs-no-preempt
sjf-bsbfs-no-preempt1:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./script/kill.sh; ./run.sh 192.158.0.11 9001 9013 1 sjf-bsbfs-no-preempt

sjf-bsbf-no-preempt:
	./script/clean.sh; ./script/kill.sh; ./run.sh 127.0.0.1 9002 9013 0 sjf-bsbf-no-preempt
sjf-bsbf-no-preempt-m:
	./script/clean.sh; ./script/kill.sh; ./run.sh 127.0.0.1 9002 9013 0 sjf-bsbf-no-preempt-m

sjf-test:
	./script/clean.sh; ./script/kill.sh; ./run.sh 127.0.0.1 9002 9013 0 sjf-test
sjf-test1:
	./script/kill.sh; ./run.sh 10.244.17.22 9002 9013 1 sjf-test


# fifo,sjf,Tiresias,sjf-ffs,sjf-bsbf,sjf-ffs-no-preempt,sjf-bsbf-no-preempt,sjf-ffss,sjf-bsbfs,sjf-ffss-no-preempt,sjf-bsbfs-no-preempt,sjf-ffss-no-preempt-m,sjf-bsbfs-no-preempt-m,
all::
	./script/clean.sh; ./script/kill.sh; ./run.sh 192.158.0.12 9001 9013 0 sjf-ffs-no-preempt,sjf-bsbf-no-preempt
all1:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./script/kill.sh; ./run.sh 192.158.0.12 9001 9013 1 sjf-ffs-no-preempt,sjf-bsbf-no-preempt
all2:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./script/kill.sh; ./run.sh 192.158.0.12 9001 9013 2 sjf-ffs-no-preempt,sjf-bsbf-no-preempt
all3:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./script/kill.sh; ./run.sh 192.158.0.12 9001 9013 3 sjf-ffs-no-preempt,sjf-bsbf-no-preempt
