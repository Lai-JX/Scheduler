rpc:
	python3 -m grpc_tools.protoc -Iruntime/proto --python_out=runtime/rpc_stubs --grpc_python_out=runtime/rpc_stubs runtime/proto/worker_to_master.proto
	python3 -m grpc_tools.protoc -Iruntime/proto --python_out=runtime/rpc_stubs --grpc_python_out=runtime/rpc_stubs runtime/proto/master_to_worker.proto
	python3 -m grpc_tools.protoc -Iruntime/proto --python_out=runtime/rpc_stubs --grpc_python_out=runtime/rpc_stubs runtime/proto/trainer_to_scheduler.proto
	python3 -m grpc_tools.protoc -Iruntime/proto --python_out=runtime/rpc_stubs --grpc_python_out=runtime/rpc_stubs runtime/proto/scheduler_to_trainer.proto

clean:
	rm -rf runtime/rpc_stubs/*_pb2.py runtime/rpc_stubs/*_pb2_grpc.py

push:
	git add .; git commit -m update; git push -u origin main;
pull:
	git pull origin main;

run:
	rm ./workloads/*.xml; ./kill.sh; ./run.sh 10.249.41.142 9001 9013 1 dlas-gpu
run1:
	./kill.sh; ./run.sh 10.249.41.142 9001 9013 2 dlas-gpu
reset:
	git fetch; git reset --hard HEAD; git merge origin/master

rm:
	rm ./workloads/*.txt; rm ./workloads/*.out; rm ./workloads/*.xml; rm ./workloads/hostfiles/hostfile-*; rm ./tmp/*.out; rm ./tmp/*.xml; rm ./*.out; rm ./*.xml

# rm ./workloads/*.txt; rm ./workloads/*.out; rm ./workloads/*.xml; rm ./workloads/hostfiles/hostfile-*; 

# srtf
srtf:
	rm ./workloads/*.txt; rm ./workloads/*.out; rm ./workloads/*.xml; rm ./workloads/hostfiles/hostfile-*; ./kill.sh; ./run.sh 10.249.41.142 9001 9013 1 shortest
srtf1:
	./kill.sh; ./run.sh 10.249.41.142 9001 9013 2 shortest

# srsf
srsf:
	rm ./workloads/*.txt; rm ./workloads/*.out; rm ./workloads/*.xml; rm ./workloads/hostfiles/hostfile-*; ./kill.sh; ./run.sh 10.249.41.142 9001 9013 1 shortest-gpu
srsf1:
	./kill.sh; ./run.sh 10.249.41.142 9001 9013 2 shortest-gpu

# MuriS
MuriS:
	./kill.sh; ./run.sh 10.249.41.142 9001 9013 1 multi-resource-blossom-same-gpu
MuriS1:
	./kill.sh; ./run.sh 10.249.41.142 9001 9013 2 multi-resource-blossom-same-gpu

# MuriL
MuriL:
	./kill.sh; ./run.sh 10.249.41.142 9001 9013 1 multi-resource-blossom-same-gpu-unaware
MuriL1:
	./kill.sh; ./run.sh 10.249.41.142 9001 9013 2 multi-resource-blossom-same-gpu-unaware
	
# Themis
themis:
	./kill.sh; ./run.sh 10.249.41.142 9001 9013 1 themis
themis1:
	./kill.sh; ./run.sh 10.249.41.142 9001 9013 2 themis

######################### 整理

fifo:
	./clean.sh; ./kill.sh; ./run.sh 192.158.0.11 9001 9013 0 fifo
fifo1:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./kill.sh; ./run.sh 192.158.0.11 9001 9013 1 fifo

sjf:
	./clean.sh; ./kill.sh; ./run.sh 192.158.0.11 9001 9013 0 sjf
sjf1:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./kill.sh; ./run.sh 192.158.0.11 9001 9013 1 sjf

Tiresias:
	./clean.sh; ./kill.sh; ./run.sh 192.158.0.11 9001 9013 0 Tiresias
Tiresias1:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./kill.sh; ./run.sh 192.158.0.11 9001 9013 1 Tiresias

# merge 2
sjf-ffs:
	./clean.sh; ./kill.sh; ./run.sh 192.158.0.11 9001 9013 0 sjf-ffs
sjf-ffs1:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./kill.sh; ./run.sh 192.158.0.11 9001 9013 1 sjf-ffs

sjf-ffss:
	./clean.sh; ./kill.sh; ./run.sh 192.158.0.11 9001 9013 0 sjf-ffss
sjf-ffss1:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./kill.sh; ./run.sh 192.158.0.11 9001 9013 1 sjf-ffss

sjf-ffss-no-preempt:
	./clean.sh; ./kill.sh; ./run.sh 192.158.0.11 9001 9013 0 sjf-ffss-no-preempt
sjf-ffss-no-preempt1:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./kill.sh; ./run.sh 192.158.0.11 9001 9013 1 sjf-ffss-no-preempt

sjf-ffs-m:
	./clean.sh; ./kill.sh; ./run.sh 192.158.0.11 9002 9013 0 sjf-ffs-m

sjf-bsbf:
	./clean.sh; ./kill.sh; ./run.sh 192.158.0.11 9002 9013 0 sjf-bsbf
sjf-bsbf1:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./kill.sh; ./run.sh 192.158.0.11 9002 9013 1 sjf-bsbf

sjf-bsbfs:
	./clean.sh; ./kill.sh; ./run.sh 192.158.0.11 9001 9013 0 sjf-bsbfs
sjf-bsbfs1:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./kill.sh; ./run.sh 192.158.0.11 9001 9013 1 sjf-bsbfs

sjf-bsbf-m:
	./clean.sh; ./kill.sh; ./run.sh 127.0.0.1 9002 9013 0 sjf-bsbf-m

sjf-bsbfs-no-preempt:
	./clean.sh; ./kill.sh; ./run.sh 192.158.0.11 9001 9013 0 sjf-bsbfs-no-preempt
sjf-bsbfs-no-preempt1:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./kill.sh; ./run.sh 192.158.0.11 9001 9013 1 sjf-bsbfs-no-preempt

sjf-bsbf-no-preempt:
	./clean.sh; ./kill.sh; ./run.sh 127.0.0.1 9002 9013 0 sjf-bsbf-no-preempt
sjf-bsbf-no-preempt-m:
	./clean.sh; ./kill.sh; ./run.sh 127.0.0.1 9002 9013 0 sjf-bsbf-no-preempt-m

sjf-test:
	./clean.sh; ./kill.sh; ./run.sh 127.0.0.1 9002 9013 0 sjf-test
sjf-test1:
	./kill.sh; ./run.sh 10.244.17.22 9002 9013 1 sjf-test

# all::
# 	./clean.sh; ./kill.sh; ./run.sh 192.158.0.11 9001 9013 0 sjf-ffss-no-preempt-m,sjf-bsbfs-no-preempt-m,sjf-ffss-m,sjf-bsbfs-m
# all1:
# 	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./kill.sh; ./run.sh 192.158.0.11 9001 9013 1 sjf-ffss-no-preempt-m,sjf-bsbfs-no-preempt-m,sjf-ffss-m,sjf-bsbfs-m
# Tiresias,sjf,fifo	
all::
	./clean.sh; ./kill.sh; ./run.sh 192.158.0.11 9001 9013 0 fifo
all1:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./kill.sh; ./run.sh 192.158.0.11 9001 9013 1 fifo
all2:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./kill.sh; ./run.sh 192.158.0.11 9001 9013 2 fifo
all3:
	rm /root/tmp/*.out; rm /root/tmp/*.xml; ./kill.sh; ./run.sh 192.158.0.11 9001 9013 3 fifo
