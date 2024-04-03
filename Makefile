rpc:
	python3 -m grpc_tools.protoc -Iruntime/proto --python_out=runtime/rpc_stubs --grpc_python_out=runtime/rpc_stubs runtime/proto/worker_to_master.proto
	python3 -m grpc_tools.protoc -Iruntime/proto --python_out=runtime/rpc_stubs --grpc_python_out=runtime/rpc_stubs runtime/proto/master_to_worker.proto
	python3 -m grpc_tools.protoc -Iruntime/proto --python_out=runtime/rpc_stubs --grpc_python_out=runtime/rpc_stubs runtime/proto/trainer_to_scheduler.proto
	python3 -m grpc_tools.protoc -Iruntime/proto --python_out=runtime/rpc_stubs --grpc_python_out=runtime/rpc_stubs runtime/proto/scheduler_to_trainer.proto

clean:
	rm -rf runtime/rpc_stubs/*_pb2.py runtime/rpc_stubs/*_pb2_grpc.py

push:
	git add .; git commit -m update; git push -u gitee master;
pull:
	git pull gitee master;
push_all:
	git add .; git commit -m update; git push -u origin master; git push -u gitee master;
run:
	rm ./workloads/*.xml; ./kill.sh; ./run.sh 10.249.41.142 9001 9013 1 dlas-gpu
run1:
	./kill.sh; ./run.sh 10.249.41.142 9001 9013 2 dlas-gpu
reset:
	git fetch; git reset --hard HEAD; git merge origin/master

rm:
	rm ./workloads/*.txt; rm ./workloads/*.out; rm ./workloads/*.xml; rm ./workloads/hostfiles/hostfile-*; rm ../../tmp/job/*.out; rm ../../tmp/job/*.xml; rm ./*.out; rm ./*.xml

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


# nps 其实就是规整后的mps
nps:
	rm ./workloads/*.txt; rm ./workloads/*.out; rm ./workloads/*.xml; rm ./workloads/hostfiles/hostfile-*; rm ../../tmp/job/*.out; rm ../../tmp/job/*.xml; rm ./*.out; rm ./*.xml; ./kill.sh; ./run.sh 10.249.41.142 9002 9013 1 nps,merge-antman
nps1:
	./kill.sh; ./run.sh 10.249.41.142 9002 9013 2 nps,merge-antman

# merge 2
merge:
	rm ./workloads/*.txt; rm ./workloads/*.out; rm ./workloads/*.xml; rm ./workloads/hostfiles/hostfile-*; rm ../../tmp/job/*.out; rm ../../tmp/job/*.xml; rm ./*.out; rm ./*.xml; ./kill.sh; ./run.sh 10.249.41.142 9002 9013 1 merge-antman
merge1:
	./kill.sh; ./run.sh 10.249.41.142 9002 9013 2 merge-antman

fifo:
	rm ./workloads/*.txt; rm ./workloads/*.out; rm ./workloads/*.xml; rm ./workloads/hostfiles/hostfile-*; rm ../../tmp/job/*.out; rm ../../tmp/job/*.xml; rm ./*.out; rm ./*.xml; ./kill.sh; ./run.sh 10.249.41.142 9002 9013 1 fifo
