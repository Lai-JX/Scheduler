SRCHOME=/home/jxlai/project/muri
MPIPATH=/home/jxlai/share/openmpi
PY=/home/jxlai/anaconda3/envs/muri/bin/python
rdma=0

# pml: point-to-point messaging layer

if [ "$rdma" = "0" ]; then
params="-mca pml ob1 -mca btl ^openib \         
    -mca btl_tcp_if_include 192.168.0.11/24 \
    -mca mpi_warn_on_fork 0 \
    -bind-to none -map-by slot \
    -x PYTHONPATH=$SRCHOME \
    -x RDMA=1 \
    -x LD_LIBRARY_PATH  \
    -x NCCL_DEBUG=VERSION \
    -x NCCL_SOCKET_IFNAME=ens5f0 \
    -x NCCL_BUFFSIZE=262144 \
    -x NCCL_IB_DISABLE=1"
else
params="--mca pml ob1 --mca btl openib,vader,self --mca btl_openib_allow_ib 1 \
    -mca btl_tcp_if_include ib0 \
    --mca btl_openib_want_fork_support 1 \
    -mca mpi_warn_on_fork 0 \
    -bind-to none -map-by slot \
    -x PYTHONPATH=$SRCHOME \
    -x RDMA=1 \
    -x LD_LIBRARY_PATH  \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_SOCKET_IFNAME=ib0 \
    -x NCCL_BUFFSIZE=262144 \
    -x NCCL_DEBUG=VERSION"
fi

#params="--mca pml ob1 --mca btl openib,vader,self --mca btl_openib_allow_ib 1 \
#    -mca btl_tcp_if_include ib0 \
#    --mca btl_openib_want_fork_support 1 \
#    -mca mpi_warn_on_fork 0 \
#    -bind-to none -map-by slot \
#    -x RDMA=1 \
#    -x PYTHONPATH=$SRCHOME \
#    -x LD_LIBRARY_PATH  \
#    -x NCCL_IB_DISABLE=0 \
#    -x NCCL_SOCKET_IFNAME=ib0 \
#    -x NCCL_DEBUG=VERSION \
#    -x HOROVOD_CACHE_CAPACITY=0"
ngpus=8
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -hostfile ./scripts/cluster$ngpus -np $ngpus \
    $params \
    $PY examples/deepspeech/main.py
    #$PY examples/cifar10_resnet20/main.py 
    #$PY examples/imagenet_ofa/main.py
    #$PY examples/imagenet_resnet50/main.py
    #$PY examples/bert/main.py --yaml /home/esetstore/repos/ddl-platform/job_configs/microsoft/job_86.yaml
    #$PY examples/yolov3/main.py
    #$PY examples/cifar10_resnet110/main.py
    #$PY examples/imagenet_vgg16/main.py
    #$PY examples/imagenet_densenet201/main.py


#mpirun -x NCCL_DEBUG=WARN --mca btl vader,self -np 4 python examples/cifar10_resnet20/main.py
