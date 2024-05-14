
# 0. 代码
- **Scheduler/** contains code for real-cluster experiment.
  - **cluster/** 存放集群管理相关代码
  - **cluster_spec/** 存放集群配置文件，如节点数量、每个节点GPU数
  - **runtime/** 存放gRPC相关的proto和封装类，包括 scheduler, trainer, master, and worker 四个端点
  - **scheduler_impl/** 各调度算法的具体实现代码
  - **script/** 辅助脚本，用于启动dcgm、mps，清理，重置等
  - **trace-data/** 存放任务集
  - **workloads/** 深度学习任务的实现代码
  - **calc.py** 指标计算，如 avg. JCT, Makespan, and 99th JCT.
  - **jobs.py** and **model.py** 任务管理相关代码
  - **flags.py** 包含集群配置参数
  - **log.py** and **utils.py** 日志和工具函数
  - **run.py** 主程序入口
  - **controller.py**, **scheduler.py**, **trainer.py**, **worker.py**, and **task.py** 包含调度器组件和调度任务的实现
  - **operate.py** 向系统发送操作命令
  - **Makefile** 准备gRPC、启动入口

# 1. 环境搭建
## 方式1：从0开始构建
### Step 1: 集群搭建

### Step 2: create conda environment
```
# create conda env
conda create -n muri python=3.8
conda activate muri
```

### Step 3: 安装 Open MPI
[Install Open MPI](https://www.open-mpi.org/faq/?category=building#easy-build) or other MPI implementation.

### Step 4: 安装 python 库
```
# gRPC
python -m pip install grpcio
python -m pip install grpcio-tools

# prepare rpc
make rpc

# other dependencies
conda install numpy
conda install -c conda-forge cvxpy
conda install pytorch torchvision torchaudio cudatoolkit -c pytorch
HOROVOD_GPU_OPERATIONS=NCCL python -m pip install horovod

# dependencies for workloads
# NLP
conda install -c huggingface transformers
# RL
python -m pip install -r Scheduler/workloads/requirements.txt
```
## 方式2：使用我们构建的镜像
```
docker pull laijx1/muri:latest
```

## 数据集
- CV模型：[Imagenet-1k](https://academictorrents.com/details/a306397ccf9c2ead27155983c254227c0fd938e2)
- NLP模型：[Wikitext](https://huggingface.co/datasets/wikitext). 存储在 ```Scheduler/workloads/```
- 修改数据集存储路径
  
  修改```Scheduler/workloads/run.sh```中的```TRAIN_FILE,judge_path```


# 2. 运行

- `cd Scheduler/`
- 设置启动参数
  - 设置集群配置文件：`run.sh`的`setups`参数
  - 设置任务集：`run.sh`的`jobs`参数
  - 设置调度间隔：`run.sh`的`schedule_intervals`参数
- 设置`Makefile`中的`IP`、`WORKER_PORT`、`TRAINER_PORT`、`SCHEDULERS`等参数
- 在master节点
  ``` shell
  make run
  ```
- 在worker节点
  ``` shell
  # worker节点1
  make run1
  # worker节点2
  make run2
  # worker节点3
  make run3
  ```
# 备注
- 改编自 [Muri/cluster_exp](https://github.com/Rivendile/Muri/tree/main/cluster_exp)
- 由于执行脚本与具体使用的集群高度相关，我们只展示具体功能并展示相关脚本（例如，run.sh）。如果应用与具体平台，请适当调整脚本。
- 我们的测试实验在4个节点上执行，每个节点有8个 A6000 GPU。对于其他集群设置，请在 run.sh 中更改 setups。
