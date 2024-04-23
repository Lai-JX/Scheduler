#!/bin/bash

# 获取脚本所在的完整路径
script_path=$(dirname "$0")

# 将路径转换为绝对路径
script_path=$(realpath "$script_path")

echo "Script path is: $script_path"
cd $script_path && cd ../

rm ./workloads/*.txt; rm ./workloads/*.out; rm ./workloads/*.xml; rm ./workloads/hostfiles/hostfile-*; rm /root/tmp/*.out; rm /root/tmp/*.xml; rm ./tmp/job/*; rm ./*.out; rm ./*.xml; 
rm ./workloads/gpus/gpu-*;