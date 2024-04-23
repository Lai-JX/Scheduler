#! /bin/bash

# 1. 获取参数
deviceMemoryScaling=1
while getopts "sqh" arg #选项后面的冒号表示该选项需要参数
do
        case $arg in
             s)
                sudo nvidia-smi -c EXCLUSIVE_PROCESS
                sudo nvidia-cuda-mps-control -d
                ;;
             h)
                echo "-h                Print this help message."
                echo "-s                open mps mode"
                echo "-q                quit mps mode"
                exit
                ;;
             q)
                echo quit | sudo nvidia-cuda-mps-control # echo quit | nvidia-cuda-mps-control
                sudo nvidia-smi -c DEFAULT
                ;;
             ?)  #当有不认识的选项的时候arg为?
            echo "unkonw argument"
        exit 1
        ;;
        esac
done


