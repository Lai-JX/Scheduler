#!/bin/bash
# 根据节点名设置网络接口
case "$(hostname)" in
  haigpu1)
    export OMPI_MCA_btl_tcp_if_include="ibs42"
    ;;
  haigpu2)
    export OMPI_MCA_btl_tcp_if_include="ibs0"
    ;;
  haigpu3)
    export OMPI_MCA_btl_tcp_if_include="ibs0"
    ;;
  haigpu4)
    export OMPI_MCA_btl_tcp_if_include="ibs27"
    ;;
  haigpu5)
    export OMPI_MCA_btl_tcp_if_include="ibs27"
    ;;
  haigpu6)
    export OMPI_MCA_btl_tcp_if_include="ibs2"
    ;;
  haigpu7)
    export OMPI_MCA_btl_tcp_if_include="ibs0"
    ;;
  haigpu8)
    export OMPI_MCA_btl_tcp_if_include="ibs2"
    ;;
esac
# echo $OMPI_MCA_btl_tcp_if_include
# 执行应用程序
exec "$@"
