nv-hostengine 

# 创建一个名为scheduler的group
dcgmi group -c scheduler

# 查看所有group
dcgmi group -l

# 将我们的0号GPU加入到2号group
dcgmi group -g 2 -a 0,1,2,3,4,5,6,7

# 再次查看2号group
dcgmi group -g 2 -i

# 启动group
dcgmi stats -g 2 --enable