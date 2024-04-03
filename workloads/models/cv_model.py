import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
# import torch.profiler
# from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
import time         # ljx


class CVModel:
    def __init__(self, idx, args, sargs):
        self.idx = idx
        self.args = args
        self.sargs = sargs # specific args for this model
    
    def prepare(self, hvd):
        '''
        prepare dataloader, model, optimizer for training
        '''
        self.device = torch.device("cuda")
        # print(2)
        time0 = time.time()
        train_dataset = \
            datasets.ImageFolder(self.sargs["train_dir"],
                            transform=transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                            ]))
        # print(3)
        # print("datasets.ImageFolder cost time:",time.time()-time0)        # ljx 大概需要380s
        # self.train_sampler是一个分布式采样器，用于在多个GPU上并行训练模型。它从train_dataset中抽取样本，并将其分配给不同的GPU （数据并行）
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        # print(4)
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.sargs["batch_size"],
            sampler=self.train_sampler, num_workers=self.sargs["num_workers"],
            prefetch_factor=self.sargs["prefetch_factor"])  # prefetch_factor：预取数据的批量数

        # print(5)
        self.model = getattr(models, self.sargs["model_name"])(num_classes=self.args.num_classes)   # 获取预训练模型
        if self.sargs["resume"]:
            filename = f'{self.args.model_path}/{self.sargs["job_id"]}-{self.sargs["model_name"]}'
            self.load(filename)

        if self.args.cuda:
            self.model.cuda()
        
        optimizer = optim.SGD(self.model.parameters(), lr=(self.args.base_lr),
                    momentum=self.args.momentum, weight_decay=self.args.wd)
        # compression是一个压缩算法，用于在分布式环境中压缩梯度更新以减少通信开销
        compression = hvd.Compression.fp16 if self.args.fp16_allreduce else hvd.Compression.none
        # 分布式优化器
        self.optimizer = hvd.DistributedOptimizer(
            optimizer, named_parameters=self.model.named_parameters(prefix='model'+str(self.idx)),
            compression=compression,
            op=hvd.Adasum if self.args.use_adasum else hvd.Average)
        # 用于在分布式环境中广播模型参数和优化器状态，初始化时需要先同步一下
        hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)

        # 迭代器，用于遍历数据加载器中的批量数据
        self.dataloader_iter = iter(self.train_loader)

        self.cur_epoch = 0
        self.batch_idx = -1

        self.model.train()

    def get_data(self):
        '''
        get data
        '''
        try:
            data,target = next(self.dataloader_iter)
        except StopIteration:
            self.cur_epoch += 1
            self.train_sampler.set_epoch(self.cur_epoch)
            self.dataloader_iter = iter(self.train_loader)
            data,target = next(self.dataloader_iter)
            self.batch_idx = -1
        self.batch_idx +=1
        
        return data,target
    
    def forward_backward(self, thread):
        '''
        forward, calculate loss and backward
        '''
        thread.join()
        data, target = thread.get_result()
        if self.args.cuda:
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()

    def comm(self):
        '''
        sync for communication
        '''
        self.optimizer.step()
    
    def print_info(self):
        print("Model ", self.idx, ": ", self.sargs["model_name"], "; batch size: ", self.sargs["batch_size"])

    def data_size(self):
        # each image is 108.6kb on average
        return self.sargs["batch_size"] * 108.6
    
    def save(self, filename):
        with open('%s.model' % (filename),'wb') as f:
            torch.save(self.model.state_dict(), f)

    def load(self, filename):
        print("load model from ", filename)
        with open('%s.model' % (filename),'rb') as f:
            state_dict = torch.load(f, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
