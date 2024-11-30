# from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
# from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb
from data_provider.data_factory import data_loader
from utils.Sample import Sampler
from utils.EarlyStopping import EarlyStopping
import torch.nn.functional as F
from utils.metric import accuracy, roc_auc_compute_fn

warnings.filterwarnings('ignore')

# 节点分类实验类  在类中定义的实例方法，它应该至少接受 self 作为第一个参数
class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)
    # 模型构建 这个方法在Basic类的init方法中调用
    def _build_model(self):
        # 获取训练和测试数据 得到的是一个字典
        self.dataset = self._get_data()
        # 是dropedge使用采样器获取数据
        if self.args.model == "dropedge" :
            # get labels and indexes
            self.sampler = Sampler(self.args.dataset, self.args.datapath, self.args.task_type)
            self.labels, self.idx_train, self.idx_val, self.idx_test = self.sampler.get_label_and_idxes(self.args.cuda)
            self.args.nfeat = self.sampler.nfeat
            self.args.nclass = self.sampler.nclass
        else :
            self.labels, self.idx_train, self.idx_val, self.idx_test = self.dataset["labels"], self.dataset["idx_train"], self.dataset["idx_val"], self.dataset["idx_test"]
            self.args.nfeat = self.dataset["features"].shape[1]
            self.args.nclass = int(self.dataset["labels"].max().item() + 1)
        # model初始化 传入模型名 这里是利用父类的 model_dict
        model = self.model_dict[self.args.model].Model(self.args).float()
        # convert to cuda
        if self.args.cuda:
            model.cuda()
        # 下面这里会报错  因为调用_select_optimizer方法时 model尚未返回 需要在train中调用方法定义优化器
        # self.model_optim = self._select_optimizer()
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.model_optim, milestones=[200, 300, 400, 500, 600, 700], gamma=0.5)
        
        # 放cuda上训练
        if self.args.cuda or self.args.mixmode:
            self.labels = self.labels.cuda()
            self.idx_train = self.idx_train.cuda()
            self.idx_val = self.idx_val.cuda()
            self.idx_test = self.idx_test.cuda()
        # set warm_start
        if self.args.warm_start is not None and self.args.warm_start != "":
            self.early_stopping = EarlyStopping(fname=self.args.warm_start, verbose=False)
            print("Restore checkpoint from %s" % (self.early_stopping.fname))
            model.load_state_dict(self.early_stopping.load_checkpoint())

        # set early_stopping
        if self.args.early_stopping > 0:
            self.early_stopping = EarlyStopping(patience=self.args.early_stopping, verbose=False)
            print("Model is saving to: %s" % (self.early_stopping.fname))

        # 使用tensorboard查看结果
        # if self.args.no_tensorboard is False:
        #     tb_writer = SummaryWriter(
        #         comment=f"-dataset_{self.args.dataset}-type_{self.args.type}"
        #     )
        
        return model


    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.Adam(self.model.parameters(),
                       lr=self.args.lr, weight_decay=self.args.weight_decay)
        # model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    # 获取给定优化器的学习率
    def _get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    # 传入命令行参数 获取数据集字典 flag 为了区分训练测试等 预设参数
    def _get_data(self, flag):
        return data_loader(self.args)

    def train(self):
        # Train model
        t_total = time.time()
        sampling_t = 0
        model_optim = self._select_optimizer()
        # 学习率调度器 用于动态调整优化器的学习率 milestones表示到这些训练epoch时，调整学习率 gamma表示到达一个milestones时 学习率乘以0.5
        scheduler = optim.lr_scheduler.MultiStepLR(model_optim, milestones=[200, 300, 400, 500, 600, 700], gamma=0.5)

        for epoch in range(self.args.epochs):
            input_idx_train = self.idx_train
            sampling_t = time.time()
            # no sampling


            # randomedge sampling if args.sampling_percent >= 1.0, it behaves the same as stub_sampler.
            if (self.args.model == "dropedge") :
                (train_adj, train_fea) = self.sampler.randomedge_sampler(percent=self.args.sampling_percent, normalization=self.args.normalization,
                                                                cuda=self.args.cuda)
            else :
                (train_adj, train_fea) = self.dataset["train_adj"], self.dataset["train_features"]

            if self.args.mixmode:
                train_adj = train_adj.cuda()

            sampling_t = time.time() - sampling_t
            if (self.args.model == 'DropEdge'):
                (val_adj, val_fea) = self.sampler.get_test_set(normalization=self.args.normalization, cuda=self.args.cuda)
            else :
                (val_adj, val_fea) = self.dataset["val_adj"], self.dataset["val_features"]

            if self.args.mixmode:
                val_adj = val_adj.cuda()

            # 下面分if else训练 把公共部分提到前面来
            if val_adj is None:
                val_adj = train_adj
                val_fea = train_fea
            
            
            
            # The validation set is controlled by idx_val
            # if sampler.learning_type == "transductive":
            # if False:
            #     outputs = train(epoch, train_adj, train_fea, input_idx_train)  # TODO 这一块先不处理
            # else:
            #     (val_adj, val_fea) = sampler.get_test_set(normalization=self.args.normalization, cuda=self.args.cuda)
            #     if self.args.mixmode:
            #         val_adj = val_adj.cuda()
            #     # outputs = train(epoch, train_adj, train_fea, input_idx_train, val_adj, val_fea)


            t = time.time()
            self.model.train()
            model_optim.zero_grad()
            # 输入特征矩阵和邻接矩阵 后续增加模型
            output = self.model(train_fea, train_adj)
            # 如果是归纳学习 直接基于整个输出output和对应标签计算损失
            if self.sampler.learning_type == "inductive":
                # 负对数似然损失函数 常用于分类任务
                loss_train = F.nll_loss(output, self.labels[self.idx_train])
                acc_train = accuracy(output, self.labels[self.idx_train])
            # 其他类型学习只使用训练集索引部分预测值和对应标签计算损失
            else:
                loss_train = F.nll_loss(output[self.idx_train], self.labels[self.idx_train])
                acc_train = accuracy(output[self.idx_train], self.labels[self.idx_train])
            # 损失反向传播
            loss_train.backward()
            # 优化器更新参数
            model_optim.step()
            # 记录训练时间
            train_t = time.time() - t
            # 记录验证开始时间
            val_t = time.time()
            # We can not apply the fastmode for the reddit dataset.
            # if sampler.learning_type == "inductive" or not args.fastmode:

            if self.args.early_stopping > 0 and self.sampler.dataset != "reddit":
                loss_val = F.nll_loss(output[self.idx_val], self.labels[self.idx_val]).item()
                self.early_stopping(loss_val, self.model)

            if not self.args.fastmode:
                #    # Evaluate validation set performance separately,
                #    # deactivates dropout during validation run.
                self.model.eval()
                output = self.model(val_fea, val_adj)
                loss_val = F.nll_loss(output[self.idx_val], self.labels[self.idx_val]).item()
                acc_val = accuracy(output[self.idx_val],self.labels[self.idx_val]).item()
                if self.sampler.dataset == "reddit":
                    self.early_stopping(loss_val, self.model)
            else:
                loss_val = 0
                acc_val = 0

            if self.args.lradjust:
                scheduler.step()

            val_t = time.time() - val_t

            # return (loss_train.item(), acc_train.item(), loss_val, acc_val, get_lr(optimizer), train_t, val_t)
            # 这里的outputs仅用于接受变量 没实质作用 后续修改打印信息时只需改这里和打印部分就行
            outputs = (loss_train.item(), acc_train.item(), loss_val, acc_val, self._get_lr(model_optim), train_t, val_t)
            if self.args.debug and epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch + 1),
                    'loss_train: {:.4f}'.format(outputs[0]),
                    'acc_train: {:.4f}'.format(outputs[1]),
                    'loss_val: {:.4f}'.format(outputs[2]),
                    'acc_val: {:.4f}'.format(outputs[3]),
                    'cur_lr: {:.5f}'.format(outputs[4]),
                    's_time: {:.4f}s'.format(sampling_t),
                    't_time: {:.4f}s'.format(outputs[5]),
                    'v_time: {:.4f}s'.format(outputs[6]))
            
            # if args.no_tensorboard is False:
            #     tb_writer.add_scalars('Loss', {'train': outputs[0], 'val': outputs[2]}, epoch)
            #     tb_writer.add_scalars('Accuracy', {'train': outputs[1], 'val': outputs[3]}, epoch)
            #     tb_writer.add_scalar('lr', outputs[4], epoch)
            #     tb_writer.add_scalars('Time', {'train': outputs[5], 'val': outputs[6]}, epoch)

            loss_train = np.zeros((self.args.epochs,))
            acc_train = np.zeros((self.args.epochs,))
            loss_val = np.zeros((self.args.epochs,))
            acc_val = np.zeros((self.args.epochs,))

            loss_train[epoch], acc_train[epoch], loss_val[epoch], acc_val[epoch] = outputs[0], outputs[1], outputs[2], outputs[3]

            if self.args.early_stopping > 0 and self.early_stopping.early_stop:
                print("Early stopping.")
                self.model.load_state_dict(self.early_stopping.load_checkpoint())
                break

        if self.args.early_stopping > 0:
            self.model.load_state_dict(self.early_stopping.load_checkpoint())

        if self.args.debug:
            print("Optimization Finished!")
            print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
            
            
            
    # test用到的 sampler idx_test labels 
    def test(self):
        # 取测试adj和fea矩阵
        if (self.args.model == 'DropEdge'):
            (test_adj, test_fea) = self.sampler.get_test_set(normalization=self.args.normalization, cuda=self.args.cuda)
        else :
            (test_adj, test_fea) = self.dataset["test_adj"], self.dataset["test_features"]

        if self.args.mixmode:
            test_adj = test_adj.cuda()
        # 进行测试    
        self.model.eval()
        # 测试和训练走的都是模型的forward方法
        output = self.model(test_fea, test_adj)
        loss_test = F.nll_loss(output[self.idx_test], self.labels[self.idx_test])
        acc_test = accuracy(output[self.idx_test], self.labels[self.idx_test])
        auc_test = roc_auc_compute_fn(output[self.idx_test], self.labels[self.idx_test])
        if self.args.debug:
            print("Test set results:",
                "loss= {:.4f}".format(loss_test.item()),
                "auc= {:.4f}".format(auc_test),
                "accuracy= {:.4f}".format(acc_test.item()))
            print("accuracy=%.5f" % (acc_test.item()))
        
