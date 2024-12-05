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
from torch_geometric.utils import negative_sampling, train_test_split_edges
from sklearn.metrics import roc_auc_score


warnings.filterwarnings('ignore')


# 节点分类实验类  在类中定义的实例方法，它应该至少接受 self 作为第一个参数
class Exp_Link_Prediction(Exp_Basic):
    def __init__(self, args):
        super(Exp_Link_Prediction, self).__init__(args)

    # 模型构建 这个方法在Basic类的init方法中调用
    def _build_model(self):
        # 加载数据集
        self.args.dataset = self._get_data("train")
        # 获取数据集中的信息 链路预测需要的信息： edge_index edge_label edge_label_index
        self.args.edge_index = self.args.dataset['edge_index']
        self.features = self.args.dataset["features"]
        # # 特征维度 类别维度
        self.args.nfeat = self.args.dataset["features"].shape[1]
        self.args.nclass = int(self.args.dataset["labels"].max().item() + 1)
        self.args.num_nodes = self.args.dataset["num_nodes"]
        # todo  加载训练和测试集

        # model初始化 传入模型名 这里是利用父类的 model_dict
        model = self.model_dict[self.args.model].Model(self.args).float()
        # convert to cuda
        if self.args.cuda:
            model.cuda()
        # 下面这里会报错  因为调用_select_optimizer方法时 model尚未返回 需要在train中调用方法定义优化器
        # self.model_optim = self._select_optimizer()
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.model_optim, milestones=[200, 300, 400, 500, 600, 700], gamma=0.5)

        # For the mix mode, lables and indexes are in cuda.
        if self.args.cuda or self.args.mixmode:
            self.labels = self.labels.cuda()
            self.idx_train = self.idx_train.cuda()
            self.idx_val = self.idx_val.cuda()
            self.idx_test = self.idx_test.cuda()
        # warm_start
        if self.args.warm_start is not None and self.args.warm_start != "":
            self.early_stopping = EarlyStopping(fname=self.args.warm_start, verbose=False)
            print("Restore checkpoint from %s" % (self.early_stopping.fname))
            model.load_state_dict(self.early_stopping.load_checkpoint())

        # set early_stopping
        if self.args.early_stopping > 0:
            self.early_stopping = EarlyStopping(patience=self.args.early_stopping, verbose=False)
            print("Model is saving to: %s" % (self.early_stopping.fname))

        # if self.args.no_tensorboard is False:
        #     tb_writer = SummaryWriter(
        #         comment=f"-dataset_{self.args.dataset}-type_{self.args.type}"
        #     )

        return model

    # def _get_data(self, flag):
    #     data_set, data_loader = data_provider(self.args, flag)
    #     return data_set, data_loader

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

    def _get_data(self, flag):
        return data_loader(self.args)

    # 训练过程
    def train(self):
        # 获取时间、优化器、scheduler、训练数据
        t_total = time.time()
        sampling_t = 0
        model_optim = self._select_optimizer()
        scheduler = optim.lr_scheduler.MultiStepLR(model_optim, milestones=[200, 300, 400, 500, 600, 700], gamma=0.5)
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        train_edge_index = self.args.dataset['train_edge_index']
        train_edge_label = self.args.dataset['train_edge_label']
        train_edge_label_index = self.args.dataset['train_edge_label_index']
        num_nodes = self.args.dataset['num_nodes']
        train_index = self.args.dataset['train_index']
        val_index = self.args.dataset['val_index']
        val_edge_index = self.args.dataset['val_edge_index']
        val_edge_label = self.args.dataset['val_edge_label']
        val_edge_label_index = self.args.dataset['val_edge_label_index']
        test_index = self.args.dataset['test_index']

        # 生成负样本和正样本
        # 正样本边是真实存在的边 表示图中的实际连接关系 即 edge_index
        # 负样本边是图中不存在的边，用于训练模型了解哪些连接是无效的
        # negative_sampling是pyg的工具函数 用于生成负样本 传入 原图的边索引、节点数量、负样本数量
        neg_edge_index = negative_sampling(
            edge_index=train_edge_index, num_nodes=num_nodes,
            num_neg_samples=train_edge_label_index.size(1), method='sparse')
        # 合并正负样本 生成最终的 edge_label_index 表示每个边的索引
        self.model.edge_label_index = torch.cat(
                [train_edge_label_index, neg_edge_index],
                dim=-1,
            )

        # 边标签 1表示存在 0表示不存在 正边和负边
        self.model.edge_label = torch.cat([
                train_edge_label,
                train_edge_label.new_zeros(neg_edge_index.size(1))
            ], dim=0)
        for epoch in range(self.args.epochs):
            sampling_t = time.time()
            # if self.args.mixmode:
            #     train_adj = train_adj.cuda()

            sampling_t = time.time() - sampling_t
            # (val_adj, val_fea) = self.sampler.get_test_set(normalization=self.args.normalization, cuda=self.args.cuda)
            # if self.args.mixmode:
            #     val_adj = val_adj.cuda()

            # 模型训练的核心
            output = self.model(self.features, self.args.edge_index)

            t = time.time()
            self.model.train()
            model_optim.zero_grad()

            # 计算损失 需要变成链路预测的损失
            # loss_train = F.nll_loss(output, self.labels[self.idx_train])
            # acc_train = accuracy(output, self.labels[self.idx_train])
            # todo 计算了 五个损失
            # loss1: supervised loss with original graph 监督损失
            loss_train = criterion(output, self.model.edge_label.float()).mean()

            loss_train.backward()
            model_optim.step()
            train_t = time.time() - t
            val_t = time.time()
            # We can not apply the fastmode for the reddit dataset.
            # if sampler.learning_type == "inductive" or not args.fastmode:

            # 改成链路预测早停法的逻辑
            # if self.args.early_stopping > 0 and self.sampler.dataset != "reddit":
            #     loss_val = F.nll_loss(output[val_index], self.labels[self.idx_val]).item()
            #     self.early_stopping(loss_val, self.model)
            #

            # 验证集的逻辑
            if not self.args.fastmode:
                #  Evaluate validation set performance separately,
                #  deactivates dropout during validation run.
                self.model.eval()
                output_val = self.model(self.features, val_edge_index)
                loss_val = criterion(output_val, self.model.edge_label.float()).mean()
                acc_val = roc_auc_score(self.model.edge_label.cpu(), output_val.detach().cpu().numpy())
            else:
                loss_val = 0
                acc_val = 0

            if self.args.lradjust:
                scheduler.step()

            val_t = time.time() - val_t
            output = output.view(-1).sigmoid()
            # 二分类准确率 边存在或缺失
            acc_train = roc_auc_score(self.model.edge_label.cpu(), output.detach().cpu().numpy())
            # return (loss_train.item(), acc_train.item(), loss_val, acc_val, get_lr(optimizer), train_t, val_t)
            outputs = (
            loss_train.item(), acc_train, loss_val, acc_val, self._get_lr(model_optim), train_t, val_t)
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

            loss_train[epoch], acc_train[epoch], loss_val[epoch], acc_val[epoch] = outputs[0], outputs[1], outputs[2], \
            outputs[3]

            if self.args.early_stopping > 0 and self.early_stopping.early_stop:
                print("Early stopping.")
                self.model.load_state_dict(self.early_stopping.load_checkpoint())
                break

        if self.args.early_stopping > 0:
            self.model.load_state_dict(self.early_stopping.load_checkpoint())

        if self.args.debug:
            print("Optimization Finished!")
            print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

#    test用到的 sampler idx_test labels
    def test(self):
        # 取测试数据
        test_index = self.args.dataset['test_index']
        test_edge_index = self.args.dataset['test_edge_index']
        test_edge_label = self.args.dataset['test_edge_label']
        test_edge_label_index = self.args.dataset['test_edge_label_index']
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

        # 数据转移到cuda训练的逻辑
        # if self.args.mixmode:
        #     test_adj = test_adj.cuda()
        # 进行测试
        self.model.eval()
        output_test = self.model(self.features, test_edge_index)
        loss_test = criterion(output_test, self.model.edge_label.float()).mean()
        acc_test = roc_auc_score(self.model.edge_label.cpu().numpy(), output_test.detach().cpu().numpy())
        if self.args.debug:
            print("Test set results:",
                  "loss= {:.4f}".format(loss_test.item()),
                  "accuracy= {:.4f}".format(acc_test.item()))
            print("accuracy=%.5f" % (acc_test.item()))

