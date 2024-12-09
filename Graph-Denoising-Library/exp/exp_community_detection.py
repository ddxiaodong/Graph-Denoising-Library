# from data_provider.data_factory import data_provider
from email.policy import default

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
from utils.metric import accuracy, roc_auc_compute_fn, evaluate_community_detection
from torch_geometric.utils import negative_sampling, train_test_split_edges
from sklearn.metrics import roc_auc_score


warnings.filterwarnings('ignore')


# 社团检测实验类
class Exp_Community_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Community_Detection, self).__init__(args)

    # 模型构建 这个方法在Basic类的init方法中调用
    def _build_model(self):
        # 加载数据集
        self.args.dataset = self._get_data("train")
        self.args.datasetPyg = self.args.dataset['datasetPyg']
        # 获取数据集中的信息 链路预测需要的信息： edge_index edge_label edge_label_index
        self.args.edge_index = self.args.dataset['edge_index']
        self.features = self.args.dataset["features"]
        # 社团检测的label 有些和节点分类的label一致 如cora
        self.labels = self.args.dataset["labels"]
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


        return model

    def _select_optimizer(self):
        print("Model parameters:", list(self.model.parameters()))
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        if not list(self.model.parameters()):
            print("No model parameters found, return NullOptimizer")
            model_optim = NullOptimizer()
        else:
            model_optim = optim.Adam(self.model.parameters(),
                                 lr=self.args.lr, weight_decay=self.args.weight_decay)
        # model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim


    # 获取给定优化器的学习率
    def _get_lr(self, optimizer):
        default_lr = 0
        if not optimizer.param_groups:
            return default_lr
        else:
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

        # scheduler = optim.lr_scheduler.MultiStepLR(model_optim, milestones=[200, 300, 400, 500, 600, 700], gamma=0.5)
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

        # graph = graph_reader(args.edge_path)
        # model = EdMot(graph, args.components, args.cutoff)
        # memberships = model.fit()
        # membership_saver(args.membership_path, memberships)

        for epoch in range(self.args.epochs):
            sampling_t = time.time()
            # if self.args.mixmode:
            #     train_adj = train_adj.cuda()

            sampling_t = time.time() - sampling_t
            # (val_adj, val_fea) = self.sampler.get_test_set(normalization=self.args.normalization, cuda=self.args.cuda)
            # if self.args.mixmode:
            #     val_adj = val_adj.cuda()

            # 这里输出的是每个节点划分到不同的社团列表
            output = self.model(self.features, self.args.edge_index)

            # todo 损失函数 得到的格式不适用于交叉熵损失
            # loss_fn = nn.CrossEntropyLoss()
            # loss = loss_fn(output, self.labels)
            # print(f"Loss: {loss.item()}")
            loss_train = 0
            # 输出社团检测各指标，包括ARI NMI Modularity
            ARI, NMI, Modularity = evaluate_community_detection(output, self.labels)


            t = time.time()
            self.model.train()
            model_optim.zero_grad()

            # loss_train.backward()
            model_optim.step()
            train_t = time.time() - t
            val_t = time.time()

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
                loss_val = 0
                ARI_val, NMI_val, Modularity_val = evaluate_community_detection(output_val, self.labels)
            else:
                loss_val = 0
                acc_val = 0

            if self.args.lradjust:
                scheduler.step()

            val_t = time.time() - val_t

            # 二分类准确率 边存在或缺失

            # return (loss_train.item(), acc_train.item(), loss_val, acc_val, get_lr(optimizer), train_t, val_t)
            outputs = (
            loss_train, ARI, NMI, loss_val, ARI_val, NMI_val, self._get_lr(model_optim), train_t, val_t)
            if self.args.debug and epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(loss_train),
                      'ARI: {:.4f}'.format(ARI),
                      'NMI: {:.4f}'.format(NMI),
                      'loss_val: {:.4f}'.format(loss_val),
                      'ARI_val: {:.4f}'.format(ARI_val),
                      'NMI_val: {:.4f}'.format(NMI_val),
                      'cur_lr: {:.5f}'.format(outputs[7]),
                      's_time: {:.4f}s'.format(sampling_t),
                      't_time: {:.4f}s'.format(outputs[8]),
                      'v_time: {:.4f}s'.format(outputs[7]))

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
        loss_test = 0
        ARI_test, NMI_test, Modularity_test = evaluate_community_detection(output_test, self.labels)
        if self.args.debug:
            print("Test set results:",
                  "loss= {:.4f}".format(loss_test),
                  "ARI= {:.4f}".format(ARI_test),
                  "NMI= {:.4f}".format(NMI_test)
                  )

            print("NMI=%.5f" % (NMI_test))



class NullOptimizer:
    def __init__(self):
        self.param_groups = []

    def step(self):
        pass  # 空操作，什么都不做

    def zero_grad(self):
        pass  # 空操作，什么都不做
