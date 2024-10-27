# from data_provider.data_factory import data_provider
from exp_basic import Exp_Basic
# from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb

from utils.sample import Sampler
from utils.earlystopping import EarlyStopping
import torch.nn.functional as F
from utils.metric import accuracy, roc_auc_compute_fn

warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # 添加参数
        # should we need fix random seed here?
        self.sampler = Sampler(self.args.dataset, self.args.datapath, self.args.task_type)

        # get labels and indexes
        self.lables, self.idx_train, self.idx_val, self.idx_test = self.sampler.get_label_and_idxes(self.args.cuda)
        self.nfeat = self.sampler.nfeat
        self.nclass = self.sampler.nclass

        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        # convert to cuda
        if self.args.cuda:
            model.cuda()
        self.model_optim = self._select_optimizer()
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.model_optim, milestones=[200, 300, 400, 500, 600, 700], gamma=0.5)
        
        # For the mix mode, lables and indexes are in cuda. 
        if self.args.cuda or self.args.mixmode:
            self.labels = self.labels.cuda()
            self.idx_train = self.idx_train.cuda()
            self.idx_val = self.idx_val.cuda()
            self.idx_test = self.idx_test.cuda()

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
    
    
    def _get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']



    # define the training function.
    def train(self):
        # Train model
        t_total = time.time()
        loss_train = np.zeros((self.args.epochs,))
        acc_train = np.zeros((self.args.epochs,))
        loss_val = np.zeros((self.args.epochs,))
        acc_val = np.zeros((self.args.epochs,))

        sampling_t = 0
        model_optim = self._select_optimizer()
        for epoch in range(self.args.epochs):
            input_idx_train = self.idx_train
            sampling_t = time.time()
            # no sampling
            # randomedge sampling if args.sampling_percent >= 1.0, it behaves the same as stub_sampler.
            (train_adj, train_fea) = self.sampler.randomedge_sampler(percent=self.args.sampling_percent, normalization=self.args.normalization,
                                                                cuda=self.args.cuda)
            if self.args.mixmode:
                train_adj = train_adj.cuda()

            sampling_t = time.time() - sampling_t
            
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
            output = self.model(train_fea, train_adj)
            # special for reddit
            if self.sampler.learning_type == "inductive":
                loss_train = F.nll_loss(output, self.labels[self.idx_train])
                acc_train = accuracy(output, self.labels[self.idx_train])
            else:
                loss_train = F.nll_loss(output[self.idx_train], self.labels[self.idx_train])
                acc_train = accuracy(output[self.idx_train], self.labels[self.idx_train])

            loss_train.backward()
            model_optim.step()
            train_t = time.time() - t
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
                self.scheduler.step()

            val_t = time.time() - val_t
            
            # return (loss_train.item(), acc_train.item(), loss_val, acc_val, get_lr(optimizer), train_t, val_t)
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
                

            loss_train[epoch], acc_train[epoch], loss_val[epoch], acc_val[epoch] = outputs[0], outputs[1], outputs[2], outputs[
                3]

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
        (test_adj, test_fea) = self.sampler.get_test_set(normalization=self.args.normalization, cuda=self.args.cuda)
        if self.args.mixmode:
            test_adj = test_adj.cuda()
        # 进行测试    
        self.model.eval()
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
        
