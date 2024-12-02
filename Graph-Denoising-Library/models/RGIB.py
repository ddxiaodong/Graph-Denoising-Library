import os.path as osp
import numpy as np
import torch
from torch.nn import ModuleList
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from utils.RGIB_utils import generate_augmentation_operator
import random
import math


# 需要知道模型输入参数实际传入的数据集哪些值
# model = Net(dataset.num_features, 128, 64, num_gnn_layers).to(device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 自己创建一个名为Model的模型 在运行时需要通过命令行确定实际模型时哪一个
class Model(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()
        Net = getGNNArch(configs.gnn_model)
        self.configs = configs
        self.model = Net(configs.nfeat, 128, 64, configs.num_gnn_layers).to(device)
        test_auc_list, val_auc_list, best_epoch_list = [], [], []
        verbose = True
        MAX_EPOCH = 1000
        if verbose: print(f'==> schedule={configs.scheduler}, param={configs.scheduler_param}')
        assert configs.scheduler in ['linear', 'exp', 'sin', 'cos', 'constant']
        if configs.scheduler == 'linear':
            lamb_scheduler = np.linspace(0, 1, MAX_EPOCH) * configs.scheduler_param
        elif configs.scheduler == 'exp':
            lamb_scheduler = np.array([math.exp(-t / MAX_EPOCH) for t in range(MAX_EPOCH)]) * configs.scheduler_param
        elif configs.scheduler == 'sin':
            lamb_scheduler = np.array(
                [math.sin(t / MAX_EPOCH * math.pi * 0.5) for t in range(MAX_EPOCH)]) * configs.scheduler_param
        elif configs.scheduler == 'cos':
            lamb_scheduler = np.array(
                [math.cos(t / MAX_EPOCH * math.pi * 0.5) for t in range(MAX_EPOCH)]) * configs.scheduler_param
        elif configs.scheduler == 'constant':
            lamb_scheduler = np.array([configs.scheduler_param] * MAX_EPOCH)


    # 允许有占位符
    def forward(self, x, edge_index):
        # 每个模型调用 model(train_fea, train_adj)方法时,都要调用forward方法
        # 原模型输入的是pyg格式的数据 需要在传入时转换
        aug1 = generate_augmentation_operator()
        aug2 = generate_augmentation_operator()


        # a new round of negative sampling for every training epoch


        # forward with original graph
        z = self.model.encode(x, edge_index)
        hidden, out = self.model.decode(z, edge_label_index)
        out = out.view(-1)

        # forward with original augmented graph1
        x1, edge_index1, _ = aug1(x, edge_index)
        z1 = self.model.encode(x1, edge_index1)
        hidden1, out1 = self.model.decode(z1, edge_label_index)
        out1 = out1.view(-1)

        # forward with original augmented graph2
        x2, edge_index2, _ = aug2(x, edge_index)
        z2 = self.model.encode(x2, edge_index2)
        hidden2, out2 = self.model.decode(z2, edge_label_index)
        out2 = out2.view(-1)
        return out1


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for i in range(0, num_layers-2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def encode(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index).relu()
        x = self.convs[-1](x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        hidden = z[edge_label_index[0]] * z[edge_label_index[1]]
        logits = (hidden).sum(dim=-1)
        hidden = F.normalize(hidden, dim=1)
        return hidden, logits

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()



class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for i in range(0, num_layers-2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def encode(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index).relu()
        x = self.convs[-1](x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        hidden = z[edge_label_index[0]] * z[edge_label_index[1]]
        logits = (hidden).sum(dim=-1)
        hidden = F.normalize(hidden, dim=1)
        return hidden, logits

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads=8, att_dropout=0):
        super().__init__()
        self.convs = ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels//heads, heads=heads, dropout=att_dropout))
        for i in range(0, num_layers-2):
            self.convs.append(GATConv(hidden_channels, hidden_channels//heads, heads=heads, dropout=att_dropout))
        self.convs.append(GATConv(hidden_channels, out_channels, dropout=att_dropout))

    def encode(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index).relu()
        x = self.convs[-1](x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        hidden = z[edge_label_index[0]] * z[edge_label_index[1]]
        logits = (hidden).sum(dim=-1)
        hidden = F.normalize(hidden, dim=1)
        return hidden, logits

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.layers = ModuleList()
        self.layers.append(torch.nn.Linear(in_channels, hidden_channels))
        for i in range(0, num_layers-2):
            self.layers.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.layers.append(torch.nn.Linear(hidden_channels, out_channels))

    def encode(self, x, edge_index):
        for fc in self.layers[:-1]:
            x = fc(x).relu()
        x = self.layers[-1](x)
        return x

    def decode(self, z, edge_label_index):
        hidden = z[edge_label_index[0]] * z[edge_label_index[1]]
        logits = (hidden).sum(dim=-1)
        hidden = F.normalize(hidden, dim=1)
        return hidden, logits

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


# 根据GNN名字返回对应的模型
def getGNNArch(GNN_name):
    assert GNN_name in ['GCN', 'GAT', 'SAGE', 'MLP']
    if GNN_name == 'GCN':
        return GCN
    elif GNN_name == 'GAT':
        return GAT
    elif GNN_name == 'SAGE':
        return SAGE
    elif GNN_name == 'MLP':
        return MLP