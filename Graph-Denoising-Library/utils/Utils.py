import os.path as osp
import numpy as np
import torch
import torch_geometric.transforms as T
import random
from tqdm import tqdm
import scipy.stats
import copy
import os
import pickle as pkl
import sys
import os
import networkx as nx
import numpy as np
import scipy.sparse as sp
from utils.Normalization import fetch_normalization, row_normalize
from torch_geometric.utils import remove_self_loops, add_self_loops

"""
    工具类 所有模型额外的utils可存放在此处
"""

# 将 SciPy 的稀疏矩阵格式转换为 PyTorch 的稀疏张量格式
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# 执行简单图卷积
def sgc_precompute(features, adj, degree):
    #t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features)
    precompute_time = 0 #perf_counter()-t
    return features, precompute_time

# 设置随机种子
def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)


def checkPath(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return



def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance
    between two probability distributions
    用于衡量两个概率分布p和q之间的相似性 0表示两个分布完全相同 1表示两个分布互斥
    """
    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)
    # calculate m
    m = (p + q) / 2
    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2
    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)
    return distance


def calculateDistSim(res, savePath=None):
    """
    基于节点嵌入的相似性，计算正负类分布的统计特性和分布差异，
    返回正类相似度均值、负类相似度均值、分布差异统计量
    """
    r_edge, r_node, label, predict = res
    label = label.int().tolist()
    cos = torch.nn.CosineSimilarity(dim=0)
    pos_sim, neg_sim = [], []
    for idx in range(r_node[0].shape[0]):
        label_idx = label[idx]
        sim = float(cos(r_node[0][idx], r_node[1][idx]))
        if label_idx == 1:
            pos_sim.append(sim + 1)
        else:
            neg_sim.append(sim + 1)
    js_dis = jensen_shannon_distance(pos_sim, neg_sim)
    ks_dis = scipy.stats.kstest(pos_sim, neg_sim).statistic
    kl_dis = np.mean(scipy.special.kl_div(sorted(pos_sim), sorted(neg_sim)))
    return [np.mean(pos_sim), np.mean(neg_sim), ks_dis]


# 随机增强器 原始代码使用的部分 GCL这个库不支持在mac上使用  后面有自定义的替代版本
def generate_augmentation_operator(n=2):
    # 数据增强的操作
    search_space = [
        (A.Identity, ()),  # 不进行增强
        (A.FeatureMasking, (0.0, 0.3)),  # 遮掩特征
        (A.FeatureDropout, (0.0, 0.3)),  # 丢弃特征
        (A.EdgeRemoving, (0.0, 0.5))  # 删除边
    ]

    operator_list = []
    index = list(range(len(search_space)))
    random.shuffle(index)
    sampled_index = index[:n]
    for idx in sampled_index:
        opt, hp_range = search_space[idx]
        if hp_range == ():
            operator_list.append(opt())
        else:
            sampled_hp = random.uniform(hp_range[0], hp_range[1])
            operator_list.append(opt(sampled_hp))

    aug = A.Compose(operator_list)
    return aug




# 自定义增强函数
def identity(x, edge_index):
    return x, edge_index

def feature_masking(x, edge_index, mask_rate=0.3):
    # 随机遮掩特征
    mask = torch.rand(x.size(0)) < mask_rate
    x[mask] = 0
    return x, edge_index

def feature_dropout(x, edge_index, dropout_rate=0.3):
    # 丢弃特征
    mask = torch.rand(x.size(0)) < dropout_rate
    x[mask] = 0
    return x, edge_index

def edge_removing(x, edge_index, removal_rate=0.5):
    # 随机删除边
    edge_mask = torch.rand(edge_index.size(1)) < removal_rate
    edge_index = edge_index[:, ~edge_mask]
    return x, edge_index

# 增强操作生成函数
def generate_augmentation_operatorV2(n=2):
    # 定义可能的增强操作
    search_space = [
        (identity, ()),  # 不进行增强
        (feature_masking, (0.0, 0.3)),  # 遮掩特征
        (feature_dropout, (0.0, 0.3)),  # 丢弃特征
        (edge_removing, (0.0, 0.5))  # 删除边
    ]

    operator_list = []
    index = list(range(len(search_space)))
    random.shuffle(index)
    sampled_index = index[:n]

    for idx in sampled_index:
        opt, hp_range = search_space[idx]
        if hp_range == ():
            operator_list.append(opt)  # 直接添加操作
        else:
            sampled_hp = random.uniform(hp_range[0], hp_range[1])  # 随机选择超参数
            # 对于需要超参数的操作，传递超参数
            if opt in [feature_masking, feature_dropout, edge_removing]:
                operator_list.append(lambda x, edge_index, hp=sampled_hp: opt(x, edge_index, hp))  # 用 lambda 延迟执行
            else:
                operator_list.append(opt)  # 不需要超参数的操作，直接添加

    # 返回增强组合
    def augmentation(x, edge_index):
        for op in operator_list:
            x, edge_index = op(x, edge_index)  # 传入 x 和 edge_index 调用数据增强函数
        return x, edge_index

    return augmentation


"""Utils for reading and writing."""

import json
import pandas as pd
import networkx as nx
from texttable import Texttable


# 以表格形式显示参数
def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def graph_reader(path):
    """
    Function to read the graph from the path.
    :param path: Path to the edge list.
    :return graph: NetworkX object returned.
    """
    graph = nx.from_edgelist(pd.read_csv(path).values.tolist())
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def membership_saver(membership_path, membership):
    """
    Saving the membership dictionary as a JSON.
    :param membership_path: Path to save the JSON.
    :param membership: Membership dictionary with cluster ids.
    """
    with open(membership_path, "w") as f:
        json.dump(membership, f)
