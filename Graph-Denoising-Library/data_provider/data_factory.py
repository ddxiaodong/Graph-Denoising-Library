import pickle as pkl
import sys
import os
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric.transforms as T
from torch_geometric.data.remote_backend_utils import num_nodes
from torch_geometric.datasets import Planetoid, WikipediaNetwork, AttributedGraphDataset
from utils.Normalization import fetch_normalization, row_normalize # 必须在根目录运行才能识别到
from utils.Utils import sparse_mx_to_torch_sparse_tensor

datadir = "data/cora"

# 原始加载数据集的方式  分为加载引文数据集和其她类型数据集


# 加载引文数据集
def load_citation(dataset_str="cora", normalization="AugNormAdj", porting_to_torch=True ,data_path=datadir, task_type="full"):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str.lower(), names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        print()
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder ) +1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range -min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range -min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # degree = np.asarray(G.degree)
    degree = np.sum(adj, axis=1)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    if task_type == "full":
        print("Load full supervised task.")
        # supervised setting
        idx_test = test_idx_range.tolist()
        idx_train = range(len(ally )- 500)
        idx_val = range(len(ally) - 500, len(ally))
    elif task_type == "semi":
        print("Load semi-supervised task.")
        # semi-supervised setting
        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y ) +500)
    else:
        raise ValueError("Task type: %s is not supported. Available option: full and semi.")

    adj, features = preprocess_citation(adj, features, normalization)
    features = np.array(features.todense())
    labels = np.argmax(labels, axis=1)
    # porting to pytorch
    if porting_to_torch:
        features = torch.FloatTensor(features).float()
        labels = torch.LongTensor(labels)
        # labels = torch.max(labels, dim=1)[1]
        adj = sparse_mx_to_torch_sparse_tensor(adj).float()
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        degree = torch.LongTensor(degree)
    learning_type = "transductive"
    # 获得边标签 和 边索引用于链路预测
    edge_index = torch.tensor(sparse_mx_to_torch_sparse_tensor(adj).float().coalesce().indices(), dtype=torch.long)
    # edge_index = torch.tensor(adj.coalesce().indices(), dtype=torch.long) # 返回所有非零元素行索引或列索引
    num_edges = edge_index.size(1)
    edge_label = torch.ones(num_edges, dtype=torch.long)
    return adj, features, labels, idx_train, idx_val, idx_test, degree, learning_type

# 从 .npz 文件中加载 Reddit 数据集的邻接矩阵、特征和标签。
def loadRedditFromNPZ(dataset_dir=datadir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir +"reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']

# 加载Reddit数据
def load_reddit_data(normalization="AugNormAdj", porting_to_torch=True, data_path=datadir):
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ(data_path)
    labels = np.zeros(adj.shape[0])
    labels[train_index]  = y_train
    labels[val_index]  = y_val
    labels[test_index]  = y_test
    adj = adj + adj.T + sp.eye(adj.shape[0])
    train_adj = adj[train_index, :][:, train_index]
    degree = np.sum(train_adj, axis=1)

    features = torch.FloatTensor(np.array(features))
    features = (features-features.mean(dim=0))/features.std(dim=0)
    train_features = torch.index_select(features, 0, torch.LongTensor(train_index))
    if not porting_to_torch:
        features = features.numpy()
        train_features = train_features.numpy()

    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    train_adj = adj_normalizer(train_adj)

    if porting_to_torch:
        train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
        labels = torch.LongTensor(labels)
        adj = sparse_mx_to_torch_sparse_tensor(adj).float()
        degree = torch.LongTensor(degree)
        train_index = torch.LongTensor(train_index)
        val_index = torch.LongTensor(val_index)
        test_index = torch.LongTensor(test_index)
    learning_type = "inductive"
    return adj, train_adj, features, train_features, labels, train_index, val_index, test_index, degree, learning_type


# 根据数据集名称调用对应的加载函数 返回的数据太多太杂 使用字典存放数据？或使用pyg的格式存放？

def data_loader(args):
    '''
        输入命令行参数
        输出原始数据字典 邻接矩阵
        考虑根据传参的模型采用不同的获取数据集方法
    '''
# dataset, datapath=datadir, normalization="AugNormAdj", porting_to_torch=True, task_type = "full")
    dataset_dict = ["Cora", "CiteSeer", "PubMed"]
    normalization="AugNormAdj"
    datapath = args.datapath
    porting_to_torch=False
    dataset = args.dataset
    task_type = args.task_type
    # 对于传统获取数据的模型在此
    if args.dataBy == "classic":
        # 特殊的处理
        if args.dataset == "reddit":
            return load_reddit_data(normalization, False, datapath)
        else:
            data = load_citation(dataset, normalization, False, datapath, task_type)
            (adj, features, labels, idx_train, idx_val, idx_test, degree, learning_type) = data
            return {
                "adj": adj,
                "train_adj": adj,
                "features": features,
                "train_features": features,
                "labels": labels,
                "idx_train": idx_train,
                "idx_val": idx_val,
                "idx_test": idx_test,
                "degree": degree,
                "learning_type": learning_type
            }
    # 对于通过框架获取数据在此
    elif args.dataBy == "pyg" :
        # 如果能直接从pyg框架中获得
        if args.dataset in dataset_dict:
            data = getDatasetByPyg(args.dataset)
            (adj, features, labels, edge_index, num_nodes, degree, learning_type,
             train_index, train_edge_label, train_edge_label_index, train_edge_index,
             val_index, val_edge_label, val_edge_label_index, val_edge_index,
             test_index, test_edge_label, test_edge_label_index, test_edge_index
             ) = data
            return {
                "adj": adj,
                "features": features,
                "labels": labels,
                "edge_index": edge_index,
                "num_nodes": num_nodes,
                "degree": degree,
                "learning_type": learning_type,
                "train_index": train_index,
                "train_edge_index": train_edge_index,
                "train_edge_label": train_edge_label,
                "train_edge_label_index": train_edge_label_index,
                "val_edge_index": val_edge_index,
                "val_index": val_index,
                "val_edge_label" : val_edge_label,
                "val_edge_label_index": val_edge_label_index,
                "test_edge_index": test_edge_index,
                "test_index": test_index,
                "test_edge_label" : test_edge_label,
                "test_edge_label_index" : test_edge_label_index
            }



# 读取文件中每一行 并将每一行整数存储在一个列表中返回 解析节点索引文件
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


# 预处理引文数据集
# 对输入的邻接矩阵进行标准化 对特征矩阵进行行归一化
def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 使用pyg框架获取数据
# data的完整属性 edge_index 图的边索引 形状为2 num_edges 包含双向边
# x 节点特征属性 y节点标签 train_mask 训练集掩码 num_nodes 节点数
# 可惜邻接矩阵被边索引替代了
def getDatasetByPyg(dataset_name, data_path=datadir):
    # 注意这里的大小写要统一
    assert dataset_name in ['Cora','Citeseer','Pubmed','chameleon','squirrel','facebook']

    # 数据转换模块 将多个图数据处理步骤串联在一起，依次进行节点特征归一化、数据转移到device、随机划分边 生成训练测试验证
    # T.RandomLinkSplit变换的目的是将图中的边划分为训练集、验证集和测试集。
    # 在链路预测任务中，这通常是通过保留一部分已知的边作为正样本，并从图中随机采样一些不存在的边作为负样本来实现的。这个变换会修改原始数据集，使其不仅包含图数据，还包含这些划分后的边集。
    # 当add_negative_train_samples=False时，T.RandomLinkSplit不会向训练集中添加负样本（即不存在的边）。但是，它仍然会进行边的划分，并更新数据集对象以包含这些划分。

    # 划分训练测试
    transform_spilt = T.Compose([
                    T.NormalizeFeatures(),
                    T.ToDevice(device),
                    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                                    add_negative_train_samples=False),])

    # 获取原始图 仅归一化
    transform = T.Compose([
                    T.NormalizeFeatures(),
                    T.ToDevice(device)])
    if dataset_name in ['Cora', 'Citeseer', 'Pubmed']:
        path = os.path.join(data_path, 'Planetoid')
        preprocess_dataset = Planetoid(path, name=dataset_name, transform=transform_spilt)
        dataset = Planetoid(path, name=dataset_name, transform=transform)
    elif dataset_name in ['chameleon', 'squirrel']:
        path = os.path.join(data_path, 'WikipediaNetwork')
        preprocess_dataset = WikipediaNetwork(path, name=dataset_name, transform=transform_spilt)
        dataset = WikipediaNetwork(path, name=dataset_name, transform=transform)
    elif dataset_name in ["facebook"]:
        path = os.path.join(data_path, 'AttributedGraphDataset')
        preprocess_dataset = AttributedGraphDataset(path, name=dataset_name, transform=transform_spilt)
        dataset = AttributedGraphDataset(path, name=dataset_name, transform=transform)
    else:
        exit()
    # return adj, train_adj, features, train_features, labels, train_index, val_index, test_index, degree, learning_type
    (train_data, val_data, test_data) = preprocess_dataset[0]  # 训练验证测试数据
    data = dataset[0] # 整图
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    features = data.x
    labels = data.y # 节点标签
    train_edge_index = train_data.edge_index
    train_edge_label = train_data.edge_label
    train_edge_label_index = train_data.edge_label_index
    train_index = train_data.train_mask
    val_edge_index = val_data.edge_index
    val_edge_label = val_data.edge_label
    val_edge_label_index = val_data.edge_label_index
    val_index = val_data.val_mask
    test_edge_index = test_data.edge_index
    test_edge_label = test_data.edge_label
    test_edge_label_index = test_data.edge_label_index
    test_index = test_data.test_mask
    adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)), (num_nodes, num_nodes))
    degree = torch.sum(adj, axis=1)
    learning_type = "inductive"

    print()
    return (adj, features, labels, edge_index, num_nodes, degree, learning_type,
            train_index, train_edge_label, train_edge_label_index, train_edge_index,
            val_index, val_edge_label, val_edge_label_index, val_edge_index,
            test_index, test_edge_label, test_edge_label_index, test_edge_index
            )
