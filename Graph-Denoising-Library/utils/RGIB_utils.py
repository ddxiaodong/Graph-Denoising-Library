import os.path as osp
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WikipediaNetwork, AttributedGraphDataset
import random
from tqdm import tqdm
import scipy.stats
import copy
import os



def checkPath(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return


def getDataset(dataset_name, device, rel_path='./data'):
    assert dataset_name in ['Cora', 'Citeseer', 'Pubmed', 'chameleon', 'squirrel', 'facebook']
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          add_negative_train_samples=False), ])
    if dataset_name in ['Cora', 'Citeseer', 'Pubmed']:
        path = osp.join(rel_path, 'Planetoid')
        dataset = Planetoid(path, name=dataset_name, transform=transform)
    elif dataset_name in ['chameleon', 'squirrel']:
        path = osp.join(rel_path, 'WikipediaNetwork')
        dataset = WikipediaNetwork(path, name=dataset_name, transform=transform)
    elif dataset_name in ["facebook"]:
        path = osp.join(rel_path, 'AttributedGraphDataset')
        dataset = AttributedGraphDataset(path, name=dataset_name, transform=transform)
    else:
        exit()
    return path, dataset





def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance
    between two probability distributions
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


# 随机增强器
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