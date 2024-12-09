import numpy as np
import scipy.sparse as sp
import torch

# 负责存放相关的指标 包括三大任务的评价指标和中间指标
# 节点分类： 准确率。 f1分数。     召回率、精确率、平均精度、
# 社团检测： 监督指标 NMI   非监督指标 模块度Q
# 链路预测： 准确率
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def roc_auc_compute_fn(y_preds, y_targets):
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")

    y_true = y_targets.cpu().numpy()
    y_true = encode_onehot(y_true)
    y_pred = y_preds.cpu().detach().numpy()
    return roc_auc_score(y_true, y_pred)


def prec_recall_n(output, labels, topn):
    preds = output.detach().numpy()[-1]
    pass


from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import numpy as np


def evaluate_community_detection(predicted_labels, true_labels, graph=None, adj_matrix=None):
    """
    计算社团检测评价指标
    :param predicted_labels: list or array, 预测的社团标签
    :param true_labels: list or array, 真实的社团标签
    :param graph: networkx.Graph, 可选，图对象，用于计算模块度
    :param adj_matrix: ndarray, 可选，邻接矩阵，用于计算模块度
    :return: dict, 包含各指标的字典
    """
    # 检查输入
    if len(predicted_labels) != len(true_labels):
        raise ValueError("The length of predicted_labels and true_labels must be the same.")

    # 计算 NMI
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)

    # 计算 ARI
    ari = adjusted_rand_score(true_labels, predicted_labels)

    # 计算 Modularity（需要图或邻接矩阵）
    modularity = None
    if graph is not None or adj_matrix is not None:
        try:
            import networkx as nx
            from networkx.algorithms.community import modularity

            # 如果提供了图，直接计算
            if graph is not None:
                communities = {i: [] for i in set(predicted_labels)}
                for node, label in enumerate(predicted_labels):
                    communities[label].append(node)
                communities = [set(nodes) for nodes in communities.values()]
                modularity = nx.algorithms.community.quality.modularity(graph, communities)
            # 如果提供了邻接矩阵，构造图后计算
            elif adj_matrix is not None:
                graph = nx.from_numpy_array(adj_matrix)
                communities = {i: [] for i in set(predicted_labels)}
                for node, label in enumerate(predicted_labels):
                    communities[label].append(node)
                communities = [set(nodes) for nodes in communities.values()]
                modularity = nx.algorithms.community.quality.modularity(graph, communities)
        except ImportError:
            print("NetworkX is required to calculate modularity.")

    return ari, nmi, modularity



# 示例用法
if __name__ == "__main__":
    # 示例数据
    true_labels = [0, 0, 1, 1, 2, 2]
    predicted_labels = [0, 0, 2, 2, 1, 1]

    # 调用函数
    metrics = evaluate_community_detection(predicted_labels, true_labels)
    print(metrics)