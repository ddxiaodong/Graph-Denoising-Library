from data_provider.data_factory import *
import GCL.augmentors


if __name__ == '__main__':
    import torch
    getDataset("Cora")
    # 示例 edge_index (2, num_edges)
    edge_index = torch.tensor([[0, 1, 2, 3],
                               [1, 2, 3, 4]])

    # 节点数量
    num_nodes = 5

    # 将 edge_index 转换为稀疏矩阵
    adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)), (num_nodes, num_nodes))

    # 如果需要密集矩阵表示
    adj_dense = adj.to_dense()
    print(adj_dense)

