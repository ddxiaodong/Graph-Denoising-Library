import numpy as np
import scipy.sparse as sp

'''
    对邻接矩阵的各类操作
'''

# 生成对称归一化拉普拉斯矩阵
def normalized_laplacian(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return (sp.eye(adj.shape[0]) - d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)).tocoo()

# 生成标准的拉普拉斯矩阵
def laplacian(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1)).flatten()
   d_mat = sp.diags(row_sum)
   return (d_mat - adj).tocoo()

# 用于GCN模型的对称归一化邻接矩阵
def gcn(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return (sp.eye(adj.shape[0]) + d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)).tocoo()

# 生成增强对称归一化邻接矩阵
def aug_normalized_adjacency(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

# 增强对称归一化邻接矩阵，并添加单位矩阵
def bingge_norm_adjacency(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return (d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt) +  sp.eye(adj.shape[0])).tocoo()

# 生成对称归一化邻接矩阵
def normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return (d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)).tocoo()

# 生成随机游走归一化邻接矩阵
def random_walk_laplacian(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv = np.power(row_sum, -1.0).flatten()
   d_mat = sp.diags(d_inv)
   return (sp.eye(adj.shape[0]) - d_mat.dot(adj)).tocoo()

# 生成增强随机游走归一化邻接矩阵
def aug_random_walk(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv = np.power(row_sum, -1.0).flatten()
   d_mat = sp.diags(d_inv)
   return (d_mat.dot(adj)).tocoo()

#
def random_walk(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv = np.power(row_sum, -1.0).flatten()
   d_mat = sp.diags(d_inv)
   return d_mat.dot(adj).tocoo()

# 返回原始邻接矩阵
def no_norm(adj):
   adj = sp.coo_matrix(adj)
   return adj

# 返回邻接矩阵加自环
def i_norm(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    return adj

# 根据指定的归一化类型动态返回对应的函数
def fetch_normalization(type):
   switcher = {
       'NormLap': normalized_laplacian,  # A' = I - D^-1/2 * A * D^-1/2
       'Lap': laplacian,  # A' = D - A
       'RWalkLap': random_walk_laplacian,  # A' = I - D^-1 * A
       'FirstOrderGCN': gcn,   # A' = I + D^-1/2 * A * D^-1/2
       'AugNormAdj': aug_normalized_adjacency,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
       'BingGeNormAdj': bingge_norm_adjacency, # A' = I + (D + I)^-1/2 * (A + I) * (D + I)^-1/2
       'NormAdj': normalized_adjacency,  # D^-1/2 * A * D^-1/2
       'RWalk': random_walk,  # A' = D^-1*A
       'AugRWalk': aug_random_walk,  # A' = (D + I)^-1*(A + I)
       'NoNorm': no_norm, # A' = A
       'INorm': i_norm,  # A' = A + I
   }
   func = switcher.get(type, lambda: "Invalid normalization technique.")
   return func

# 行归一化矩阵
def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

