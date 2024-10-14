

import numpy as np
import pandas as pd
import copy

#                          adj_A
def extractEdgesFromMatrix(m, geneNames,TFmask):
    geneNames = np.array(geneNames)
    mat = copy.deepcopy(m)
    num_nodes = mat.shape[0]
    mat_indicator_all = np.zeros([num_nodes, num_nodes])
    if TFmask is not None:
        mat = mat*TFmask   #TFmask是一个布尔矩阵，用于选择或过滤调控网络中的某些元素。
    mat_indicator_all[abs(mat) > 0] = 1
    idx_rec, idx_send = np.where(mat_indicator_all)
    edges_df = pd.DataFrame(
        {'TF': geneNames[idx_send], 'Target': geneNames[idx_rec], 'EdgeWeight': (mat[idx_rec, idx_send])})
    edges_df = edges_df.sort_values('EdgeWeight', ascending=False)

    return edges_df


def evaluate(A, truth_edges, Evaluate_Mask):
    num_nodes = A.shape[0]
    num_truth_edges = len(truth_edges)
    A= abs(A)
    if Evaluate_Mask is None:
        Evaluate_Mask = np.ones_like(A) - np.eye(len(A))
    A = A * Evaluate_Mask
    # 将矩阵A的所有元素排序并转换为列表，然后反转列表
    A_val = list(np.sort(abs(A.reshape(-1, 1)), 0)[:, 0])
    A_val.reverse()
     # 获取截止阈值，即排在真实边数量位置的元素值
    cutoff_all = A_val[num_truth_edges]
    # 将绝对值大于截止阈值的元素位置设置为1，表示存在边
    A_indicator_all = np.zeros([num_nodes, num_nodes])
    A_indicator_all[abs(A) > cutoff_all] = 1
    # 获取存在边的接收节点和发送节点的索引
    idx_rec, idx_send = np.where(A_indicator_all)
    # 创建一个边的集合
    A_edges = set(zip(idx_send, idx_rec))
    # 获取预测边与真实边的交集
    overlap_A = A_edges.intersection(truth_edges)
    # 返回交集的大小和交集边占真实边的比例
    return len(overlap_A), 1. * len(overlap_A) / ((num_truth_edges ** 2) / np.sum(Evaluate_Mask))
