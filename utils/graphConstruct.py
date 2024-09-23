import torch
from torch import nn
import scipy.sparse as ss
import numpy as np
from dataLoader import Options, Read_all_cascade
import torch.nn.functional as F
from scipy.special import softmax
from datetime import datetime
def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, torch.Size(coo.shape))

'''Hypergraph'''
def ConHypergraph(data_name, user_size, window, doc,text_embedding):
    user_size, all_cascade, all_time = Read_all_cascade(data_name)
    ###context
    user_cont = {}
    for i in range(user_size):
        user_cont[i] = []

    win = window
    for i in range(len(all_cascade)):
        cas = all_cascade[i]

        if len(cas) < win:
            for idx in cas:
                user_cont[idx] = list(set(user_cont[idx] + cas))
            continue
        for j in range(len(cas) - win + 1):
            if (j + win) > len(cas):
                break
            cas_win = cas[j:j + win]
            for idx in cas_win:
                user_cont[idx] = list(set(user_cont[idx] + cas_win))

    # 构建超图结构
    indptr, indices, data = [], [], []
    indptr.append(0)
    idx = 0

    if doc:   
        for j in user_cont.keys():
            if len(user_cont[j]) == 0:
                idx = idx + 1
                continue
            source = np.unique(user_cont[j])

            length = len(source)
            s = indptr[-1]
            indptr.append((s + length))
            
            # 计算文本嵌入相关性并调整权重
        
            for user_idx in source:

                text_sim = F.cosine_similarity(text_embedding[j].unsqueeze(0), text_embedding[user_idx].unsqueeze(0), dim=1)              
                data.append(text_sim.item())  # 使用文本相似度作为超边的权重(hit好 map差)                
                indices.append(user_idx)
        # scores_tensor = torch.tensor(data)
        # data = F.softmax(scores_tensor, dim=0) 


    else:
        for j in user_cont.keys():
            if len(user_cont[j]) == 0:
                idx = idx + 1
                continue
            source = np.unique(user_cont[j])

            length = len(source)
            s = indptr[-1]
            indptr.append((s + length))
            for user_idx in source:
                indices.append(user_idx)
                data.append(1)  

    H_U = ss.csr_matrix((data, indices, indptr), shape=(len(user_cont.keys())-idx, user_size))
    H_U_sum = 1.0 / (H_U.sum(axis=1).reshape(1, -1))
    H_U_sum[H_U_sum == float("inf")] = 0

    BH_T = H_U.T.multiply(H_U_sum)#归一化每列
    BH_T = BH_T.T#每一行都被归一化了
    H = H_U.T

    H_sum = 1.0 / (H.sum(axis=1).reshape(1, -1) )
    H_sum[H_sum == float("inf")] = 0

    DH = H.T.multiply(H_sum)

    DH = DH.T
    HG_User = np.dot(DH, BH_T).tocoo()
#   HG_User是一个稀疏矩阵，它表示用户之间的连接强度。

    '''U-I hypergraph'''
    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(all_cascade)):
        items = np.unique(all_cascade[j])#每条级联

        length = len(items)#每条级联长度

        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(items[i])
            data.append(1)

    H_T = ss.csr_matrix((data, indices, indptr), shape=(len(all_cascade), user_size))

    # H_T_dense = H_T.toarray()  # 转换为密集数组以方便操作#cs us
    # H_T_softmax = softmax(H_T_dense, axis=1)  # 对每一行应用 softmax#cs us
    # # 将 softmax 结果转换回稀疏矩阵
    # BH_T = ss.csr_matrix(H_T_softmax)#cs us
    # # 对 H_T 进行转置并乘以自身
    # H = H_T.T  # 直接使用 softmax 归一化后的 H_T#us cs
    # # 对 H 进行相同的操作
    # H_dense = H.toarray()  # 将 H 转换为密集数组
    # H_softmax = softmax(H_dense, axis=1)  # 对每一行应用 softmax
    # # 将 softmax 结果转换回稀疏矩阵
    # H = ss.csr_matrix(H_softmax)
    # # 计算 DH
    # DH = H  # 直接使用 softmax 归一化后的 H#UC 
##############################################################
    # start_time = datetime.now()
    H_T_sum = 1.0 /(H_T.sum(axis=1).reshape(1, -1))#1,cs
    H_T_sum[H_T_sum == float("inf")] = 0

    BH_T = H_T.T.multiply(H_T_sum)#user_size,cs
    BH_T = BH_T.T#cs us
    H = H_T.T#us cs

    H_sum = 1.0 / (H.sum(axis=1).reshape(1, -1))#1 us
    H_sum[H_sum == float("inf")] = 0

    DH = H.T.multiply(H_sum)
    DH = DH.T#user_size,cs


    HG_Item = np.dot(DH, BH_T).tocoo()#us us
    HG_G = _convert_sp_mat_to_sp_tensor(HG_Item)#torch.Size([us, us]
    # end_time = datetime.now()
    # execution_time = end_time - start_time
    HG_L = _convert_sp_mat_to_sp_tensor(HG_User)#cs us
    return HG_G, HG_L#torch.Size([us, us])

