import torch
import scipy.sparse as ss
import numpy as np
from dataLoader import  Read_all_cascade
import torch.nn.functional as F
def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, torch.Size(coo.shape))

'''Hypergraph'''
def ConHypergraph(data_name, user_size, window, doc,text_embedding):
    user_size, all_cascade, all_time = Read_all_cascade(data_name)
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
        
            for user_idx in source:

                text_sim = F.cosine_similarity(text_embedding[j].unsqueeze(0), text_embedding[user_idx].unsqueeze(0), dim=1)              
                data.append(text_sim.item())              
                indices.append(user_idx)


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

    BH_T = H_U.T.multiply(H_U_sum)
    BH_T = BH_T.T
    H = H_U.T

    H_sum = 1.0 / (H.sum(axis=1).reshape(1, -1) )
    H_sum[H_sum == float("inf")] = 0

    DH = H.T.multiply(H_sum)

    DH = DH.T
    HG_User = np.dot(DH, BH_T).tocoo()

    '''U-I hypergraph'''
    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(all_cascade)):
        items = np.unique(all_cascade[j])

        length = len(items)

        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(items[i])
            data.append(1)

    H_T = ss.csr_matrix((data, indices, indptr), shape=(len(all_cascade), user_size))

    H_T_sum = 1.0 /(H_T.sum(axis=1).reshape(1, -1))
    H_T_sum[H_T_sum == float("inf")] = 0

    BH_T = H_T.T.multiply(H_T_sum)
    BH_T = BH_T.T
    H = H_T.T

    H_sum = 1.0 / (H.sum(axis=1).reshape(1, -1))
    H_sum[H_sum == float("inf")] = 0

    DH = H.T.multiply(H_sum)
    DH = DH.T


    HG_Item = np.dot(DH, BH_T).tocoo()
    HG_G = _convert_sp_mat_to_sp_tensor(HG_Item)
    HG_L = _convert_sp_mat_to_sp_tensor(HG_User)
    return HG_G, HG_L

