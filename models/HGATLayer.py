import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class HGATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, transfer, concat=True, bias=False):
        super(HGATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.transfer = transfer

        if self.transfer:
            self.weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        else:
            self.register_parameter('weight', None)

        self.weight2 = Parameter(torch.Tensor(self.in_features, self.out_features))            
        self.weight3 = Parameter(torch.Tensor(out_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
        self.weight3.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self, x, adj):
        if self.transfer:
            x = x.matmul(self.weight)
        else:
            x = x.matmul(self.weight2)        

        adj_dense = adj.to_dense() if not hasattr(self, 'adj_dense') else self.adj_dense
        adjt = F.softmax(adj_dense.T, dim=1)

        edge = F.dropout(torch.matmul(adjt, x), self.dropout, training=self.training)
        edge = F.relu(edge)

        node = torch.matmul(adj_dense, edge.matmul(self.weight3))
        node = F.dropout(node, self.dropout, training=self.training)
        if self.concat:
            node = F.relu(node)
        return node
