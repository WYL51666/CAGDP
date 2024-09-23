import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from Optim import ScheduledOptim
from models.HGATLayer import *
from models.TransformerBlock import *
'''To GPU'''
def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

'''To CPU'''
def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

'''Mask previous activated users'''
def get_previous_user_mask(seq, user_size):
    assert seq.dim() == 2
    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
    previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
    previous_mask = torch.from_numpy(previous_mask)
    if seq.is_cuda:
        previous_mask = previous_mask.cuda()
    masked_seq = previous_mask * seqs.data.float()

    # force the 0th dimension (PAD) to be masked
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
    if seq.is_cuda:
        PAD_tmp = PAD_tmp.cuda()
    masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
    if seq.is_cuda:
        ans_tmp = ans_tmp.cuda()
    masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float('-1000'))
    return masked_seq.cuda()

class Fusion(nn.Module):
    def __init__(self, input_size, fusion_method, dropout=0.1):
        super(Fusion, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, 1)
        self.dropout = nn.Dropout(dropout)

        self.init_weights()
        self.fusion_method = fusion_method

    def init_weights(self):
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
    def forward(self, hidden, dy_emb):
        
        if self.fusion_method == 'mean':
            emb =((hidden+ dy_emb)/2).unsqueeze(dim=0)

            emb_score = F.softmax(torch.tanh(self.linear1(emb)), dim=0)
            emb_score = self.dropout(emb_score)#torch.Size([2, 64, 199, 1])
            out = torch.sum(emb_score * emb, dim=0)#torch.Size([64, 199, 64])
        elif self.fusion_method == 'max':
            emb =  torch.max(hidden, dy_emb).unsqueeze(dim=0)
            emb_score = F.softmax(torch.tanh(self.linear1(emb)), dim=0)
            emb_score = self.dropout(emb_score)
            out = torch.sum(emb_score * emb, dim=0)
        elif self.fusion_method == 'cat':
            emb = torch.cat([hidden.unsqueeze(dim=0), dy_emb.unsqueeze(dim=0)], dim=0)#2批次
            emb_score = F.softmax(self.linear2(torch.tanh(self.linear1(emb))), dim=0)
            emb_score = self.dropout(emb_score)#torch.Size([2, 64, 199, 1])
            out = torch.sum(emb_score * emb, dim=0)#torch.Size([64, 199, 64])
        else:

            raise ValueError("Unsupported fusion method. Please choose from 'mean', 'max' or 'cat'.")

        return out

class GLUVariant(nn.Module):
    def __init__(self, input_size):
        super(GLUVariant, self).__init__()
        # 定义线性层，输出特征维度是输入的两倍
        self.linear = nn.Linear(input_size, input_size * 2)
        # # 添加Layer Normalization
        self.layer_norm = nn.LayerNorm(input_size * 2)
    def forward(self, x):
        # 通过线性层
        x = self.linear(x)
        # 应用Layer Normalization
        x = self.layer_norm(x)
        value, gate = x.chunk(2, dim=-1)
        return value * torch.relu(gate)
    
class Module(nn.Module):
    def __init__(self,  hypergraphs, args ,text_embedding,dropout=0.3):
        super(Module, self).__init__()
        # paramete
        self.emb_size = args.embSize
        self.n_node = args.n_node
        self.layers = args.layer
        self.local_layers = args.layer
        self.dropout = nn.Dropout(dropout)
        self.drop_rate = dropout
        self.n_channel = len(hypergraphs) 
        self.win_size = 5
        self.data_path = args.data_name

        self.fuse = Fusion(self.emb_size, fusion_method=args.fusion_method)

        #hypergraph
        self.H_G = hypergraphs[0]   #HG
        self.H_L =hypergraphs[1] #HL
        self.gat = HGATLayer(self.emb_size,self.emb_size, dropout=dropout,transfer=False, concat=True)#0.3
        self.glu = GLUVariant(self.emb_size)

        # user embedding
        self.user_embedding = nn.Embedding(self.n_node, self.emb_size, padding_idx=0)
        ### use text
        self.doc = args.use_doc        
        if self.doc:
            self.text_embedding = text_embedding.to(torch.float32)
        else:
            self.text_embedding = None

        ### channel self-gating parameters     
        self.weights = nn.ParameterList([nn.Parameter(torch.zeros(self.emb_size, self.emb_size)) for _ in range(self.n_channel)])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(1, self.emb_size)) for _ in range(self.n_channel)])

        ### channel self-supervised parameters
        self.ssl_weights = nn.ParameterList([nn.Parameter(torch.zeros(self.emb_size, self.emb_size)) for _ in range(self.n_channel)])
        self.ssl_bias = nn.ParameterList([nn.Parameter(torch.zeros(1, self.emb_size)) for _ in range(self.n_channel)])
        
        ### attention parameters
        self.att = nn.Parameter(torch.zeros(1, self.emb_size))
        self.att_m = nn.Parameter(torch.zeros(self.emb_size, self.emb_size))

        ### multi-head attention
        self.multi_att = TransformerBlock(input_size=self.emb_size, n_heads=4, attn_dropout=dropout)
        self.past_multi_att = TransformerBlock(input_size=self.emb_size, n_heads=4, attn_dropout=dropout)

        self.linear =  nn.Linear(in_features=768, out_features=self.emb_size)
        self.attention_norm = nn.LayerNorm(self.emb_size)
        self.reset_parameters()

        #### optimizer and loss function
        self.optimizerAdam = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-09)
        self.optimizer = ScheduledOptim(self.optimizerAdam, self.emb_size, args.n_warmup_steps, self.data_path)
        self.loss_function = nn.CrossEntropyLoss(reduction='sum', ignore_index=0)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def self_gating(self, em, channel):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.weights[channel]) + self.bias[channel]))
    def self_supervised_gating(self, em, channel):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.ssl_weights[channel]) + self.ssl_bias[channel]))

    def channel_attention(self, *channel_embeddings):
        weights = []
        for embedding in channel_embeddings:
            weights.append(
                torch.sum(
                    torch.multiply(self.att, torch.matmul(embedding, self.att_m)),
                    1))
        embs = torch.stack(weights, dim=0)
        score = F.softmax(embs.t(), dim = -1)
        mixed_embeddings = 0
        for i in range(len(weights)):
            mixed_embeddings += torch.multiply(score.t()[i], channel_embeddings[i].t()).t()
        return mixed_embeddings
    def hierarchical_ssl(self, em, adj):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            return corrupted_embedding

        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding

        def score(x1, x2):
             return torch.sum(x1 * x2, dim=1)

        user_embeddings = em#us es
        edge_embeddings = torch.sparse.mm(adj, em)#

        # Local MIM
        pos = score(user_embeddings, edge_embeddings)
        neg1 = score(row_shuffle(user_embeddings), edge_embeddings)
        neg2 = score(row_column_shuffle(edge_embeddings), user_embeddings)
        local_loss = torch.sum(-torch.log(torch.sigmoid(pos - neg1)) - torch.log(torch.sigmoid(neg1 - neg2)))

        # Global MIM
        graph = torch.mean(edge_embeddings, 0)
        pos = score(edge_embeddings, graph)
        neg1 = score(row_column_shuffle(edge_embeddings), graph)
        global_loss = torch.sum(-torch.log(torch.sigmoid(pos - neg1)))
        return global_loss + local_loss

    def _dropout_graph(self, graph, keep_prob):
        size = graph.size()
        index = graph.coalesce().indices()
        values = graph.coalesce().values()
        random_mask = torch.rand(len(values)) < keep_prob
        indices = index[:, random_mask] 
        values = values[random_mask] / keep_prob 
        g = torch.sparse_coo_tensor(indices, values, size)
        return g

    def CAG(self, cas, text):
        cas = self.attention_norm(cas)
        text = self.attention_norm(text)

        ct = self.multi_att(cas,text,text)
        tc = self.multi_att(text,cas,cas)
        cas = cas + ct
        text =text + self.glu(tc)
        return cas,text

    '''social structure and hypergraph structure embeddding'''
    def structure_embed(self, H_Time=None, H_G=None, H_L=None):
        if self.training:
            H_G = self._dropout_graph(self.H_G, keep_prob=1-self.drop_rate) 
            H_L = self._dropout_graph(self.H_L, keep_prob=1-self.drop_rate) 
        else:
            H_G = self.H_G 
            H_L = self.H_L

        u_emb_c2 = self.self_gating(self.user_embedding.weight, 0)  
        u_emb_c3 = self.self_gating(self.user_embedding.weight, 1)  

        all_emb_c2 = [u_emb_c2]  
        all_emb_c3 = [u_emb_c3]
        for k in range(self.layers):
            '''HG传播'''         
            u_emb_c2 = self.gat(u_emb_c2, H_G) 
            u_emb_c3 = torch.sparse.mm(H_L, u_emb_c3)
            norm_embeddings2 = F.normalize(u_emb_c2, p=2, dim=1)
            all_emb_c2 += [norm_embeddings2]

            norm_embeddings3 = F.normalize(u_emb_c3, p=2, dim=1)
            all_emb_c3 += [norm_embeddings3]
        u_emb_c22 = torch.stack(all_emb_c2, dim=1) 
        u_emb_c33 = torch.stack(all_emb_c3, dim=1) 

        u_emb_c2 = torch.sum(u_emb_c22, dim=1)  
        u_emb_c3 = torch.sum(u_emb_c33, dim=1)  

        # aggregating channel-specific embeddings
        high_embs = self.channel_attention(u_emb_c2, u_emb_c3) 
        return high_embs

    def forward(self, input):
        mask = (input == 0)
        if  self.doc:

            Text_emb= self.linear(self.text_embedding)
            '''structure embeddding'''
            HG_Uemb = self.structure_embed()
            '''text_emb++++++++++++++++++++++++++++++++'''  
            text_emb = F.embedding(input, Text_emb).float()
            '''past cascade embeddding'''
            cas_seq_emb = F.embedding(input, HG_Uemb)

            '''CAG'''
            cas_seq_embs,text_embs = self.CAG(cas_seq_emb,text_emb)
            '''CE'''
            cas_glu= self.glu(cas_seq_embs)
            CE = self.multi_att(cas_glu,cas_seq_embs,cas_seq_embs,mask= mask.cuda())
            cas_seq_with_text_emb = self.fuse(cas_seq_embs,text_embs) 
            '''CAG'''
            CEs,cas_seq_with_text_emb = self.CAG(CE,cas_seq_with_text_emb)
            ECS = self.fuse(CEs,cas_seq_with_text_emb)
            '''ATT'''
            output = self.past_multi_att(ECS,ECS,ECS, mask)
            '''SSL loss'''
            graph_ssl_loss = self.hierarchical_ssl(self.self_supervised_gating(HG_Uemb, 0), self.H_G)
            graph_ssl_loss += self.hierarchical_ssl(self.self_supervised_gating(HG_Uemb, 1), self.H_L)

        else:
            '''structure embeddding'''
            HG_Uemb = self.structure_embed()
            '''past cascade embeddding'''
            cas_seq_emb = F.embedding(input, HG_Uemb)

            cas_glu= self.glu(cas_seq_emb)
            CE = self.multi_att(cas_glu,cas_seq_emb,cas_seq_emb,mask= mask.cuda())
            '''biatt'''
            CEs,cas_seq_embs = self.CAG(CE,cas_seq_emb)
            ECS = self.fuse(CEs,cas_seq_embs)
            '''ATT'''
            output = self.past_multi_att(ECS,ECS,ECS, mask)
            '''SSL loss'''
            graph_ssl_loss = self.hierarchical_ssl(self.self_supervised_gating(HG_Uemb, 0), self.H_G)
            graph_ssl_loss += self.hierarchical_ssl(self.self_supervised_gating(HG_Uemb, 1), self.H_L)

        '''Prediction'''
        pre_y = torch.matmul(output, torch.transpose(HG_Uemb, 1, 0))
        mask = get_previous_user_mask(input, self.n_node)
        return (pre_y + mask).view(-1, pre_y.size(-1)).cuda(), graph_ssl_loss

