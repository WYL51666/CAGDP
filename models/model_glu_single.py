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

class fusion(nn.Module):
    def __init__(self, input_size, fusion_method, dropout=0.1):
        super(fusion, self).__init__()
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
            # emb = emb
            emb_score = F.softmax(torch.tanh(self.linear1(emb)), dim=0)
            emb_score = self.dropout(emb_score)#torch.Size([2, 64, 199, 1])
            out = torch.sum(emb_score * emb, dim=0)#torch.Size([64, 199, 64])
        elif self.fusion_method == 'max':
            emb =  torch.max(hidden, dy_emb).unsqueeze(dim=0)
            emb_score = F.softmax(torch.tanh(self.linear1(emb)), dim=0)
            emb_score = self.dropout(emb_score)
            out = torch.sum(emb_score * emb, dim=0)
        elif self.fusion_method == 'cat':
            emb = torch.cat([hidden.unsqueeze(dim=0), dy_emb.unsqueeze(dim=0)], dim=0)
            emb_score = F.softmax(self.linear2(torch.tanh(self.linear1(emb))), dim=0)
            emb_score = self.dropout(emb_score)#torch.Size([2, 64, 199, 1])
            out = torch.sum(emb_score * emb, dim=0)#torch.Size([64, 199, 64])
        elif self.fusion_method == 'sum':
            emb =  (hidden+ dy_emb).unsqueeze(dim=0)
            emb_score = F.softmax(self.linear2(torch.tanh(self.linear1(emb))), dim=0)
            emb_score = self.dropout(emb_score)#torch.Size([2, 64, 199, 1])
            out = torch.sum(emb_score * emb, dim=0)   
        else:

            raise ValueError("Unsupported fusion method. Please choose from 'sum', 'mean', or 'elementwise_multiply'.")

        return out

class GLUVariant(nn.Module):
    def __init__(self, input_size):
        super(GLUVariant, self).__init__()
        # 定义线性层，输出特征维度是输入的两倍
        self.linear = nn.Linear(input_size, input_size * 2)
        # 添加Layer Normalization
        self.layer_norm = nn.LayerNorm(input_size * 2)

    def forward(self, x):
        # 通过线性层
        x = self.linear(x)
        # 应用Layer Normalization
        x = self.layer_norm(x)
        # 分割张量为两部分
        value, gate = x.chunk(2, dim=-1)
        # 应用门控，这里我们使用tanh代替sigmoid
        return value * torch.relu(gate)
class Mlp(nn.Module):
    def __init__(self, in_size,out_size,dropout):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_size, out_size)
        self.fc2 = nn.Linear(out_size,in_size)
        self.act_fn = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
class LSTMGNN(nn.Module):
    def __init__(self,  hypergraphs, args ,text_embedding,dropout=0.2):
        super(LSTMGNN, self).__init__()
        # paramete
        self.emb_size = args.embSize
        self.n_node = args.n_node
        self.layers = args.layer
        self.dropout = nn.Dropout(dropout)
        self.drop_rate = dropout
        self.n_channel = len(hypergraphs) 
        self.win_size = 5

        self.gat = HGATLayer(self.emb_size,self.emb_size, dropout=dropout,transfer=False, concat=True)#0.3
        # self.gat1 = HGraphAttentionConvolution(self.emb_size,self.emb_size, dropout=dropout)
        self.gat2 = HGraphConvolution(self.emb_size,self.emb_size, dropout=dropout)
        self.fuse = fusion(self.emb_size, fusion_method=args.fusion_method)#
        #hypergraph
        self.H_Item = hypergraphs[0]   
        self.H_User =hypergraphs[1]

        self.past_glu = GLUVariant(self.emb_size)

        ###### user embedding
        self.user_embedding = nn.Embedding(self.n_node, self.emb_size, padding_idx=0)
        self.doc = args.use_doc
        '''text embedding'''
        if self.doc:
            self.text_embedding = text_embedding
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
        '''attttttttt'''
        # multi-head attention
        self.multi_att = TransformerBlock(input_size=self.emb_size, n_heads=4, attn_dropout=dropout)####
        self.past_multi_att = TransformerBlock(input_size=self.emb_size, n_heads=4, attn_dropout=dropout)
        self.future_multi_att = TransformerBlock(input_size=self.emb_size, n_heads=4, is_FFN=False,
                                                 is_future=True, attn_dropout=dropout)

        self.linear = nn.Linear(self.emb_size*2, self.emb_size)
        self.attention_norm = nn.LayerNorm(self.emb_size)
        self.ffn = Mlp(self.emb_size,self.emb_size*4,dropout)
        self.reset_parameters()
        #### optimizer and loss function
        self.data_path = args.data_name
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
        return mixed_embeddings, score


    def hierarchical_ssl(self, em, adj):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            return corrupted_embedding

        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding

        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)

        user_embeddings = em
        edge_embeddings = torch.sparse.mm(adj, em)

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

    def seq2seq_ssl(self, L_fea1, L_fea2, S_fea1, S_fea2):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            return corrupted_embedding

        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding

        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), -1)

        # Local MIM
        pos = score(L_fea1, L_fea2)
        neg1 = score(L_fea1, S_fea2)
        neg2 = score(L_fea2, S_fea1)
        loss1 = torch.sum(-torch.log(torch.sigmoid(pos - neg1)) - torch.log(torch.sigmoid(neg1 - neg2)))

        pos = score(S_fea1, S_fea2)
        loss2 = torch.sum(-torch.log(torch.sigmoid(pos - neg1)) - torch.log(torch.sigmoid(neg1 - neg2)))

        return loss1 + loss2
    def _dropout_graph(self, graph, keep_prob):
        size = graph.size()
        index = graph.coalesce().indices()
        values = graph.coalesce().values()
        random_mask = torch.rand(len(values)) < keep_prob
        indices = index[:, random_mask] 
        values = values[random_mask] / keep_prob 
        g = torch.sparse_coo_tensor(indices, values, size)
        return g

    def bi_att_embed(self, cas, text):
        cas = self.attention_norm(cas)
        text = self.attention_norm(text)
        # ct = self.multi_att(cas,text,cas)
        # tc = self.multi_att(text,cas,text)
        ct = self.multi_att(cas,text,text)
        tc = self.multi_att(text,cas,cas)
        return ct,tc

    def fu_bi_att_embed(self, cas, text):
        cas = self.attention_norm(cas)
        text = self.attention_norm(text)
        ct = self.multi_att(cas,text,text)
        tc = self.multi_att(text,cas,cas)
        # cas = cas + ct
        # text =text + delta*tc
        cas = cas + ct
        text =text + self.past_glu(tc)
        return cas,text
    '''social structure and hypergraph structure embeddding'''
    def structure_embed(self, H_Time=None, H_Item=None, H_User=None):

        if self.training:
            H_Item = self._dropout_graph(self.H_Item, keep_prob=1-self.drop_rate)#####所有用户us*us
            H_User = self._dropout_graph(self.H_User, keep_prob=1-self.drop_rate)#####用户窗口化us*us
        else:
            H_Item = self.H_Item#####所有用户
            H_User = self.H_User

        u_emb_c2 = self.self_gating(self.user_embedding.weight, 0)###tensor格式torch.Size([us, es])
        u_emb_c3 = self.self_gating(self.user_embedding.weight, 1)###tensor格式torch.Size([us, es])

        all_emb_c2 = [u_emb_c2]#len
        all_emb_c3 = [u_emb_c3]

        for k in range(self.layers):
            '''==========================HG传播=========================='''
            u_emb_c2 = self.gat(u_emb_c2, H_Item) #torch.Size([us, es])
            # u_emb_c3 = self.gat2(u_emb_c3, H_User) 

            u_emb_c3 = torch.sparse.mm(H_User, u_emb_c3)###########
            norm_embeddings2 = F.normalize(u_emb_c2, p=2, dim=1)####torch.Size([us, es])
            all_emb_c2 += [norm_embeddings2]####len 1+3=4

            norm_embeddings3 = F.normalize(u_emb_c3, p=2, dim=1)
            all_emb_c3 += [norm_embeddings3]

        u_emb_c22 = torch.stack(all_emb_c2, dim=1)####torch.Size([us, 4, es])
        u_emb_c33 = torch.stack(all_emb_c3, dim=1)#torch.Size([us, 4, es])

        u_emb_c2 = torch.sum(u_emb_c22, dim=1)#######torch.Size([us, es])
        u_emb_c3 = torch.sum(u_emb_c33, dim=1)#torch.Size([us, es])
        high_embs, attention_score = self.channel_attention(u_emb_c2, u_emb_c3)#####torch.Size([us,es])
        return high_embs
    def forward(self, input, label):
        mask = (input == 0)
        mask_label = (label == 0)
        Text_emb= self.text_embedding
        '''structure embeddding'''
        HG_Uemb = self.structure_embed()
        '''text_emb++++++++++++++++++++++++++++++++'''  
        text_emb = F.embedding(input, Text_emb).float()
        '''past cascade embeddding'''
        cas_seq_emb = F.embedding(input, HG_Uemb)
        '''bi_att'''

        cas_seq_embs,text_embs = self.fu_bi_att_embed(cas_seq_emb,text_emb)
        user_cas_gru= self.past_glu(cas_seq_embs)

        Cs = self.multi_att(user_cas_gru,cas_seq_embs,cas_seq_embs,mask= mask.cuda())
        '''fuse'''
        cas_seq_with_text_emb = self.fuse(cas_seq_embs,text_embs)
        '''biatt2'''

        Cs,cas_seq_with_text_emb = self.fu_bi_att_embed(Cs,cas_seq_with_text_emb)

        ECS = self.fuse(Cs,cas_seq_with_text_emb)

        '''ATT'''
        output = self.past_multi_att(ECS,ECS,ECS, mask)
        '''ffffffuture cascade'''
        future_cas_emb = F.embedding(label, HG_Uemb)
        '''future OUT ATT'''
        future_output = self.future_multi_att(future_cas_emb, future_cas_emb,future_cas_emb, mask=mask_label.cuda())######torch.Size([embSize, 199, embSize])

      
        short_emb = self.past_glu(user_cas_gru)
        shortt_emb = self.past_glu(self.past_glu(text_emb) )
        '''SSL loss'''
        graph_ssl_loss = self.hierarchical_ssl(self.self_supervised_gating(HG_Uemb, 0), self.H_Item)
        graph_ssl_loss += self.hierarchical_ssl(self.self_supervised_gating(HG_Uemb, 1), self.H_User)
        '''SSL loss'''
        seq_ssl_loss = self.seq2seq_ssl(cas_seq_with_text_emb, future_output,shortt_emb ,short_emb )
        '''Prediction'''
        pre_y = torch.matmul(output, torch.transpose(HG_Uemb, 1, 0))
        mask = get_previous_user_mask(input, self.n_node)
        return (pre_y + mask).view(-1, pre_y.size(-1)).cuda(), graph_ssl_loss, seq_ssl_loss

    def model_prediction(self, input):
        mask = (input == 0)

        Text_emb= self.text_embedding
        '''structure embeddding'''
        HG_Uemb = self.structure_embed()
        '''text_emb++++++++++++++++++++++++++++++++'''  
        text_emb = F.embedding(input, Text_emb).float()
        '''past cascade embeddding'''
        cas_seq_emb = F.embedding(input, HG_Uemb)
        '''bi_att'''    
        cas_seq_embs,text_embs = self.fu_bi_att_embed(cas_seq_emb,text_emb)
        user_cas_gru= self.past_glu(cas_seq_embs)

        Cs = self.multi_att(user_cas_gru,cas_seq_embs,cas_seq_embs,mask= mask.cuda())
        '''fuse'''
        cas_seq_with_text_emb = self.fuse(cas_seq_embs,text_embs)
        '''biatt2'''

        Cs,cas_seq_with_text_emb = self.fu_bi_att_embed(Cs,cas_seq_with_text_emb)
        ECS = self.fuse(Cs,cas_seq_with_text_emb)

        output = self.past_multi_att(ECS,ECS,ECS, mask)
        pre_y = torch.matmul(output, torch.transpose(HG_Uemb, 1, 0))
        mask = get_previous_user_mask(input, self.n_node)
        return (pre_y + mask).view(-1, pre_y.size(-1)).cuda()


