import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from Optim import ScheduledOptim
from models.HGATLayer import *
from models.TransformerBlock import *
from models.ConvBlock import *

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
    masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float('-inf'))
    return masked_seq.cuda()

class fusion(nn.Module):
    def __init__(self, input_size, dropout=0.1):
        super(fusion, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, hidden, dy_emb):
        emb = torch.cat([hidden.unsqueeze(dim=0), dy_emb.unsqueeze(dim=0)], dim=0)
        emb_score = F.softmax(self.linear2(torch.tanh(self.linear1(emb))), dim=0)
        emb_score = self.dropout(emb_score)
        out = torch.sum(emb_score * emb, dim=0)
        return out

class LSTMGNN(nn.Module):
    def __init__(self,  hypergraphs, args ,dropout=0.2):#,text_embs
        super(LSTMGNN, self).__init__()
        # paramete
        self.emb_size = args.embSize#default=64 128
        self.n_node = args.n_node#opt.n_node = user_size
        self.layers = args.layer
        self.dropout = nn.Dropout(dropout)
        self.drop_rate = dropout
        self.n_channel = len(hypergraphs) 
        self.win_size = 10
        # self.doc =args.doc
        self.gat1 = HGATLayer(self.emb_size,self.emb_size, dropout=0.2,transfer=True, concat=True, edge=False)#0.3
        # self.attention_module = ChannelAttentionModule(self.n_node,self.emb_size)
        self.bi_att_n =1#nn.Parameter(torch.tensor(0.9))
        self.fuse = fusion(self.emb_size)
        #hypergraph
        self.H_Item = hypergraphs[0]   
        self.H_User =hypergraphs[1]
        #GRU
        # self.past_gru = nn.GRU(input_size=self.emb_size, hidden_size=self.emb_size, batch_first= True)
        self.past_lstm = nn.LSTM(input_size=self.emb_size, hidden_size=self.emb_size, batch_first=True)
        ###### user embedding
        self.user_embedding = nn.Embedding(self.n_node, self.emb_size, padding_idx=0)

        ''''''
        ###### text embedding
        # self.text_embedding = torch.from_numpy(np.load('text_embs_fits_cut2.npy')).cuda()#weibo
        self.text_embedding = torch.from_numpy(np.load('data/meme/text_embs_fits_cut.npy')).cuda()#meme
        # self.text_embedding = torch.from_numpy(np.load('data/meme/text_embs_x64.npy')).cuda()#meme打乱位置
        # self.text_embedding = torch.from_numpy(np.load('data/meme/text_embs_index128fcn.npy')).cuda()#meme

        ### channel self-gating parameters     
        self.weights = nn.ParameterList([nn.Parameter(torch.zeros(self.emb_size, self.emb_size)) for _ in range(self.n_channel)])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(1, self.emb_size)) for _ in range(self.n_channel)])

        ### channel self-supervised parameters
        self.ssl_weights = nn.ParameterList([nn.Parameter(torch.zeros(self.emb_size, self.emb_size)) for _ in range(self.n_channel)])
        self.ssl_bias = nn.ParameterList([nn.Parameter(torch.zeros(1, self.emb_size)) for _ in range(self.n_channel)])

        ### attention parameters
        self.att = nn.Parameter(torch.zeros(1, self.emb_size))
        self.att_m = nn.Parameter(torch.zeros(self.emb_size, self.emb_size))

        # multi-head attention
        self.multi_att = TransformerBlock(input_size=self.emb_size, n_heads=1, attn_dropout=dropout)####
        self.past_multi_att = TransformerBlock(input_size=self.emb_size, n_heads=1, attn_dropout=dropout)
        self.future_multi_att = TransformerBlock(input_size=self.emb_size, n_heads=1, is_FFN=False,
                                                 is_future=True, attn_dropout=dropout)
        self.conv = ConvBlock(n_inputs=self.emb_size, n_outputs=self.emb_size, kernel_size=self.win_size, padding = self.win_size-1)
        self.linear = nn.Linear(self.emb_size*2, self.emb_size)#self.emb_size*3

        self.reset_parameters()
        #### optimizer and loss function
        self.optimizerAdam = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-09)
        self.optimizer = ScheduledOptim(self.optimizerAdam, self.emb_size, args.n_warmup_steps)
        self.loss_function = nn.CrossEntropyLoss(size_average=False, ignore_index=0)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def self_gating(self, em, channel):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.weights[channel]) + self.bias[channel]))
    def self_gating_with_softmax(self, em, channel):
        # 克隆 self.weights[channel] 并脱离当前计算图，以避免就地修改 self.weights
        weights_normalized = F.softmax(self.weights[channel], dim=0)
        # 使用 softmax 标准化后的权重进行矩阵乘法
        gate_scores = torch.sigmoid(torch.matmul(em, weights_normalized) + self.bias[channel])
        # 将门控应用于嵌入
        gated_em = torch.multiply(em, gate_scores)
        return gated_em

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
        index = graph.coalesce().indices().t()
        values = graph.coalesce().values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    def bi_att_embed(self, cas, text):
        ct = self.multi_att(cas,text,text)
        tc = self.multi_att(text,cas,cas)
        return ct,tc
    def fu_bi_att_embed(self, cas, text):
        ct = self.future_multi_att(cas,text,text)
        tc = self.future_multi_att(text,cas,cas)
        return ct,tc

    '''social structure and hypergraph structure embeddding'''
    def structure_embed(self, H_Time=None, H_Item=None, H_User=None):
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
       
        
        if self.training:
            H_Item = self._dropout_graph(self.H_Item, keep_prob=1-self.drop_rate)#####所有用户us*us
            H_User = self._dropout_graph(self.H_User, keep_prob=1-self.drop_rate)#####用户窗口化us*us
        else:
            H_Item = self.H_Item#####所有用户
            H_User = self.H_User

        # u_emb_c2 = self.user_embedding.weight###tensor格式torch.Size([us, es])
        # u_emb_c3 = self.user_embedding1.weight###tensor格式torch.Size([us, es])
        # u_emb_c2 = self.self_gating_with_softmax(self.user_embedding.weight, 0)###tensor格式torch.Size([us, es])
        # u_emb_c3 = self.self_gating_with_softmax(self.user_embedding.weight, 1)###tensor格式torch.Size([us, es])

        u_emb_c2 = self.self_gating(self.user_embedding.weight, 0)###tensor格式torch.Size([us, es])
        u_emb_c3 = self.self_gating(self.user_embedding.weight, 1)###tensor格式torch.Size([us, es])

        all_emb_c2 = [u_emb_c2]#len
        all_emb_c3 = [u_emb_c3]

        for k in range(self.layers):
            # Channel Item

            u_emb_c2 = self.gat1(u_emb_c2, H_Item) #torch.Size([us, es])
            u_emb_c3 = self.gat1(u_emb_c3, H_User) 
            # u_emb_c2 = torch.sparse.mm(H_Item, u_emb_c2)#######torch.Size([us, es])
            norm_embeddings2 = F.normalize(u_emb_c2, p=2, dim=1)####torch.Size([us, es])
            all_emb_c2 += [norm_embeddings2]####len 1+3=4
    

            # u_emb_c3 = torch.sparse.mm(H_User, u_emb_c3)
            norm_embeddings3 = F.normalize(u_emb_c3, p=2, dim=1)
            all_emb_c3 += [norm_embeddings3]

        ''' u_emb_c2 = torch.sparse.mm(H_Item, u_emb_c2)#######torch.Size([us, es])
            norm_embeddings2 = F.normalize(u_emb_c2, p=2, dim=1)####torch.Size([us, es])
            all_emb_c2 += [norm_embeddings2]####len 1+3=4

            u_emb_c3 = torch.sparse.mm(H_User, u_emb_c3)
            norm_embeddings2 = F.normalize(u_emb_c3, p=2, dim=1)
            all_emb_c3 += [norm_embeddings2]'''

        u_emb_c22 = torch.stack(all_emb_c2, dim=1)####torch.Size([us, 4, es])
        u_emb_c33 = torch.stack(all_emb_c3, dim=1)#torch.Size([us, 4, es])

        # c23,c32 = self.bi_att_embed(u_emb_c22, u_emb_c33)#torch.Size([us, 4, es])
        # u_emb_c22 =u_emb_c22 + c23
        # u_emb_c33 =u_emb_c33 + c32

        u_emb_c2 = torch.sum(u_emb_c22, dim=1)#######torch.Size([us, es])
        u_emb_c3 = torch.sum(u_emb_c33, dim=1)#torch.Size([us, es])
        # u_emb_c2 = u_emb_c2 + torch.sum(self.ch_bi_att_n*c23, dim=1)
        # u_emb_c3  = u_emb_c3  + torch.sum(self.ch_bi_att_n*c32, dim=1)
        '''aggregating channel-specific embeddings'''


        # high_embs = (u_emb_c2+u_emb_c3)/2

        high_embs, attention_score = self.channel_attention(u_emb_c2, u_emb_c3)#####torch.Size([us,es])

        '''fuse stra'''
        # cas_seq_with_text_emb = cas_seq_emb+ text_embs#####+
        # cas_seq_with_text_emb = torch.cat([cas_seq_emb, text_embs], -1)#####注意形状cat
        # high_embs = self.fuse(u_emb_c2, u_emb_c3)#torch.Size([embSize, 199, embSize])#fuse
        # high_embs = self.linear(high_embs)#torch.Size([embSize, 199, embSize])#cat

        return high_embs




    def forward(self, input, label):

        mask = (input == 0)
        mask_label = (label == 0)
        Text_emb= self.text_embedding#weibo 2d
        '''structure embeddding'''
        HG_Uemb = self.structure_embed()#####  2d
        '''text_emb++++++++++++++++++++++++++++++++'''  
        text_emb = F.embedding(input, Text_emb).float()#torch.Size([embSize, 199, embSize])
        '''past cascade embeddding'''
        cas_seq_emb = F.embedding(input, HG_Uemb)#####torch.Size([embSize, 199, embSize])
        '''bi_att'''
        ct,tc = self.bi_att_embed(cas_seq_emb,text_emb)
        text_embs = text_emb + self.bi_att_n*tc#A
        cas_seq_embs = cas_seq_emb + self.bi_att_n*ct#B

        # user_cas_gru, _ = self.past_gru(cas_seq_emb)#Q
        user_cas_gru, _ = self.past_lstm(cas_seq_embs)#Q
        # user_text_gru, _ = self.past_gru(text_emb)#Q
        # Ts = self.multi_att(user_text_gru,cas_seq_embs,cas_seq_embs)
        Cs = self.multi_att(user_cas_gru,cas_seq_embs,cas_seq_embs,mask= mask.cuda())
        '''GRU21'''
        # LSt = self.fuse(text_embs,Ts)
        # LSc = self.fuse(cas_seq_embs,Cs)
        # LS = LSc*LSt
        # output = self.past_multi_att(LS,LS,LS, mask)
        '''GRU22'''
        # LSt = self.fuse(text_embs,Ts)
        # LSc = self.fuse(cas_seq_embs,Cs)
        # LS = torch.cat([LSc,LSt], -1)
        # LS = self.linear(LS)
        # output = self.past_multi_att(LS,LS,LS, mask)        
        '''fuse'''
        # cas_seq_with_text_emb = cas_seq_emb+ text_embs#####+
        # cas_seq_with_text_emb = torch.cat([cas_seq_embs, text_embs], -1)#####注意形状cat
        #fuse
        cas_seq_with_text_emb = self.fuse(cas_seq_emb,text_embs)#torch.Size([embSize, 199, embSize])
        #*****

        # cas_seq_with_text_emb = cas_seq_embs*text_embs#torch.Size([embSize, 199, embSize])
        
        # text_cas_emb = self.linear(cas_seq_with_text_emb)#torch.Size cat
        ECS = self.fuse(Cs,cas_seq_with_text_emb)##LS *
        
        # ECS = self.fuse(Cs,text_cas_emb)##LS
        # output = self.past_multi_att(text_cas_emb,text_cas_emb, text_cas_emb, mask)#######cat

        # output = self.past_multi_att(cas_seq_with_text_emb,cas_seq_with_text_emb, cas_seq_with_text_emb, mask)#######fuse torch.Size([embSize, 199, embSize])
        output = self.past_multi_att(ECS,ECS,ECS, mask)#######fuse torch.Size([embSize, 199, embSize]) #GRU

        '''future cascade'''
        future_cas_emb = F.embedding(label, HG_Uemb)#labeltorch.Size([64, 199])   fu:torch.Size([embSize, 199, embSize])
        future_text_emb = F.embedding(label, Text_emb).float()#####torch.Size([embSize, 199, embSize])
        f_ct,f_tc = self.fu_bi_att_embed(future_cas_emb,future_text_emb)
        future_cas_embs = future_cas_emb + self.bi_att_n*f_ct
        future_text_embs = future_text_emb + self.bi_att_n*f_tc

        # fu_cas_with_text_emb = torch.cat([future_cas_embs, future_text_embs], -1)#####注意形状torch.Size([64, 199, 128])
        fu_cas_with_text_emb = self.fuse(future_cas_embs,future_text_embs)#torch.Size([embSize, 199, embSize])
        # fu_cas_with_text_emb = future_cas_embs*future_text_embs#torch.Size([embSize, 199, embSize])

        # fu_cas_with_text_emb = (future_cas_embs+ future_text_embs)/2#####注意形状torch.Size

        # fu_cas_with_text_emb = self.linear(fu_cas_with_text_emb)#torch.Size([64, 199, 64])
        '''future OUT'''
        future_output = self.future_multi_att(fu_cas_with_text_emb, fu_cas_with_text_emb,fu_cas_with_text_emb, mask=mask_label.cuda())######torch.Size([embSize, 199, embSize])
        # future_output = fu_cas_with_text_emb######torch.Size([embSize, 199, embSize])
        # future_output = self.future_multi_att(fu_cas_seq_wt_emb,fu_cas_seq_wt_emb,fu_cas_seq_wt_emb, mask=mask_label.cuda())######torch.Size([embSize, 199, embSize])

        
        short_emb = self.conv(cas_seq_emb)#torch.Size([embSize, 199, embSize])
        shortt_emb = self.conv(text_embs)#torch.Size([embSize, 199, embSize])
        '''SSL loss'''
        graph_ssl_loss = self.hierarchical_ssl(self.self_supervised_gating(HG_Uemb, 0), self.H_Item)
        graph_ssl_loss += self.hierarchical_ssl(self.self_supervised_gating(HG_Uemb, 1), self.H_User)

        '''SSL loss'''
        # seq_ssl_loss = self.seq2seq_ssl(text_cas_emb, future_output,shortt_emb , short_emb)######cat

        # seq_ssl_loss = self.seq2seq_ssl(cas_seq_with_text_emb ,future_output,shortt_emb , short_emb)######fusion 
        seq_ssl_loss = self.seq2seq_ssl(cas_seq_with_text_emb ,future_output, Cs, short_emb)######GRU fuse
        # seq_ssl_loss = self.seq2seq_ssl(text_cas_emb ,future_output, Cs, short_emb)######GRU cat
        # seq_ssl_loss = self.seq2seq_ssl(text_cas_emb ,future_output, Cs, short_emb)######GRU21 cat


       
        '''Prediction'''
        pre_y = torch.matmul(output, torch.transpose(HG_Uemb, 1, 0))#torch.transpose(HG_Uemb, 1, 0) :torch.Size([64, 46210])
        # pre_y = torch.matmul(cas_seq_with_text_emb, torch.transpose(HG_Uemb, 1, 0))#torch.transpose(HG_Uemb, 1, 0) :torch.Size([64, 46210])
        mask = get_previous_user_mask(input, self.n_node)
        print(self.bi_att_n)
        return (pre_y + mask).view(-1, pre_y.size(-1)).cuda(), graph_ssl_loss, seq_ssl_loss

    def model_prediction(self, input):

        mask = (input == 0)

        Text_emb= self.text_embedding#weibo 2d
        '''structure embeddding'''
        HG_Uemb = self.structure_embed()#####  2d
        '''text_emb++++++++++++++++++++++++++++++++'''  
        text_emb = F.embedding(input, Text_emb).float()#torch.Size([embSize, 199, embSize])
        '''past cascade embeddding'''
        cas_seq_emb = F.embedding(input, HG_Uemb)#####torch.Size([embSize, 199, embSize])
        '''bi_att'''
        ct,tc = self.bi_att_embed(cas_seq_emb,text_emb)
        text_embs = text_emb + self.bi_att_n*tc#A
        cas_seq_embs = cas_seq_emb + self.bi_att_n*ct#B
        user_cas_gru, _ = self.past_lstm(cas_seq_embs)
        # user_cas_gru, _ = self.past_gru(cas_seq_emb)
        # user_text_gru, _ = self.past_gru(text_emb)#Q
        # Ts = self.multi_att(user_text_gru,cas_seq_embs,cas_seq_embs)
        Cs = self.multi_att(user_cas_gru,cas_seq_embs,cas_seq_embs,mask= mask.cuda())
        '''GRU21'''
        # LSt = self.fuse(text_embs,Ts)
        # LSc = self.fuse(cas_seq_embs,Cs)
        # LS = LSc*LSt
        # output = self.past_multi_att(LS,LS,LS, mask)
        '''GRU22'''
        # LSt = self.fuse(text_embs,Ts)
        # LSc = self.fuse(cas_seq_embs,Cs)
        # LS = torch.cat([LSc,LSt], -1)
        # LS = self.linear(LS)
        # output = self.past_multi_att(LS,LS,LS, mask)         
        '''fuse'''
        # cas_seq_with_text_emb = cas_seq_emb+ text_embs#####+
        # cas_seq_with_text_emb = torch.cat([cas_seq_embs, text_embs], -1)#####注意形状cat

        #fuse
        cas_seq_with_text_emb = self.fuse(cas_seq_embs,text_embs)#torch.Size([embSize, 199, embSize])
        #*****

        # cas_seq_with_text_emb = cas_seq_embs*text_embs#torch.Size([embSize, 199, embSize])
        
        # text_cas_emb = self.linear(cas_seq_with_text_emb)#torch.Size cat
        ECS = self.fuse(Cs,cas_seq_with_text_emb)##LS
        # ECS = self.fuse(Cs,text_cas_emb)##LS
        # output = self.past_multi_att(text_cas_emb,text_cas_emb, text_cas_emb, mask)#######cat

        # output = self.past_multi_att(cas_seq_with_text_emb,cas_seq_with_text_emb, cas_seq_with_text_emb, mask)#######fuse torch.Size([embSize, 199, embSize])
        output = self.past_multi_att(ECS,ECS,ECS, mask)#######fuse torch.Size([embSize, 199, embSize]) #GRU
        # output = self.past_multi_att(cas_seq_wt_emb,cas_seq_wt_emb,cas_seq_wt_emb, mask)#######fuse torch.Size([embSize, 199, embSize])
        pre_y = torch.matmul(output, torch.transpose(HG_Uemb, 1, 0))
        # pre_y = torch.matmul(cas_seq_with_text_emb, torch.transpose(HG_Uemb, 1, 0))
        mask = get_previous_user_mask(input, self.n_node)

        return (pre_y + mask).view(-1, pre_y.size(-1)).cuda()


