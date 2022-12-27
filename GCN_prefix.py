import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
import random
import scipy.sparse as sp
import math

class GCNLayer(nn.Module):
    """
        Simple GCN layer, similar to 
        https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=False):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
                  torch.empty(size=(in_features, out_features)))
        if bias:
            self.bias = nn.Parameter(
                         torch.empty(size=(1, 1, out_features)))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
 
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
 
    def forward(self, input, adj):
        support = torch.matmul(input.float(),
                               self.weight.float())
        output = torch.matmul(adj.float(), support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
 
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GATLayer(nn.Module):
    """GAT层"""
    def __init__(self,input_feature,output_feature,dropout,alpha,concat=True):
        super(GATLayer,self).__init__()
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.alpha = alpha
        self.dropout = dropout
        self.concat = concat
        self.a = nn.Parameter(torch.empty(size=(2*output_feature,1)))
        self.w = nn.Parameter(torch.empty(size=(input_feature,output_feature)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w.data,gain=1.414)
        nn.init.xavier_uniform_(self.a.data,gain=1.414)
    
    def forward(self,h,adj):
        Wh = torch.matmul(h,self.w)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec) 
        # adj>0的位置使用e对应位置的值替换，其余都为-9e15，
        # 这样设定经过Softmax后每个节点对应的行非邻居都会变为0。
        attention = F.softmax(attention, dim=1) # 每行做Softmax，相当于每个节点做softmax
        attention = F.dropout(attention, self.dropout, training=self.training) #几乎dropout了大多的点
        # print("attention.shape,attention[:5,:5]:",attention.shape,attention[:5,:5])
        h_prime = torch.matmul(attention, Wh) # 得到下一层的输入
        
        if self.concat:
            return F.elu(h_prime) #激活
        else:
            return h_prime
        
    def _prepare_attentional_mechanism_input(self,Wh):
        
        Wh1 = torch.matmul(Wh,self.a[:self.output_feature,:]) # N*out_size @ out_size*1 = N*1
        
        Wh2 = torch.matmul(Wh,self.a[self.output_feature:,:]) # N*1
        
        e = Wh1+Wh2.permute(0, 2, 1) # Wh1的每个原始与Wh2的所有元素相加，生成N*N的矩阵
        return self.leakyrelu(e)
class CodeGraphPrefix(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config, weight, args):#加一个bert embedding
        super().__init__()
        self.prefix_hidden_size = 128
        # adj, features, labels, idx_train, idx_val, idx_test = load_data()
        # (hidden_size*nheads, output_size, dropout=dropout, alpha=alpha,concat=False)
        self.args=args
        if self.args.prefix_tuning:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.vocab_size, config.hidden_size)#Embedding(51416,768)
            self.embedding.weight= weight
            # self.gcn_layer=GATLayer(config.hidden_size,config.hidden_size,dropout=0.6,alpha=0.2,concat=False)
            self.gcn_layer=GCNLayer(config.hidden_size,config.hidden_size)
            #config.pre_seq_len 新随机初始化 换成bert embedding 希望和此表有关
            #传进来是词表大小 token_id对应词表向量
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, self.prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(self.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )
            #在这个seq改成gat或者trans前接一个gat 每个token一个节点 先随机mask矩阵套在gat
            #
        else:
            self.embedding = torch.nn.Embedding(config.vocab_size, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        #prefix传进来可能是[17053,18,3516,4492] shape是[2*batch,len(token)]拿embedding
        #用GAT 改成传邻接矩阵
        if self.args.prefix_tuning:
            prefix_tokens = self.embedding(prefix) ##就是GAT的x 生成的codeprompt [batch_size,pre_seq_len, num_hidden_layers]
            #但我们还要加edge_index 只保留存在边的缩影   #初始化定死但边可以动！（启发式边可以动）
            #code embedding只是初始化 可以调 但索引不希望定死 就是不知道能不能带权重 就是attention
            prefix_tokens=self.gcn_layer(prefix_tokens,matrix)#[batch_size,pre_seq_len, num_hidden_layers=768]
            
            prefix_tokens_repeat=prefix_tokens.repeat(1,self.args.max_source_length//prefix_tokens.shape[1],1)
            if self.args.max_source_length%prefix_tokens.shape[1]!=0:
                prefix_tokens_repeat=torch.cat([prefix_tokens[:,:self.args.max_source_length%prefix_tokens.shape[1],:],prefix_tokens_repeat],dim=1)
            past_key_values = self.trans(prefix_tokens_repeat)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values#shape:[batch_size,pre_seq_len, num_hidden_layers768 * 2 * hidden_size 12]