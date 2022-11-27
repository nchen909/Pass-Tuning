import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
import random
import scipy.sparse as sp

# # from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification
# # from transformers.models.bert.modeling_bert import BertForSequenceClassification
# from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
# # from transformers.models.bart.modeling_bart import BartForConditionalGeneration
# # from transformers import AutoTokenizer, AutoModelForSequenceClassification


# # model4 = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
# model3 = T5ForConditionalGeneration.from_pretrained('t5-base')
# # model = RobertaForSequenceClassification.from_pretrained('roberta-base')
# # model2 = BertForSequenceClassification.from_pretrained('bert-base-uncased')
# # model.roberta.embeddings.word_embeddings.weight.shape
# # model2.bert.embeddings.word_embeddings.weight.shape
# # model4.parameters



# def encode_onehot(labels):
#     classes = set(labels)
#     classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
#                     enumerate(classes)}
#     labels_onehot = np.array(list(map(classes_dict.get, labels)),
#                              dtype=np.int32)
#     return labels_onehot

# def normalize_adj(mx):
#     """Row-normalize sparse matrix"""
#     rowsum = np.array(mx.sum(1))
#     r_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
#     r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
#     return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

# def normalize(mx):
#     """Row-normalize sparse matrix"""
#     rowsum = np.array(mx.sum(1))
#     r_inv = np.power(rowsum, -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0.
#     r_mat_inv = sp.diags(r_inv)
#     mx = r_mat_inv.dot(mx)
#     return mx

# def accuracy(output, labels):
#     preds = output.max(1)[1].type_as(labels)
#     correct = preds.eq(labels).double()
#     correct = correct.sum()
#     return correct / len(labels)

# def sparse_mx_to_torch_sparse_tensor(sparse_mx):
#     """Convert a scipy sparse matrix to a torch sparse tensor."""
#     sparse_mx = sparse_mx.tocoo().astype(np.float32)
#     indices = torch.from_numpy(
#         np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
#     values = torch.from_numpy(sparse_mx.data)
#     shape = torch.Size(sparse_mx.shape)
#     return torch.sparse.FloatTensor(indices, values, shape)

# def load_data(path="/data/pretrain-attention/P-tuning-v2/GNN/cora/", dataset="cora"):
#     """读取引文网络数据cora"""
#     print('Loading {} dataset...'.format(dataset))
#     idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
#                                         dtype=np.dtype(str)) # 使用numpy读取.txt文件
#     features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32) # 获取特征矩阵
#     labels = encode_onehot(idx_features_labels[:, -1]) # 获取标签

#     # build graph
#     idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
#     idx_map = {j: i for i, j in enumerate(idx)}
#     edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
#                                     dtype=np.int32)
#     edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
#                      dtype=np.int32).reshape(edges_unordered.shape)
#     adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
#                         shape=(labels.shape[0], labels.shape[0]),
#                         dtype=np.float32)

#     # build symmetric adjacency matrix
#     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

#     features = normalize(features)
#     adj = normalize_adj(adj + sp.eye(adj.shape[0]))

#     idx_train = range(140)
#     idx_val = range(200, 500)
#     idx_test = range(500, 1500)

#     features = torch.FloatTensor(np.array(features.todense()))
#     labels = torch.LongTensor(np.where(labels)[1])
#     adj = torch.FloatTensor(np.array(adj.todense()))

#     idx_train = torch.LongTensor(idx_train)
#     idx_val = torch.LongTensor(idx_val)
#     idx_test = torch.LongTensor(idx_test)

#     return adj, features, labels, idx_train, idx_val, idx_test

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


class GAT(nn.Module):
    """GAT模型"""
    def __init__(self,input_size,hidden_size,output_size,dropout,alpha,nheads,concat=True):
        super(GAT,self).__init__()
        self.dropout= dropout
        self.attention = [GATLayer(input_size, hidden_size, dropout=dropout, alpha=alpha,concat=True) for _ in range(nheads)]
        for i,attention in enumerate(self.attention):
            # print("attention.shape:",attention.shape)
            self.add_module('attention_{}'.format(i),attention)
        
        self.out_att = GATLayer(hidden_size*nheads, output_size, dropout=dropout, alpha=alpha,concat=False)
        
    def forward(self,x,adj):
        x = F.dropout(x,self.dropout,training=self.training)
        x = torch.cat([att(x,adj) for att in self.attention],dim=1)
        x = F.dropout(x,self.dropout,training=self.training)
        x = F.elu(self.out_att(x,adj))
        # print('#####x.shape,adj.shape,len(self.attention):',x.shape,adj.shape,len(self.attention))#(2708, 7)  (2708,2708)
        return F.log_softmax(x,dim=1)


class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config, weight):#加一个bert embedding
        super().__init__()
        self.prefix_projection = True#config.prefix_projection
        self.prefix_hidden_size = 128
        # adj, features, labels, idx_train, idx_val, idx_test = load_data()
        # (hidden_size*nheads, output_size, dropout=dropout, alpha=alpha,concat=False)
        
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.vocab_size, config.hidden_size)#50264,768
            # self.embedding.weight= weight
            self.gat_layer=GATLayer(config.hidden_size,config.hidden_size,dropout=0.6,alpha=0.2,concat=False)
            #config.pre_seq_len 新随机初始化 换成bert embedding 希望和此表有关
            #传进来是词表大小 token_id对应词表向量
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, self.prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(self.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )
            #在这个seq改成gnn或者trans前接一个gnn 每个token一个节点 先随机mask矩阵套在gnn
            #
        else:
            self.embedding = torch.nn.Embedding(config.vocab_size, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        #prefix传进来可能是[17053,18,3516,4492]拿embedding
        #用GNN 改成传邻接矩阵
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix) ##就是GNN的x 生成的codeprompt [batch_size,pre_seq_len, num_hidden_layers]
            #但我们还要加edge_index 只保留存在边的缩影   #初始化定死但边可以动！（启发式边可以动）
            #code embedding只是初始化 可以调 但索引不希望定死 就是不知道能不能带权重 就是attention
            prefix_tokens=self.gat_layer(prefix_tokens,matrix)#[batch_size,pre_seq_len, num_hidden_layers=768]
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values#shape:[batch_size,pre_seq_len, num_hidden_layers768 * 2 * hidden_size 12]