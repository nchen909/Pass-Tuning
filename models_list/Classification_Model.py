        
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig, T5ForConditionalGeneration, BartForConditionalGeneration, AutoModelForSeq2SeqLM, RobertaConfig, RobertaModel, RobertaTokenizer
from transformers import PLBartForConditionalGeneration
import logging
import sys

from utils import get_graph_metadata


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, x):
        x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Model4UniXcoder(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model4UniXcoder, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
    
        
    def forward(self, input_ids=None,labels=None): 
        input_ids = input_ids.view(-1,self.args.max_source_length)
        outputs = self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
        outputs = (outputs * input_ids.ne(1)[:,:,None]).sum(1)/input_ids.ne(1).sum(1)[:,None]
        outputs = outputs.reshape(-1,2,outputs.size(-1))
        outputs = torch.nn.functional.normalize(outputs, p=2, dim=-1)
        cos_sim = (outputs[:,0]*outputs[:,1]).sum(-1)

        if labels is not None:
            loss = ((cos_sim-labels.float())**2).mean()
            return loss,cos_sim
        else:
            return cos_sim

class CloneModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(CloneModel, self).__init__()
        # checkpoint = os.path.join(args.huggingface_locals, MODEL_LOCALS[args.model_name])
        # config = AutoConfig.from_pretrained(checkpoint)
        # tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
        if self.args.prefix_tuning:
            if self.args.model_name in ['t5','codet5']:
                embeddings_weight = self.encoder.shared.weight
            elif self.args.model_name in ['bart','plbart']:
                embeddings_weight = self.encoder.model.shared.weight
            else:
                embeddings_weight = self.encoder.embeddings.word_embeddings.weight
            if self.args.fix_model_param:
                for param in self.encoder.parameters():
                    param.requires_grad = False
            # load retrieved prefix code tokens
            # self.code_prefix_tokens = torch.Tensor([10, 11, 12, 13, 14, 15]).long().cuda()
            # self.code_prefix_matrix = torch.ones(6, 6).long().cuda()

            # if self.args.model_name in ['t5','codet5','bart','plbart']:
            #     # self.code_prefix_tokens = torch.arange(10,self.args.max_target_length+10, dtype=torch.long).cuda()
            #     self.code_prefix_tokens = torch.tensor(get_graph_metadata(self.args,self.tokenizer), dtype=torch.long).cuda()
            # else:
            #     # self.code_prefix_tokens = torch.arange(10,self.args.max_source_length+10, dtype=torch.long).cuda()
            #     self.code_prefix_tokens = torch.tensor(get_graph_metadata(self.args,self.tokenizer), dtype=torch.long).cuda()
            # #now len:args.max_source_length=max_target_length=code_prefix_tokens for t5&bart
            # self.pre_seq_len = self.args.max_source_length # 5
            # self.code_prefix_matrix = torch.ones(len(self.code_prefix_tokens), len(self.code_prefix_tokens)).long().cuda()

            self.code_prefix_tokens, self.code_prefix_matrix = get_graph_metadata(self.args,self.tokenizer)
            self.code_prefix_tokens = torch.tensor(self.code_prefix_tokens, dtype=torch.long).cuda()
            self.code_prefix_matrix = torch.tensor(self.code_prefix_matrix, dtype=torch.long).cuda()
            self.pre_seq_len = self.args.max_source_length

            self.n_layer = config.num_hidden_layers
            self.n_head = config.num_attention_heads
            self.n_embd = config.hidden_size // config.num_attention_heads
            # add prefix encoder

            if self.args.prefix_tuning == 'pass_tuning':
                from GAT_prefix import CodeGraphPrefix
                self.code_prefix = CodeGraphPrefix(self.config, embeddings_weight,self.args)
            elif self.args.prefix_tuning == 'GCN_tuning':
                from GCN_prefix import CodeGraphPrefix
                self.code_prefix = CodeGraphPrefix(self.config, embeddings_weight,self.args)
            
            if self.args.model_name in ['t5','codet5']:
                self.dropout = torch.nn.Dropout(config.dropout_rate)
            elif self.args.model_name in ['bart','plbart']:
                self.dropout = torch.nn.Dropout(config.dropout)
            else:
                self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def get_prompt(self, batch_size):
        code_prefix_tokens = self.code_prefix_tokens.unsqueeze(0).expand(batch_size, -1)
        code_prefix_matrix = self.code_prefix_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        if self.args.adjcency_mode=='fully-connected':
            code_prefix_matrix = torch.where(code_prefix_matrix >0,  torch.ones_like(code_prefix_matrix), torch.zeros_like(code_prefix_matrix))
        elif self.args.adjcency_mode=='sast':
            code_prefix_matrix = torch.where(code_prefix_matrix ==1, torch.ones_like(code_prefix_matrix), torch.zeros_like(code_prefix_matrix)) 
        
        past_key_values = self.code_prefix(code_prefix_tokens, code_prefix_matrix)
        # bsz, seqlen, _ = past_key_values.shape

        past_key_values = past_key_values.view(
            batch_size, #1 (8)
            self.pre_seq_len, #3 (seq_len)512
            self.n_layer * 2, #0 (2)
            self.n_head, #2 (12)
            self.n_embd #4 (64)
        ).contiguous()#注意这里加了contiguous()!

        past_key_values = self.dropout(past_key_values)
        if self.args.model_name in ['t5','codet5']:
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).contiguous().split(4)
        else:
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).contiguous().split(2)
        return past_key_values
    
    def get_t5_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        if 0 and self.args.prefix_tuning: # use overwritten modeling_t5
            # batch_size = attention_mask.shape[0]
            # past_key_values = self.get_prompt(batch_size=batch_size) # add
            # prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len,dtype=attention_mask.dtype).to(self.encoder.device)
            # encoder_attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            # outputs = self.encoder(
            #     input_ids=source_ids, # 512
            #     postion_ids=position_ids,
            #     attention_mask=attention_mask, # 512
            #     labels=source_ids, 
            #     decoder_attention_mask=encoder_attention_mask, # 518
            #     output_hidden_states=True, 
            #     past_key_values=past_key_values # size
            #     )
            batch_size = attention_mask.shape[0]
            past_key_values = self.get_prompt(batch_size=batch_size) # add
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len,dtype=attention_mask.dtype).to(self.encoder.device)
            prefix_attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            # encoder_source_ids = torch.cat((self.code_prefix_tokens.expand_as(prefix_attention_mask),source_ids),dim=1)
            outputs = self.encoder(#encoder is T5ForConditionalGeneration(T5Model)!是T5模型不是里面的encoder
                input_ids=source_ids, 
                attention_mask=attention_mask,
                labels=source_ids, 
                decoder_attention_mask=prefix_attention_mask, 
                output_hidden_states=True,
                past_key_values=past_key_values#tuple((i.contiguous() for i in past_key_values)) # [2,16,12,6,64]
                )#只训了encoder部分 吐出来encoder_hidden_states decoder没输入
        else:
            outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                                    labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def get_bart_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        # position_ids = torch.arange(1,source_ids.size(1)+1, dtype=torch.long, device=source_ids.device).expand_as(source_ids).cuda()
        # position_ids = position_ids*attention_mask
        if 0 and self.args.prefix_tuning:# use overwritten modeling_t5
            batch_size = attention_mask.shape[0]
            past_key_values = self.get_prompt(batch_size=batch_size) # add
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len,dtype=attention_mask.dtype).to(self.encoder.device)
            prefix_attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            # encoder_source_ids = torch.cat((self.code_prefix_tokens.expand_as(prefix_attention_mask),source_ids),dim=1)
            outputs = self.encoder(
                input_ids=source_ids, 
                attention_mask=attention_mask,
                labels=source_ids, 
                decoder_attention_mask=prefix_attention_mask, 
                output_hidden_states=True,
                past_key_values=past_key_values#tuple((i.contiguous() for i in past_key_values)) # [2,16,12,6,64]
                )
        else:
            outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                                    labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def get_roberta_vec(self, source_ids):
        # attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        position_ids = torch.arange(1,source_ids.size(1)+1, dtype=torch.long, device=source_ids.device).expand_as(source_ids).cuda()
        position_ids = position_ids*attention_mask
        if self.args.prefix_tuning:
            batch_size = attention_mask.shape[0] #batch_size   attention_mask.shape (batch_size,512)
            past_key_values = self.get_prompt(batch_size=batch_size) # batch_size*[2,12,12,nodenum,64]
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len,dtype=attention_mask.dtype).cuda()#[batch_size,nodenum]
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            vec = self.encoder(
                input_ids=source_ids, 
                position_ids=position_ids,
                attention_mask=attention_mask, 
                past_key_values=past_key_values # add
                )[0][:, 0, :]
        else:
            vec = self.encoder(input_ids=source_ids, attention_mask=attention_mask)[0][:, 0, :]
        return vec

    def get_unixcoder_vec(self, source_ids):
        attention_mask = source_ids.ne(1)
        position_ids = torch.arange(1,source_ids.size(1)+1, dtype=torch.long, device=source_ids.device).expand_as(source_ids).cuda()
        position_ids = position_ids*attention_mask
        if self.args.prefix_tuning:
            batch_size = attention_mask.shape[0]
            past_key_values = self.get_prompt(batch_size=batch_size) # add
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len,dtype=attention_mask.dtype).to(self.encoder.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            outputs = self.encoder(
                source_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values # add
            )[0]
        else:
            outputs = self.encoder(source_ids,attention_mask=attention_mask)[0]#shape:batch_size*max_len512*hidden_size768
        
        outputs = (outputs * source_ids.ne(1)[:,:,None]).sum(1)/source_ids.ne(1).sum(1)[:,None]#shape:batch_size*hidden_size
        outputs = outputs.reshape(-1,2,outputs.size(-1))#shape:batch_size/2 *2*hidden_size
        outputs = torch.nn.functional.normalize(outputs, p=2, dim=-1)
        cos_sim = (outputs[:,0]*outputs[:,1]).sum(-1)

        return cos_sim #cos_sim, labels
        
    def forward(self, source_ids=None, labels=None):
        source_ids = source_ids.view(-1, self.args.max_source_length)#[batch*2,512]

        if self.args.model_name in ['t5','codet5']:
            vec = self.get_t5_vec(source_ids)#[batch*2,768]
            logits = self.classifier(vec)#[batch,2]
            prob = nn.functional.softmax(logits)
        elif self.args.model_name in ['bart','plbart']:
            vec = self.get_bart_vec(source_ids)
            logits = self.classifier(vec)
            prob = nn.functional.softmax(logits)
        elif self.args.model_name in ['roberta', 'codebert', 'graphcodebert']:
            vec = self.get_roberta_vec(source_ids)
            logits = self.classifier(vec)
            prob = nn.functional.softmax(logits)
        elif self.args.model_name in ['unixcoder']:
            logits = self.get_unixcoder_vec(source_ids)
            prob = logits #=cos_sim

        if labels is not None:
            if self.args.model_name not in ['unixcoder']:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob
            else:
                loss = ((logits-labels.float())**2).mean()
                return loss, prob
        else:
            return prob


class DefectModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(DefectModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.args = args
        if self.args.prefix_tuning:
            if self.args.model_name in ['t5','codet5']:
                embeddings_weight = self.encoder.shared.weight
            elif self.args.model_name in ['bart','plbart']:
                embeddings_weight = self.encoder.model.shared.weight
            else:
                embeddings_weight = self.encoder.embeddings.word_embeddings.weight
            if self.args.fix_model_param:
                for param in self.encoder.parameters():
                    param.requires_grad = False
            self.code_prefix_tokens, self.code_prefix_matrix = get_graph_metadata(self.args,self.tokenizer)
            self.code_prefix_tokens = torch.tensor(self.code_prefix_tokens, dtype=torch.long).cuda()
            self.code_prefix_matrix = torch.tensor(self.code_prefix_matrix, dtype=torch.long).cuda()
            self.pre_seq_len = self.args.max_source_length

            self.n_layer = config.num_hidden_layers
            self.n_head = config.num_attention_heads
            self.n_embd = config.hidden_size // config.num_attention_heads
            # add prefix encoder
            if self.args.prefix_tuning == 'pass_tuning':
                from GAT_prefix import CodeGraphPrefix
                self.code_prefix = CodeGraphPrefix(self.config, embeddings_weight,self.args)
            elif self.args.prefix_tuning == 'GCN_tuning':
                from GCN_prefix import CodeGraphPrefix
                self.code_prefix = CodeGraphPrefix(self.config, embeddings_weight,self.args)
            
            if self.args.model_name in ['t5','codet5']:
                self.dropout = torch.nn.Dropout(config.dropout_rate)
            elif self.args.model_name in ['bart','plbart']:
                self.dropout = torch.nn.Dropout(config.dropout)
            else:
                self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
            
    def get_prompt(self, batch_size):
        code_prefix_tokens = self.code_prefix_tokens.unsqueeze(0).expand(batch_size, -1)
        code_prefix_matrix = self.code_prefix_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        if self.args.adjcency_mode=='fully-connected':
            code_prefix_matrix = torch.where(code_prefix_matrix >0,  torch.ones_like(code_prefix_matrix), torch.zeros_like(code_prefix_matrix))
        elif self.args.adjcency_mode=='sast':
            code_prefix_matrix = torch.where(code_prefix_matrix ==1, torch.ones_like(code_prefix_matrix), torch.zeros_like(code_prefix_matrix)) 
        
        past_key_values = self.code_prefix(code_prefix_tokens, code_prefix_matrix)
        # bsz, seqlen, _ = past_key_values.shape

        past_key_values = past_key_values.view(
            batch_size, #1 (8)
            self.pre_seq_len, #3 (seq_len)512
            self.n_layer * 2, #0 (2)
            self.n_head, #2 (12)
            self.n_embd #4 (64)
        ).contiguous()#注意这里加了contiguous()!

        past_key_values = self.dropout(past_key_values)
        if self.args.model_name in ['t5','codet5']:
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).contiguous().split(4)
        else:
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).contiguous().split(2)
        return past_key_values

    def get_origin_prompt(self, batch_size):
        code_prefix_tokens = self.code_prefix_tokens.unsqueeze(0).expand(batch_size, -1)
        code_prefix_matrix = self.code_prefix_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        if self.args.adjcency_mode=='fully-connected':
            code_prefix_matrix = torch.where(code_prefix_matrix >0,  torch.ones_like(code_prefix_matrix), torch.zeros_like(code_prefix_matrix))
        elif self.args.adjcency_mode=='sast':
            code_prefix_matrix = torch.where(code_prefix_matrix ==1, torch.ones_like(code_prefix_matrix), torch.zeros_like(code_prefix_matrix)) 
        
        past_key_values = self.code_prefix(code_prefix_tokens, code_prefix_matrix)
        # bsz, seqlen, _ = past_key_values.shape

        past_key_values = past_key_values.view(
            batch_size, #1 (8)
            self.pre_seq_len, #3 (seq_len)512
            self.n_layer * 2, #0 (2)
            self.n_head, #2 (12)
            self.n_embd #4 (64)
        ).contiguous()#注意这里加了contiguous()!

        past_key_values = self.dropout(past_key_values)
        if self.args.model_name in ['t5','codet5']:
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).contiguous().split(4)
        else:
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).contiguous().split(2)
        return past_key_values

    def get_t5_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        if 0 and self.args.prefix_tuning:# use overwritten modeling_t5
            batch_size = attention_mask.shape[0]
            past_key_values = self.get_prompt(batch_size=batch_size) # add
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len,dtype=attention_mask.dtype).to(self.encoder.device)
            prefix_attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            # encoder_source_ids = torch.cat((self.code_prefix_tokens.expand_as(prefix_attention_mask),source_ids),dim=1)
            outputs = self.encoder(
                input_ids=source_ids, 
                attention_mask=attention_mask,
                labels=source_ids, 
                decoder_attention_mask=prefix_attention_mask, 
                output_hidden_states=True,
                past_key_values=past_key_values#tuple((i.contiguous() for i in past_key_values)) # [2,16,12,6,64]
                )
        else:
            outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                                    labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        # print(outputs)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def get_bart_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        if 0 and self.args.prefix_tuning:# use overwritten modeling_t5
            batch_size = attention_mask.shape[0]
            past_key_values = self.get_prompt(batch_size=batch_size) # add
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len,dtype=attention_mask.dtype).to(self.encoder.device)
            prefix_attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            # encoder_source_ids = torch.cat((self.code_prefix_tokens.expand_as(prefix_attention_mask),source_ids),dim=1)
            outputs = self.encoder(
                input_ids=source_ids, 
                attention_mask=attention_mask,
                labels=source_ids, 
                decoder_attention_mask=prefix_attention_mask, 
                output_hidden_states=True,
                past_key_values=past_key_values#tuple((i.contiguous() for i in past_key_values)) # [2,16,12,6,64]
                )
        else:
            outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                                    labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def get_roberta_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        position_ids = torch.arange(1,source_ids.size(1)+1, dtype=torch.long, device=source_ids.device).expand_as(source_ids).cuda()
        position_ids = position_ids*attention_mask
        if self.args.prefix_tuning:
            batch_size = attention_mask.shape[0] #batch_size   attention_mask.shape (batch_size,512)
            past_key_values = self.get_prompt(batch_size=batch_size) # batch_size*[2,12,12,nodenum,64]
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len,dtype=attention_mask.dtype).cuda()#[batch_size,nodenum]
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            vec = self.encoder(
                input_ids=source_ids, 
                position_ids=position_ids,
                attention_mask=attention_mask, 
                past_key_values=past_key_values # add
                )[0][:, 0, :]
        else:
            vec = self.encoder(input_ids=source_ids, attention_mask=attention_mask)[0][:, 0, :]
        return vec

    def get_unixcoder_vec(self, source_ids):
        attention_mask = source_ids.ne(1)
        position_ids = torch.arange(1,source_ids.size(1)+1, dtype=torch.long, device=source_ids.device).expand_as(source_ids).cuda()
        position_ids = position_ids*attention_mask
        if self.args.prefix_tuning:
            batch_size = attention_mask.shape[0]
            past_key_values = self.get_prompt(batch_size=batch_size) # add
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len,dtype=attention_mask.dtype).to(self.encoder.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            outputs = self.encoder(
                source_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values # add
            )[0]
        else:
            outputs = self.encoder(source_ids,attention_mask=attention_mask)[0]#shape:batch_size*max_len512*hidden_size768
        outputs = (outputs * source_ids.ne(1)[:,:,None]).sum(1)/attention_mask.sum(1)[:,None]#shape:batch_size*hidden_size
        # outputs = outputs.reshape(-1,2,outputs.size(-1))
        # outputs = torch.nn.functional.normalize(outputs, p=2, dim=-1)
        # cos_sim = (outputs[:,0]*outputs[:,1]).sum(-1)

        # return cos_sim #cos_sim, labels
        outputs = self.classifier(outputs)
        return outputs
    
    def forward(self, source_ids=None, labels=None):
        source_ids = source_ids.view(-1, self.args.max_source_length)

        if self.args.model_name in ['t5','codet5']:
            vec = self.get_t5_vec(source_ids)#[batch_size,hidden_size]
            logits = self.classifier(vec)#[batch_size,2]
            prob = nn.functional.softmax(logits)#[batch_size,2]
        elif self.args.model_name in ['bart','plbart']:
            vec = self.get_bart_vec(source_ids)
            logits = self.classifier(vec)
            prob = nn.functional.softmax(logits)
        elif self.args.model_name in ['roberta', 'codebert', 'graphcodebert']:
            vec = self.get_roberta_vec(source_ids)
            logits = self.classifier(vec)
            prob = nn.functional.softmax(logits)
        elif self.args.model_name in ['unixcoder']:
            logits = self.get_unixcoder_vec(source_ids)
            prob = logits #=cos_sim


        if labels is not None:
            # if self.args.model_name not in ['unixcoder']:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)#[batchsize,2] [batchsize]
            return loss, prob
            # else:
            #     loss = ((logits-labels.float())**2).mean()
            #     return loss, prob
        else:
            return prob

