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

# https://github.com/microsoft/CodeBERT/blob/master/CodeBERT/code2nl/model.py
class Seq2Seq(nn.Module):
    """
        Build Seqence-to-Sequence.

        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model.
        * `beam_size`- beam size for beam search.
        * `max_length`- max length of target for beam search.
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search.
    """

    def __init__(self, encoder, decoder, config, tokenizer, args, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()

        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id

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
    
    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)

    def forward(self, source_ids=None, source_mask=None, target_ids=None, target_mask=None, args=None):
        if self.args.prefix_tuning:
            attention_mask = source_mask
            position_ids = torch.arange(1,source_ids.size(1)+1, dtype=torch.long, device=source_ids.device).expand_as(source_ids).cuda()
            position_ids = position_ids*attention_mask
            batch_size = attention_mask.shape[0]
            past_key_values = self.get_prompt(batch_size=batch_size) # add
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len,dtype=attention_mask.dtype).to(self.encoder.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            # encoder_source_ids = torch.cat((self.code_prefix_tokens.expand_as(prefix_attention_mask),source_ids),dim=1)
            outputs = self.encoder(
                input_ids=source_ids,
                position_ids=position_ids, 
                attention_mask=attention_mask,
                past_key_values=past_key_values#tuple((i.contiguous() for i in past_key_values)) # [2,16,12,6,64]
                )
        else:
            outputs = self.encoder(source_ids, attention_mask=source_mask)#source_mask size: [batch_size, source_length=256]
        encoder_attention = outputs[-1]#[batch, 256, 768]
        encoder_output = outputs[0].permute([1, 0, 2]).contiguous()#[256, batch, 768]
        if target_ids is not None:
            attn_mask = -1e4 * \
                (1 - self.bias[:target_ids.shape[1], :target_ids.shape[1]])#[128,128] upper triangular=-10000 lower=0 mask upper
            tgt_embeddings = self.encoder.embeddings(
                target_ids).permute([1, 0, 2]).contiguous()#[128, batch, 768]
            out = self.decoder(tgt_embeddings, encoder_output, tgt_mask=attn_mask,
                            memory_key_padding_mask=~source_mask)#[128, batch, 768]
            # memory_key_padding_mask=(1 - source_mask).bool())
            hidden_states = torch.tanh(self.dense(
                out)).permute([1, 0, 2]).contiguous()
            lm_logits = self.lm_head(hidden_states)
            # Shift so that tokens < n predict n
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])
            return loss, loss * active_loss.sum(), active_loss.sum(), encoder_attention
        else:
            # Predict
            preds = []
            zero = torch.cuda.LongTensor(1).fill_(0)
            for i in range(source_ids.shape[0]):
                context = encoder_output[:, i:i + 1]
                context_mask = source_mask[i:i + 1, :]
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.getCurrentState()
                context = context.repeat(1, self.beam_size, 1)
                context_mask = context_mask.repeat(self.beam_size, 1)
                for _ in range(self.max_length):
                    if beam.done():
                        break
                    attn_mask = -1e4 * \
                        (1 - self.bias[:input_ids.shape[1],
                         :input_ids.shape[1]])
                    tgt_embeddings = self.encoder.embeddings(
                        input_ids).permute([1, 0, 2]).contiguous()
                    out = self.decoder(tgt_embeddings, context, tgt_mask=attn_mask,
                                    memory_key_padding_mask=~context_mask)
                    # memory_key_padding_mask=(1 - context_mask).bool())
                    out = torch.tanh(self.dense(out))
                    hidden_states = out.permute(
                        [1, 0, 2]).contiguous()[:, -1, :]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(
                        0, beam.getCurrentOrigin()))
                    input_ids = torch.cat(
                        (input_ids, beam.getCurrentState()), -1)
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:self.beam_size]
                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p in
                        pred]
                preds.append(torch.cat(pred, 0).unsqueeze(0))

            preds = torch.cat(preds, 0)
            return preds, encoder_attention



class Seq2Seq4UniXcoder_e2d(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:
        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model. 
        * `beam_size`- beam size for beam search. 
        * `max_length`- max length of target for beam search. 
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search. 
    """
    def __init__(self, encoder,decoder, config, tokenizer, args, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(Seq2Seq4UniXcoder_e2d, self).__init__()
        self.encoder = encoder
        self.decoder=decoder
        self.config=config
        self.tokenizer = tokenizer
        self.args = args
        self.register_buffer(
            "bias", torch.tril(torch.ones((1024, 1024), dtype=torch.uint8)).view(1,1024, 1024)
        )
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.encoder.embeddings.word_embeddings.weight
        self.lsm = nn.LogSoftmax(dim=-1)
        
        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id
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
    def forward(self, source_ids, target_ids=None):   
        if target_ids is None:
            return self.generate(source_ids)
        
        mask = source_ids.ne(1)[:,None,:]*source_ids.ne(1)[:,:,None]
        #[batch,256,256] case: upper left 70*70(source) true other false

        if self.args.prefix_tuning:
            attention_mask = mask
            position_ids = torch.arange(1,source_ids.size(1)+1, dtype=torch.long, device=source_ids.device).expand_as(source_ids).cuda()
            position_ids = position_ids*attention_mask
            batch_size = attention_mask.shape[0]
            past_key_values = self.get_prompt(batch_size=batch_size) # add
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len,dtype=attention_mask.dtype).to(self.encoder.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            # encoder_source_ids = torch.cat((self.code_prefix_tokens.expand_as(prefix_attention_mask),source_ids),dim=1)
            encoder_output = self.encoder(
                input_ids=source_ids,
                position_ids=position_ids, 
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values#tuple((i.contiguous() for i in past_key_values)) # [2,16,12,6,64]
                )
        else:
            encoder_output = self.encoder(source_ids,attention_mask=mask,use_cache=True)
        ids = torch.cat((source_ids,target_ids),-1)
        #[batch,384] case: total source70 not 1 and target 15 not 1=85
        mask = self.bias[:,source_ids.size(-1):ids.size(-1),:ids.size(-1)].bool()
        #[batch,256:384,0:384]=[batch,128,384],upper left 384*256 true,lower right 128*128 lower triangle 
        mask = mask & ids[:,None,:].ne(1)
        #[batch,128,384] set redundance 1 to false
        out = self.decoder(target_ids,attention_mask=mask,past_key_values=encoder_output.past_key_values).last_hidden_state
        lm_logits = self.lm_head(out)
        # Shift so that tokens < n predict n
        active_loss = target_ids[..., 1:].ne(1).view(-1)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                        shift_labels.view(-1)[active_loss])

        outputs = loss,loss*active_loss.sum(),active_loss.sum()
        return outputs
    
    def generate(self, source_ids):
        mask = source_ids.ne(1)[:,None,:]*source_ids.ne(1)[:,:,None]
        if self.args.prefix_tuning:
            attention_mask = mask
            position_ids = torch.arange(1,source_ids.size(1)+1, dtype=torch.long, device=source_ids.device).expand_as(source_ids).cuda()
            position_ids = position_ids*attention_mask
            batch_size = attention_mask.shape[0]
            past_key_values = self.get_prompt(batch_size=batch_size) # add
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len,dtype=attention_mask.dtype).to(self.encoder.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            # encoder_source_ids = torch.cat((self.code_prefix_tokens.expand_as(prefix_attention_mask),source_ids),dim=1)
            encoder_output = self.encoder(
                input_ids=source_ids,
                position_ids=position_ids, 
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values#tuple((i.contiguous() for i in past_key_values)) # [2,16,12,6,64]
                )
        else:
            encoder_output = self.encoder(source_ids,attention_mask=mask,use_cache=True)      
        preds = []       
        zero = torch.cuda.LongTensor(1).fill_(0)   
        source_len = list(source_ids.ne(1).sum(-1).cpu().numpy())
        for i in range(source_ids.shape[0]):
            context = [[x[i:i+1,:,:source_len[i]].repeat(self.beam_size,1,1,1) for x in y] 
                     for y in encoder_output.past_key_values]
            beam = Beam(self.beam_size,self.sos_id,self.eos_id)
            input_ids = beam.getCurrentState()
            context_ids = source_ids[i:i+1,:source_len[i]].repeat(self.beam_size,1)
            for _ in range(self.max_length): 
                if beam.done():
                    break

                ids = torch.cat((context_ids,input_ids),-1)
                mask = self.bias[:,context_ids.size(-1):ids.size(-1),:ids.size(-1)].bool()
                mask = mask & ids[:,None,:].ne(1)
                out = self.decoder(input_ids,attention_mask=mask,past_key_values=context).last_hidden_state
                hidden_states = out[:,-1,:]
                out = self.lsm(self.lm_head(hidden_states)).data
                beam.advance(out)
                input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                input_ids = torch.cat((input_ids,beam.getCurrentState()),-1)
            hyp = beam.getHyp(beam.getFinal())
            pred = beam.buildTargetTokens(hyp)[:self.beam_size]
            pred = [torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred]
            preds.append(torch.cat(pred,0).unsqueeze(0))

        preds = torch.cat(preds,0)    

        return preds   

class Seq2Seq4UniXcoder_completion(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:
        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model. 
        * `beam_size`- beam size for beam search. 
        * `max_length`- max length of target for beam search. 
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search. 
    """
    def __init__(self, encoder,decoder,config, tokenizer, args,beam_size=None,max_length=None,sos_id=None,eos_id=None):
        super(Seq2Seq4UniXcoder_completion, self).__init__()
        self.encoder = encoder
        self.decoder=decoder
        self.config=config
        self.tokenizer = tokenizer
        self.args = args
        self.register_buffer(
            "bias", torch.tril(torch.ones((1024, 1024), dtype=torch.uint8)).view(1,1024, 1024)
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight=self.encoder.embeddings.word_embeddings.weight
        self.lsm = nn.LogSoftmax(dim=-1)
        
        self.beam_size=beam_size
        self.max_length=max_length
        self.sos_id=sos_id
        self.eos_id=eos_id
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

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)        
        
    def forward(self, source_ids,train=False): 
        max_length = source_ids.ne(1).sum(-1).max()
        source_ids = source_ids[:,:max_length]        
        if train:  
            length = source_ids.size(-1)
            if self.args.prefix_tuning:
                attention_mask = self.bias[:,:length,:length]
                position_ids = torch.arange(1,source_ids.size(1)+1, dtype=torch.long, device=source_ids.device).expand_as(source_ids).cuda()
                position_ids = position_ids*attention_mask
                batch_size = attention_mask.shape[0]
                past_key_values = self.get_prompt(batch_size=batch_size) # add
                prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len,dtype=attention_mask.dtype).to(self.encoder.device)
                attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
                # encoder_source_ids = torch.cat((self.code_prefix_tokens.expand_as(prefix_attention_mask),source_ids),dim=1)
                out = self.decoder(
                    input_ids=source_ids,
                    position_ids=position_ids, 
                    attention_mask=attention_mask,
                    use_cache=True,
                    past_key_values=past_key_values#tuple((i.contiguous() for i in past_key_values)) # [2,16,12,6,64]
                    ).last_hidden_state
            else:
                out = self.decoder(source_ids,attention_mask=self.bias[:,:length,:length],use_cache=True).last_hidden_state
            lm_logits = self.lm_head(out)
            # Shift so that tokens < n predict n
            active_loss = source_ids[..., 1:].ne(1).view(-1)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = source_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])

            outputs = loss,loss*active_loss.sum(),active_loss.sum()
            return outputs
        else:
            #Predict 
            preds=[]       
            zero=torch.cuda.LongTensor(1).fill_(0)   
            source_len = list(source_ids.ne(1).sum(-1).cpu().numpy())
            length = source_ids.size(-1)
            if self.args.prefix_tuning:
                attention_mask = self.bias[:,:length,:length]
                position_ids = torch.arange(1,source_ids.size(1)+1, dtype=torch.long, device=source_ids.device).expand_as(source_ids).cuda()
                position_ids = position_ids*attention_mask
                batch_size = attention_mask.shape[0]
                past_key_values = self.get_prompt(batch_size=batch_size) # add
                prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len,dtype=attention_mask.dtype).to(self.encoder.device)
                attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
                # encoder_source_ids = torch.cat((self.code_prefix_tokens.expand_as(prefix_attention_mask),source_ids),dim=1)
                encoder_output = self.decoder(
                    input_ids=source_ids,
                    position_ids=position_ids, 
                    attention_mask=attention_mask,
                    past_key_values=past_key_values#tuple((i.contiguous() for i in past_key_values)) # [2,16,12,6,64]
                    )
            else:
                encoder_output = self.decoder(source_ids,attention_mask=self.bias[:,:length,:length])
            for i in range(source_ids.shape[0]):
                context=[[x[i:i+1,:,:source_len[i]].repeat(self.beam_size,1,1,1) for x in y] 
                         for y in encoder_output.past_key_values]
                beam = Beam(self.beam_size,self.sos_id,self.eos_id)
                input_ids=beam.getCurrentState()
                context_ids = source_ids[i:i+1,:source_len[i]].repeat(self.beam_size,1)
                out = encoder_output.last_hidden_state[i:i+1,:source_len[i]].repeat(self.beam_size,1,1)
                for _ in range(self.max_length): 
                    if beam.done():
                        break
                    if _ == 0: 
                        hidden_states=out[:,-1,:]
                        out = self.lsm(self.lm_head(hidden_states)).data
                        beam.advance(out)
                        input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                        input_ids=beam.getCurrentState()
                    else:
                        length = context_ids.size(-1)+input_ids.size(-1)
                        out = self.decoder(input_ids,attention_mask=self.bias[:,context_ids.size(-1):length,:length],
                                           past_key_values=context).last_hidden_state
                        hidden_states=out[:,-1,:]
                        out = self.lsm(self.lm_head(hidden_states)).data
                        beam.advance(out)
                        input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                        input_ids=torch.cat((input_ids,beam.getCurrentState()),-1)
                hyp= beam.getHyp(beam.getFinal())
                pred=beam.buildTargetTokens(hyp)[:self.beam_size]
                pred=[torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred]
                preds.append(torch.cat(pred,0).unsqueeze(0))
                
            preds=torch.cat(preds,0)    

            return preds   

class Seq2Seq4UniXcoder_generation(nn.Module):
    """
        Build Seqence-to-Sequence.
        
        Parameters:
        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model. 
        * `beam_size`- beam size for beam search. 
        * `max_length`- max length of target for beam search. 
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search. 
    """
    def __init__(self, encoder,decoder, config, tokenizer, args, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(Seq2Seq4UniXcoder_generation, self).__init__()
        self.encoder = encoder
        self.decoder=decoder
        self.config=config
        self.tokenizer = tokenizer
        self.args = args
        self.register_buffer(
            "bias", torch.tril(torch.ones((1024, 1024), dtype=torch.uint8)).view(1,1024, 1024)
        )
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.encoder.embeddings.word_embeddings.weight
        self.lsm = nn.LogSoftmax(dim=-1)
        
        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id
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

    def forward(self, source_ids, target_ids=None):   
        if target_ids is None:
            return self.generate(source_ids)
        
        mask = source_ids.ne(1)[:,None,:]*source_ids.ne(1)[:,:,None]
        if self.args.prefix_tuning:
            attention_mask = mask
            position_ids = torch.arange(1,source_ids.size(1)+1, dtype=torch.long, device=source_ids.device).expand_as(source_ids).cuda()
            position_ids = position_ids*attention_mask
            batch_size = attention_mask.shape[0]
            past_key_values = self.get_prompt(batch_size=batch_size) # add
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len,dtype=attention_mask.dtype).to(self.encoder.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            # encoder_source_ids = torch.cat((self.code_prefix_tokens.expand_as(prefix_attention_mask),source_ids),dim=1)
            encoder_output = self.encoder(
                input_ids=source_ids,
                position_ids=position_ids, 
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values#tuple((i.contiguous() for i in past_key_values)) # [2,16,12,6,64]
                )
        else:
            encoder_output = self.encoder(source_ids,attention_mask=mask,use_cache=True)
        ids = torch.cat((source_ids,target_ids),-1)
        mask = self.bias[:,source_ids.size(-1):ids.size(-1),:ids.size(-1)].bool()
        mask = mask & ids[:,None,:].ne(1)

        out = self.decoder(target_ids,attention_mask=mask,past_key_values=encoder_output.past_key_values).last_hidden_state
        lm_logits = self.lm_head(out)
        # Shift so that tokens < n predict n
        active_loss = target_ids[..., 1:].ne(1).view(-1)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                        shift_labels.view(-1)[active_loss])

        outputs = loss,loss*active_loss.sum(),active_loss.sum()
        return outputs
    
    def generate(self, source_ids):
        mask = source_ids.ne(1)[:,None,:]*source_ids.ne(1)[:,:,None]
        if self.args.prefix_tuning:
            attention_mask = mask
            position_ids = torch.arange(1,source_ids.size(1)+1, dtype=torch.long, device=source_ids.device).expand_as(source_ids).cuda()
            position_ids = position_ids*attention_mask
            batch_size = attention_mask.shape[0]
            past_key_values = self.get_prompt(batch_size=batch_size) # add
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len,dtype=attention_mask.dtype).to(self.encoder.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            # encoder_source_ids = torch.cat((self.code_prefix_tokens.expand_as(prefix_attention_mask),source_ids),dim=1)
            encoder_output = self.encoder(
                input_ids=source_ids,
                position_ids=position_ids, 
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values#tuple((i.contiguous() for i in past_key_values)) # [2,16,12,6,64]
                )
        else:
            encoder_output = self.encoder(source_ids,attention_mask=mask,use_cache=True)       
        preds = []       
        zero = torch.cuda.LongTensor(1).fill_(0)   
        source_len = list(source_ids.ne(1).sum(-1).cpu().numpy())
        for i in range(source_ids.shape[0]):
            context = [[x[i:i+1,:,:source_len[i]].repeat(self.beam_size,1,1,1) for x in y] 
                     for y in encoder_output.past_key_values]
            beam = Beam(self.beam_size,self.sos_id,self.eos_id)
            input_ids = beam.getCurrentState()
            context_ids = source_ids[i:i+1,:source_len[i]].repeat(self.beam_size,1)
            for _ in range(self.max_length): 
                if beam.done():
                    break

                ids = torch.cat((context_ids,input_ids),-1)
                mask = self.bias[:,context_ids.size(-1):ids.size(-1),:ids.size(-1)].bool()
                mask = mask & ids[:,None,:].ne(1)
                out = self.decoder(input_ids,attention_mask=mask,past_key_values=context).last_hidden_state
                hidden_states = out[:,-1,:]
                out = self.lsm(self.lm_head(hidden_states)).data
                beam.advance(out)
                input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                input_ids = torch.cat((input_ids,beam.getCurrentState()),-1)
            hyp = beam.getHyp(beam.getFinal())
            pred = beam.buildTargetTokens(hyp)[:self.beam_size] #len:10
            pred = [torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred]
            preds.append(torch.cat(pred,0).unsqueeze(0))

        preds = torch.cat(preds,0)    

        return preds   



class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                           .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size - len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
