from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import PLBartForConditionalGeneration,PLBartConfig
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from utils import load_prefix_code
from code_prefix import CodePrefix

# Copied from transformers.models.mbart.modeling_mbart.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int):
    """
    Shift input ids one token to the right, and wrap the last non pad token (the <LID> token) Note that MBart does not
    have a single `decoder_start_token_id` in contrast to other Bart-like models.
    """
    prev_output_tokens = input_ids.clone()

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    prev_output_tokens.masked_fill_(prev_output_tokens == -100, pad_token_id)

    index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    decoder_start_tokens = prev_output_tokens.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].clone()
    prev_output_tokens[:, 0] = decoder_start_tokens

    return prev_output_tokens

class PLBartForConditionalGeneration_Prefix(PLBartForConditionalGeneration):

    def __init__(self, config: PLBartConfig,tokenizer,args):
        PLBartForConditionalGeneration.__init__(self,config)
        self.tokenizer = tokenizer
        self.args = args
        if self.args.prefix_tuning:
            if self.args.model_name in ['t5','codet5']:
                embeddings_weight = self.shared.weight
            elif self.args.model_name in ['bart','plbart']:
                embeddings_weight = self.model.shared.weight
            else:
                embeddings_weight = self.decoder.embeddings.word_embeddings.weight
            if self.args.fix_model_param:
                for param in self.model.encoder.parameters():
                    param.requires_grad = False
                for param in self.model.decoder.parameters():
                    param.requires_grad = False
            self.code_prefix_tokens, self.code_prefix_matrix = load_prefix_code(self.args,self.tokenizer)
            self.code_prefix_tokens = torch.tensor(self.code_prefix_tokens, dtype=torch.long).cuda()
            self.code_prefix_matrix = torch.tensor(self.code_prefix_matrix, dtype=torch.long).cuda()
            self.pre_seq_len = self.args.max_source_length

            self.n_layer = config.num_hidden_layers
            self.n_head = config.num_attention_heads
            self.n_embd = config.hidden_size // config.num_attention_heads
            # add prefix encoder
            self.code_prefix = CodePrefix(self.config, embeddings_weight,self.args)
            if self.args.model_name in ['t5','codet5']:
                self.dropout = torch.nn.Dropout(config.dropout_rate)
            elif self.args.model_name in ['bart','plbart']:
                self.dropout = torch.nn.Dropout(config.dropout)
            else:
                self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def get_prompt(self, batch_size, is_generate=False):
        if is_generate:
            batch_size = batch_size // self.args.beam_size
        
        code_prefix_tokens = self.code_prefix_tokens.unsqueeze(0).expand(batch_size, -1)
        code_prefix_matrix = self.code_prefix_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        past_key_values = self.code_prefix(code_prefix_tokens, code_prefix_matrix)
        # bsz, seqlen, _ = past_key_values.shape

        past_key_values = past_key_values.view(
            batch_size, #1 (8)
            self.pre_seq_len, #3 (seq_len)512
            self.n_layer * 2, #0 (2)
            self.n_head, #2 (12)
            self.n_embd #4 (64)
        ).contiguous()#注意这里加了contiguous()!

        if is_generate:
            past_key_values = past_key_values.repeat(self.args.beam_size,1,1,1,1)

        past_key_values = self.dropout(past_key_values)
        if self.args.model_name in ['t5','codet5']:
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).contiguous().split(4)
        else:
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).contiguous().split(2)
        return past_key_values
    
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.LongTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds=None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)

        if self.args.prefix_tuning:
            # decoder_attention_mask = source_mask
            # position_ids = torch.arange(1,source_ids.size(1)+1, dtype=torch.long, device=source_ids.device).expand_as(source_ids).cuda()
            # position_ids = position_ids*decoder_attention_mask
            batch_size = attention_mask.shape[0]
            if decoder_attention_mask is not None:# refers to model.generate()
                past_key_values = self.get_prompt(batch_size=batch_size) # add
                prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len,dtype=decoder_attention_mask.dtype).to(self.model.decoder.device)
                decoder_attention_mask = torch.cat((prefix_attention_mask, decoder_attention_mask), dim=1)
            else:
                past_key_values = self.get_prompt(batch_size=batch_size,is_generate=True) # add
            # encoder_source_ids = torch.cat((self.code_prefix_tokens.expand_as(prefix_attention_mask),source_ids),dim=1)
            # outputs = self.encoder(
            #     input_ids=source_ids,
            #     # position_ids=position_ids, 
            #     attention_mask=decoder_attention_mask,
            #     past_key_values=past_key_values#tuple((i.contiguous() for i in past_key_values)) # [2,16,12,6,64]
            #     )
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
