from transformers import T5ForConditionalGeneration,T5Config
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from torch.nn import CrossEntropyLoss
import torch
from torch import nn
from typing import Optional, Tuple, Union
from utils import load_prefix_code
from code_prefix import CodePrefix
import logging
logger = logging.getLogger(__name__)
class T5ForConditionalGeneration_Prefix(T5ForConditionalGeneration):
    def __init__(self,config: T5Config,tokenizer,args):
        T5ForConditionalGeneration.__init__(self,config)
        self.tokenizer = tokenizer
        self.args = args
        if self.args.prefix_tuning:
            if self.args.model_name in ['t5','codet5']:
                embeddings_weight = self.shared.weight
            elif self.args.model_name in ['bart','plbart']:
                embeddings_weight = self.shared.weight
            else:
                embeddings_weight = self.decoder.embeddings.word_embeddings.weight
            if self.args.fix_model_param:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                for param in self.decoder.parameters():
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
    def get_prompt(self, batch_size):
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

        past_key_values = self.dropout(past_key_values)
        if self.args.model_name in ['t5','codet5']:
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).contiguous().split(4)
        else:
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).contiguous().split(2)
        return past_key_values
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                # warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        if self.args.prefix_tuning:

            # decoder_attention_mask = source_mask
            # position_ids = torch.arange(1,source_ids.size(1)+1, dtype=torch.long, device=source_ids.device).expand_as(source_ids).cuda()
            # position_ids = position_ids*decoder_attention_mask
            
            # if True:#decoder_attention_mask is None:
            #     print(self.args.max_source_length)
            #     print(self.args.max_target_length)
            #     print(self.pre_seq_len)
            #     print(attention_mask.shape)
            #     print(self.n_head)
            #     print(self.args.batch_size)
            #     print(decoder_attention_mask.shape[0])
            batch_size = attention_mask.shape[0]#decoder_attention_mask.shape[0]
            past_key_values = self.get_prompt(batch_size=batch_size) # add
            if decoder_attention_mask is not None:
                prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len,dtype=attention_mask.dtype).to(self.decoder.device)
                decoder_attention_mask = torch.cat((prefix_attention_mask, decoder_attention_mask), dim=1)
            # encoder_source_ids = torch.cat((self.code_prefix_tokens.expand_as(prefix_attention_mask),source_ids),dim=1)
            # outputs = self.encoder(
            #     input_ids=source_ids,
            #     # position_ids=position_ids, 
            #     attention_mask=decoder_attention_mask,
            #     past_key_values=past_key_values#tuple((i.contiguous() for i in past_key_values)) # [2,16,12,6,64]
            #     )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
