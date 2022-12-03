import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig, T5ForConditionalGeneration, BartForConditionalGeneration, AutoModelForSeq2SeqLM, RobertaConfig, RobertaModel, RobertaTokenizer
from transformers import PLBartForConditionalGeneration
import logging
import sys
from code_prefix import CodePrefix
from utils import load_prefix_code
from models_list.T5ForConditionalGeneration_Prefix import T5ForConditionalGeneration_Prefix
from models_list.T5ForConditionalGeneration_Prefix_2 import T5ForConditionalGeneration_Prefix_2
from models_list.PLBartForConditionalGeneration_Prefix import PLBartForConditionalGeneration_Prefix
from models_list.Seq2Seq import Seq2Seq, Seq2Seq4UniXcoder_completion, Seq2Seq4UniXcoder_generation, Seq2Seq4UniXcoder_e2d
from models_list.Classification_Model import Model4UniXcoder, CloneModel, DefectModel
#import codecs
#sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
logger = logging.getLogger(__name__)

MODEL_CHECKPOINTS = {'roberta': 'roberta-base',
                     'codebert': 'microsoft/codebert-base',
                     'graphcodebert': 'microsoft/graphcodebert-base',
                     't5': 't5-base',
                     'codet5': 'Salesforce/codet5-base',
                     'bart': 'facebook/bart-base',
                     'plbart': 'uclanlp/plbart-base',
                     'unixcoder':'microsoft/unixcoder-base'}


MODEL_LOCALS = {
    'roberta': 'roberta-base',
    'codebert': 'codebert-base',
    'graphcodebert': 'graphcodebert-base',
    't5': 't5-base',
    'codet5':'codet5-base',
    'bart': 'bart-base',
    'plbart': 'plbart-base',
    'unixcoder':'unixcoder-base',
}
MODEL_CLASSES = {'roberta': (AutoConfig, AutoModel, AutoTokenizer),
                 'codebert': (AutoConfig, AutoModel, AutoTokenizer),
                 'graphcodebert': (AutoConfig, AutoModel, AutoTokenizer),
                 'unixcoder':(AutoConfig, AutoModel, AutoTokenizer),
                 't5': (AutoConfig, T5ForConditionalGeneration, AutoTokenizer),
                 'codet5': (AutoConfig, T5ForConditionalGeneration, AutoTokenizer),
                 'bart': (AutoConfig, BartForConditionalGeneration, AutoTokenizer),
                 'plbart':(AutoConfig, PLBartForConditionalGeneration, AutoTokenizer)}
MODEL_CLASSES_PLG = {'roberta': (AutoConfig, AutoModel, AutoTokenizer),
                 'codebert': (AutoConfig, AutoModel, AutoTokenizer),
                 'graphcodebert': (AutoConfig, AutoModel, AutoTokenizer),
                 'unixcoder':(AutoConfig, AutoModel, AutoTokenizer),
                 't5': (AutoConfig, T5ForConditionalGeneration_Prefix, AutoTokenizer),
                 'codet5': (AutoConfig, T5ForConditionalGeneration_Prefix, AutoTokenizer),
                 'bart': (AutoConfig, BartForConditionalGeneration, AutoTokenizer),
                 'plbart':(AutoConfig, PLBartForConditionalGeneration_Prefix, AutoTokenizer)}
# MODEL_CLASSES = {'roberta': (AutoConfig, AutoModel, AutoTokenizer),
#                  'codebert': (AutoConfig, AutoModel, AutoTokenizer),
#                  'graphcodebert': (AutoConfig, AutoModel, AutoTokenizer),
#                  'unixcoder':(AutoConfig, AutoModel, AutoTokenizer),
#                  't5': (AutoConfig, T5ForConditionalGeneration, AutoTokenizer),
#                  'codet5': (AutoConfig, T5ForConditionalGeneration, AutoTokenizer),
#                  'bart': (AutoConfig, BartForConditionalGeneration, AutoTokenizer),
#                  'plbart':(AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer)}

# MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
#                  't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
#                  'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
#                  'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer)}

def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6,3))

#如果addargument了prompt 在encoder的输出前就先加上prompt 用past_key_values 照着deltatuning加

#unixcoder seq2seq unilm 怎么实现三个mask
def bulid_or_load_gen_model(args):
    # checkpoint = MODEL_CHECKPOINTS[args.model_name]
    checkpoint = os.path.join(args.huggingface_locals, MODEL_LOCALS[args.model_name])
    config_class, model_class, tokenizer_class = MODEL_CLASSES_PLG[args.model_name]
    
    config = config_class.from_pretrained(checkpoint)
    tokenizer = tokenizer_class.from_pretrained(checkpoint)
    print(config.model_type)
    if args.model_name in ['roberta', 'codebert', 'graphcodebert']:
        encoder = model_class.from_pretrained(checkpoint, output_attentions=True)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        model = Seq2Seq(encoder=encoder, decoder=decoder, tokenizer=tokenizer, args=args,
                        config=config, beam_size=args.beam_size, max_length=args.max_target_length,
                        sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
    elif args.model_name in ['unixcoder']:
        # import！！！you must set is_decoder as True for generation in unixcoder！！！
        config.is_decoder = True
        encoder = model_class.from_pretrained(checkpoint, config=config)
        if args.task in ['complete']:
            if args.sub_task == "python":
                eos_ids = [tokenizer.sep_token_id]
            else:
                eos_ids = [tokenizer.convert_tokens_to_ids('Ġ;'), tokenizer.convert_tokens_to_ids('Ġ}'), tokenizer.convert_tokens_to_ids('Ġ{')]
            model=Seq2Seq4UniXcoder_completion(encoder=encoder,decoder=encoder,config=config,
                        beam_size=args.beam_size,max_length=args.max_target_length,
                        sos_id=tokenizer.cls_token_id,eos_id=eos_ids)
        elif args.task in ['generate']:
            model = Seq2Seq4UniXcoder_generation(encoder=encoder,decoder=encoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0],eos_id=tokenizer.sep_token_id)
        elif args.task in ['summarize','translate','refine']:
            model = Seq2Seq4UniXcoder_e2d(encoder=encoder,decoder=encoder,config=config,
                        beam_size=args.beam_size,max_length=args.max_target_length,
                        sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0],eos_id=tokenizer.sep_token_id)
            
    elif args.model_name in ['t5', 'codet5','bart','plbart']:
        model = model_class.from_pretrained(checkpoint, output_attentions=True,args=args,tokenizer=tokenizer)
    if args.prefix_tuning:
        logger.info("Finish loading model [%s] parameters from %s", get_model_size(
            model.code_prefix.gat_layer), args.model_name)
    else:
        logger.info("Finish loading model [%s] parameters from %s", get_model_size(
            model), args.model_name)

    return config, model, tokenizer

def bulid_or_load_cls_model(args):
    # checkpoint = MODEL_CHECKPOINTS[args.model_name]
    checkpoint = os.path.join(args.huggingface_locals, MODEL_LOCALS[args.model_name])
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_name]
    config = config_class.from_pretrained(checkpoint)
    tokenizer = tokenizer_class.from_pretrained(checkpoint)
    # if args.model_name in ['unixcoder']:
    #     model = model_class.from_pretrained(checkpoint, output_attentions=True)
    #     model = Model4UniXcoder(model,config,tokenizer,args)
    if args.model_name not in ['t5', 'codet5','bart','plbart']:
        model = model_class.from_pretrained(checkpoint, output_attentions=True)
    else:
        model = model_class.from_pretrained(checkpoint, output_attentions=True)
    if args.task == 'defect':
        model = DefectModel(model, config, tokenizer, args)
    elif args.task == 'clone':
        # model.resize_token_embeddings(32000)
        model = CloneModel(model, config, tokenizer, args)

    if args.prefix_tuning:
        logger.info("Finish loading model [%s] parameters from %s", get_model_size(
            model.code_prefix.gat_layer), args.model_name)
    else:
        logger.info("Finish loading model [%s] parameters from %s", get_model_size(
            model), args.model_name)

    return config, model, tokenizer
