import json
import pdb
from torch.nn.init import xavier_uniform_
from torch.utils.data import TensorDataset
import numpy as np
import logging
import os
import random
import torch
import time
from tqdm import tqdm
import networkx as nx
import re
from io import StringIO
import tokenize
from functools import partial
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from prefixcode import PrefixCode
import multiprocessing
logger = logging.getLogger(__name__)

def get_lang_by_task(task, sub_task):
    if task in ['summarize','complete']:
        return sub_task
    elif task in ['refine','generate','clone']:
        return 'java'
    elif task == 'translate':
        if sub_task == 'cs-java':
            return 'c_sharp'
        else:
            return 'java'
    elif task == 'defect':
        return 'c'
    else:
        raise 'java'
def add_lang_by_task(target_str, task, sub_task):
    if task == 'summarize':
        target_str = '<en> ' + target_str
    elif task == 'refine':
        target_str = '<java> ' + target_str
    elif task == 'translate':
        if sub_task == 'java-cs':
            target_str = '<c_sharp> ' + target_str
        else:
            target_str = '<java> ' + target_str
    elif task == 'generate':
        target_str = '<java> ' + target_str
    elif task == 'defect':
        target_str = target_str
    return target_str


def convert_examples_to_features(args,item):
    example, example_index, tokenizer, args, stage = item

    if args.model_name in ['t5', 'codet5'] and args.add_task_prefix:
        if args.sub_task != 'none':
            source_str = "{} {}: {}".format(
                args.task, args.sub_task, example.source)
        else:
            source_str = "{}: {}".format(args.task, example.source)
    elif args.model_name in ['unixcoder'] and args.task == 'complete':
        source_str = format_special_chars(tokenizer.tokenize(example.source[:args.max_source_length-3]))
        source_str =[tokenizer.sep_token,"<decoder-only>",tokenizer.sep_token]+source_str
    elif args.model_name in ['unixcoder']:
        source_str = format_special_chars(tokenizer.tokenize(example.source[:args.max_source_length-5]))
        source_str =[tokenizer.cls_token,"<encoder-decoder>",tokenizer.sep_token]+source_str+["<mask0>",tokenizer.sep_token]
        # in https://github.com/microsoft/CodeBERT when args.task == 'summarize' they put <mask0> before source_str, which performs not better
    else:
        source_str = example.source

    if args.model_name in ['unixcoder']:
        source_ids = tokenizer.convert_tokens_to_ids(source_str) 
        padding_length = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id]*padding_length
    else:
        source_str = source_str.replace('</s>', '<unk>')
        source_ids = tokenizer.encode(
            source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
        assert source_ids.count(tokenizer.eos_token_id) == 1
    if stage == 'test':
        target_ids = []
        if args.model_name in ['unixcoder'] and args.task != 'complete':
            target_str = tokenizer.tokenize("None")
            target_str = ["<mask0>"] + target_str + [tokenizer.sep_token]            
            target_ids = tokenizer.convert_tokens_to_ids(target_str)
            padding_length = args.max_target_length - len(target_ids)
            target_ids += [tokenizer.pad_token_id] * padding_length
    else:
        if args.model_name in ['unixcoder']:
            target_str = format_special_chars(tokenizer.tokenize(example.target)[:args.max_target_length-2])
        else:
            target_str = example.target
        if args.add_lang_ids:
            target_str = add_lang_by_task(
                example.target, args.task, args.sub_task)
        if args.model_name in ['unixcoder'] and args.task != 'complete':
            target_str = ["<mask0>"] + target_str + [tokenizer.sep_token]            
            target_ids = tokenizer.convert_tokens_to_ids(target_str)
            padding_length = args.max_target_length - len(target_ids)
            target_ids += [tokenizer.pad_token_id] * padding_length
        else:
            target_str = target_str.replace('</s>', '<unk>')
            target_ids = tokenizer.encode(target_str, max_length=args.max_target_length, padding='max_length',
                                        truncation=True)
            assert target_ids.count(tokenizer.eos_token_id) == 1

    return InputFeatures(
        example_index,
        source_ids,
        target_ids,
        url=example.url
    )


def convert_clone_examples_to_features(args,item):
    example, example_index, tokenizer, args = item
    if args.model_name in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
        target_str = "{}: {}".format(args.task, example.target)
    elif args.model_name in ['unixcoder']:
        source_str = format_special_chars(tokenizer.tokenize(example.source[:args.max_source_length-4]))
        source_str =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+source_str+[tokenizer.sep_token]
        target_str = format_special_chars(tokenizer.tokenize(example.target[:args.max_target_length-4]))
        target_str =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+target_str+[tokenizer.sep_token]
        example_index = source_str + target_str
    else:
        
        source_str = example.source
        target_str = example.target
    if args.model_name in ['unixcoder']:
        code1 = tokenizer.convert_tokens_to_ids(source_str)
        padding_length = args.max_source_length - len(code1)
        code1 += [tokenizer.pad_token_id]*padding_length
        
        code2 = tokenizer.convert_tokens_to_ids(target_str)
        padding_length = args.max_source_length - len(code2)
        code2 += [tokenizer.pad_token_id]*padding_length
        source_ids = code1 + code2
    else:
        code1 = tokenizer.encode(
            source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
        code2 = tokenizer.encode(
            target_str, max_length=args.max_target_length, padding='max_length', truncation=True)
        source_ids = code1 + code2
    return CloneInputFeatures(example_index, source_ids, example.label, example.url1, example.url2)


def convert_defect_examples_to_features(args,item):
    example, example_index, tokenizer, args = item
    if args.model_name in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
    elif args.model_name in ['unixcoder']:
        source_str = format_special_chars(tokenizer.tokenize(example.source[:args.max_source_length-4]))
        source_str =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+source_str+[tokenizer.sep_token]
    else:
        source_str = example.source
    if args.model_name in ['unixcoder']:
        code = tokenizer.convert_tokens_to_ids(source_str)
        padding_length = args.max_source_length - len(code)
        code += [tokenizer.pad_token_id]*padding_length
    else:
        code = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    return DefectInputFeatures(example_index, code, example.target)

class CloneInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label=None,
                 url1=None,
                 url2=None
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label
        self.url1 = url1
        self.url2 = url2

class DefectInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label=None
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label

class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids=None,
                 url=None
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.url = url


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,#
                 source,
                 target='',#
                 url=None,
                 task='',
                 sub_task='',
                 ast=None,
                 raw_code=None
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.url = url
        self.task = task
        self.sub_task = sub_task
        self.ast = ast
        self.raw_code = raw_code


class CloneExample(object):
    """A single training/test example."""

    def __init__(self,
                 code1,
                 code2=None,
                 label=None,
                 url1=None,
                 url2=None
                 ):
        self.source = code1
        self.target = code2
        self.label = label
        self.url1 = url1
        self.url2 = url2


def read_translate_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0
    with open(src_filename, encoding="utf-8") as f1, open(trg_filename, encoding="utf-8") as f2:
        for line1, line2 in tqdm(zip(f1, f2),desc="Read examples"):
            src = line1.strip()
            trg = line2.strip()
            examples.append(
                Example(
                    idx=idx,
                    source=src,
                    target=trg,
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_refine_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0

    with open(src_filename, encoding="utf-8") as f1, open(trg_filename, encoding="utf-8") as f2:
        for line1, line2 in tqdm(zip(f1, f2),desc="Read examples"):
            examples.append(
                Example(
                    idx=idx,
                    source=line1.strip(),
                    target=line2.strip(),
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_generate_examples(filename, data_num):
    """Read examples from filename."""
    examples = []

    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f,desc="Read examples")):
            x = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=x["nl"].strip(),
                    target=x["code"].strip()
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_summarize_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f,desc="Read examples")):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = ' '.join(js['code_tokens']).replace('\n', ' ')
            code = ' '.join(code.strip().split())
            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                    raw_code=js['code']
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def read_defect_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f,desc="Read examples")):
            line = line.strip()
            js = json.loads(line)

            code = ' '.join(js['func'].split())
            examples.append(
                Example(
                    idx=js['idx'],
                    source=code,
                    target=js['target']
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def read_clone_examples(filename, data_num):
    """Read examples from filename."""
    index_filename = filename
    url_to_code = {}
    with open('/'.join(index_filename.split('/')[:-1]) + '/data.jsonl', encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            code = ' '.join(js['func'].split())
            # code_tokens, dfg = extract_dataflow(js['func'], parsers['java'], 'java')
            # code = ' '.join(code_tokens)
            # pdb.set_trace()
            url_to_code[js['idx']] = code

    data = []
    with open(index_filename, encoding="utf-8") as f:
        idx = 0
        for line in tqdm(f,desc="Read examples"):
            line = line.strip()
            url1, url2, label = line.split('\t')
            if url1 not in url_to_code or url2 not in url_to_code:
                continue
            if label == '0':
                label = 0
            else:
                label = 1
            data.append(CloneExample(
                url_to_code[url1], url_to_code[url2], label, url1, url2))
            idx += 1
            if idx == data_num:
                break
    return data


def load_and_cache_gen_data(args, filename, pool, tokenizer, split_tag, only_src=False, is_sample=False):
    # cache the data into args.cache_path except it is sampled
    # only_src: control whether to return only source ids for bleu evaluating (dev/test)
    # return: examples (Example object), data (TensorDataset)
    data_tag = '_all' if args.data_num == -1 else '_%d' % args.data_num
    cache_fn = '{}/{}.pt'.format(args.cache_path,
                                 split_tag + ('_src' if only_src else '') + data_tag)

    examples = read_examples(filename, -1, args.task)
    
    # if is_sample and is_attention:
    #     if args.few_shot <= len(examples):
    #         examples = random.sample(examples, min(3000, len(examples)) if args.few_shot == -1 else args.few_shot)
    #     else:
    #         # for CodeTrans dataset, dev&test example len = 500, may smaller than few-shot case
    #         # we compensate some examples from train set to fill examples to args.few_shot
    #         examples_train = read_examples(args.train_filename, -1, args.task)
    #         examples += random.sample(examples_train, args.few_shot - len(examples))
    #         assert len(examples) == args.few_shot
        
    #     args.warmup_steps = len(examples) / 100
    if split_tag!='test' and is_sample or args.few_shot != -1 :
        if args.few_shot <= len(examples):
            sample_num = min(5000, len(examples))
            # if args.task=='generate':#evalnum_before2000
            #     sample_num = min(1500, len(examples)//2)
            # elif args.task=='refine':#evalnum_before5000
            #     sample_num = min(1500, len(examples)//4)
            if split_tag=='train':
                examples = random.sample(examples, sample_num if args.few_shot == -1 else args.few_shot)
            else:
                examples = random.sample(examples, sample_num if args.few_shot == -1 else args.few_shot)
        else:
            # for CodeTrans dataset, dev&test example len = 500, may smaller than few-shot case
            # we compensate some examples from train set to fill examples to args.few_shot
            examples_train = read_examples(args.train_filename, -1, args.task)
            examples += random.sample(examples_train, args.few_shot - len(examples))
            assert len(examples) == args.few_shot
        args.warmup_steps = len(examples) / 100
    if split_tag == 'train':
        calc_stats(examples, tokenizer, is_tokenize=True)
    else:
        calc_stats(examples)
    if os.path.exists(cache_fn) and not is_sample and args.few_shot == -1:
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info(
                "Sample %d data for computing bleu/attention from %s", len(examples),filename)
        elif args.data_num == -1:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args, split_tag)
                          for idx, example in enumerate(examples)]
        f_=partial(convert_examples_to_features,args)
        features = pool.map(f_, tqdm(
            tuple_examples, total=len(tuple_examples)))
        all_source_ids = torch.tensor(
            [f.source_ids for f in features], dtype=torch.long)
        if split_tag == 'test' or only_src:
            data = TensorDataset(all_source_ids)
        else:
            all_target_ids = torch.tensor(
                [f.target_ids for f in features], dtype=torch.long)
            data = TensorDataset(all_source_ids, all_target_ids)
        if args.local_rank in [-1, 0] and not is_sample and args.few_shot == -1:
            torch.save(data, cache_fn)
    return examples, data


# def load_and_cache_multi_gen_data(args, split_tag, pool, tokenizer, encode_target=True, is_sample=False):
#     cache_fn = os.path.join(args.cache_path, split_tag)
#     if os.path.exists(cache_fn) and not is_sample:
#         logger.info("Load cache data from %s", cache_fn)
#         examples_data_dict = torch.load(cache_fn)
#     else:
#         examples_data_dict = {}

#         task_list = ['summarize', 'translate', 'refine', 'generate', 'defect', 'clone']
#         for task in task_list:
#             if task == 'summarize':
#                 sub_tasks = ['ruby', 'javascript',
#                              'go', 'python', 'java', 'php']
#             elif task == 'translate':
#                 sub_tasks = ['java-cs', 'cs-java']
#             elif task == 'refine':
#                 sub_tasks = ['small', 'medium']
#             else:
#                 sub_tasks = []
#             args.task = task
#             for sub_task in sub_tasks:
#                 args.sub_task = sub_task
#                 if task == 'summarize':
#                     args.max_source_length = 256
#                     args.max_target_length = 128
#                 elif task == 'translate':
#                     args.max_source_length = 320
#                     args.max_target_length = 256
#                 elif task == 'refine':
#                     if sub_task == 'small':
#                         args.max_source_length = 130
#                         args.max_target_length = 120
#                     else:
#                         args.max_source_length = 240
#                         args.max_target_length = 240
#                 elif task == 'generate':
#                     args.max_source_length = 320
#                     args.max_target_length = 150
#                 elif task == 'defect':
#                     args.max_source_length = 512
#                     args.max_target_length = 3  # as do not need to add lang ids
#                 elif task == 'clone':
#                     args.max_source_length = 256
#                     args.max_target_length = 256

#                 filename = get_filenames(
#                     args.data_dir, args.task, args.sub_task, split_tag)
#                 examples = read_examples(filename, args.data_num, args.task)
#                 if is_sample:
#                     examples = random.sample(
#                         examples, min(5000, len(examples)))
#                 if split_tag == 'train':
#                     calc_stats(examples, tokenizer, is_tokenize=True)
#                 else:
#                     calc_stats(examples)

#                 tuple_examples = [(example, idx, tokenizer, args, split_tag)
#                                   for idx, example in enumerate(examples)]
#                 f_=partial(convert_examples_to_features,args)
#                 if args.data_num == -1:
#                     features = pool.map(f_, tqdm(
#                         tuple_examples, total=len(tuple_examples)))
#                 else:
#                     features = [f_(
#                         x) for x in tuple_examples]
#                 all_source_ids = torch.tensor(
#                     [f.source_ids for f in features], dtype=torch.long)
#                 if encode_target:
#                     all_target_ids = torch.tensor(
#                         [f.target_ids for f in features], dtype=torch.long)
#                     data = TensorDataset(all_source_ids, all_target_ids)
#                 else:
#                     data = TensorDataset(all_source_ids)
#                 examples_data_dict['{}_{}'.format(
#                     task, sub_task) if sub_task != 'none' else task] = (examples, data)

#         if args.local_rank in [-1, 0] and not is_sample:
#             torch.save(examples_data_dict, cache_fn)
#             logger.info("Save data into %s", cache_fn)
#     return examples_data_dict


def load_prefix_code(args, tokenizer):
    filename = get_filenames(
                args.prefix_dir, args.task, args.sub_task, 'prefix')
    # examples, data = load_and_cache_clone_data(args, filename, pool, tokenizer, 'train') 
    if args.task == 'clone':
        # examples = read_examples(filename, args.data_num, args.task)
        # index_filename = filename
        # url_to_code = {}
        with open(filename, encoding="utf-8") as f:
            line = f.readline().strip()
            js = json.loads(line)
            js['func']=js['func'].replace('</s>', '<unk>')
            examples=' '.join(js['func'].split())
            examples=[CloneExample(examples)]
            feature= CloneInputFeatures(example_id=1,source_ids=tokenizer.encode(examples[0].source, max_length=args.max_source_length, padding='max_length', truncation=True))
            data= torch.tensor([feature.source_ids], dtype=torch.long)
            prefix_code=PrefixCode(args, examples, data, 'java')
    elif args.task == 'defect':
        with open(filename, encoding="utf-8") as f:
            line = f.readline().strip()
            js = json.loads(line)
            js['func']=js['func'].replace('</s>', '<unk>')
            examples=' '.join(js['func'].split())
            examples=[Example(1,examples)]
            feature= DefectInputFeatures(example_id=1,source_ids=tokenizer.encode(examples[0].source, max_length=args.max_source_length, padding='max_length', truncation=True))
            data= torch.tensor([feature.source_ids], dtype=torch.long)
            prefix_code=PrefixCode(args, examples, data, 'c')
            # if args.prefix_token_level == 'token':
            #     tokens_ids=tokenizer.encode(js['func'], max_length=args.gnn_token_num, padding='max_length', truncation=True)
            #     print(tokens_list)
            #     weight_matrix=distance_list[0]
            #     return tokens_ids#,weight_matrix
            # elif args.prefix_token_level == 'subtoken':
            #     return None,None
            # tokens_list = js['func'].split()
    elif args.task == 'generate':
        with open(filename, encoding="utf-8") as f:
            line = f.readline().strip()
            js = json.loads(line)
            js['code']=js['code'].replace('</s>', '<unk>')
            examples=' '.join(js['code'].split())
            examples=[Example(1,examples)]
    #         InputFeatures(
    #     example_index,
    #     source_ids,
    #     target_ids,
    #     url=example.url
    # )
            feature= InputFeatures(example_id=1,source_ids=tokenizer.encode(examples[0].source, max_length=args.max_source_length, padding='max_length', truncation=True))
            data= torch.tensor([feature.source_ids], dtype=torch.long)
            prefix_code=PrefixCode(args, examples, data, 'java')
    elif args.task == 'refine':
        with open(filename, encoding="utf-8") as f:
            line = f.readline().strip()
            examples=[Example(1,line)]
            feature= InputFeatures(example_id=1,source_ids=tokenizer.encode(examples[0].source, max_length=args.max_source_length, padding='max_length', truncation=True))
            data= torch.tensor([feature.source_ids], dtype=torch.long)
            prefix_code=PrefixCode(args, examples, data, 'java')
    elif args.task == 'translate':
        with open(filename, encoding="utf-8") as f:
            line = f.readline().strip()
            examples=[Example(1,line)]
            feature= InputFeatures(example_id=1,source_ids=tokenizer.encode(examples[0].source, max_length=args.max_source_length, padding='max_length', truncation=True))
            data= torch.tensor([feature.source_ids], dtype=torch.long)
            if args.sub_task == 'java-cs':
                prefix_code=PrefixCode(args, examples, data, 'c_sharp')
            else:
                prefix_code=PrefixCode(args, examples, data, 'java')
    elif args.task == 'summarize':
        with open(filename, encoding="utf-8") as f:
            line = f.readline().strip()
            js = json.loads(line)
            js['code_tokens']=' '.join(js['code_tokens']).replace('</s>', '<unk>').replace('\n',' ')
            examples=' '.join(js['code_tokens'].strip().split())
            js['docstring_tokens']=' '.join(js['docstring_tokens']).replace('</s>', '<unk>').replace('\n',' ')
            nl=' '.join(js['docstring_tokens'].strip().split())
            examples=[Example(1,examples)]
            feature= InputFeatures(example_id=1,source_ids=tokenizer.encode(examples[0].source, max_length=args.max_source_length, padding='max_length', truncation=True))
            data= torch.tensor([feature.source_ids], dtype=torch.long)
            prefix_code=PrefixCode(args, examples, data, args.sub_task)
    
    ast_list, sast_list, tokens_list, tokens_type_list, leaves =prefix_code.get_ast_and_token(prefix_code.examples, prefix_code.parser, prefix_code.lang)
    tokens_ids=tokenizer.convert_tokens_to_ids(tokens_list[0].values())
    distance_list=prefix_code.get_token_distance(args, leaves, ast_list, sast_list, 'shortest_path_length')[0]
    assert len(tokens_ids)==distance_list.shape[0]
    if len(tokens_ids)>=args.gnn_token_num:
        return tokens_ids[:args.gnn_token_num], distance_list[:args.gnn_token_num,:args.gnn_token_num]
    else:
        distance_list=np.pad(distance_list,((0,args.gnn_token_num-len(tokens_ids)),(0,args.gnn_token_num-len(tokens_ids))),'constant')
        tokens_ids=tokens_ids+[tokenizer.pad_token_id]*(args.gnn_token_num-len(tokens_ids))
        assert len(tokens_ids)==distance_list.shape[0]
        return tokens_ids, distance_list
        # token_ids = tokenizer.convert_tokens_to_ids(tokens_list) 
        # if len(token_ids)<=args.max_source_length:
        #     padding_length = args.max_source_length - len(token_ids)
        #     token_ids += [tokenizer.pad_token_id]*padding_length
        # else:
        #     token_ids = token_ids[:args.max_source_length]
        # return token_ids
        



def get_distance(args,tokenizer):
    pool=multiprocessing.Pool(args.cpu_count)
    examples, data = load_and_cache_clone_data(args, args.train_filename, pool, tokenizer, 'train') 

class TextDataset_POJ104(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        # super(TensorDataset, self).__init__(tokenizer, args, file_path)
        self.examples = []
        data = []
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                data.append(js)
        for js in data:
            self.examples.append(convert_examples_to_features(js,tokenizer,args))
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                    logger.info("*** Example ***")
                    logger.info("idx: {}".format(idx))
                    logger.info("label: {}".format(example.label))
                    logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
        self.label_examples = {}
        for e in self.examples:
            if e.label not in self.label_examples:
                self.label_examples[e.label]=[]
            self.label_examples[e.label].append(e)
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        label = self.examples[i].label
        index = self.examples[i].index
        labels = list(self.label_examples)
        labels.remove(label)
        while True:
            shuffle_example = random.sample(self.label_examples[label],1)[0]
            if shuffle_example.index != index:
                p_example = shuffle_example
                break
        n_example = random.sample(self.label_examples[random.sample(labels,1)[0]],1)[0]
        
        return (torch.tensor(self.examples[i].input_ids),torch.tensor(p_example.input_ids),
                torch.tensor(n_example.input_ids),torch.tensor(label))

# class TextDataset2(Dataset):
#     def __init__(self, features, args):
#         # super(TensorDataset, self).__init__(tokenizer, args, file_path)
#         self.examples = features
#         self.label_examples = {}
#         for e in self.examples:
#             if e.label not in self.label_examples:
#                 self.label_examples[e.label]=[]
#             self.label_examples[e.label].append(e)
        
#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, i):   
#         label = self.examples[i].label
#         index = self.examples[i].example_id
#         labels = list(self.label_examples)
#         labels.remove(label)
#         while True:
#             shuffle_example = random.sample(self.label_examples[label],1)[0]
#             if shuffle_example.example_id != index:
#                 p_example = shuffle_example
#                 break
#         n_example = random.sample(self.label_examples[random.sample(labels,1)[0]],1)[0]
        
#         return (torch.tensor(self.examples[i].source_ids),torch.tensor(p_example.source_ids),
#                 torch.tensor(n_example.source_ids),torch.tensor(label))
class TextDataset_BCB(Dataset):
    def __init__(self, features, args):
        self.examples = features
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item].source_ids),torch.tensor(self.examples[item].label)

def load_and_cache_clone_data(args, filename, pool, tokenizer, split_tag, is_sample=False):

    cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag +
                                '_all' if args.data_num == -1 else '_%d' % args.data_num)
    examples = read_examples(filename, -1, args.task)
    if is_sample or args.is_clone_sample:
        examples = random.sample(examples,  int(len(examples) * 0.1))
    if split_tag!='test' and args.few_shot!=-1:
        examples_True = [e for e in examples if e.label == 1]
        examples_False = [e for e in examples if e.label == 0]
        examples_True = random.sample(examples_True,args.few_shot)
        examples_False = random.sample(examples_False,args.few_shot)
        examples = examples_True + examples_False

    if split_tag!='test' and args.few_shot != -1:
        calc_stats(examples, tokenizer, is_tokenize=True)

    if os.path.exists(cache_fn) and args.few_shot == -1:
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if split_tag!='test' and args.few_shot == -1:
            logger.info("Sample 10 percent of data from %s", filename)
        elif args.data_num == -1:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args)
                        for idx, example in enumerate(examples)]
        f_=partial(convert_clone_examples_to_features,args)
        features = pool.map(f_, tqdm(
            tuple_examples, total=len(tuple_examples)))
        
        # if args.sub_task == "POJ":
        #     train_dataset = TextDataset_POJ104(features, args)
        #     return train_dataset, train_dataset
        # features = [convert_clone_examples_to_features(x) for x in tuple_examples]
        if args.model_name in ['unixcoder']:
            train_dataset = TextDataset_BCB(features, args)
            return examples, train_dataset
        all_source_ids = torch.tensor(
            [f.source_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor(
            [f.label for f in features], dtype=torch.long)
        data = TensorDataset(all_source_ids, all_labels)

        if args.local_rank in [-1, 0] and args.data_num == -1 and args.few_shot == -1:
            torch.save(data, cache_fn)
    return examples, data


def load_and_cache_defect_data(args, filename, pool, tokenizer, split_tag, is_sample=False):
    cache_fn = os.path.join(args.cache_path, split_tag)
    examples = read_examples(filename, -1, args.task)
    if is_sample:
        sample_num = min(5000, len(examples))
        examples = random.sample(examples, sample_num)
    elif split_tag!='test' and args.few_shot != -1:
        examples_True = [e for e in examples if e.target == 1]
        examples_False = [e for e in examples if e.target == 0]
        examples_True = random.sample(examples_True,args.few_shot)
        examples_False = random.sample(examples_False,args.few_shot)
        examples = examples_True + examples_False
    calc_stats(examples, tokenizer, is_tokenize=True)
    if os.path.exists(cache_fn) and args.few_shot == -1:
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if split_tag!='test' and is_sample:
            logger.info("Sample min(5000, len(examples)) of data from %s", filename)
        elif args.data_num == -1:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args) for idx, example in enumerate(examples)]
        f_=partial(convert_defect_examples_to_features,args)
        features = pool.map(f_, tqdm(tuple_examples, total=len(tuple_examples)))
        # if args.sub_task == "POJ":
        #     train_dataset = TextDataset_POJ104(features, args)
        #     return train_dataset, train_dataset
        if args.model_name in ['unixcoder']:
            train_dataset = TextDataset_BCB(features, args)
            return examples, train_dataset
        # features = [convert_clone_examples_to_features(x) for x in tuple_examples]
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        data = TensorDataset(all_source_ids, all_labels)

        if args.local_rank in [-1, 0] and args.data_num == -1 and args.few_shot == -1:
            torch.save(data, cache_fn)
    return examples, data

def get_filenames(data_root, task, sub_task, split=''):
    if task == 'generate':
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.json'.format(data_dir)
        dev_fn = '{}/dev.json'.format(data_dir)
        test_fn = '{}/test.json'.format(data_dir)
        prefix_fn = '{}/prefix.json'.format(data_dir)
    elif task == 'summarize':
        data_dir = '{}/{}/{}'.format(data_root, task, sub_task)
        train_fn = '{}/train.jsonl'.format(data_dir)
        dev_fn = '{}/valid.jsonl'.format(data_dir)
        test_fn = '{}/test.jsonl'.format(data_dir)
        prefix_fn = '{}/prefix.jsonl'.format(data_dir)
    elif task == 'refine':
        data_dir = '{}/{}/{}'.format(data_root, task, sub_task)
        train_fn = '{}/train.buggy-fixed.buggy,{}/train.buggy-fixed.fixed'.format(
            data_dir, data_dir)
        dev_fn = '{}/valid.buggy-fixed.buggy,{}/valid.buggy-fixed.fixed'.format(
            data_dir, data_dir)
        test_fn = '{}/test.buggy-fixed.buggy,{}/test.buggy-fixed.fixed'.format(
            data_dir, data_dir)
        prefix_fn = '{}/prefix.java'.format(
                data_dir)
    elif task == 'translate':
        data_dir = '{}/{}'.format(data_root, task)
        if sub_task == 'cs-java':
            train_fn = '{}/train.java-cs.txt.cs,{}/train.java-cs.txt.java'.format(
                data_dir, data_dir)
            dev_fn = '{}/valid.java-cs.txt.cs,{}/valid.java-cs.txt.java'.format(
                data_dir, data_dir)
            test_fn = '{}/test.java-cs.txt.cs,{}/test.java-cs.txt.java'.format(
                data_dir, data_dir)
            prefix_fn = '{}/prefix.txt.java'.format(
                data_dir)
        else:
            train_fn = '{}/train.java-cs.txt.java,{}/train.java-cs.txt.cs'.format(
                data_dir, data_dir)
            dev_fn = '{}/valid.java-cs.txt.java,{}/valid.java-cs.txt.cs'.format(
                data_dir, data_dir)
            test_fn = '{}/test.java-cs.txt.java,{}/test.java-cs.txt.cs'.format(
                data_dir, data_dir)
            prefix_fn = '{}/prefix.txt.cs'.format(
                data_dir)
    elif task == 'clone':
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.txt'.format(data_dir)
        dev_fn = '{}/valid.txt'.format(data_dir)
        test_fn = '{}/test.txt'.format(data_dir)
        prefix_fn = '{}/prefix.txt'.format(data_dir)
    elif task == 'defect':
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.jsonl'.format(data_dir)
        dev_fn = '{}/valid.jsonl'.format(data_dir)
        test_fn = '{}/test.jsonl'.format(data_dir)
        prefix_fn = '{}/prefix.jsonl'.format(data_dir)
    if split == 'train':
        return train_fn
    elif split == 'dev':
        return dev_fn
    elif split == 'test':
        return test_fn
    elif split == 'prefix':
        return prefix_fn
    else:
        return train_fn, dev_fn, test_fn


def read_examples(filename, data_num, task):
    read_example_dict = {
        # read_summarize_examples， read_summarize_indent_examples
        'summarize': read_summarize_examples,
        'refine': read_refine_examples,
        'translate': read_translate_examples,
        'generate': read_generate_examples,
        'clone': read_clone_examples,
        'defect': read_defect_examples,
    }
    return read_example_dict[task](filename, data_num)


def calc_stats(examples, tokenizer=None, is_tokenize=False):
    avg_src_len = []
    avg_trg_len = []
    avg_src_len_tokenize = []
    avg_trg_len_tokenize = []
    for ex in examples:
        if is_tokenize:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
            avg_src_len_tokenize.append(len(tokenizer.tokenize(ex.source)))
            avg_trg_len_tokenize.append(
                len(tokenizer.tokenize(str(ex.target))))
        else:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
    if is_tokenize:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))
        logger.info("[TOKENIZE] avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    np.mean(avg_src_len_tokenize), np.mean(
                        avg_trg_len_tokenize), max(avg_src_len_tokenize),
                    max(avg_trg_len_tokenize))
    else:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))


def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)


def remove_comments_and_docstrings(source, lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp = []
        for x in out.split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)


# def format_attention(attention, layers=None, heads=None):
#     """[format attention whose batch size > 1]

#     Args:
#         attention ([type]): [description]
#         layers ([type], optional): [description]. Defaults to None.
#         heads ([type], optional): [description]. Defaults to None.

#     Raises:
#         ValueError: [description]

#     Returns:
#         [type]: [description]
#     """

#     if type(layers) == int:
#         layers = [layers]
#     if layers:
#         attention = [attention[layer_index] for layer_index in layers]
#     squeezed = []
#     for layer_attention in attention:
#         # batch_size x num_heads x seq_len x seq_len
#         # print('layer_attention', layer_attention.shape)
#         if len(layer_attention.shape) != 4:
#             raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
#                              "output_attentions=True when initializing your model.")
#         # num_heads x batch_size x seq_len x seq_len
#         layer_attention = layer_attention.permute((1, 0, 2, 3))

#         if heads:
#             layer_attention = layer_attention[heads]
#         squeezed.append(layer_attention)
#     # num_layers x num_heads x batch_size x seq_len x seq_len
#     return torch.stack(squeezed).permute((2, 0, 1, 3, 4))
#     # batch_size x num_layers x num_heads x seq_len x seq_len


# def num_layers(attention):
#     return len(attention)


# def num_heads(attention):
#     return attention[0][0].size(0)


def format_special_chars(tokens):
    return [t.replace('Ġ', '') for t in tokens]#.replace(u"\u2581", u" ")


def index_to_code_token(index, code):
    code = code.split('\n')
    start_point = index[0]
    end_point = index[1]
    if start_point[0] == end_point[0]:
        s = code[start_point[0]][start_point[1]:end_point[1]]
    else:
        s = ""
        s += code[start_point[0]][start_point[1]:]
        for i in range(start_point[0] + 1, end_point[0]):
            s += code[i]
        s += code[end_point[0]][:end_point[1]]
    return s

def is_frequent_type(token, lang):
    #get frequent type from model_free_frequent_type.ipynb
    frequent_type = {}
    frequent_type['javascript'] = ['function',
                                   ')', 'string_fragment', 'identifier', '(', ';', '{', '}']
    frequent_type['go'] = ['package_identifier',
                           'type_identifier', 'field_identifier', 'if', 'return', '=']
    frequent_type['java'] = [')', 'public', 'string_literal',
                             'identifier', '}', 'return', 'type_identifier', 'if']
    frequent_type['python'] = [')', 'def', 'return',
                               'identifier', 'if', 'for', ':', ']']
    if lang in frequent_type:
        return token in frequent_type[lang]
    else:
        return True  # if lang is not provided by frequent_type, assume all token types are frequent

def top_n_scores(n, score_dict):
    ''' returns keys which match the top n scores of values from a name:score dict'''
    lot = [(k, v)
           for k, v in score_dict.items()]  # make list of tuple from scores dict
    nl = []
    while len(lot) > 0:
        nl.append(max(lot, key=lambda x: x[1]))
        lot.remove(nl[-1])
    return [i[0] for i in nl[0:n]]