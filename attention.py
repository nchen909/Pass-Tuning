from utils import (get_filenames, get_elapse_time,
                   load_and_cache_gen_data, get_ast_nx, format_attention, num_layers,
                   index_to_code_token, format_special_chars, is_frequent_type,top_n_scores)
from models import bulid_or_load_gen_model
from configs import add_args, set_dist, set_seed, set_hyperparas

import pickle
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch.nn as nn
import json
import random
import argparse
import multiprocessing
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm, trange
import os
import logging
from tree_sitter import Language, Parser
import networkx as nx
import numpy as np
import sys
import codecs
from collections import Counter
from functools import reduce

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
sys.setrecursionlimit(5000)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_attention_and_subtoken(args, data, model, tokenizer):
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=args.attention_batch_size,
                            num_workers=4, pin_memory=True)
    model.eval()
    attention_list = []
    subtokens_list = []
    logger.info("Obtain subtokens and their attention")
    for batch in tqdm(dataloader, total=len(dataloader), desc="Computing attention"):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, target_ids = batch
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        target_mask = target_ids.ne(tokenizer.pad_token_id)
        for source_id in source_ids:
            subtokens = tokenizer.convert_ids_to_tokens(source_id)
            subtokens_list.append(subtokens)

        with torch.no_grad():
            if args.model_name in ['roberta', 'codebert', 'graphcodebert', 'unixcoder']:
                _, _, _, attention = model(source_ids=source_ids, source_mask=source_mask,
                                           target_ids=target_ids, target_mask=target_mask)
            else:
                outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask)
                attention = outputs.encoder_attentions
        includer_layers = list(range(num_layers(attention)))
        attention = format_attention(
            attention, layers=includer_layers[args.layer_num])

        attention = attention.detach().cpu().numpy()
        attention_list.append(attention)
    print("len(layer):", len(includer_layers))
    print("len(attention_list)", len(attention_list))
    attention_numpy = np.concatenate(attention_list, axis=0)
    return attention_numpy, subtokens_list


def number_subtoken(args, subtokens_list, tokens_list, tokens_type_list, tokenizer, freq_type_list=False):
    print('special tokens: ', tokenizer.additional_special_tokens)
    assert len(subtokens_list) == len(tokens_list)
    subtoken_numbers_list = []
    # tokens_list like {3: 'private', 7: '',10: '(',...}may include ''
    # so convert to tokens_list_output like ['private','(',...]
    tokens_list_output = []
    tokens_list_type_output = []
    for i in trange(len(subtokens_list), desc="Getting subtoken number"):
        subtokens = subtokens_list[i]
        token_numbers = list(tokens_list[i].keys())
        tokens = list(tokens_list[i].values())
        tokens_type = list(tokens_type_list[i].values())
        assert len(token_numbers) == len(tokens)
        assert len(tokens) == len(tokens_type)
        subtoken_numbers = []
        subtokens = format_special_chars(subtokens)
        if i == 0:
            print('after formatting, subtokens 0:', )
            print(subtokens)
        pos = 0
        pos_old = 0
        token_list_output = []
        token_list_type_output = []
        for j in range(len(subtokens)):
            if subtokens[j] in ['<s>', '</s>', '<pad>'] or subtokens[j] in tokenizer.additional_special_tokens:
                # the special tokens of tokenizer is not involved in AST tree, we use -1 to tag it
                subtoken_numbers.append(-1)
            else:
                if pos == len(tokens):
                    pos = pos_old
                if freq_type_list:
                    while subtokens[j] not in tokens[pos] or tokens_type[pos] not in freq_type_list:
                        # (language like java) use while because token may exist "", which not includes any subtoken
                        pos += 1
                        # subtoken like "%3d" may not be included in any token
                        if pos == len(tokens):
                            subtoken_numbers.append(-1)
                            break
                else:
                    while subtokens[j] not in tokens[pos]:
                        pos += 1
                        # subtoken like "%3d" may not be included in any token
                        if pos == len(tokens):
                            subtoken_numbers.append(-1)
                            break
                if pos == len(tokens):
                    continue
                if pos_old != pos or pos_old == 0:
                    # (language like go) it will prevent a combined token
                    # (which pair more than 1 subtoken) append to token_list_output several times
                    token_list_output.append(tokens[pos])
                    token_list_type_output.append(tokens_type[pos])
                subtoken_numbers.append(token_numbers[pos])
                pos_old = pos
        tokens_list_output.append(token_list_output)
        tokens_list_type_output.append(token_list_type_output)
        subtoken_numbers_list.append(subtoken_numbers)

    # 写文件
    pickle.dump(tokens_list_output, open(args.cache_path +
                "/tokens_list_mf"+args.pickle_suffix, 'wb'))
    pickle.dump(tokens_list_type_output, open(args.cache_path +
                "/tokens_list_type_mf"+args.pickle_suffix, 'wb'))
    print("wrote tokens_list to", args.cache_path +
          "/tokens_list_mf"+args.pickle_suffix)
    print("wrote tokens_list_type to", args.cache_path +
          "/tokens_list_type_mf"+args.pickle_suffix)

    print('subtoken_numbers_list length: ', len(subtoken_numbers_list))
    print('subtoken_numbers_list 0: ')
    print(subtoken_numbers_list[0])
    return subtoken_numbers_list, tokens_list_type_output


def get_token_attention(args, attention_numpy, subtoken_numbers_list, action_per_head):  
    leaves = []
    attention_list = []
    if action_per_head == "mean":
        attention_numpy = attention_numpy.mean(axis=2)
    elif action_per_head == "max":
        attention_numpy = attention_numpy.max(axis=2)
    for i in trange(len(attention_numpy), desc="Getting token attention"):  
        # reason why not just use leaves from get_ast_and_token :
        # leaves may include null token like "", which not includes any subtoken
        leaf = []
        #subtoken_numbers = subtoken_numbers_list[i]
        attention_numpy_case = attention_numpy[i][0]
        dict_subtoken_numbers_list = {}
        for index, value in enumerate(subtoken_numbers_list[i]):
            if value == -1:
                continue
            if value not in dict_subtoken_numbers_list:
                leaf.append(value)
                dict_subtoken_numbers_list[value] = [index]
            else:
                dict_subtoken_numbers_list[value].append(index)
        token_num = len(leaf)
        attention = np.zeros((token_num, token_num))
        leaves.append(leaf)
        for j in range(token_num):
            for k in range(token_num):
                token_attention_from_mean_subtoken_list = []
                for left_indices in dict_subtoken_numbers_list[leaf[j]]:
                    for right_indices in dict_subtoken_numbers_list[leaf[k]]:
                        token_attention_from_mean_subtoken_list.append(
                            attention_numpy_case[left_indices][right_indices])
                attention[j][k] = np.mean(token_attention_from_mean_subtoken_list)
        attention_list.append(attention)
    print("len(attention_list)", len(attention_list))

    pickle.dump(attention_list, open(args.cache_path +
                "/attention_list_mf"+args.pickle_suffix, 'wb'))
    print("wrote attention_list to", args.cache_path +
          "/attention_list_mf"+args.pickle_suffix)

    return attention_list, leaves


def get_freq_type_list(args, attention_list, token_type_list):
    quantile_threshold = args.quantile_threshold
    print('len(attention_list):', len(attention_list))
    case_count = len(attention_list)
    token_type_len = 0
    loss_list = []
    token_type_ge_quantile = {}
    token_type_num = {}
    for i in trange(case_count, desc="Get freq type list"):
        attention = attention_list[i]

        token_type = token_type_list[i]
        if not attention.any():  # code may not include any "identifier" type
            continue
        mask_att = attention > np.quantile(attention, quantile_threshold)
        token_type_len += mask_att.shape[0]
        for length in range(mask_att.shape[0]):
            for width in range(mask_att.shape[1]):
                token_type_l = token_type[length]
                token_type_w = token_type[width]
                if mask_att[length][width]:
                    if token_type_l not in token_type_ge_quantile:
                        token_type_ge_quantile[token_type_l] = 1
                    else:
                        token_type_ge_quantile[token_type_l] += 1
                    if token_type_w not in token_type_ge_quantile:
                        token_type_ge_quantile[token_type_w] = 1
                    else:
                        token_type_ge_quantile[token_type_w] += 1
                if token_type_l not in token_type_num:
                    token_type_num[token_type_l] = 1
                else:
                    token_type_num[token_type_l] += 1
                if token_type_w not in token_type_num:
                    token_type_num[token_type_w] = 1
                else:
                    token_type_num[token_type_w] += 1

    token_type_ge_quantile_rate = {}
    print("token_type_ge_quantile:", token_type_ge_quantile)
    print("token_type_num:", token_type_num)
    for key in token_type_ge_quantile:
        # select type that averages more than once per sample
        if token_type_num[key] > token_type_len:
            token_type_ge_quantile_rate[key] = float(
                token_type_ge_quantile[key])/token_type_num[key]
    print("Counter(token_type_ge_quantile_rate):",
          Counter(token_type_ge_quantile_rate))
    freq_type_list = top_n_scores(8, token_type_ge_quantile_rate)
    with open(os.path.join("./", "freq_type_list_mf.txt"), 'a') as type_txt:
        type_txt.write(str(args.layer_num)+' '+args.model_name+' ' +
                       args.task+' '+args.sub_task+' '+str(freq_type_list)+"\n")
    return freq_type_list


def get_subtoken_distance(args, ast_list, subtoken_numbers_list, distance_metric):
    print('get subtoken distance')
    assert len(ast_list) == len(subtoken_numbers_list)
    if distance_metric == 'shortest_path_length':
        ast_distance_list = [nx.shortest_path_length(ast) for ast in ast_list]
    elif distance_metric == 'simrank_similarity':
        ast_distance_list = [nx.simrank_similarity(ast) for ast in ast_list]
    subtoken_num = len(subtoken_numbers_list[0])

    distance_list = []
    for i in trange(len(subtoken_numbers_list), desc="Getting subtoken distance"):
        distance = np.zeros((subtoken_num, subtoken_num))
        subtoken_numbers = subtoken_numbers_list[i]
        ast_distance = dict(ast_distance_list[i])
        for j in range(subtoken_num):
            if subtoken_numbers[j] in ast_distance.keys():  # no in 'cls' etc,att=0
                for k in range(subtoken_num):
                    if subtoken_numbers[k] in ast_distance[subtoken_numbers[j]].keys():
                        distance[j][k] = ast_distance[subtoken_numbers[j]
                                                      ][subtoken_numbers[k]]
        distance_list.append(distance)

    distance_numpy = np.array(distance_list)
    print('distance_numpy shape: ', distance_numpy.shape)
    print('distance_numpy 0: ')
    print(distance_numpy[0])
    np.save(os.path.join(str(args.cache_path),
            "distance_numpy.npy"), distance_numpy)
    print("wrote distance_numpy to", os.path.join(
        str(args.cache_path), "distance_numpy.npy"))
    return distance_numpy


def get_token_distance(args, leaves, ast_list, sast_list, distance_metric):  # 4min
    print('get token distance')
    if distance_metric == 'shortest_path_length':
        ast_distance_list = [nx.shortest_path_length(ast) for ast in sast_list]
    elif distance_metric == 'simrank_similarity':
        ast_distance_list = [nx.simrank_similarity(ast) for ast in sast_list]
    distance_list = []
    for i in trange(len(leaves), desc="Getting token distance"):
        leaf = leaves[i]
        token_num = len(leaf)
        distance = np.zeros((token_num, token_num))
        ast_distance = dict(ast_distance_list[i])
        for j in range(token_num):
            for k in range(token_num):
                if leaf[k] in ast_distance[leaf[j]].keys():
                    distance[j][k] = ast_distance[leaf[j]
                                                  ][leaf[k]]  # just token distance
        distance_list.append(distance)

    print('distance_list 0: ')
    print(distance_list[0])

    pickle.dump(distance_list, open(args.cache_path +
                "/distance_list_mf"+args.pickle_suffix, 'wb'))
    print("wrote distance_list to", args.cache_path +
          "/distance_list_mf"+args.pickle_suffix)
    return distance_list

def compare_token_attention_and_distance_by_case(args, attention_list, distance_list, token_type_list):
    quantile_threshold = args.quantile_threshold
    print('len(attention_list):', len(attention_list))
    print('len(distance_list):', len(distance_list))
    assert len(attention_list) == len(distance_list)
    case_count = len(attention_list)
    token_type_len = 0
    loss_list = []

    for i in trange(case_count, desc="Compareing token attention and distance by case"):
        attention = attention_list[i]
        distance = distance_list[i]
        token_type = token_type_list[i]
        if not attention.any():  # code may not include any "identifier" type
            continue
        mask_att = attention > np.quantile(attention, quantile_threshold)
        token_type_len += mask_att.shape[0]
        mask_dist = distance < np.quantile(distance, 1-quantile_threshold)
        total_num = reduce(lambda x, y: x * y, attention.shape)
        weight_and = mask_att & mask_dist
        weight_or = mask_att | mask_dist
        weight_and_sum = sum(weight_and.sum(axis=0))
        weight_or_sum = sum(weight_or.sum(axis=0))
        if weight_or_sum == 0:  # identifier token num smaller than 3, so nothing will be larger than quantile, so that weight_or_sum=0
            continue
        loss = float(weight_and_sum)/weight_or_sum
        loss_list.append(loss)

    print("loss=sum_and/sum_or:", np.mean(loss_list))
    with open(os.path.join(str(args.cache_path), "loss_mf.txt"), 'w') as loss_txt:
        loss_txt.write(
            "\n attention threshold[0]:"+str(np.quantile(attention_list[0], quantile_threshold))+"\n")
        loss_txt.write(
            "distance threshold[0]:"+str(np.quantile(distance_list[0], quantile_threshold))+"\n")
        loss_txt.write("loss=sum_and/sum_or:"+str(np.mean(loss_list))+"\n")
    with open(os.path.join("./", "score_per_head_mf.txt"), 'a') as loss_txt:
        loss_txt.write(str(args.layer_num)+' '+args.model_name+' ' +
                       args.task+' '+args.sub_task+' '+str(np.mean(loss_list))+"\n")
    return np.mean(loss_list)


def get_sast(T, leaves, tokens_dict, tokens_type_dict):
    # add subtoken edges and Data flow edges to T
    T = nx.Graph(T)
    subtoken_edges = []
    dataflow_edges = []
    identifier_dict = {}
    i = 0
    for leaf in leaves:
        token = tokens_dict[leaf]
        token_type = tokens_type_dict[leaf]
        if token_type == 'identifier':
            if token not in identifier_dict:
                identifier_dict[token] = leaf
            else:
                dataflow_edges.append((identifier_dict[token], leaf))
                identifier_dict[token] = leaf
        if i > 0:
            subtoken_edges.append((old_leaf, leaf))
        old_leaf = leaf
        i += 1
    T.add_edges_from(subtoken_edges)
    T.add_edges_from(dataflow_edges)
    return T  # new_T


def get_ast_and_token(args, examples, parser, lang):
    ast_list = []
    sast_list = []
    tokens_list = []
    tokens_type_list = []
    logger.info("Parse AST trees and obtain leaf tokens")

    i = 0
    j = 0
    leaves_list = []
    for example in tqdm(examples, desc="Getting ast and token"):
        ast_example = get_ast_nx(example, parser, lang)
        G = ast_example.ast
        ast_list.append(G)
        T = nx.dfs_tree(G, 0)
        leaves = [x for x in T.nodes() if T.out_degree(x) ==
                  0 and T.in_degree(x) == 1]  # save all child node
        leaves_list.append(leaves)
        tokens_dict = {}
        tokens_type_dict = {}
        for leaf in leaves:
            feature = G.nodes[leaf]['features']
            if feature.type != 'comment':
                start = feature.start_point
                end = feature.end_point
                token = index_to_code_token([start, end], ast_example.source)
                if i == 0:
                    print('leaf: ', leaf, 'start: ', start,
                          ', end: ', end, ', token: ', token)
                tokens_dict[leaf] = token
                tokens_type_dict[leaf] = feature.type
        new_T = get_sast(T, tokens_dict.keys(), tokens_dict, tokens_type_dict)
        sast_list.append(new_T)
        if i == 0:
            nx.write_gpickle(T, os.path.join(
                str(args.cache_path), "T_example0.pickle"))
            nx.write_gpickle(new_T, os.path.join(
                str(args.cache_path), "sasT_example0.pickle"))
        if i == 0 or j == 348:
            print(j, "th code of T:")
            print(T.nodes)
            print(T.edges)

            print(j, "th code of sast T:")
            print(new_T.nodes)
            print(new_T.edges)
            print('raw_code: ', ast_example.source)
            print('leaves: ', leaves)
            i += 1

        tokens_list.append(tokens_dict)
        tokens_type_list.append(tokens_type_dict)
        j += 1
    
    print('tokens 0: ')
    print(tokens_list[0])
    print('tokens type 0: ')
    print(tokens_type_list[0])
    if not args.upgraded_ast:
        print("use old ast")
        sast_list = ast_list
    return ast_list, sast_list, tokens_list, tokens_type_list


def main():
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    set_dist(args)
    set_seed(args)
    set_hyperparas(args)

    print("frequent_type or not:", args.frequent_type)
    print("upgraded_ast or not:", args.upgraded_ast)
    suffix_dict = {1: {1: ".pickle", 0: "_ft_sa.pickle"},
                   0: {1: "_at_ua.pickle", 0: "_at_sa.pickle"}}
    args.pickle_suffix = suffix_dict[int(
        args.frequent_type)][int(args.upgraded_ast)]
    # rename pickle suffix as args.pickle_suffix[args.frequent_type][args.pickle_suffix]
    # to discriminate freqent_type or not and upgraded_ast or not
    # ft:frequent_type ua:upgraded_ast at:all_type sa:standard_ast

    logger.info(args)

    if args.task in ['summarize', 'translate']:
        config, model, tokenizer = bulid_or_load_gen_model(args)

    model_dict = os.path.join(
        args.output_dir, 'checkpoint-best-ppl/pytorch_model.bin')
    model.load_state_dict(torch.load(model_dict))

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    pool = multiprocessing.Pool(args.cpu_count)
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(
        args.data_dir, args.task, args.sub_task)
    examples, data = load_and_cache_gen_data(
        args, args.train_filename, pool, tokenizer, 'attention', is_sample=True, is_attention=True)

    Language.build_library(
        'build/my-language.so',
        [
            '/data/code/tree-sitter/tree-sitter-ruby',
            '/data/code/tree-sitter/tree-sitter-javascript',
            '/data/code/tree-sitter/tree-sitter-go',
            '/data/code/tree-sitter/tree-sitter-python',
            '/data/code/tree-sitter/tree-sitter-java',
            '/data/code/tree-sitter/tree-sitter-php',
        ]
    )
    language = Language('build/my-language.so', args.sub_task)
    parser = Parser()
    parser.set_language(language)

    ast_list, sast_list, tokens_list, tokens_type_list = get_ast_and_token(
        args, examples, parser, args.sub_task)
    attention_numpy, subtokens_list = get_attention_and_subtoken(
        args, data, model, tokenizer)

    subtoken_numbers_list, tokens_list_type_output = number_subtoken(
        args, subtokens_list, tokens_list, tokens_type_list, tokenizer, False)
    attention_list, leaves = get_token_attention(
        args, attention_numpy,  subtoken_numbers_list, 'mean')
    if args.frequent_type:
        freq_type_list = get_freq_type_list(
            args, attention_list, tokens_list_type_output)
        subtoken_numbers_list, tokens_list_type_output = number_subtoken(
            args, subtokens_list, tokens_list, tokens_type_list, tokenizer, freq_type_list)
    attention_list, leaves = get_token_attention(
        args, attention_numpy,  subtoken_numbers_list, 'mean')

    distance_list = get_token_distance(
        args, leaves, ast_list, sast_list,  distance_metric='shortest_path_length') 

#     attention_list = pickle.load(open(args.cache_path + "/attention_list_mf"+args.pickle_suffix, "rb"))
#     distance_list = pickle.load(open(args.cache_path + '/distance_list_mf'+args.pickle_suffix, "rb"))
#     tokens_list = pickle.load(open(args.cache_path + '/tokens_list_mf'+args.pickle_suffix, "rb"))
#     tokens_list_type_output = pickle.load(open(args.cache_path + '/tokens_list_type_mf'+args.pickle_suffix, "rb"))
    compare_token_attention_and_distance_by_case(
        args, attention_list, distance_list, tokens_list_type_output)


if __name__ == "__main__":
    main()
