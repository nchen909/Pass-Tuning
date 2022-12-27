import jieba
import re
import numpy as np
import re
from GraphMetadata import GraphMetadata
import multiprocessing
class Retriever(object):

    def __init__(self, args, examples, data, k=1):
        self.args = args
        self.examples = examples
        self.data = data
        self.k = k

    def get_retriever_metadata(self,args,examples, data):
        '''
        read prefix code from args.old_prefix_dir
        load to examples(raw code) and data(after encode to ids)
        GraphMetadata convert it to token_ids and distance_list
        return token_ids and distance_list for GAT init
        '''
        
        pool=multiprocessing.Pool(args.cpu_count)
        args.lang = get_lang_by_task(args.task, args.sub_task)
        graphmetadata=GraphMetadata(args, examples, data, args.lang)
        
        ast_list, sast_list, tokens_list, tokens_type_list, leaves =graphmetadata.get_ast_and_token(graphmetadata.examples, graphmetadata.parser, graphmetadata.lang)
        return tokens_list, tokens_type_list
        # tokens_ids=tokenizer.convert_tokens_to_ids(tokens_list[0].values())
        # distance_list=graphmetadata.get_token_distance(args, leaves, ast_list, sast_list, 'shortest_path_length')[0]
        # assert len(tokens_ids)==distance_list.shape[0]
        # if len(tokens_ids)>=args.gat_token_num:
        #     return tokens_ids[:args.gat_token_num], distance_list[:args.gat_token_num,:args.gat_token_num]
        # else:
        #     distance_list=np.pad(distance_list,((0,args.gat_token_num-len(tokens_ids)),(0,args.gat_token_num-len(tokens_ids))),'constant')
        #     tokens_ids=tokens_ids+[tokenizer.pad_token_id]*(args.gat_token_num-len(tokens_ids))
        #     assert len(tokens_ids)==distance_list.shape[0]
        #     return tokens_ids, distance_list
        #     # token_ids = tokenizer.convert_tokens_to_ids(tokens_list) 
        #     # if len(token_ids)<=args.max_source_length:
        #     #     padding_length = args.max_source_length - len(token_ids)
        #     #     token_ids += [tokenizer.pad_token_id]*padding_length
        #     # else:
        #     #     token_ids = token_ids[:args.max_source_length]
        #     # return token_ids

    def get_retrieve_id_list(self,args, examples, data, k=1):
        tokens_list, tokens_type_list = get_retriever_metadata(args,examples, data)
        from retriever.BM25 import BM25
        print("aaaaaaa")
        bm25 = BM25([i.values() for i in tokens_type_list])
        print("bbbbbbb")
        freq_types=bm25.get_freq_word(8)
        print("cccc")
        print("freq_token_types:{} for task:{}, lang:{}".format(freq_types,args.task, args.lang ))
        return bm25.get_top_k_related_ids(freq_types, k)

    def retrieve2file(self,args, examples, data, k=1):
        print('\n Retrieving...')
        if args.retriever_mode == 'random':
            retrieve_id_list = np.random.choice(len(examples), k, replace=False)
        elif args.retriever_mode == 'retrieve':
            retrieve_id_list = get_retrieve_id_list(args, examples, data, k)
        filename = get_filenames(
                    args.old_prefix_dir, args.task, args.sub_task, 'prefix')
        with open(filename,'w',encoding="utf-8") as f:
            for id_ in retrieve_id_list:
                print("retrieve case:\n")
                print(examples[id_].source,'\n')
                print('Writing retrieve file...')
                f.write(examples[id_].raw_line+'\n')

def get_sentences(doc):
    line_break = re.compile('[\r\n]')
    delimiter = re.compile('[，。？！；]')
    sentences = []
    for line in line_break.split(doc):
        line = line.strip()
        if not line:
            continue
        for sent in delimiter.split(line):
            sent = sent.strip()
            if not sent:
                continue
            sentences.append(sent)
    return sentences

if __name__ == '__main__':
    # 测试文本
    text = '''
    自然语言处理是计算机科学领域与人工智能领域中的一个重要方向。
    它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。
    自然语言处理是一门融语言学、计算机科学、数学于一体的科学。
    因此，这一领域的研究将涉及自然语言，即人们日常使用的语言，
    所以它与语言学的研究有着密切的联系，但又有重要的区别。
    自然语言处理并不是一般地研究自然语言，
    而在于研制能有效地实现自然语言通信的计算机系统，
    特别是其中的软件系统。因而它是计算机科学的一部分。
    '''
    sents = get_sentences(text)
    doc = []
    for sent in sents:
        words = list(jieba.cut(sent))
        doc.append(words)
    print(doc)
    s = BM25(doc)
    print(s.f)
    print(s.df)
    print(s.idf)
    print(s.simall(['自然语言', '计算机科学', '领域', '人工智能', '领域']))
    print(s.get_top_k_related_ids(['自然语言', '计算机科学', '领域', '人工智能', '领域'], 2))
    freq_word=s.get_freq_word(8)
    print(freq_word)#task,lang,freq_word
    print(s.get_top_k_related_ids(freq_word, 2))