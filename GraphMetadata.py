from tree_sitter import Language, Parser
import numpy as np
import networkx as nx
import sys
import tqdm
sys.setrecursionlimit(5000)
# depth-first traverse
def traverse( cursor, G, came_up, node_tag, node_sum, parent_dict):
    '''
        cursor: the pointer of tree-sitter. An AST cursor is an object that is used to traverse an AST one node at a time
        G: the graph stored in the format of networkx
        came_up: used to denote whether the node is the first glance
        node_tag: the tag of this node
        node_sum: the number of distinct nodes
        parent_dict: used to store the parent nodes of all traversed nodes
    '''
    if not came_up:
        G.add_node(node_sum, features=cursor.node, label=node_tag)
        if node_tag in parent_dict.keys():
            G.add_edge(parent_dict[node_tag], node_tag)
        if cursor.goto_first_child():
            node_sum += 1
            parent_dict[node_sum] = node_tag
            traverse(cursor, G, came_up=False, node_tag=node_sum,
                    node_sum=node_sum, parent_dict=parent_dict)
        elif cursor.goto_next_sibling():
            node_sum += 1
            parent_dict[node_sum] = parent_dict[node_tag]
            traverse(cursor, G, came_up=False, node_tag=node_sum,
                    node_sum=node_sum, parent_dict=parent_dict)
        elif cursor.goto_parent():
            node_tag = parent_dict[node_tag]
            traverse(cursor, G, came_up=True, node_tag=node_tag,
                    node_sum=node_sum, parent_dict=parent_dict)
    else:
        if cursor.goto_next_sibling():
            node_sum += 1
            parent_dict[node_sum] = parent_dict[node_tag]
            traverse(cursor, G, came_up=False, node_tag=node_sum,
                    node_sum=node_sum, parent_dict=parent_dict)
        elif cursor.goto_parent():
            node_tag = parent_dict[node_tag]
            traverse(cursor, G, came_up=True, node_tag=node_tag,
                    node_sum=node_sum,  parent_dict=parent_dict)
class GraphMetadata():
    def __init__(self, args,examples, data, lang):
        self.args = args
        self.examples = examples
        self.data = data
        self.lang = lang
        LANGUAGE = Language('build/my-languages.so', self.lang)
        parser = Parser()
        parser.set_language(LANGUAGE)
        self.parser = parser

    def get_sast(self,T, leaves, tokens_dict, tokens_type_dict):
        # print("len(leaves), len(tokens_dict), len(tokens_type_dict)", len(leaves), len(tokens_dict), len(tokens_type_dict))
        
        # add subtoken edges and Data flow edges to T
        T = nx.Graph(T)
        subtoken_edges = []
        dataflow_edges = []
        identifier_dict = {}
        i = 0
        for leaf in leaves:
            token_type = tokens_type_dict[leaf]
            token = tokens_dict[leaf]
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
    def index_to_code_token(self,index, code):
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
    def get_ast_and_token(self,examples, parser, lang):
        ast_list = []
        sast_list = []
        tokens_list = []
        tokens_type_list = []
        for example in examples:#tqdm(examples,desc="Get ast, tokens and token types"):
            example_code = example.source
            tree = parser.parse(bytes(example_code, 'utf-8'))
            G = nx.Graph()
            cursor = tree.walk()
            try:
                traverse(cursor, G, came_up=False, node_tag=0, node_sum=0, parent_dict={})
            except RecursionError as e:
                continue
            ast_list.append(G)
            
            T = nx.dfs_tree(G, 0)
            leaves = [x for x in T.nodes() if T.out_degree(x) ==
                    0 and T.in_degree(x) == 1]
            tokens_dict = {}
            tokens_type_dict = {}
            for leaf in leaves[:]:
                feature = G.nodes[leaf]['features']
                if feature.type == 'comment':
                    leaves.remove(leaf)
                    T.remove_node(leaf)
                else:
                    start = feature.start_point
                    end = feature.end_point
                    token = self.index_to_code_token([start, end], example_code)
                    # print('leaf: ', leaf, 'start: ', start,
                    #     ', end: ', end, ', token: ', token)
                    tokens_dict[leaf] = token
                    tokens_type_dict[leaf] = feature.type
            assert len(leaves) == len(tokens_dict)
            new_T = self.get_sast(T, leaves, tokens_dict, tokens_type_dict)
            
            sast_list.append(new_T)
            tokens_list.append(tokens_dict)
            tokens_type_list.append(tokens_type_dict)
        print('ast list length', len(ast_list))
        print('tokens list length', len(tokens_list))
        print('tokens 0: ')
        print(tokens_list[0])
        print('tokens_type_list 0: ')
        print(tokens_type_list[0])
        return ast_list, sast_list, tokens_list, tokens_type_list, leaves
    def get_token_distance(self, args, leaves, ast_list, sast_list, distance_metric='shortest_path_length'):  # 4min
        print('get token distance')
        if distance_metric == 'shortest_path_length':
            ast_distance_list = [nx.shortest_path_length(ast) for ast in sast_list][0]
        elif distance_metric == 'simrank_similarity':
            ast_distance_list = [nx.simrank_similarity(ast) for ast in sast_list][0]
        distance_list = []
        leaf=leaves
        token_num = len(leaves)
        distance = np.zeros((token_num, token_num))
        ast_distance = dict(ast_distance_list)
        for j in range(token_num):
            for k in range(token_num):
                if leaf[k] in ast_distance[leaf[j]].keys():
                    distance[j][k] = ast_distance[leaf[j]
                                                ][leaf[k]]  # just token distance
        distance_list.append(distance)

        print('distance_list 0: ')
        print(distance_list[0])

        return distance_list