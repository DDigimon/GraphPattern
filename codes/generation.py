

import os
import networkx as nx
import pickle
import warnings
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Union, Any, List
import numpy as np
import random

import networkx as nx
import random
from networkx.algorithms import isomorphism

from graph_patterns import find_pattern_list,direct_judge


from typing import Callable



###
def get_potential_edges(directed: bool, nodes):
    potential_edges = []
    if directed:
        for u in nodes:
            for v in nodes:
                if u != v:
                    potential_edges.append((u, v))
    else:
        sorted_node = sorted(nodes)
        for i in range(len(sorted_node) - 1):
            for j in range(i + 1, len(sorted_node)):
                potential_edges.append((sorted_node[i], sorted_node[j]))

    return potential_edges

    
def generate_graph(N, M, directed = False, weight_func: Callable = lambda:None, seed = None):
    if M < 0 or (M > N*(N-1)//2 and not directed) or (M > N*(N-1) and directed):
            raise ValueError("M must be between N-1 and N*(N-1)/2 for a simple graph.")
    
    if seed is not None:
        random.seed(seed)

    potential_edges = get_potential_edges(directed, list(range(N)))
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.add_nodes_from(range(N))

   
    while G.number_of_edges() < M:
        chosen_edge = potential_edges.pop(random.randint(0, len(potential_edges) - 1))
        G.add_edge(chosen_edge[0], chosen_edge[1], weight=weight_func())

    return G



def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    

from tqdm import tqdm
num_chaingraph = 0
num_treegraph = 0
num_normalgraph = 50
randbare_max = 50
difficulty='hard'

def iso_list_check(g,g_list):
    flag=True
    for g_in in g_list:
        # print(g_in)
        if len(g.edges())!=len(g_in.edges()):
            continue
        else:
            GM = isomorphism.GraphMatcher(g_in, g)
            if GM.is_isomorphic():
                # print(g_in,g)
                flag=False
    return flag



def single_generation(difficulty,source_path='',target_path='',given_condition='',target_condition='',max_num=25,max_density=1,directed=False):
    N_range_dict={'easy':list(range(5,15)),'mid':list(range(15,25)),'hard':list(range(25,35)),'ex':list(range(250,500))}
    M_num = 10
    g_list=[]
    g_list_dicts={}
    ori_list=[]
    edge_list=[]
    if source_path!='':
        print(source_path)
        with open(os.path.join(source_path,f'{difficulty}.pkl'),'rb')as f:
            ori_list=pickle.load(f)
        for g in ori_list:
            edge_num=len(g.edges())
            if edge_num not in edge_list:
                edge_list.append(edge_num)

    for n in N_range_dict[difficulty]:
        max_m=(n*(n-1)//2)*max_density
        if difficulty=='ex':
            max_num=1
        else:max_num=max_num
        # print(n)
        
        for i in tqdm(range(max_num)):
            patient_count=0
            patient=10
            while True:
                if patient_count>=patient:
                    break
                if difficulty!='ex':
                    m=random.choice([i for i in range(int(max_m*0.1) if int(max_m*0.1)>0 else 1,int(max_m))])
                else:
                    m=0.6
                # print(n,m)
                g=generate_graph(n,m,directed=directed)
                # print(g)
                if (n,m) not in g_list_dicts:
                    g_list_dicts[(n,m)]=[]
                # print(iso_list_check(g,g_list_dicts[(n,m)]),iso_list_check(g,ori_list))
                edge_num=len(g.edges())
                if given_condition!='':
                    condition_num=len(find_pattern_list(g,given_condition))
                else:
                    condition_num=1
                if target_condition!='':
                    target_num=len(find_pattern_list(g,target_condition))
                else:
                    target_num=0
                if edge_num not in edge_list and condition_num>0 and target_num==0:
                    g_list.append(g)
                    edge_list.append(edge_num)
                    g_list_dicts[(n,m)].append(g)
                    patient_count=0
                    break
                else:
                    if edge_num<50:
                        if iso_list_check(g,g_list_dicts[(n,m)]) and iso_list_check(g,ori_list) and condition_num>0 and target_num==0:
                            g_list.append(g)
                            g_list_dicts[(n,m)].append(g)
                            patient_count=0
                            break
                        else:
                            patient_count+=1
                    else:
                        patient_count+=1
                    
    print(len(g_list))
    if os.path.exists(target_path)==False:
        os.makedirs(target_path)
    with open(os.path.join(target_path,f'{difficulty}.pkl'),'wb') as f:
        pickle.dump(g_list,f)
    return g_list

def training_set_single():
    g_list=[]
    source_path='Dataset/Basic_tasks'
    target_path='Dataset/Basic_tasks/training'
    for difficult in ['easy','mid','hard']:
        g=single_generation(difficult,source_path=source_path,target_path=target_path,max_num=50,directed=True)
        g_list.extend(g)
    with open(os.path.join(target_path,'all_di.pkl'),'wb') as f:
        pickle.dump(g_list,f)
training_set_single()

def get_modifying_graph():
    g_list=[]
    for condition in  [('square','house'),('square','house'),('square','diamond'),('FFL','FBL')]:
        source_path=f'Dataset/Modify/{condition}'
        target_path=f'Dataset/Modify/training/{condition}'
        for difficult in ['easy','mid','hard']:
            g=single_generation(difficult,source_path=source_path,target_path=target_path,given_condition=condition[0],target_condition=condition[1],max_num=80,max_density=0.6,directed=direct_judge(condition[0]))
            g_list.append(g)
        with open(os.path.join(target_path,'all.pkl'),'wb') as f:
            pickle.dump(g_list,f)


def get_condition_graph():
    g_list=[]
    for condition in ['triangle','square','diamond','house','FBL','FFL','d-diamond']:#  ['triangle','square','diamond','house']:
        # print(direct_judge(condition))
        source_path=f'Dataset/Fresub/{condition}'
        target_path=f'Dataset/Fresub/training/{condition}'
        for difficult in ['easy','mid','hard']:
            g=single_generation(difficult,source_path=source_path,target_path=target_path,given_condition=condition,max_num=80,max_density=0.6,directed=direct_judge(condition))
            g_list.append(g)
        if os.path.exists(target_path)==False:
            os.makedirs(target_path)
        with open(os.path.join(target_path,'all.pkl'),'wb') as f:
            pickle.dump(g_list,f)



def get_iso_graphs():
    source_path='Dataset/Basic_tasks/training'
    target_path='Dataset/Iso/training'
    global_list=[]
    for difficulty in ['easy','mid','hard']:
        g_list=[]
        with open(os.path.join(source_path,f'{difficulty}.pkl'),'rb')as f:
            ori_list=pickle.load(f)

        for g1 in ori_list:
            nodes = list(g1.nodes())
            new_node_names = random.sample(range(100, 100 + len(nodes)), len(nodes))  # Creates unique random numbers as names
            mapping = dict(zip(nodes, new_node_names))
            g2 = nx.relabel_nodes(g1, mapping)
            g_list.append((g1,g2))
            global_list.append((g1,g2))
        if os.path.exists(target_path)==False:
            os.makedirs(target_path)

        with open(os.path.join(target_path,f'{difficulty}.pkl'),'wb') as f:
            pickle.dump(g_list,f)
    if os.path.exists(target_path)==False:
        os.makedirs(target_path)
        
    with open(os.path.join(target_path,'all.pkl'),'wb') as f:
        pickle.dump(global_list,f)

# get_modifying_graph()
# get_condition_graph()
# get_iso_graphs()

# get_iso_graphs()

