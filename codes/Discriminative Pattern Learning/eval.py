import pickle as pkl
import networkx as nx
import numpy as np
import random
import json
import numpy as np
import sys
from tqdm import tqdm
sys.path.append('../')
from graph_nlp import graph_txt
import re
from networkx.algorithms import isomorphism

with open('../Dataset/Summary/BA-2motif.pkl','rb') as fin:
    adjs,features,labels = pkl.load(fin)

with open('../Dataset/Summary/BA-train.pkl','rb') as fin:
    all_train_list=pkl.load(fin)

gold_set='Set2'
method='edge'
deployment_name='o1m'
data_base='BA2'
with open(f'zero_response/BA2_{gold_set}_{method}_{deployment_name}.json','r') as f:
    response_data=json.load(f)

all_trained_graph=[]
all_trained_inv=[]
for train_list in all_train_list:
    graph_train_lists=[]
    graph_train_inv=[]
    nums=int(len(train_list)/2)
    if gold_set=='Set2':
        train_inv=train_list[:nums]
        train_list=train_list[nums:]
    else:
        train_inv=train_list[nums:]
        train_list=train_list[:nums]
    for i in train_list:
        graph_train_lists.append(nx.from_numpy_array(adjs[i,:,:]))
    for i in train_inv:
        graph_train_inv.append(nx.from_numpy_array(adjs[i,:,:]))
    all_trained_graph.append(graph_train_lists)
    all_trained_inv.append(graph_train_inv)

def clean(lists):
    dicts={}
    new_list=[]
    for l in lists:
        if l not in dicts:
            dicts[l]=0
        dicts[l]+=1
    for key in dicts.keys():
        if dicts[key]==1:
            new_list.append(key)
    return new_list

def extract_graph_nodes(text):
    pattern = r"Node \d+: \[[\d, ]+\]"
    matches = re.findall(pattern, text, re.MULTILINE)
    if len(matches)==0:
        pattern = r"- \*\*Node [\d]:\*\* [\d, ]+"
        matches = re.findall(pattern, text, re.MULTILINE)
    edges=[]
    print(matches)
    if len(matches)==0:print(text)
    for lines in matches:
        neighbor_pattern=r'\d+'
        neighbor_string=lines
        neighbors=re.findall(neighbor_pattern,neighbor_string)
        numbers=[int(num) for num in neighbors]
        start_node=numbers[0]
        numbers=numbers[1:]
        for n in numbers:
            edges.append((start_node,n))
        clean(edges)
    return edges

def extraction_node(idx,text):
    pattern = re.findall(r'The pattern[s]? graphs is:\nNode [\d]+: \[[\d, ]+\](?:\nNode [\d]+: \[[\d, ]+\])+', text)
    if len(pattern)!=0:
        return extract_graph_nodes(pattern[0])
    else:
        if 'summary' or 'Summary' in text:
            print(text)
            print(idx)
        return pattern
    
def extract_text_edge(text):
    pattern = r'\(\d+, \d+\)'
    pattern2 = r'\(Node \d+, Node \d+\)'
    edges = re.findall(pattern, text)+re.findall(pattern2, text)
    edge_list=[]
    for e in edges:
        pattern=r'\d+'
        edge_pattern=re.findall(pattern,e)
        edge_list.append((int(edge_pattern[0]),int(edge_pattern[1])))
    clean(edge_list)
    return edge_list

def extraction_edge(idx,text):
    pattern = re.findall(r'\[([^\[\]]+(?:\[[^\[\]]+\][^\[\]]*)*)\]', text)
    if len(pattern)!=0:
        return extract_text_edge(pattern[-1])
    else:
        # print(text)
        pattern=extract_text_edge(text)
        return pattern
    
def get_graph(extracted_list):
    M = nx.Graph()
    M.add_edges_from(extracted_list)
    return M
'''
@Qihao
'''
def graph_compare(graph_list,M):
    flag=[]
    # print(len(graph_list))
    for g in graph_list:
        # print(len(g.edges()),len(M.edges()))
        # 25
        # edge number > 10
        if len(M.edges())>10:
            flag.append(0)
            continue
        # print(g,M)
        GM = isomorphism.GraphMatcher(g, M)
        if GM.subgraph_is_isomorphic()==False:
            flag.append(0)
        else:
            flag.append(1)
    if sum(flag)/len(flag)>0.9:return True
    else: return False

scussed_num=0
right_counting=0
pattern_path=f'./patterns/{deployment_name}_{method}_{gold_set}_patterns.pkl'
precision=0
recall=0
graph_list=[]
def graph_adding(graph_list,added_graph):
    flag=False
    for g in graph_list:
        # print(g)
        if nx.is_isomorphic(g,added_graph):
            flag=True
    if flag==False:
        graph_list.append(added_graph)
    return graph_list
for idx,data in tqdm(enumerate(response_data),total=len(response_data)):
    try:
        response=data['response']# [0][0][0]
    except:
        print(idx)
        continue
    # print(response)
    # print(data['text'])
    extracted_graph=extraction_edge(idx,response)
    extracted_graph=clean(extracted_graph)
    print(extracted_graph)
    if len(extracted_graph)!=0:
        scussed_num+=1
        graph=get_graph(extracted_graph)
        # print(graph)
        
        true_pattern=graph_compare(all_trained_graph[idx],graph)
        false_pattern=graph_compare(all_trained_inv[idx],graph)
        
        print(true_pattern,false_pattern)
        if true_pattern==1 and false_pattern==0:
            right_counting+=1
            graph_list=graph_adding(graph_list,graph)
        # print(right_counting)
    print(extracted_graph)
    # break
print(scussed_num,right_counting)
# with open(pattern_path,'wb') as f:
#     pkl.dump(graph_list,f)
    
                

