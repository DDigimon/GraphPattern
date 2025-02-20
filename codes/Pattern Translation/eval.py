import json
import pickle as pkl
import numpy as np
import re
import networkx as nx
import itertools
from networkx.algorithms import isomorphism
import sys
sys.path.append('../')
from graph_patterns import pattern_generation,find_pattern_list,direct_judge
from graph_nlp import graph_txt
from util import iso_list_check,graph_iso_check



ground_truth=[]
preds=[]
correct=[]

def extract_text_node(text):
    pattern = r"Node \#\d+: \[[\d, ]+\]"
    matches = re.findall(pattern, text, re.MULTILINE)
    if len(matches)==0:
        pattern = r"- \*\*Node [\d]:\*\* [\d, ]+"
        matches = re.findall(pattern, text, re.MULTILINE)
    edges=[]
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

def extract_text_edge(text):
    patterns=[r'\(\d+, \d+\)', r'\(\d+,\d+\)', r'\(Node \d+, Node \d+\)', r'\[\d+, \d+\]',r'\(N\d+, N\d+\)',r'\(N_\d+, N_\d+\)',r'\(N_\d+, N_\{\d+\}\)',r'\(N_\{\d+\}, N_\{\d+\}\)',r'\(\w, \w\)']
    edges=[]
    for pattern in patterns:
        edges.extend(re.findall(pattern,text))
    edge_list=[]
    for e in edges:
        pattern=r'\d+'
        edge_pattern=re.findall(pattern,e)
        if len(edge_pattern)!=0:
            edge_list.append((int(edge_pattern[0]),int(edge_pattern[1])))
        else:
            pattern=r'\w'
            edge_pattern=re.findall(pattern,e)
            edge_list.append((edge_pattern[0],edge_pattern[1]))
    clean(edge_list)
    return edge_list

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

def extract(text):
    try:
        match = re.search(r'It should be in set Set(\d+)', text)
        label = match.group(1)
        print(label)
        return label
    except:
        # print(text)
        return None

def construct_graphs(edges,direction=False):
    if direction:
        G=nx.DiGraph()
    else:
        G=nx.Graph()
    G.add_edges_from(edges)
    return G


def is_diamond_motif(G, nodes):
    """Check if a set of 4 nodes forms a diamond motif."""
    subgraph = G.subgraph(nodes)
    return subgraph.number_of_nodes() == 4 and subgraph.number_of_edges() == 5

def count_diamond_motifs(G):
    """Count the number of diamond motifs in the graph."""
    count = 0
    # Generate all combinations of 4 nodes
    for nodes in itertools.combinations(G.nodes(), 4):
        # print(nodes)
        if is_diamond_motif(G, nodes):
            count += 1
    return count

def check_triangle_num(G):
    triangles_dict = nx.triangles(G)
    total_triangles = sum(triangles_dict.values()) // 3
    return total_triangles

def counting_graph(G,pattern):
    lists=find_pattern_list(G,pattern)
    return len(lists)


prompt='zero'
deployment_name='llama'
dataset='BA2'
train_samples=30
method='edge'
template='claim_FBL'

with open(f'{prompt}_response/{deployment_name}_{template}_'+method+'.json','r') as f:
    response_file=json.load(f)

g_list=[]
for idx,data in enumerate(response_file):
    if 'response' in data and len(data['response'])>0:
        # print(data['text'])
        if 'gpt' in deployment_name.lower():
            response=data['response'][0][0][0]
        else:
            response=data['response']
        # print(response)
        if 'node' in method:
            graph_text=extract_text_node(response)
        elif 'edge' in method:
            graph_text=extract_text_edge(response)
        
        graph=construct_graphs(graph_text,direct_judge(template))
        
        # print(graph)
        triangle_num=len(find_pattern_list(graph,template))
        # triangle_num=counting_graph(graph,template)

        print(triangle_num)
        g_list.append(graph)
        if triangle_num==1:
            correct.append(1)
        else:
            print(response)
            print(graph_text)
            print(idx)
            correct.append(0)
counting=0
compares=0
from tqdm import tqdm
for idx_i,graph1 in tqdm(enumerate(g_list),total=len(g_list)):
    for idx_j,graph2 in enumerate(g_list):
        if idx_i!=idx_j:
            in_score=graph_iso_check(graph1,graph2)
            if in_score==0:
                counting+=1
            compares+=1
print(sum(correct)/len(correct),len(correct))
if counting==0:diversity=0
else:
    diversity=counting/compares
print(diversity)