import pickle as pkl
import networkx as nx
import pickle
import numpy as np
import sys
from tqdm import tqdm
import os
sys.path.append('../')
from graph_nlp import graph_txt
from graph_patterns import pattern_generation,pattern_description
from response import get_response

# method='adj_n'
prompt='zero'
pattern='triangle'

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='GPT4o')
parser.add_argument("--given_api", type=str, default='')
parser.add_argument("--pattern", type=str, default='')
parser.add_argument("--difficulty", type=str, default='')
parser.add_argument("--method", type=str, default='')
args = parser.parse_args()
pattern=args.pattern
method=args.method
difficulty=args.difficulty
given_api=args.given_api

import json

# for deployment_name in ['GPT-4']:
deployment_name=args.model_name
if 'claim' in pattern:
    pattern_name=pattern.split('_')[1]
else:
    pattern_name=pattern
target_pattern=pattern_name.split(',')[1][2:-2]
print(target_pattern)

if 'FFL' in pattern or 'FBL' in pattern or 'd-diamond' in pattern or 'vs' in pattern:
    graph_mode='directed'
else:
    graph_mode='undirected'
# path="/egr/research-dselab/daixinna/LLMGB/Dataset/Modify/('square', 'diamond')/easy.pkl"# 
path=f'../Dataset/Modify/{pattern_name}/{difficulty}.pkl'

with open(path,'rb') as f:
    graphs=pickle.load(f)
if difficulty!='easy':
    new_graphs=[]
    with open(f'../Dataset/Modify/{pattern_name}/{difficulty}_idx.pkl','rb') as f:
        random_indices=pickle.load(f)
    for i in random_indices:
        new_graphs.append(graphs[i])
    graphs=new_graphs
input_txt=[]
for g in graphs:
    # print(g)
    based_graph=graph_txt(g,method)
    print(based_graph)
    txt=f'Modify the graph to include the given pattern. The graph is shown as {based_graph}'
    if 'claim' in pattern:
        txt+=f'The given pattern is {pattern_description[target_pattern]}\n'
    else:
        pattern_graph=graph_txt(pattern_generation(target_pattern),method)
        txt+=f'The given pattern is {pattern_graph}\n'
    txt+='The number of modified edges should be as less as possible.'
    graph_form='The edges of the final graph should be list as: [(Node #1, Node #2), ...]'
    txt+=graph_form
    input_txt.append(txt)

response_file=[]
# api_key, deployment_name, resource_name = get_secret_env_value()
if os.path.exists(f'{prompt}_response/{deployment_name}_{difficulty}_{pattern}_{method}.json'):
    with open(f'{prompt}_response/{deployment_name}_{difficulty}_{pattern}_{method}.json','r') as f:
        ori_data=json.load(f)

for idx,input_text in tqdm(enumerate(input_txt),total=len(input_txt)):
    responses_dicts={}
    responses_dicts['response']=get_response(deployment_name,input_text,given_api=given_api)
    responses_dicts['text']=input_text
    responses_dicts['idx']=idx
    print(responses_dicts['response'])
    response_file.append(responses_dicts)
if os.path.exists(f'{prompt}_response/{deployment_name}')==False:
    os.makedirs(f'{prompt}_response/{deployment_name}')
with open(f'{prompt}_response/{deployment_name}/{deployment_name}_{difficulty}_{pattern}_{method}.json','w') as f:
    json.dump(response_file,f)