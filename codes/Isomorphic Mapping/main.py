import os
import pickle
import sys
import json
from tqdm import tqdm
sys.path.append('../')
from graph_nlp import graph_txt
from graph_patterns import pattern_generation,pattern_description
from response import get_response
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='GPT4o')
parser.add_argument("--given_api", type=str, default='')
parser.add_argument("--difficulty", type=str, default='')
parser.add_argument("--method", type=str, default='')
args = parser.parse_args()
given_api=args.given_api
difficulty=args.difficulty
method=args.method

path='../Dataset/Iso'
prompt='zero'

deployment_name=args.model_name
if deployment_name=='gpt':
    deployment_name='GPT-4'
if deployment_name=='gpto':
    deployment_name='GPT-4o'
print(deployment_name)

with open(os.path.join(path,f'{difficulty}.pkl'),'rb') as f:
    g_list=pickle.load(f)

if difficulty!='easy':
    new_graphs=[]
    with open('../Dataset/sub_idx.pkl','rb')as f:
        sub_idx_list=pickle.load(f)
    for i in sub_idx_list:
        new_graphs.append(g_list[i])
    g_list=new_graphs
input_txt=[]
for g in g_list:
    g1,g2=g
    based_graph1=graph_txt(g1,method)
    based_graph2=graph_txt(g2,method)

    states_begin='Given a pair of isomorphic graphs, determine the node correspondence between the two graphs. The first graph is:\n'
    states_begin+=f'{based_graph1} The second graph is: {based_graph2}'
    states_begin+='Provide a node matching list such as "[Graph 1: Node #1 -> Graph 2: Node #2, ...]"'
    input_txt.append(states_begin)
response_file=[]


for idx,input_text in tqdm(enumerate(input_txt),total=len(input_txt)):
    responses_dicts={}
    responses_dicts['response']=get_response(deployment_name,input_text,given_api=given_api)
    responses_dicts['text']=input_text
    responses_dicts['idx']=idx
    response_file.append(responses_dicts)
if os.path.exists(f'{prompt}_response/{deployment_name}')==False:
    os.makedirs(f'{prompt}_response/{deployment_name}')
with open(f'{prompt}_response/{deployment_name}/{deployment_name}_{difficulty}_{method}.json','w') as f:
    json.dump(response_file,f)

