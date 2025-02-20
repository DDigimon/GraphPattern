import pickle as pkl
import networkx as nx

import numpy as np
import sys
from tqdm import tqdm
sys.path.append('../')
from graph_nlp import graph_txt
from graph_patterns import pattern_generation,pattern_description
from response import get_response

# method='adj_n'
prompt='zero'
pattern='triangle'


import json
            
for pattern in ['claim_triangle']:
    for method in ['edge']:
        input_txt=[]
        if 'FFL' in pattern or 'FBL' in pattern or 'd-diamond' in pattern or 'vs' in pattern:
            graph_mode='directed'
        else:
            graph_mode='undirected'
        for _ in range(50):
            if 'claim' in pattern:
                my_pattern=pattern.split('_')[1]
                txt=f'Generate a {graph_mode} graph that includes only one {pattern_description[my_pattern]}, the node number is 20. Each node at least has one edge. The formulate of the graph should be:'
                if method=='edge':
                    if graph_mode=='directed':
                        graph_form='We define the direction of edges by indicating Node #1 pointing to Node #2 as (Node #1, Node #2). The edges of the graph should be list as: (Node #1, Node #2), ...'    
                    else:
                        graph_form='The edges of the graph should be list as: (Node #1, Node #2), ...'
                elif method=='node':
                    graph_form='For each node of the graph, the neighboring nodes are listed as follows: "Node #1: []\n Node #2: []"'
            else:
                pattern_graph=graph_txt(pattern_generation(pattern),method)
                txt=f'Generate a {graph_mode} graph that includes only one given pattern shown like: {pattern_graph}, the node number is 20. Each node at least has one edge. The formulate of the graph should be:'
                if method=='edge':

                    graph_form='The edges of the graph should be list as: (Node #1, Node #2), ...'
                elif method=='node':
                    graph_form='For each node of the graph, the neighboring nodes are listed as follows: "Node #1: []\n Node #2: []"'

            txt+=graph_form
            input_txt.append(txt)

        response_file=[]
        api_key, deployment_name, resource_name = get_secret_env_value()

        # while responses[0][0][0]=='Unknow error. No':
        with open(f'{prompt}_response/{deployment_name}_{pattern}_{method}.json','r') as f:
            ori_data=json.load(f)

        for idx,input_text in tqdm(enumerate(input_txt),total=len(input_txt)):
            responses_dicts={}
            responses_dicts['response']=get_response(deployment_name,input_text)
            responses_dicts['text']=input_text
            responses_dicts['idx']=idx
            # responses_dicts=take_response(input_text)
                
            response_file.append(responses_dicts)

        with open(f'{prompt}_response/{deployment_name}_{pattern}_{method}.json','w') as f:
            json.dump(response_file,f)