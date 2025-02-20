import pickle
import os
import sys
import numpy as np
from tqdm import tqdm
import json
sys.path.append('../')

from graph_nlp import graph_txt
from graph_patterns import pattern_generation,pattern_description,pattern_output_form
from responses import get_response
deployment_name='GPT-4'
prompt='zero'
for pattern in ['diamond']:
    for difficulty in ['easy','mid','hard']:
        for method in ['edge','node']:

            path=f'../Dataset/Fresub/{pattern}'
            with open(os.path.join(path,f'{difficulty}_idx.pkl'),'rb') as f:
                idx_list=pickle.load(f)
            with open(os.path.join(path,f'{difficulty}.pkl'),'rb') as f:
                graph_data=pickle.load(f)
            if difficulty!='easy':
                idx_list=idx_list[:50]
            input_txt=[]
            for selected_idx in idx_list:
                print(selected_idx)
                text='Consider the following graphs and summarize the common patterns in them.'
                for i,idx in enumerate(selected_idx):
                    graphs=graph_txt(graph_data[idx],method)
                    text+=f'No. {i+1}. {graphs}'
                text+='Show the common patterns as list as: [The pattern graphs are: Pattern #1: [(Node #1, Node #2), ...]; Pattern #2: [(Node #1, Node #2), ...]]'
                input_txt.append(text)
                
            print(len(input_txt))

            response_file=[]
            if os.path.exists(f'{prompt}_response/{deployment_name}_{pattern}_{method}.json'):
                continue

            for idx,input_text in tqdm(enumerate(input_txt),total=len(input_txt)):
                responses_dicts={}
                responses_dicts['response']=get_response(deployment_name,input_text)
                responses_dicts['text']=input_text
                responses_dicts['idx']=idx
                response_file.append(responses_dicts)
            if os.path.exists(f'{prompt}_response/{deployment_name}')==False:
                os.makedirs(f'{prompt}_response/{deployment_name}')
            with open(f'{prompt}_response/{deployment_name}/{deployment_name}_{difficulty}_{pattern}_{method}.json','w') as f:
                json.dump(response_file,f)