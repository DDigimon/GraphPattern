from datasets import load_dataset
import sys
import utils
from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
import networkx as nx
from tqdm import tqdm 
import json 
import pickle as pkl
import random
import re 

from graph_nlp import graph_txt
from graph_patterns import pattern_generation
from response import get_response

import warnings
warnings.filterwarnings("ignore")

# --------------------- Parse -------------------------
# hyper-params
config = {}

config['dataset'] = 'IMDB-MULTI'  # ENZYMES, Fingerprint, IMDB-MULTI
config['process'] = 'summary'
config['num'] = 100
dataname = config['dataset']
role = 'are an expert in identifying patterns within graphs'
max_epoch = 8
deployment_name = 'GPT-4o'
model_name = f'{deployment_name}-{dataname}-summary'
feature = 'none'

# fix config
config['seed'] = 2024
config['batch_size'] = 5  # 188 * 0.8 / 20 = 9
config['split'] = 0.8  # training and test split
role_play = f"You {role}. "
skip_summary = False
utils.set_seed(config['seed'])

dataloader, labels = utils.read_3(config)
utils.statistics_3(dataloader, config)

for method in ['node', 'edge']:
    for gold_set in ['Set1', 'Set2', 'Set3']:
        print('\nsetting =', method, gold_set)
        # -------------------- sampling ----------------------
        input_list = []
        for epoch in range(max_epoch):                
            for i, batch in enumerate(dataloader):
                txt = role_play + f'Here are several graphs in the {gold_set}:\n'
                for databatch in batch:
                    current_batch = len(databatch)
                    graphs, y = utils.read_graph_3(databatch, config['dataset'])
                    # prompting
                    count = 0
                    for idx, graph in enumerate(graphs):  # Set1
                        count += 1
                        if feature == 'atom':
                            txt += f'\nNo. {count}:\n' + graph_txt(graph, method)
                        else:
                            txt += f'\nNo. {count}:\n' + graph_txt(graph, method)
                            
                    txt+=f'Please show the significant pattern in {gold_set} as list as: [(Node 1, Node 2), (Node 2, Node 3), ...]'
                input_list.append(txt)


        # -------------------- summary ----------------------
        if skip_summary == False:
            response_file=[]
            for idx, input_text in tqdm(enumerate(input_list), total=len(input_list), desc=f'LLM Inference on {dataname} using {deployment_name} and {method} description'):
                
                responses = get_response(deployment_name, input_text)
                
                responses_dicts={}
                responses_dicts['response']=responses
                responses_dicts['text']=input_text
                responses_dicts['idx']=idx
                response_file.append(responses_dicts)

            with open(f'response/summary/{dataname}_{gold_set}_{method}_{deployment_name}.json','w') as f:
                json.dump(response_file, f)


        else:  # read summary responses from files.
            with open(f'response/summary/{dataname}_{gold_set}_{method}_{deployment_name}.json','r', encoding='utf-8') as f:
                response_file = json.load(f)


        # ------------------ Pattern Filtering ---------------------------
        skip_filtering = False
        if skip_filtering == False:
                             
            pattern_list = ['pattern']
            
            unique_pattern_num = 0
            available_pattern_num = 0
            for idx, summary in enumerate(response_file):
                # print('\n epoch', idx)
                input_ = summary['text']
                try:
                    txt = summary['response']
                except:
                    txt = 'Response Error.'
                    
                pattern = re.findall(r'\[(.*?)\]', txt)  # extract all content within [ ]
                try:
                    edge_list = utils.extract_edge(pattern[0])
                    current_unique = utils.graph_unique(pattern[0], pattern_list)  # remove duplicates
                    available_pattern_num += 1
                    if current_unique == True:
                        unique_pattern_num += 1
                        pattern_list.append(pattern[0])
                except:
                    continue
            print(pattern_list)
            print('available pattern number =', available_pattern_num)
            print('unique patttern number =', unique_pattern_num)
            with open(f'response/pattern/{dataname}_{gold_set}_{method}_{deployment_name}.pkl','wb') as f:
                pkl.dump(pattern_list, f)

        else:
            with open(f'response/pattern/{dataname}_{gold_set}_{method}_{deployment_name}.pkl','rb') as f:
                pattern_list = pkl.load(f)