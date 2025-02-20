from datasets import load_dataset
import sys
import utils
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import networkx as nx
from tqdm import tqdm 
import json 
import pickle as pkl
import re 

from graph_nlp import graph_txt
from graph_patterns import pattern_generation
from response import get_response

import warnings
warnings.filterwarnings("ignore")

# --------------------- Parse -------------------------
# hyper-params
config = {}

config['dataset'] = 'ogbg-molbbbp'
config['process'] = 'summary'
config['num'] = 250
dataname = config['dataset']
role = 'an expert in molecular chemistry'
max_epoch = 2
feature = 'atom'
data_des = 'The graphs of compounds include binary labels on their permeability properties to the blood brain barrier.'

# fix config
deployment_name = 'GPT-4'
model_name = f"{deployment_name}-bbbp-all"
config['seed'] = 2024
config['batch_size'] = 5
config['split'] = 0.8  # if work, training and test split
role_play = f"You are {role}. "
skip_summary = False

# input data
loader = utils.read_dataset(config)
utils.statistics(loader, config, feature)

for method in ['node', 'edge']:
    for gold_set in ['Set1', 'Set2']:
        print('\nsetting =', method, gold_set)
        # -------------------- sampling ----------------------
        input_list = []
        for epoch in range(max_epoch):
            for i, batch in enumerate(loader):
                pos_databatch, neg_databatch = batch
                current_batch = len(pos_databatch)
                pos_graphs, pos_atoms, pos_y = utils.read_graph(pos_databatch, config['dataset'])
                neg_graphs, neg_atoms, neg_y = utils.read_graph(neg_databatch, config['dataset'])
                # prompting
                txt = role_play + f'Here are two sets of graphs. {data_des}. The first set are:\n'
                count = 0
                if feature == 'atom':
                    for idx, (graph, atom) in enumerate(zip(pos_graphs, pos_atoms)):  # Set1
                        print(graph)
                        sys.exit()
                        count += 1
                        txt += f'\nSet1: No. {count}:\n' + graph_txt(graph, method, dicts=atom, nodes_feature=feature)
                else:
                    for idx, graph in enumerate(pos_graphs):  # Set1
                        txt += f'\nSet1: No. {count}:\n' + graph_txt(graph, method)
                
                txt+='The second set are:\n'
                count = 0
                if feature == 'atom':
                    for idx, (graph, atom) in enumerate(zip(neg_graphs, neg_atoms)):  # Set2
                        count += 1
                        txt += f'\nSet2: No. {count}:\n' + graph_txt(graph, method, dicts=atom, nodes_feature=feature)
                else:
                    for idx, graph in enumerate(neg_graphs):
                        txt += f'\nSet2: No. {count}:\n' + graph_txt(graph, method)
                        
                if feature == 'atom':
                    txt+=f'Please find out the differences between the two sets and show the significant pattern in {gold_set} as list as: [(Node 0 Atom C, Node 1 Atom N), (Node 1 Atom N, Node 2 Atom O), ...]. '
                else:
                    txt+=f'Please find out the differences between the two sets and show the significant pattern in {gold_set} as list as: [(Node 1, Node 2), (Node 2, Node 3), ...]'
                input_list.append(txt)
                print(txt)
                sys.exit()

        # -------------------- summary ----------------------
        if skip_summary == False:
            response_file=[]
            for idx, input_text in tqdm(enumerate(input_list), total=len(input_list), desc=f'LLM Inference on {dataname} using {deployment_name} and {method} description.'):
                responses = get_response(deployment_name, input_text)
                responses_dicts={}
                responses_dicts['response']=responses
                responses_dicts['text']=input_text
                responses_dicts['idx']=idx
                response_file.append(responses_dicts)
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
                    if feature == 'atom':
                        edge_list, atom_dict = utils.extract_text_edge(pattern[0])
                    else:
                        edge_list = utils.extract_edge(pattern[0])

                    current_unique = utils.graph_unique(pattern[0], pattern_list)
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