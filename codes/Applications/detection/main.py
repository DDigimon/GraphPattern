from datasets import load_dataset
import sys
import networkx as nx
from networkx.algorithms import isomorphism
from tqdm import tqdm 
import json 
import pickle as pkl
import re 
import random 
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# PyG
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid

# local sources
import utils
from graph_nlp import graph_txt
from graph_patterns import pattern_generation
from response import get_response

from torch_geometric.datasets import Planetoid  # "Cora", "CiteSeer" and "PubMed"
from torch_geometric.datasets import AttributedGraphDataset  # "Wiki", "Cora" "CiteSeer", "PubMed", "BlogCatalog", "PPI", "Flickr", "Facebook", "Twitter", "TWeibo", "MAG"

import warnings
warnings.filterwarnings("ignore")


# input data
dataname = 'alkane_carbonyl'
data = np.load(f'data/{dataname}.npz', allow_pickle=True)
lst = data.files
smiles = data['smiles']
api = ''

# hyper params
seed = 2024
utils.setseed(seed)
test_size = 200
feature = 'atom'
deployment_name = 'GPT-4'
model_name = f'{deployment_name}-{dataname}-all'
desc = dict()
if dataname == 'benzene':
    pattern_name = 'benzene ring'
    desc = '(Node 0 Atom C, Node 1 Atom C), (Node 1 Atom C, Node 2 Atom C), (Node 2 Atom C, Node 3 Atom C), (Node 3 Atom C, Node 4 Atom C), (Node 4 Atom C, Node 5 Atom C)'
elif dataname == 'alkane_carbonyl':
    pattern_name = 'Alkane Carbonyl which contains an unbranched alkane and a carbonyl functional group'
    desc = '(Node 0 Atom C, Node 1 Atom O), (Node 0 Atom C, Node 2 Atom C), (Node 2 Atom C, Node 3 Atom O)'
    
elif dataname == 'fluoride_carbonyl':
    pattern_name = 'Fluoride Carbonyl whcih contains a fluoride (F) and a carbonyl (C=O) functional group'
    desc = '(Node 0 Atom C, Node 1 Atom O), (Node 2 Atom C, Node 3 Atom F)'


edge, atom = utils.extract_text_edge(desc)
P = nx.Graph()
P.add_edges_from(edge)

pos_smiles, neg_smiles = utils.sampling(smiles)
sample_smiles = pos_smiles[:int(test_size/2)]
temp_smiles = neg_smiles[:int(test_size/2)]
print('pos sample number =', len(sample_smiles))
print('neg sample number =', len(temp_smiles))
sample_smiles.extend(temp_smiles)


avg_node_num, avg_edge_num, avg_density = utils.statistics(pos_smiles, dataname)
print('positive node, edge, density =', avg_node_num, avg_edge_num, avg_density)

avg_node_num, avg_edge_num, avg_density = utils.statistics(neg_smiles, dataname)
print('negative node, edge, density =', avg_node_num, avg_edge_num, avg_density)

avg_node_num, avg_edge_num, avg_density = utils.statistics(sample_smiles, dataname)
print('node, edge, density =', avg_node_num, avg_edge_num, avg_density)

inference = True
for method in ['node', 'edge']:
    pattern = graph_txt(P, method, dicts=atom, nodes_feature=feature)
    response_file = []
    confuse_count = 0
    y_true = []
    y_pred = []
    idx_list = np.array(np.arange(len(smiles)))
    for i in tqdm(range(len(sample_smiles)), total=test_size, desc=f'Detecting on {dataname} using {deployment_name} and {method} description ...'):
        sample = sample_smiles[i]
        txt = f'In the context of molecular biology, you have been provided with a pattern motif to compare against a test molecule graph. \nFirst, the pattern is a {pattern_name}, represented as ' + pattern
        idx = sample[0]
        smile = sample[1]
        label = sample[-1]
        
        
        G, atoms = utils.smile2graph(smile, dataname)
        
        txt += f'\nSecond, the test molecule {smile} can be shown as' + graph_txt(G, method, dicts=atoms, nodes_feature=feature)
        txt += '\nNow, please determine whether the pattern motif exists in the molecule graph by selecting either **The pattern does exist** or **The pattern does not exist**. Please consider from the perspective of molecular biology, and do not be too rigid.'
        input_text=txt

        responses = get_response(deployment_name, input_text)
        responses_dicts={}
        responses_dicts['response']=responses
        responses_dicts['text']=input_text
        responses_dicts['idx']=idx
        response_file.append(responses_dicts)
        response_file.append(responses_dicts)
        

        found_text = re.findall(r'\*\*(.*?)\*\*', responses)
        try:
            if 'not exist' not in found_text[0]:
                y_pred.append(1)
                y_true.append(label)
            elif 'not exist' in found_text[0]:
                y_pred.append(0)
                y_true.append(label)
            else:
                confuse_count += 1
        except:
            confuse_count += 1
        
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    print("{:.2f}".format(auc), "{:.2f}".format(f1), "{:.2f}".format(acc))
    print('confuse =', confuse_count)
    with open(f'response/summary/{dataname}_{method}_{deployment_name}.json','w') as f:
        json.dump(response_file, f)
