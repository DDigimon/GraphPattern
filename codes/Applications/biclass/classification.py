from datasets import load_dataset
import sys
import utils
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import networkx as nx
from tqdm import tqdm 
import json 
import pickle as pkl
import torch
import re 
import interface

from graph_nlp import graph_txt
from graph_patterns import pattern_generation

import warnings
warnings.filterwarnings("ignore")


# --------------------- Parse -------------------------
# hyper-params
config = {}


config['dataset'] = 'ogbg-molbbbp'
config['process'] = 'classification'
config['num'] = 25
dataname = config['dataset']
role = 'an expert in molecular chemistry'
feature = 'atom'


deployment_name = 'GPT-4o'
model_name = f"{deployment_name}-{dataname}-all"

method = 'edge'
config['seed'] = 2024
config['batch_size'] = 5  # 188 * 0.8 / 20 = 9
config['split'] = 0.8  # training and test split
role_play = f"You are {role}. "
gold_set_list = ['Set1', 'Set2']
data_des = 'The graphs of compounds include binary labels on their permeability properties to the blood brain barrier.'

# input data
# loader = utils.read_dataset(config)
# utils.statistics(loader, config, feature)

pattern_dict = {}
for gold_set in gold_set_list:  # read patterns
    with open(f'response/pattern/{dataname}_{gold_set}_{method}_{deployment_name}.pkl','rb') as f:
            pattern_dict[gold_set] = pkl.load(f)


# ------------------- Prompting ----------------------------
node_num = 0
graph_num = 0
prompt = f'You are an expert at classifying different types of graphs based on whether they contain specific patterns. The first type of patterns such as:'
count = 0
pattern_list = []
for idx, pattern in enumerate(pattern_dict[gold_set_list[0]][1:]):
    if feature == 'atom':
        edge_list, atom_dict = utils.extract_text_edge(pattern)
        G = nx.Graph()
        G.add_edges_from(edge_list)
        G, atom_dict = utils.remapping(G, atom_dict)  # re-rank node ID
        # edge_num = len(G.edges)
        graph_text = graph_txt(G, method, atom_dict, feature)
    else:
        edge_list = utils.extract_edge(pattern)
        G = nx.Graph()
        G.add_edges_from(edge_list)
        G = utils.remapping_(G)
        graph_text = graph_txt(G, method)
    
    node_num += len(G.nodes)
    graph_num += 1
    temp = utils.graph_unique(graph_text, pattern_list)
    if temp == True:
        pattern_list.append(graph_text)
        count += 1
        prompt+= f'\nPatterns for Type 1: No.{count}:\n'+graph_text  # +f'Label:{np.argmax(labels[train_idx],-1)}'

# prompt += '\n There is no significant patterns in the second type of molecules.'
prompt += '\nThe second type of patterns are like'
count = 0
for idx, pattern in enumerate(pattern_dict[gold_set_list[1]][1:]):
    if feature == 'atom':
        edge_list, atom_dict = utils.extract_text_edge(pattern)
        G = nx.Graph()
        G.add_edges_from(edge_list)
        G, atom_dict = utils.remapping(G, atom_dict)  # re-rank node ID
        # edge_num = len(G.edges)
        graph_text = graph_txt(G, method, atom_dict, feature)
    else:
        edge_list = utils.extract_edge(pattern)
        G = nx.Graph()
        G.add_edges_from(edge_list)
        G = utils.remapping_(G)
        graph_text = graph_txt(G, method)
    
    node_num += len(G.nodes)
    graph_num += 1
    temp = utils.graph_unique(graph_text, pattern_list)
    if temp == True:
    # edge_num = len(G.edges)
        pattern_list.append(graph_text)
        count += 1
        prompt += f'\nPatterns for Type 2: No.{count}:\n'+graph_text  # +f'Label:{np.argmax(labels[train_idx],-1)}'
    
    if count > 10:  # prevent from out of token limitation.
        break

avg_nodes = node_num / graph_num
print(avg_nodes)
# print(prompt)
sys.exit()

#  --------------- downstream task -------------------------
input_txt=[]
labels = []
for idx, batch in enumerate(loader):
    pos_databatch, neg_databatch = batch
    current_batch = len(pos_databatch)

    pos_graphs, pos_atoms, pos_y = utils.read_graph(pos_databatch, config['dataset'])
    neg_graphs, neg_atoms, neg_y = utils.read_graph(neg_databatch, config['dataset'])
    
    # positive test samples
    if feature == 'atom':
        for idx, (graph, atom) in enumerate(zip(pos_graphs, pos_atoms)):  # Set1
            txt = prompt + '\nNow, please identify which type the given graph is most likely to belong to. The graph is shown as: ' + graph_txt(graph, method, atom, feature)
            txt+='Choosing the answer from "It should be in Type 1" or "It should be in Type 2". Notably, the presence of distinct patterns in a graph suggests a higher probability of its association with a specific class. '
            input_txt.append(txt)
            labels.append(pos_y[idx][0])

    else:
        for idx, graph in enumerate(pos_graphs):  # Set1
            txt = prompt + '\nNow, please identify which type the given graph is most likely to belong to. The graph is shown as: ' + graph_txt(graph, method)
            txt+='Choosing the answer from "It should be in Type 1" or "It should be in Type 2". Notably, the presence of distinct patterns in a graph suggests a higher probability of its association with a specific class. '
            input_txt.append(txt)
            labels.append(pos_y[idx][0])            

    
    # negative test samples
    if feature == 'atom':
        for idx, (graph, atom) in enumerate(zip(neg_graphs, neg_atoms)):  # Set2
            txt = prompt + '\nNow, please identify which type the given graph is most likely to belong to. The graph is shown as: ' + graph_txt(graph, method, atom, feature)
            txt+='Choosing the answer from "It should be in Type 1" or "It should be in Type 2". Notably, the presence of distinct patterns in a graph suggests a higher probability of its association with a specific class. '
            input_txt.append(txt)
            labels.append(neg_y[idx][0])
    else:
        for idx, graph in enumerate(neg_graphs):
            txt = prompt + '\nNow, please identify which type the given graph is most likely to belong to. The graph is shown as: ' + graph_txt(graph, method)
            txt+='Choosing the answer from "It should be in Type 1" or "It should be in Type 2". Notably, the presence of distinct patterns in a graph suggests a higher probability of its association with a specific class. '
            input_txt.append(txt)
            labels.append(neg_y[idx][0])    

    

response_file= []
right_count = 0
confuse_count = 0
for idx, input_text in tqdm(enumerate(input_txt),total=len(input_txt)):
    if 'o1m' in deployment_name:
        responses = interface._get_response(deployment_name, input_text)
    else:
        responses = interface._get_gpt(input_text, deployment_name, model_name)
    
    responses_dicts={}
    responses_dicts['response']= responses 
    responses_dicts['text']=input_text
    responses_dicts['idx']=idx
    if torch.is_tensor(labels[idx]):
        label = str(labels[idx].numpy())
    else:
        label = labels[idx]
    responses_dicts['label'] = label
    response_file.append(responses_dicts)
    
    try:
        answer = re.findall(r'It should be in Type \d', responses)
        predict = re.findall(r'\d', answer[0])
        if int(predict[0]) == 1 and labels[idx] == 1:
            right_count += 1
        elif int(predict[0]) != 1 and labels[idx] != 1:
            right_count += 1
    except:
        confuse_count += 1
accuracy = right_count / (len(input_txt) - confuse_count)
print('accuracy=', accuracy)
print('confuse_case =', confuse_count)
with open(f'response/classification/{dataname}_{method}_{deployment_name}.json','w') as f:
    json.dump(response_file, f)