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
from response import get_response
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from graph_nlp import graph_txt
from graph_patterns import pattern_generation

import warnings
warnings.filterwarnings("ignore")


# --------------------- Parse -------------------------
# hyper-params
config = {}

config['dataset'] = 'IMDB-MULTI'  # IMDB-MULTI, ENZYMES, Fingerprint
config['process'] = 'classification'
config['num'] = 20
dataname = config['dataset']
role = 'an expert in uncovering graph patterns'
feature = 'none'
deployment_name = 'GPT-4o'
model_name = f'{deployment_name}-{dataname}-cls3'
config['seed'] = 2024
config['batch_size'] = 5
config['split'] = 0.8  # training and test split
role_play = f"You are {role}. "
gold_set_list = ['Set1', 'Set2', 'Set3']
utils.set_seed(config['seed'])

# input data
labels = [1,2,3]
loader = utils.read_3_test(config, labels)
utils.statistics_3(loader, config)


for method in ['node', 'edge']:
    pattern_dict = {}
    for gold_set in gold_set_list:  # read patterns
        with open(f'response/pattern/{dataname}_{gold_set}_{method}_{deployment_name}.pkl','rb') as f:
                pattern_dict[gold_set] = pkl.load(f)


    # ------------------- Prompting ----------------------------
    prompt = 'You are an expert at classifying different types of graphs based on the presence of specific patterns. The first type of patterns such as:'
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
        
        temp = utils.graph_unique(graph_text, pattern_list)
        if temp == True:
            pattern_list.append(graph_text)
            count += 1
            prompt+= f'\nPattern Type 1: No.{count}:\n'+graph_text  # +f'Label:{np.argmax(labels[train_idx],-1)}'
        
        if count > 10:
            break

    # prompt += '\n There is no significant patterns in the second type of molecules.'
    prompt += '\nThe second type of patterns are'
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
            
        temp = utils.graph_unique(graph_text, pattern_list)
        if temp == True:
        # edge_num = len(G.edges)
            pattern_list.append(graph_text)
            count += 1
            prompt += f'\nPattern Type 2: No.{count}:\n'+graph_text  # +f'Label:{np.argmax(labels[train_idx],-1)}'
        
        if count > 10:  # prevent from out of token limitation.
            break

    # prompt += '\n There is no significant patterns in the second type of molecules.'
    prompt += '\nThe third type of patterns are'
    count = 0
    for idx, pattern in enumerate(pattern_dict[gold_set_list[2]][1:]):
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
            
        temp = utils.graph_unique(graph_text, pattern_list)
        if temp == True:
        # edge_num = len(G.edges)
            pattern_list.append(graph_text)
            count += 1
            prompt += f'\nPattern Type 3: No.{count}:\n'+graph_text  # +f'Label:{np.argmax(labels[train_idx],-1)}'
        
        if count > 10:  # prevent from out of token limitation.
            break

    #  --------------- downstream task -------------------------
    input_txt=[]
    labels = []
    for idx, batch in enumerate(loader):
        a_databatch, b_databatch, c_databatch = batch
        current_batch = len(a_databatch)

        a_graphs, a_y = utils.read_graph_3(a_databatch, config['dataset'])
        b_graphs, b_y = utils.read_graph_3(b_databatch, config['dataset'])
        c_graphs, c_y = utils.read_graph_3(c_databatch, config['dataset'])

        # positive test samples

        for idx, graph in enumerate(a_graphs):  # Set1
            txt = prompt + '\nPay attention, the test graph is shown as: ' + graph_txt(graph, method)
            txt+='\nNow, please identify which type the given graph is most likely to belong to and choose the answer from "It should be in Type 1", "It should be in Type 2", or "It should be in Type 3".'
            input_txt.append(txt)
            labels.append(1)            
        
        # negative test samples
        for idx, graph in enumerate(b_graphs):
            txt = prompt + '\nNow, please identify which type the given graph is most likely to belong to. The graph is shown as: ' + graph_txt(graph, method)
            txt+='Now, please identify which type the given graph is most likely to belong to and choose the answer from "It should be in Type 1", "It should be in Type 2", or "It should be in Type 3".'
            input_txt.append(txt)
            labels.append(2)    


        # negative test samples
        for idx, graph in enumerate(c_graphs):
            txt = prompt + '\nNow, please identify which type the given graph is most likely to belong to. The graph is shown as: ' + graph_txt(graph, method)
            txt+='Now, please identify which type the given graph is most likely to belong to and choose the answer from "It should be in Type 1", "It should be in Type 2", or "It should be in Type 3".'
            input_txt.append(txt)
            labels.append(3)    
        

    self_sum = True
    inference = False
    response_file = []
    y_true = []
    y_pred = []
    confuse_count = 0
    for idx, input_text in tqdm(enumerate(input_txt),total=len(input_txt), desc=f'Classification on {dataname} using {deployment_name} and {method} description.'):
        responses = get_response(deployment_name, input_text)
        responses_dicts={}
        responses_dicts['response']=responses
        responses_dicts['text']=input_text
        responses_dicts['idx']=idx
        response_file.append(responses_dicts)
        if torch.is_tensor(labels[idx]):
            label = labels[idx].numpy()
        else:
            label = labels[idx]
        responses_dicts['label'] = label
        response_file.append(responses_dicts)
        
        try:
            answer = re.findall(r'It should be in Type \d', responses)
            predict = re.findall(r'\d', answer[0])
            y_pred.append(int(predict[0]))
            y_true.append(label)
        except:
            confuse_count += 1

    # f1 = f1_score(y_true, y_pred)
    # auc = roc_auc_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    # print("F1 score: {:.2f}".format(f1))
    # print("Auc score :{:.2f}".format(auc))
    print("Accuracy :{:.2f}".format(acc))
    print('confuse =', confuse_count)
    with open(f'response/classification/{dataname}_{method}_{deployment_name}.json','w') as f:
        json.dump(response_file, f)
        