import pandas as pd 
import numpy as np 
import networkx as nx
import torch
import random
from tqdm import tqdm
import re
import sys
import math
from rdkit import Chem
from rdkit.Chem import rdmolops




def statistics(samples, dataname):
    node_num = 0
    edge_num = 0
    density = 0
    test_size = len(samples)
    for i in range(test_size):
        sample = samples[i]
        smile = sample[1]
        
        G, atoms = smile2graph(smile, dataname)
        node_num += len(G.nodes)
        edge_num += len(G.edges)
        density += len(G.edges) / (len(G.nodes) * len(G.nodes))

    avg_node_num = node_num / test_size
    avg_edge_num = edge_num / test_size
    avg_density = density / test_size
    return avg_node_num, avg_edge_num, avg_density


def extract_communities(data, cls_dict):
    edge_list = data.edge_index
    cms_dict = dict()
    for i, cls in tqdm(enumerate(cls_dict), total=len(cls_dict), desc='Sampling communities ...'):
        cls = cls_dict[i]
        cms_edges = []
        for j in range(edge_list.shape[1]):
            edge = edge_list[:, j]
            if edge[0] in cls and edge[1] in cls:
                cms_edges.append((str(edge[0].item()), str(edge[1].item())))
        cms_dict[i] = cms_edges 
    
    return cms_dict


def wholegraph(data):
    edge_list = data.edge_index
    output = []
    for j in range(edge_list.shape[1]):
        edge = edge_list[:, j]
        output.append((str(edge[0].item()), str(edge[1].item())))
    G = nx.DiGraph()
    G.add_edges_from(output)
    return G


def subgraph_sampling(data, size, num, seed=2024, sample_mode='node'):
    torch.manual_seed(seed)
    edge_list = data.edge_index  # torch.tensor([2, edge_num])
    node_list = data.x
    node_num, node_feat_num = node_list.shape
    _, edge_num = edge_list.shape
    
    # 
    if sample_mode == 'node':
        sample_dict = dict()
        for n in range(num):
            cms_edges = []
            nodes = torch.randint(node_num, (size, ))
            for j in range(edge_num):
                edge = edge_list[:, j]
                if edge[0] in nodes and edge[1] in nodes:
                    cms_edges.append((str(edge[0].item()), str(edge[1].item())))
            sample_dict[n] = cms_edges 
                
    elif sample_mode == 'edge':
        sample_dict = dict()
        for n in range(num):
            subgraph_edges = []
            index = torch.randint(edge_num, (size, ))
            subgraph = edge_list[:, index]
            for l in range(len(index)):
                subgraph_edges.append((str(subgraph[0, l].item()), str(subgraph[1, l].item())))
            sample_dict[n] = subgraph_edges
    
    else:
        print('Error. Please input correct sample_mode, [node, edge]')
        sys.exit()
    
    return sample_dict  


def remapping_(G):
    node_list = list(G.nodes())
    mapping = {}
    for i in range(len(node_list)):
        mapping[node_list[i]] = str(i)
    remap_G = nx.relabel_nodes(G, mapping)
    return remap_G


def num2letter(G):
    num_to_letter = {
    0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j',
    10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's',
    19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z'}
    mapping = {}
    node_list = list(G.nodes())
    for i in range(len(node_list)):
        mapping[node_list[i]] = num_to_letter[int(node_list[i])]
    remap_G = nx.relabel_nodes(G, mapping)
    return remap_G


def division(G, ratio=0.8, train_split=100, test_split=20):
    nodes = G.nodes()
    split = round(len(nodes)*ratio)
    if split > train_split:
        train = random.sample(nodes, train_split)
        test = random.sample(nodes, test_split)
    else:
        train = random.sample(nodes, split)
        test = random.sample(nodes, len(nodes)-split)
    return train, test


def split(G, size=50):
    nodes = list(G.nodes())
    num = math.ceil(len(nodes) / size)
    split_dict = {}
    for n in range(num):
        if n != num-1:
            split_dict[n] = nodes[n*size:(n+1)*size]
            
        else:
            split_dict[n] = nodes[n*size:]
    
    return split_dict


def smile2graph(smile, dataname):
    if 'alkane' in dataname:
        temp = Chem.MolFromSmiles(smile)
        mol = Chem.AddHs(temp)
    else:
        mol = Chem.MolFromSmiles(smile)
    atoms = dict()
    for i, atom in enumerate(mol.GetAtoms()):
        atoms[i]=atom.GetSymbol()
    G = nx.Graph(rdmolops.GetAdjacencyMatrix(mol))
    return G, atoms

def extract_text_edge(text):
    pattern = r'\(\w+, \w+\)'
    pattern2 = r'\(Node \w+ Atom \w+, Node \w+ Atom \w+\)'
    edges = re.findall(pattern, text)+re.findall(pattern2, text)
    edge_list = []
    atom_dict = {}
    for idx, e in enumerate(edges):
        # edge
        edge=r'Node \w+'
        edge_pattern = re.findall(edge, e)       
        edge_list.append((edge_pattern[0].replace('Node ', ''), edge_pattern[1].replace('Node ', '')))
        # atom
        atom = r'Atom \w+'
        atom_pattern = re.findall(atom, e)
        atom_dict[edge_pattern[0].replace('Node ', '')] = atom_pattern[0].replace('Atom ', '')
        atom_dict[edge_pattern[1].replace('Node ', '')] = atom_pattern[1].replace('Atom ', '')

    return edge_list, atom_dict


def setseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sampling(smiles):
    np.random.shuffle(smiles)
    pos_smiles = []
    neg_smiles = []
    for smile in smiles:
        if smile[-1] == 1:
            pos_smiles.append(smile)
        else:
            neg_smiles.append(smile)
            
            
    return pos_smiles, neg_smiles
            
