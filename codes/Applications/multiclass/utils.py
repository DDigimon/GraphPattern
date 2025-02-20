import pandas as pd 
import numpy as np 
import networkx as nx
import re 
import random 
from rdkit import Chem
from rdkit.Chem import rdmolops
import torch
from tqdm import tqdm
import sys 
from graph_nlp import graph_txt
from graph_patterns import pattern_generation
from networkx.algorithms import isomorphism

from datasets import load_dataset
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data, DataLoader, Dataset
# from torch_geometric.loader import DataLoader
from torch.utils.data.sampler import Sampler
from ogb.graphproppred import GraphPropPredDataset, PygGraphPropPredDataset


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    
def sparse2dense(edge_indx, n_nodes, start):
    num_edges = edge_indx.shape[1]
    adj = np.zeros([n_nodes, n_nodes])
    for i in range(num_edges):
        adj[edge_indx[0][i]-start, edge_indx[1][i]-start] = 1
    return adj

def smile2graph(smiles):
    graphs = []
    atoms = []
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        dict = {}
        for i, atom in enumerate(mol.GetAtoms()):
            # G.nodes[i]['atom'] = atom.GetSymbol()
            # atoms.append(atom.GetSymbol())
            dict[i] = atom.GetSymbol()
        atoms.append(dict)
        G1 = nx.Graph(rdmolops.GetAdjacencyMatrix(mol))
        graphs.append(G1)
    return graphs, atoms


def node2atom(node_feat, num_nodes, dataset):  # dataset - data name
    atom_dict = {}
    if dataset == 'MUTAG':  # [C, N, O, F, I, Cl, Br]
        NODE = ["C", "N", "O", "F", "I", "Cl", "Br"]
        for n in range(num_nodes):
            idx = np.where(np.array(node_feat[n]) == 1.0)
            atom_dict[n] = NODE[idx[0][0]]
    elif dataset == 'ogbg-molhiv' or dataset == 'ogbg-molbbbp':
        periodic_table = {
                            1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
                            11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar',
                            19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe',
                            27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se',
                            35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo',
                            43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
                            51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce',
                            59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy',
                            67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W',
                            75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb',
                            83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th',
                            91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf',
                            99: 'Es', 100: 'Fm', 101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Db',
                            106: 'Sg', 107: 'Bh', 108: 'Hs', 109: 'Mt', 110: 'Ds', 111: 'Rg', 112: 'Cn',
                            113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts', 118: 'Og'
                        }
        for n in range(num_nodes):
            idx = node_feat[n][0] + 1   # ogbg-molhiv (atomic_num-1) 
            atom_dict[n] = periodic_table[idx]       
    return atom_dict


def clean(lists):
    dicts={}
    new_list=[]
    for l in lists:
        if l not in dicts:
            dicts[l]=0
        dicts[l]+=1
    for key in dicts.keys():
        if dicts[key]==1:
            new_list.append(key)
    return new_list


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
    clean(edge_list)
    return edge_list, atom_dict


def extract_edge(text):
    pattern = r'\(Node \w+, Node \w+\)'
    edges = re.findall(pattern, text)
    edge_list = []
    for idx, e in enumerate(edges):
        # edge
        edge=r'Node \w+'
        edge_pattern = re.findall(edge, e)
        edge_list.append((edge_pattern[0].replace('Node ', ''), edge_pattern[1].replace('Node ', '')))
        
    clean(edge_list)
    return edge_list


def remapping(G, atom_dict):
    node_list = list(G.nodes())
    remap_atom_dict = {}
    mapping = {}
    for i in range(len(node_list)):
        mapping[node_list[i]] = str(i+1)
        remap_atom_dict[str(i+1)] = atom_dict[node_list[i]]
    remap_G = nx.relabel_nodes(G, mapping)
    return remap_G, remap_atom_dict


def remapping_(G):
    node_list = list(G.nodes())
    mapping = {}
    for i in range(len(node_list)):
        mapping[node_list[i]] = str(i+1)
    remap_G = nx.relabel_nodes(G, mapping)
    return remap_G
    


def graph_unique(pattern, pattern_list):
    flag = True
    for p in pattern_list:
        if pattern == p:
            flag = False
            
    return flag
    
    
def pattern_filtering(graphs, graphs_inv, pattern):
    shred = 0.9
    flag = []
    for g in graphs:
        GM = isomorphism.GraphMatcher(g, pattern)
        if GM.subgraph_is_isomorphic() == False:
            flag.append(0)
        else:
            flag.append(1)
    flag_value = sum(flag)/len(flag)
    
    flag = []
    for g in graphs_inv:
        GM = isomorphism.GraphMatcher(g, pattern)
        if GM.subgraph_is_isomorphic() == False:
            flag.append(0)
        else:
            flag.append(1)
    inv_flag_value = sum(flag)/len(flag)
    
    print('values =', flag_value, inv_flag_value)
    if flag_value > shred and inv_flag_value <= shred:
        return True
    else: return False
        
    
def read_graph(databatch, dataset):
    if dataset in ['IMDB-BINARY']:
        x = databatch['x']
        edge_indx = x['edge_index']
        num_nodes = x['num_nodes']
        y = x['y']
        current_batch = len(databatch)
        graphs = []
        _ = []
        for b in range(current_batch):
            adj = sparse2dense(edge_indx[b], num_nodes[b])
            G = nx.from_numpy_array(adj)
            graphs.append(G)
        
        return graphs, _, y
            
    else: 
        x = databatch['x']
        edge_indx = x['edge_index']
        node_feat = x['node_feat']  # MUTAG = [C, N, O, F, I, Cl, Br], ogbg-molhiv (atomic_num-1) = [C:5, N:6, O:7, F:8, S:15, Cl:16, Br:34, I:52]
        edge_attr = x['edge_attr']  #  aromatic, single, double, and triple bonds
        num_nodes = x['num_nodes']
        y = x['y']
        current_batch = len(databatch)

        # create Graph
        graphs = []
        atoms = []
        for b in range(current_batch):
            adj = sparse2dense(edge_indx[b], num_nodes[b])
            G = nx.from_numpy_array(adj)
            N = node2atom(node_feat[b], num_nodes[b], dataset)
            graphs.append(G)
            atoms.append(N)
            
        return graphs, atoms, y
        

# for balance sampling
class LabelSampler(Sampler):
    def __init__(self, data_list, config, label):
        # self.data_list = random.shuffle(data_list)
        self.label = label
        self.indices = []
        for i, data in enumerate(data_list):
            y = data.y.item()

            if len(self.indices) >= config['num']:  # truncation
                break
            
            if y == label:
                self.indices.append(i)
        print('sample number =', len(self.indices))
    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
    
    
def read_dataset(config):
    if config['process'] == 'summary':
        if config['dataset'] == 'MUTAG':
            dataset_hf = load_dataset("graphs-datasets/" + config['dataset'])
            dataset_hf = dataset_hf.shuffle(seed=config['seed'])  # shuffle the data
            dataset_pg_list = [Data(graph) for graph in dataset_hf["train"]]  # valid, test, 188 samples, 2 rounds, Whether Set 2 has not significant patterns.
            split = round(len(dataset_pg_list) * config['split'])
            split_dataset = dataset_pg_list[:split]
            # balance sampling
            pos_sampler = LabelSampler(split_dataset, config, label=1)
            neg_sampler = LabelSampler(split_dataset, config, label=0)
            pos_dataset_pg = DataLoader(split_dataset, batch_size=config['batch_size'], sampler=pos_sampler)
            neg_dataset_pg = DataLoader(split_dataset, batch_size=config['batch_size'], sampler=neg_sampler)
            
        elif config['dataset'] == 'ogbg-molhiv':
            dataset_hf = load_dataset("graphs-datasets/" + config['dataset'])
            dataset_hf = dataset_hf.shuffle(seed=config['seed'])  # shuffle the data
            dataset_pg_list = [Data(graph) for graph in dataset_hf["test"]]  # test, 4113 samples
            split = round(len(dataset_pg_list) * config['split'])
            split_dataset = dataset_pg_list[:split]

            # balance sampling
            pos_sampler = LabelSampler(split_dataset, config, label=1)
            neg_sampler = LabelSampler(split_dataset, config, label=0)
            pos_dataset_pg = DataLoader(split_dataset, batch_size=config['batch_size'], sampler=pos_sampler)
            neg_dataset_pg = DataLoader(split_dataset, batch_size=config['batch_size'], sampler=neg_sampler)
            
        elif config['dataset'] == 'IMDB-BINARY':
            dataset_hf = load_dataset("graphs-datasets/" + config['dataset'])
            dataset_hf = dataset_hf.shuffle(seed=config['seed'])  # shuffle the data
            dataset_pg_list = [Data(graph) for graph in dataset_hf["train"]]  # train, 1000 samples
            split = round(len(dataset_pg_list) * config['split'])
            split_dataset = dataset_pg_list[:split]

            # balance sampling
            pos_sampler = LabelSampler(split_dataset, config, label=1)
            neg_sampler = LabelSampler(split_dataset, config, label=0)
            pos_dataset_pg = DataLoader(split_dataset, batch_size=config['batch_size'], sampler=pos_sampler)
            neg_dataset_pg = DataLoader(split_dataset, batch_size=config['batch_size'], sampler=neg_sampler)
        
        elif config['dataset'] == 'ogbg-molbbbp':
            dataset_hf = PygGraphPropPredDataset(name = config['dataset'])
            split = dataset_hf.get_idx_split()
            train_dataset = dataset_hf[split['train']]  # train 1631
            split_dataset = [graph for graph in train_dataset]
            data_list = []
            for i, data in enumerate(split_dataset):
                x = dict()
                x['edge_index'] = [data['edge_index'][0, :].numpy(), data['edge_index'][1, :].numpy()]
                x['edge_attr'] = [data['edge_attr'][k, :].numpy() for k in range(data['edge_attr'].shape[0])]
                x['node_feat'] = [data['x'][k, :].numpy() for k in range(data['x'].shape[0])]
                x['num_nodes'] = data['num_nodes']
                x['y'] = data['y'][0]
                data_list.append(Data(x))
            # balance sampling
            pos_sampler = LabelSampler(data_list, config, label=1)
            neg_sampler = LabelSampler(data_list, config, label=0)
            pos_dataset_pg = DataLoader(data_list, batch_size=config['batch_size'], sampler=pos_sampler)
            neg_dataset_pg = DataLoader(data_list, batch_size=config['batch_size'], sampler=neg_sampler)
            
        else:
            NotImplementedError
            
    elif config['process'] == 'classification':
        if config['dataset'] == 'MUTAG':
            dataset_hf = load_dataset("graphs-datasets/" + config['dataset'])
            dataset_hf = dataset_hf.shuffle(seed=config['seed'])  # shuffle the data
            dataset_pg_list = [Data(graph) for graph in dataset_hf["train"]]  # valid, test, 188 samples, 2 rounds, Whether Set 2 has not significant patterns.
            split = round(len(dataset_pg_list) * config['split'])
            split_dataset = dataset_pg_list[:split]
            # balance sampling
            pos_sampler = LabelSampler(split_dataset, config, label=1)
            neg_sampler = LabelSampler(split_dataset, config, label=0)
            pos_dataset_pg = DataLoader(split_dataset, batch_size=config['batch_size'], sampler=pos_sampler)
            neg_dataset_pg = DataLoader(split_dataset, batch_size=config['batch_size'], sampler=neg_sampler)
            
        elif config['dataset'] == 'ogbg-molhiv':
            dataset_hf = load_dataset("graphs-datasets/" + config['dataset'])
            dataset_hf = dataset_hf.shuffle(seed=config['seed'])  # shuffle the data
            dataset_pg_list = [Data(graph) for graph in dataset_hf["test"]]  # test, 4113 samples
            split = round(len(dataset_pg_list) * config['split'])
            split_dataset = dataset_pg_list[split:]  # !!! the last part
            # balance sampling
            pos_sampler = LabelSampler(split_dataset, config, label=1)
            neg_sampler = LabelSampler(split_dataset, config, label=0)
            pos_dataset_pg = DataLoader(split_dataset, batch_size=config['batch_size'], sampler=pos_sampler)
            neg_dataset_pg = DataLoader(split_dataset, batch_size=config['batch_size'], sampler=neg_sampler)
            
        elif config['dataset'] == 'IMDB-BINARY':
            dataset_hf = load_dataset("graphs-datasets/" + config['dataset'])
            dataset_hf = dataset_hf.shuffle(seed=config['seed'])  # shuffle the data
            dataset_pg_list = [Data(graph) for graph in dataset_hf["train"]]  # train, 1000 samples
            split = round(len(dataset_pg_list) * config['split'])
            split_dataset = dataset_pg_list[split:]

            # balance sampling
            pos_sampler = LabelSampler(split_dataset, config, label=1)
            neg_sampler = LabelSampler(split_dataset, config, label=0)
            pos_dataset_pg = DataLoader(split_dataset, batch_size=config['batch_size'], sampler=pos_sampler)
            neg_dataset_pg = DataLoader(split_dataset, batch_size=config['batch_size'], sampler=neg_sampler)
        
        
        elif config['dataset'] == 'ogbg-molbbbp':
            dataset_hf = PygGraphPropPredDataset(name = config['dataset'])
            split = dataset_hf.get_idx_split()
            test_dataset = dataset_hf[split['test']]
            split_dataset = [graph for graph in test_dataset]
            data_list = []
            for i, data in enumerate(split_dataset):
                x = dict()
                x['edge_index'] = [data['edge_index'][0, :].numpy(), data['edge_index'][1, :].numpy()]
                x['edge_attr'] = [data['edge_attr'][k, :].numpy() for k in range(data['edge_attr'].shape[0])]
                x['node_feat'] = [data['x'][k, :].numpy() for k in range(data['x'].shape[0])]
                x['num_nodes'] = data['num_nodes']
                x['y'] = data['y']
                data_list.append(Data(x))
            # balance sampling
            pos_sampler = LabelSampler(data_list, config, label=1)
            neg_sampler = LabelSampler(data_list, config, label=0)
            pos_dataset_pg = DataLoader(data_list, batch_size=config['batch_size'], sampler=pos_sampler)
            neg_dataset_pg = DataLoader(data_list, batch_size=config['batch_size'], sampler=neg_sampler)
        
        else:
            NotImplementedError
    else:
        NotImplementedError
        
    return pos_dataset_pg, neg_dataset_pg


def statistics(pos_loader, neg_loader, config, feature):
    node_num = 0
    edge_num = 0
    graph_num = 0
    density = 0
    for i, (pos_databatch, neg_databatch) in enumerate(zip(pos_loader, neg_loader)):
        current_batch = len(pos_databatch)
        pos_graphs, pos_atoms, pos_y = read_graph(pos_databatch, config['dataset'])
        neg_graphs, neg_atoms, neg_y = read_graph(neg_databatch, config['dataset'])
        
        # positive samples
        if feature == 'atom':
            for idx, (graph, atom) in enumerate(zip(pos_graphs, pos_atoms)):  # Set1
                graph_num += 1
                node_num += len(graph.nodes)
                edge_num += len(graph.edges)
                density += len(graph.edges) / np.power(len(graph.nodes), 2)
                
        else:
            for idx, graph in enumerate(pos_graphs):  # Set1
                graph_num += 1
                node_num += len(graph.nodes)
                edge_num += len(graph.edges)
                density += len(graph.edges) / np.power(len(graph.nodes), 2)

        # negative samples
        if feature == 'atom':
            for idx, (graph, atom) in enumerate(zip(neg_graphs, neg_atoms)):  # Set2
                graph_num += 1
                node_num += len(graph.nodes)
                edge_num += len(graph.edges)
                density += len(graph.edges) / np.power(len(graph.nodes), 2)
        else:
            for idx, graph in enumerate(neg_graphs):
                graph_num += 1
                node_num += len(graph.nodes)
                edge_num += len(graph.edges)
                density += len(graph.edges) / np.power(len(graph.nodes), 2)
        
    ave_density = density / graph_num
    ave_node_num = node_num / graph_num
    ave_edge_num = edge_num / graph_num
    print('ave_node_num =', ave_node_num, 'ave_edge_num =', ave_edge_num, 'ave_density_num =', ave_density)
    

class CombinedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return min([len(dataset) for dataset in self.datasets])

    def __getitem__(self, idx):
        data_list = [dataset[idx] for dataset in self.datasets]
        return data_list
    
    
def read_3(config, split=100):
    if config['dataset'] == 'ENZYMES':
        labels = [random.randint(0, 5) for _ in range(3)]
        dataset = TUDataset(root='data/TUDataset', name=config['dataset'])
        a_dataset = [graph for graph in dataset if graph.y.item() == labels[0]]
        b_dataset = [graph for graph in dataset if graph.y.item() == labels[1]]
        c_dataset = [graph for graph in dataset if graph.y.item() == labels[2]]
        a_dataset = a_dataset[:round(len(a_dataset) * 0.8)]
        b_dataset = b_dataset[:round(len(b_dataset) * 0.8)]
        c_dataset = c_dataset[:round(len(c_dataset) * 0.8)]
        data = CombinedDataset([a_dataset, b_dataset, c_dataset])
        dataloader = DataLoader(data, batch_size=config['batch_size'], shuffle=True)
        
    elif config['dataset'] == 'Fingerprint':
        labels = []
        dataset = TUDataset(root='data/TUDataset', name=config['dataset'])
        dataset_list = []
        for i in range(14):
            if len(dataset_list) > 2:
                break
            label = i
            # labels = [1, 2, 3]
            data = [graph for graph in dataset if graph.y.item() == label]
            if len(data) > split:
                split_dataset = data[:split]
                dataset_list.append(split_dataset)
                labels.append(label)
        
        data = CombinedDataset(dataset_list)
        dataloader = DataLoader(data, batch_size=config['batch_size'], shuffle=True)
    elif config['dataset'] == 'IMDB-MULTI':
        labels = []
        dataset = TUDataset(root='data/TUDataset', name=config['dataset'])
        dataset_list = []
        for i in range(14):
            if len(dataset_list) > 2:
                break
            label = i
            # labels = [1, 2, 3]
            data = [graph for graph in dataset if graph.y.item() == label]
            if len(data) > split:
                split_dataset = data[:split]
                dataset_list.append(split_dataset)
                labels.append(label)
        
        data = CombinedDataset(dataset_list)
        dataloader = DataLoader(data, batch_size=config['batch_size'], shuffle=True)
    return dataloader, labels


def read_3_test(config, labels, split=100, ratio=0.8):
    if config['dataset'] == 'ENZYMES':
        dataset = TUDataset(root='data/TUDataset', name=config['dataset'])
        a_dataset = [graph for graph in dataset if graph.y.item() == labels[0]]
        b_dataset = [graph for graph in dataset if graph.y.item() == labels[1]]
        c_dataset = [graph for graph in dataset if graph.y.item() == labels[2]]
        a_dataset = a_dataset[round(len(a_dataset) * 0.8):]
        b_dataset = b_dataset[round(len(b_dataset) * 0.8):]
        c_dataset = c_dataset[round(len(c_dataset) * 0.8):]
        data = CombinedDataset([a_dataset, b_dataset, c_dataset])
        dataloader = DataLoader(data, batch_size=config['batch_size'], shuffle=True)
    elif config['dataset'] == 'Fingerprint':
        labels = []
        dataset = TUDataset(root='data/TUDataset', name=config['dataset'])
        dataset_list = []
        for i in range(14):
            if len(dataset_list) > 2:
                break
            label = i
            # labels = [1, 2, 3]
            data = [graph for graph in dataset if graph.y.item() == label]
            if len(data) > 100:
                split_dataset = data[split:split+20]
                dataset_list.append(split_dataset)
                labels.append(label)
        
        data = CombinedDataset(dataset_list)
        dataloader = DataLoader(data, batch_size=config['batch_size'], shuffle=True)
    elif config['dataset'] == 'IMDB-MULTI':
        labels = []
        dataset = TUDataset(root='data/TUDataset', name=config['dataset'])
        dataset_list = []
        for i in range(14):
            if len(dataset_list) > 2:
                break
            label = i
            # labels = [1, 2, 3]
            data = [graph for graph in dataset if graph.y.item() == label]
            if len(data) > 100:
                split_dataset = data[split:split+20]
                dataset_list.append(split_dataset)
                labels.append(label)
        
        data = CombinedDataset(dataset_list)
        dataloader = DataLoader(data, batch_size=config['batch_size'], shuffle=True)
    return dataloader


def read_graph_3(databatch, dataset):
    batch = databatch.batch
    edge_index = databatch.edge_index
    batch = databatch.batch
    ptr = databatch.ptr
    x = databatch.x
    y = databatch.y
    batch_size = len(ptr)-1

    # create Graph
    graphs = []
    for b in range(batch_size):
        condition = (edge_index >= ptr[b]) & (edge_index < ptr[b+1])
        edge = edge_index[condition].view(2, -1)
        adj = sparse2dense(edge, ptr[b+1]-ptr[b], ptr[b])
        G = nx.from_numpy_array(adj)
        # N = node2atom(node_feat[b], ptr[b+1]-ptr[b], dataset)
        graphs.append(G)
            
    
    return graphs, y
        
        
def statistics_3(loader, config):
    node_num = 0
    edge_num = 0
    graph_num = 0
    density = 0
    for i, batch in enumerate(loader):
        a_databatch, b_databatch, c_databatch = batch
        current_batch = len(a_databatch)
        a_graphs, a_y = read_graph_3(a_databatch, config['dataset'])
        b_graphs, b_y = read_graph_3(b_databatch, config['dataset'])
        c_graphs, c_y = read_graph_3(c_databatch, config['dataset'])
        

        for idx, graph in enumerate(a_graphs):  # Set1
            graph_num += 1
            node_num += len(graph.nodes)
            edge_num += len(graph.edges)
            density += len(graph.edges) / np.power(len(graph.nodes), 2)


        for idx, graph in enumerate(b_graphs):
            graph_num += 1
            node_num += len(graph.nodes)
            edge_num += len(graph.edges)
            density += len(graph.edges) / np.power(len(graph.nodes), 2)


        for idx, graph in enumerate(c_graphs):
            graph_num += 1
            node_num += len(graph.nodes)
            edge_num += len(graph.edges)
            density += len(graph.edges) / np.power(len(graph.nodes), 2)
        
    ave_density = density / graph_num
    ave_node_num = node_num / graph_num
    ave_edge_num = edge_num / graph_num
    print('ave_node_num =', ave_node_num, 'ave_edge_num =', ave_edge_num, 'ave_density_num =', ave_density)

    