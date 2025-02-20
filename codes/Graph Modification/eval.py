import json
import pickle
import re
import networkx as nx
import sys
sys.path.append('../')
from graph_patterns import find_pattern_list



def extract_text_edge(text):
    patterns=[r'\(\d+, \d+\)', r'\(\d+,\d+\)', r'\(Node \d+, Node \d+\)', r'\[\d+, \d+\]',r'\(N\d+, N\d+\)',r'\(N_\d+, N_\d+\)',r'\(N_\d+, N_\{\d+\}\)',r'\(N_\{\d+\}, N_\{\d+\}\)',r'\(\w, \w\)']
    edges=[]

    for pattern in patterns:
        edges.extend(re.findall(pattern,text))
    # edges = re.findall(pattern, text)+re.findall(pattern2, text)+re.findall(pattern3, text)
    edge_list=[]
    for e in edges:
        pattern=r'\d+'
        edge_pattern=re.findall(pattern,e)
        if len(edge_pattern)==2:
            edge_list.append((int(edge_pattern[0]),int(edge_pattern[1])))
        else:
            pattern=r'\w'
            edge_pattern=re.findall(pattern,e)
            edge_list.append((edge_pattern[0],edge_pattern[1]))
    clean(edge_list)
    return edge_list

def clean(lists):
    dicts={}
    new_list=[]
    for l in lists:
        if l not in dicts:
            dicts[l]=0
        dicts[l]+=1
    for key in dicts.keys():
        if dicts[key]>=1:
            new_list.append(key)
    return new_list

difficulty='easy'
prompt='zero'
pattern="claim_('square', 'house')"
deployment_name='GPT-4o'
method='edge'

mat_dictions={}
for deployment_name in ['o1m']:
    for pattern in ["('square', 'house')"]:#,"claim_('square', 'diamond')","claim_('diamond', 'square')","claim_('FFL', 'FBL')"]:
        for method in ['edge','node']:
    
            if 'claim' in pattern:
                pattern_name=pattern.split('_')[1]
            else:
                pattern_name=pattern
            target_pattern=pattern_name.split(',')[1][2:-2]

            if 'FFL' in pattern or 'FBL' in pattern or 'd-diamond' in pattern or 'vs' in pattern:
                graph_mode='directed'
                direction=True
            else:
                graph_mode='undirected'
                direction=False
            path=f'../Dataset/Modify/{pattern_name}/{difficulty}.pkl'

            with open(path,'rb') as f:
                graphs=pickle.load(f)

            if difficulty!='easy':
                with open(f'{prompt}_response/{deployment_name}/{deployment_name}_{difficulty}_{pattern}_{method}.json','r') as f:
                    data=json.load(f)
            else:
                try:
                    with open(f'{prompt}_response/{deployment_name}/{deployment_name}_{pattern}_{method}.json','r') as f:
                        data=json.load(f)
                except:
                    try:
                        with open(f'{prompt}_response/{deployment_name}/{deployment_name}_{difficulty}_{pattern}_{method}.json','r') as f:
                            data=json.load(f)
                    except:
                        print('no file')
                        exit()

            def extraction(text):
                matches = re.findall(r'\[([^\[\]]+(?:\[[^\[\]]+\][^\[\]]*)*)\]', text)
                return matches

            def construct_graphs(edges,direction=False):

                if direction:
                    G=nx.DiGraph()
                else:
                    G=nx.Graph()
                G.add_edges_from(edges)
                return G


            # def accuracy(g,ground_trut)
            accuracy=[]
            def modified_edges(graph,ori_graph):
                ori_edge=ori_graph.edges()
                edge=graph.edges()
                unique_edge=0
                for e in ori_edge:
                    if e not in edge and (e[1],e[0]) not in edge:
                        unique_edge+=1
                for e in edge:
                    if e not in ori_edge and (e[1],e[0]) not in ori_edge:
                        unique_edge+=1
                return unique_edge

            modified_edges_list=[]
            if difficulty!='easy':
                new_graphs=[]
                with open(f'../Dataset/Modify/{pattern_name}/{difficulty}_idx.pkl','rb') as f:
                    random_indices=pickle.load(f)
                for i in random_indices:
                    new_graphs.append(graphs[i])
                graphs=new_graphs

            for idx,(g,d) in enumerate(zip(graphs,data)):
                response=d['response']
                # print(response)
                graph_pairs=extraction(response)
                # print(graph_pairs)
                if len(graph_pairs)==0:
                    try:
                        graph_text=extract_text_edge(response)
                        graph=construct_graphs(graph_text,direction=direction)
                        modified_edges_list.append(modified_edges(graph,g))
                        if len(find_pattern_list(graph,target_pattern))>=1:
                            accuracy.append(1)
                        else:
                            accuracy.append(0)
                    except:
                        # print(response)
                        modified_edges_list.append(0)
                        accuracy.append(0)
                else:
                    # print(response)
                    graph_text=extract_text_edge(graph_pairs[-1])
                    graph_text=clean(graph_text)
                    graph=construct_graphs(graph_text,direction=direction)
                    modified_edges_list.append(modified_edges(graph,g))
                    if len(find_pattern_list(graph,target_pattern))>=1:
                        accuracy.append(1)

                    else:
                        accuracy.append(0)
            print(len(accuracy))
            print(difficulty,method,deployment_name,pattern,sum(accuracy)/len(accuracy))
            if deployment_name not in mat_dictions:
                mat_dictions[deployment_name]={}
            if pattern not in mat_dictions[deployment_name]:
                mat_dictions[deployment_name][pattern]={}
            mat_dictions[deployment_name][pattern][method]=sum(accuracy)/len(accuracy)
            print(sum(modified_edges_list)/len(modified_edges_list))

print(mat_dictions)