import random
import networkx as nx

def _graph_discribe_node(graph,dicts=None,nodes_feature='',weight=False):
    if graph.is_directed():
        directed_str = 'a directed'
        connect_str = 'directed'
    else:
        directed_str = 'an undirected'
        connect_str = 'connected'
    strings=f'G describes {directed_str} graph among '
    nodes=graph.nodes
    if dicts==None:
        dicts={}
        for i in nodes:
            dicts[i]=i
    degrees=list(graph.degree())
    sorted_nodes = sorted(degrees, key=lambda x: x[1], reverse=True)
    name_dicts={}
    for idx,i in enumerate(sorted_nodes):
        name_dicts[i[0]]=idx
    nodes_list=list(graph.nodes)
    for idx,i in enumerate(range(len(nodes))):
        if idx==len(nodes)-1:
            strings+='and '+str(nodes_list[idx])+'.\n'
        else:
            # print(nodes_list,idx)
            strings+=str(nodes_list[idx])+', '
    strings+='In this graph:\n'
    edge_dicts={}
    for e in graph.edges(data=True):
        if e[0] not in edge_dicts:
            edge_dicts[e[0]]=[]
        if e[1] not in edge_dicts:
            edge_dicts[e[1]]=[]
        edge_dicts[e[0]].append((e[1], e[2]))
        if graph.is_directed()==False:
            edge_dicts[e[1]].append((e[0], e[2]))
    edge_list=[]
    edge_dicts = {key: edge_dicts[key] for key in sorted(edge_dicts)}
        
    for key in edge_dicts.keys():
        edge_sentence=''
        if len(edge_dicts[key])!=0:
            if nodes_feature!='':
                edge_sentence+=f'Node {str(key)} ({nodes_feature}: {str(dicts[key])}) is {connect_str} to nodes '
            else:
                edge_sentence+='Node '+str(key)+' is ' + connect_str + ' to nodes '

            for i in range(len(edge_dicts[key])):
                if nodes_feature!='':
                    edge_sentence+=str(edge_dicts[key][i][0])+f' ({nodes_feature}: '+str(dicts[edge_dicts[key][i][0]])+')'
                else:
                    edge_sentence+=str(edge_dicts[key][i][0])

                if weight:
                    if 'weight' in edge_dicts[key][i][1] and edge_dicts[key][i][1]['weight'] is not None:
                        edge_sentence += f" (weight: {edge_dicts[key][i][1]['weight']})"
                if i==len(edge_dicts[key])-1:
                    edge_sentence+='.'
                else:
                    edge_sentence+=', '
                
            edge_list.append(edge_sentence)
    
    for i in range(len(edge_list)):
        strings+=edge_list[i]

        strings+='\n'
    return strings


def _graph_discribe_adj_n(graph,dicts=None, nodes_feature='',weight=False):
    begins=f"G describes {'a directed' if graph.is_directed() else 'an undirected'} graph among node "
    nodes=graph.nodes
    if dicts==None or len(dicts)==0:
        dicts={}
        for i in nodes:
            dicts[i]=i

    nodes_list=list(graph.nodes)
    for idx,i in enumerate(range(len(nodes))):
        # print(nodes_list[idx])
        if idx==len(nodes)-1:
            begins+='and '+str(nodes_list[idx])+'.\n'
        else:
            begins+=str(nodes_list[idx])+', '

    edges=graph.edges
    edge_list=[]
    for idx,e in enumerate(edges):
        edge_list.append(e)
    data_list=[]
    for idx,e in enumerate(graph.edges(data=True)):
        data_list.append(e[2])
    for e, data in zip(edge_list, data_list):
        weight_str = ''
        if weight:
            if 'weight' in data and data['weight'] is not None:
                weight_str = f" with weight {data['weight']}"
        if nodes_feature!='':
            begins+='Node '+f'{str(e[0])} ({nodes_feature}: {str(dicts[e[0]])})'+f" is {'directed' if graph.is_directed() else 'connected'} to Node "+f'{str(e[1])} ({nodes_feature}: {str(dicts[e[1]])})'+weight_str+'.\n'
        else:
            begins+='Node '+f'{str(e[0])}'+f" is {'directed' if graph.is_directed() else 'connected'} to Node "+f'{str(e[1])} '+weight_str+'.\n'
    # begins+=str(edges)
    return begins


def graph_txt(G,method,dicts=None,nodes_feature=''):
    if method=='adj':
        return _graph_discribe_adj_n(G,dicts=dicts,nodes_feature=nodes_feature)
    
    if method=='node':
        return _graph_discribe_node(G,dicts=dicts,nodes_feature=nodes_feature)
