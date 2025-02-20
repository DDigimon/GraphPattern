import json
import pickle
import re
import networkx as nx

prompt='zero'
deployment_name='o1m'
difficulty='easy'
method='edge'

def extract_nodes(text):
    matches = re.findall(r'\[([^\[\]]+(?:\[[^\[\]]+\][^\[\]]*)*)\]', text)
    nodes=[]
    if len(matches)!=0:
        pattern=r'\d+'
        nodes=re.findall(pattern,matches[0])
    else:
        matched_nodes=[]
        match = re.findall(r'\([^\)]*\)', text)
        if len(match):
            for content in match:
                numbers = re.findall(r'\d+', content)
                for n in numbers:
                    matched_nodes.append(int(n))
            nodes=list(set(matched_nodes))
        else:
            print(text)
    return nodes

with open(f'{prompt}_response/{deployment_name}/{deployment_name}_{difficulty}_{method}_.json','r') as f:
    data=json.load(f)

path=f'../../Dataset/Basic_tasks/{difficulty}.pkl'
with open(path,'rb') as f:
    graphs=pickle.load(f)


def extract_text_edge(text):
    matches = re.findall(r'\[([^\[\]]+(?:\[[^\[\]]+\][^\[\]]*)*)\]', text)
    if len(matches)>0:
        text=matches[0]
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
        if dicts[key]==1:
            new_list.append(key)
    return new_list
ans_path=f'ans/{difficulty}.pkl'
with open(ans_path,'rb') as f:
    ans=pickle.load(f)

if difficulty!='easy':
    new_graphs=[]
    new_ans=[]
    with open('../../Dataset/sub_idx.pkl','rb')as f:
        sub_idx_list=pickle.load(f)
    for i in sub_idx_list:
        new_graphs.append(graphs[i])
        new_ans.append(ans[i])
    graphs=new_graphs
    ans=new_ans

def node_precision(preds,ground_truth):
    if len(preds)==0 and ground_truth!=0:return 0
    rights=0
    for i in preds:
        if int(i) in ground_truth:
            rights+=1
    return rights/len(preds)

def edge_precision(preds,ground_truth):
    if len(preds)==0 and ground_truth!=0:return 0
    rights=0
    for e in preds:
        try:
            e=(int(e[0]),int(e[1]))
            if e in ground_truth or (e[1],e[0]) in ground_truth:
                rights+=1
        except:continue
    return rights/len(preds)

precision=[]
degrees=[]
countings={}
over_all_counting={}
for idx,(d,g,a) in enumerate(zip(data,graphs,ans)):
    degrees.append(nx.density(g))
    response=d['response']# [0][0][0]
    nodes=extract_nodes(response)
    ground_truth_nodes=list(a.nodes())
    nodes_degrees={}# (nodes: degree num)
    for n in g.nodes():
        nodes_degrees[n]=nx.degree(g,n)
    # print(nodes_degrees)
    ground_truth_edges=list(a.edges())
    precision.append(node_precision(nodes,ground_truth_nodes))

    for n in nodes:
        n=int(n)
        if nodes_degrees[n] not in over_all_counting:
            over_all_counting[nodes_degrees[n]]=0
        # print('over all1',over_all_counting,nodes_degrees[n])
        over_all_counting[nodes_degrees[n]]=over_all_counting[nodes_degrees[n]]+1

        if nodes_degrees[n] not in countings:
            countings[nodes_degrees[n]]=0

        # for p_n in nodes:
        if n in ground_truth_nodes:
            countings[nodes_degrees[n]]+=1
new_counting={}
for i in countings:
    new_counting[i]=countings[i]/over_all_counting[i]




