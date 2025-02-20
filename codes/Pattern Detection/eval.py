import networkx as nx
import pickle
import sys
from tqdm import tqdm
import os
import re
import numpy as np
import json
from networkx.algorithms import isomorphism
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
from graph_patterns import direct_judge

def find_feed_forward_loops(G):
    # List to store all detected FFLs
    ffl_list = []
    
    # Iterate over all possible triples (A, B, C) in the graph
    for A in G.nodes:
        for B in G.successors(A):  # B is a successor of A
            for C in G.successors(A):  # C is also a successor of A
                if B != C and G.has_edge(B, C):  # Check B -> C edge
                    ffl_list.append(tuple(sorted((A, B, C))))
    ffl_list=set(ffl_list)
    
    return ffl_list


pattern_output_form={'house':5,'triangle':3,'diamond':4,'tailed-triangle':4,'square':4,'FFL':3,'FBL':3,'vs':3,'d-diamond':4}

def extract_edges(text,patterns,directed=False):

     
    matches=re.findall(r'\[([^\[\]]+(?:\[[^\[\]]+\][^\[\]]*)*)\]',text)
    # print('given matched',matches)
    if pattern_output_form[patterns]==4:
        match_sample=r'\*{0,2}\\?\(?\\?\(\s*(?:\\?(?:Node\s+|#))?(\d+)\s*,\s*(?:\\?(?:Node\s+|#))?(\d+)\s*,\s*(?:\\?(?:Node\s+|#))?(\d+)\s*,\s*(?:\\?(?:Node\s+|#))?(\d+)\s*\\?\)\\?\)?\*{0,2}'
    elif pattern_output_form[patterns]==3:
        match_sample=r'\*{0,2}\\?\(?\\?\(\s*(?:\\?(?:Node\s+|#))?(\d+)\s*,\s*(?:\\?(?:Node\s+|#))?(\d+)\s*,\s*(?:\\?(?:Node\s+|#))?(\d+)\s*\\?\)\\?\)?\*{0,2}'
    elif pattern_output_form[patterns]==5:
        match_sample=r'\*{0,2}\\?\(?\\?\(\s*(?:\\?(?:Node\s+|#))?(\d+)\s*,\s*(?:\\?(?:Node\s+|#))?(\d+)\s*,\s*(?:\\?(?:Node\s+|#))?(\d+)\s*,\s*(?:\\?(?:Node\s+|#))?(\d+)\s*,,\s*(?:\\?(?:Node\s+|#))?(\d+)\s*\\?\)\\?\)?\*{0,2}'
    triangles=set()
    if len(matches)>0:
        matched_nodes_list=[]
        for recog_pattern in matches:
            recog_pattern=matches[-1]
            matched_nodes=re.findall(match_sample,recog_pattern)
            # print(matched_nodes)
            matched_nodes_list.extend(matched_nodes)
        if len (matched_nodes_list)==0:
            matched_nodes=re.findall(match_sample,text)
            matches=matched_nodes
        else:
            matches=matched_nodes_list
    else:
        matched_nodes=re.findall(match_sample,text)
        matches=matched_nodes

    # print('matched',matches)
    if len(matches)==0:
        txt=text.split('\n')
        extract_patterns=[]
        for s in txt:
            s=s.lower()
            if 'no ' in s or 'n\'t' in s or 'not ' in s:
                continue
            if 'connected' in s or 'form a pattern' in s:
                numbers = re.findall(r'\d+', s)
                if pattern_output_form[patterns]==3:
                    extract_patterns.append(tuple(map(int,numbers[1:4])))
                if pattern_output_form[patterns]==4:
                    extract_patterns.append(tuple(map(int,numbers[1:5])))
                if pattern_output_form[patterns]==5:
                    extract_patterns.append(tuple(map(int,numbers[1:6])))
        if len(extract_patterns)==0:
            text=text.lower()
            pattern_list =[r'\d+\.\s\*\*nodes\s([\d,\s]+):\*\*.*?this set has \d+ edges, (a (\w+)|not a (\w+))\.',
                           r'\d+\.\s\*\*nodes\s([\d,\s]+)\*\*.*?this forms (a (\w+)|not a (\w+))',
                           r'\d+\.\s\*\*nodes\s([\d,\s]+)\*\*.*?(a (\w+)|not a (\w+))',
                           r'\d+\.\s\*\*nodes\s([\d,\s]+):\*\*.*?this forms a complete graph \(\d+ edges\), (a (\w+)|not a (\w+))\.',
                           r'nodes\s([\d,\s]+) forms (a (\w+)|not a (\w+)):']
            

            # Use re.DOTALL to make '.' match any character including newline
            matches=[]
            for pattern in pattern_list:
                matches.extend(re.finditer(pattern, text, re.DOTALL))

            for match in matches:
                nodes_str = match.group(1)
                status = match.group(2)
                # Split the string by commas and convert each number to an integer
                nodes = [int(num.strip()) for num in nodes_str.split(',')]
                if 'no' not in status.lower() and 'n\'t' not in status.lower():
                    extract_patterns.append(tuple(map(int,nodes)))
                # print(f"Nodes: {nodes} - {status}")
            
    else:
        extract_patterns=[]
        for nodes in matches:
            # print(nodes)
            nodes=str(nodes)
            numbers = re.findall(r'\d+', nodes)
            # print(numbers)
            triangle=[]
            for n in numbers:
                triangle.append(int(n))
            extract_patterns.append(triangle)
                
            # triangle = tuple(map(int, nodes))
    for t in extract_patterns:
        if directed==False:
            triangle = tuple(sorted(t))
        else:
            triangle = tuple(t)
        triangles.add(triangle)
    # print('get triangles',triangles)
    return triangles




def jaccard_index(set1, set2):
    # Calculate the intersection of two sets
    intersection = set1.intersection(set2)
    # Calculate the union of two sets
    union = set1.union(set2)
    # Compute the Jaccard Index
    if len(union)==0:
        if len(intersection)!=0:return 0
        else:
            return 1
    index = len(intersection) / len(union)
    return index  

def graph_pattern_check(graph,pattern):
    if 'FFL' in pattern:
        ffl_list=find_feed_forward_loops(graph)
    elif 'diamond' in pattern:
        ffl_list=find_diamonds(graph)
    else:
        triangles_networkx = [tuple(sorted(triangle)) for triangle in nx.enumerate_all_cliques(graph) if len(triangle) == 3]
        ffl_list = set(triangles_networkx)
    return ffl_list

def find_diamonds(graph):
    M = nx.Graph()
    M.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A'),('D','A',),('D','B')])

    # Initialize the GraphMatcher
    GM = isomorphism.GraphMatcher(g, M)

    # Find subgraph isomorphisms
    matches = list(GM.subgraph_isomorphisms_iter())
    triangles_networkx=set()
    
    for m in matches:
        pattern=tuple(sorted(m.keys()))
        # print(pattern)
        triangles_networkx.add(pattern)
    return triangles_networkx


def precision_and_recall(actual, predicted):
    relevant_and_recommended = set(predicted) & set(actual)
    if len(predicted)==0:
        if len(actual)==0:
            return 1,1
        else:
            return 0,0
    if len(actual)==0:
        if len(predicted)==0:
            return 1,1
        else:
            return 0,0
    precision = len(relevant_and_recommended) / len(predicted)
    recall = len(relevant_and_recommended) / len(actual)
    return precision,recall

def get_score(pred_list,ground_truth_list):
    p,r=precision_and_recall(ground_truth_list,pred_list)
    jaccard=jaccard_index(ground_truth_list,pred_list)
    return p,r,jaccard

graph_list=[]

m='node'
model='GPT-4o'
patterns='square'
direction=direct_judge(patterns)
density=[]
ground_turth_flag=False
over_all_F1=[]


graph_list=[]
data_list=[]
F1_scores=[]
code_ex=[]

model_names= ['GPT-4','GPT-4o','mix','llama','gemini','claude','o1m']
counts=0
for model in model_names:
    # if model!='claude':continue
    F1_scores=[]
    code_ex=[]
    # for patterns in ['triangle','tailed-triangle','square','diamond','house','vs','FFL','FBL','d-diamond']:
    for patterns in ['claim_triangle','claim_tailed-triangle','claim_square','claim_diamond','claim_house','claim_vs','claim_FFL','claim_FBL','claim_d-diamond']:
    # for patterns in ['claim_d-diamond']:
    # for patterns in ['vs']:    
        # if patterns!='claim_square':continue
        for m in ['node','edge']:
            graph_list=[]
            data_list=[]
            for difficulty in ['mid']:

                if direction==False:
                    path=f'../../Dataset/Basic_tasks/{difficulty}.pkl'
                else:
                    path=f'../../Dataset/Basic_tasks/di_{difficulty}.pkl'
                
                with open(path,'rb') as f:
                    graphs=pickle.load(f)

                if difficulty!='easy':
                    new_graphs=[]
                    with open('../../Dataset/sub_idx.pkl','rb')as f:
                        sub_idx_list=pickle.load(f)
                    for i in sub_idx_list:
                        new_graphs.append(graphs[i])
                    graphs=new_graphs
                graph_list.extend(graphs)
                file_path=f'zero_response/{difficulty}/{model}/{model}_{patterns}_'+difficulty+'_'+m+'_.json'
                try:
                    
                    with open(file_path,'r') as f:
                        data=json.load(f)
                except:
                    F1_scores.append('no_file')
                    code_ex.append('no_file')
                    continue

                if 'claim' in patterns:
                    ans_patterns=patterns.split('_')[-1]
                else:
                    ans_patterns=patterns
                ans_path=f'ans/{ans_patterns}_{difficulty}.pkl'

                ground_turth=[]
                if os.path.exists(ans_path):
                    with open(ans_path,'rb') as f:
                        ground_turth=pickle.load(f)
                        ground_turth_flag=True
                    new_ground=[]
                    if difficulty!='easy':
                        for i in sub_idx_list:
                            new_ground.append(ground_turth[i])
                        ground_turth=new_ground
                
                data_list.extend(data)

                jaccard_list=[]
                precisions=[]
                recalls=[]
                density_list=[]
                code_solution=0
                # print(len(data_list),len(graph_list))
                for idx,(g,d) in tqdm(enumerate(zip(graph_list,data_list)),total=len(data_list)):
                    ans_list=ground_turth[idx]
                    density_list.append(nx.density(g))
                    response=d['response']

                    # if 'python' in response:
                    #     code_solution+=1
                    #     continue
                    pred_triangles=extract_edges(response,ans_patterns)
                    p,r,jaccrad=get_score(pred_triangles,ans_list)
                    jaccard_list.append(jaccrad)
                    precisions.append(p)
                    recalls.append(r)

                    if len(ans_list)==0:counts+=1
                    # num=extract_num(response)
                    num=len(set(pred_triangles))
                    # if idx==10:break
                precision_score=sum(precisions)/len(precisions)
                recall_score=sum(recalls)/len(recalls)
                jaccard_score=sum(jaccard_list)/len(jaccard_list)

                if ground_turth_flag==False:
                    with open(ans_path,'wb') as f:
                        pickle.dump(ground_turth,f)
                try:
                    f1=2*precision_score*recall_score/(precision_score+recall_score)
                except:
                    f1=0
                F1_scores.append(int(f1*10000)/10000)
                print('Precision',precision_score,'Recall',recall_score,'Jaccard',jaccard_score,'F1',f1)
                print(np.mean(np.array(density_list)))
                print(len(recalls))
                
                if difficulty=='easy':
                    print('code_solution',code_solution/250)
                    code_ex.append(code_solution/250)
                else:
                    print('code_solution',code_solution/50)
                    code_ex.append(code_solution/50)
    over_all_F1.append(F1_scores)

print(counts)

for name,F1_scores in zip(model_names,over_all_F1):
    print(difficulty,name,end='\t')
    for idx,f in enumerate(F1_scores):
        print(f,end='\t')
    print()
