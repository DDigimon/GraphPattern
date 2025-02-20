import os
import pickle
import json
import re
import networkx as nx
import sys
sys.path.append('../')
from graph_patterns import pattern_generation,direct_judge,find_pattern_list
from util import graph_iso_check
from tqdm import tqdm
from networkx.algorithms import isomorphism
from tqdm import tqdm

difficulty='hard'
model_name='o1m'
method='node'
pattern='triangle'
model_names=['GPT-4','GPT-4o','mix','llama','gemini','claude','o1m']
over_all_scores={}
overall_pattern_score={}
for model_name in model_names:
    over_all_scores[model_name]=[]
    overall_pattern_score[model_name]=[]
    for pattern in ['triangle','square','diamond','house','FFL','FBL','d-diamond']:
        for method in ['node','edge']:
            path=f'../Dataset/Fresub/{pattern}'
            # print('Iso/zero_response/GPT-4/GPT-4_easy_node.json')
            with open(os.path.join(path,f'{difficulty}.pkl'),'rb') as f:
                graphs=pickle.load(f)


            with open(os.path.join(path,f'{difficulty}_idx.pkl'),'rb') as f:
                idx_list=pickle.load(f)
            if difficulty!='easy':
                idx_list=idx_list[:50]
            else:
                idx_list=idx_list[:100]
            try:
                with open(f'./zero_response/{model_name}/{model_name}_{difficulty}_{pattern}_{method}.json','r') as f:
                    data=json.load(f)
            except:
                print(f'./zero_response/{model_name}/{model_name}_{difficulty}_{pattern}_{method}.json')
                over_all_scores[model_name].append('no_file')
                overall_pattern_score[model_name].append('no_file')



            def extract_text_edge(text):
                patterns=[r'\(\d+, \d+\)', r'\(\d+,\d+\)', r'\(Node \d+, Node \d+\)', r'\[\d+, \d+\]',r'\(N\d+, N\d+\)',r'\(N_\d+, N_\d+\)',r'\(N_\d+, N_\{\d+\}\)',r'\(N_\{\d+\}, N_\{\d+\}\)',r'\(\w, \w\)']
                edges=[]
                for pattern in patterns:
                    edges.extend(re.findall(pattern,text))
                edge_list=[]
                for e in edges:
                    pattern=r'\d+'
                    edge_pattern=re.findall(pattern,e)
                    if len(edge_pattern)!=0:
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

            def construct_graphs(edges,pattern):
                direction=direct_judge(pattern)
                if direction:
                    G=nx.DiGraph()
                else:
                    G=nx.Graph()
                G.add_edges_from(edges)
                return G

            def extraction(text,patterns):
                matches=re.findall(r'\[((?:\([^()]+\)(?:, )?)+)\]',text)
                # g_list=[]
                flag_list=[]
                pattern_list=[]
                pattern = [r'\[\s*(?:\(\s*(?:node\s+)?\d+\s*,\s*(?:node\s+)?\d+\s*\)\s*,?\s*)+\]']

                # Find all matches in the text
                matches=[]
                for p in pattern:
                    matches.extend(re.findall(p, text))
                if len(matches)>0:
                    for motif in matches:
                        graph=extract_text_edge(motif)
                        graph=construct_graphs(graph,patterns)
                        # flag=graph_iso_check(graph,pattern_structure)
                        # flag_list.append(flag)
                        pattern_list.append(graph)
                else:
                    # print(text)
                    return []
                clean_patterns=[]
                for g in pattern_list:
                    flag=False
                    for g_in in clean_patterns:
                        if graph_iso_check(g,g_in):
                            flag=True
                    if flag==False:
                        clean_patterns.append(g)
                return clean_patterns

                        # g_list.append(graph)


            pattern_structure=pattern_generation(pattern)

            def graph_check(given_graph_list,pattern_graph,direction=False):
                counts=0
                for g in given_graph_list:
                    # print(len(pattern_graph.nodes()))
                    if len(pattern_graph.nodes())>=8:
                        continue
                    # print(g,pattern_graph)
                    if direction==False:
                        GM = isomorphism.GraphMatcher(g, pattern_graph)
                    else:
                        GM = isomorphism.DiGraphMatcher(g, pattern_graph)
                    matches = list(GM.subgraph_isomorphisms_iter())
                    if len(matches)>0:
                        counts+=1
                if counts/len(given_graph_list)>0.6:
                    return True
                else:
                    return False

            directions={}
            given_pattern_name=['triangle','cycle','star','loop','clique']
            for n in given_pattern_name:
                if n not in directions:
                    directions[n]=0

            def text_ex(text,given_pattern):
                # print(text)
                numbers = re.findall(r"pattern #(\d+)", text)
                numbers=set(numbers)
                # print(numbers)
                patterns=len(numbers)
                
                counts=0
                rights=0
                if given_pattern in text:
                    rights+=1

                for names in given_pattern_name:
                    if names in text:
                        counts+=1
                        directions[names]+=1

                if patterns>0:
                    return rights,counts/patterns
                else:
                    return rights,0

            truth=[]
            given_pattern=[]
            for idx,(g_idx,d) in tqdm(enumerate(zip(idx_list,data))):
                if idx==100:break
                response=d['response']
                response=response.lower()
                pattern_list=extraction(response,pattern)
                get_pattern=False
                get_pattern,rights_score=text_ex(response,pattern)
                # print(rights_score)
                if rights_score>0:
                    given_pattern.append(get_pattern)
                    truth.append(1)
                    continue
                else:
                    # print(response)
                    for p in pattern_list:
                        if graph_iso_check(p,pattern_generation(pattern)):
                            get_pattern=True
                        given_g_list=[]
                        pattern_score=[]
                        for gs in g_idx:
                            # print(gs,graphs[gs])
                            given_g_list.append(graphs[gs])
                        pattern_score.append(graph_check(given_g_list,p,direct_judge(pattern)))
                        given_pattern.append(get_pattern)
                        truth.append(sum(pattern_score)/len(pattern_score))
            
            print(model_name)
            if len(truth)==0:score=0
            else:
                score=sum(truth)/len(truth)
            print('overall',score)
            over_all_scores[model_name].append(score)
            # print(given_pattern)
            if len(given_pattern)==0:
                p_s=0
            else:
                p_s=sum(given_pattern)/len(given_pattern)
            print('given pattern',p_s)
            overall_pattern_score[model_name].append(p_s)


for m in model_names:
    print(m,end='\t')
    for s in over_all_scores[m]:
        print(s,end='\t')
    print()

for m in model_names:
    print(m,end='\t')
    for s in overall_pattern_score[m]:
        print(s,end='\t')
    print()