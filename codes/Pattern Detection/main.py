import networkx as nx
import pickle
import sys
from tqdm import tqdm
import json
sys.path.append('../../')
from graph_nlp import graph_txt
from graph_patterns import pattern_generation,pattern_description,direct_judge
from responses import get_response
import os

'''
# claim: give a named triangle pattern
# trianlge: generated triangle
'''
# pattern='diamond' 

nodes_form={'diamond':'The detected patterns are: [(#1, #2, #3, #4), ...]',
            'square':'The detected patterns are: [(#1, #2, #3, #4), ...]',
            'tailed-triangle':'The detected patterns are: [(#1, #2, #3, #4), ...]',
            'house':'The detected patterns are: [(#1, #2, #3, #4, #5), ...]',
            'triangle':'The detected patterns are: [(#1, #2, #3), ...]',
            'FFL':'The detected patterns are: [(#1, #2, #3), ...]',
            'FBL':'The detected patterns are: [(#1, #2, #3), ...]',
            'vs':'The detected patterns are: [(#1, #2, #3), ...]',
            'd-diamond':'The detected patterns are: [(#1, #2, #3, #4), ...]',
            'diamond_a':'The detected patterns are: [(#1, #2, #3, #4, #5), ...]'}

deployment_name='GPT-4'
for pattern in ['d-diamond']:#,'claim_tailed-triangle','tailed-triangle']:
    direction=direct_judge(pattern)
    for difficulty in ['mid']:
        for method in ['edge']:
            if direction==False:
                path=f'../../Dataset/Basic_tasks/{difficulty}.pkl'
            else:
                path=f'../../Dataset/Basic_tasks/di_{difficulty}.pkl'
            with open(path,'rb') as f:
                graphs=pickle.load(f)
            
            if difficulty!='easy':
                new_graphs=[]
                # if 'd-diamond' not in pattern:
                with open('../../Dataset/sub_idx.pkl','rb')as f:
                    sub_idx_list=pickle.load(f)
                for i in sub_idx_list:
                    new_graphs.append(graphs[i])
                graphs=new_graphs

            input_txt=[]
            prompt='zero'
            for g in graphs:
                txt=graph_txt(g,method)
                if 'claim' in pattern:
                    pattern_name=pattern.split('_')[1]
                    claim=f'Identify the occurrence patterns of the given motif in the graph. The given patterns is {pattern_description[pattern_name]}.\n'
                    states=f"Please identify the patterns for each node and list all of them as follows: {nodes_form[pattern_name]}."
                else:
                    claim='Identify the occurrence patterns of the given motif in the graph.\nThe pattern is:'
                    pattern_graph=graph_txt(pattern_generation(pattern),method)
                    claim+=pattern_graph
                    claim+='\n The graph is:\n'
                    states=f'Please identify the patterns for each node and list all of them as follows: {nodes_form[pattern]}.'
                txt=claim+txt
                txt=txt+states
                
                input_txt.append(txt)

            ori_data=[]
            if os.path.exists(f'{prompt}_response/{difficulty}/{deployment_name}/{deployment_name}_{pattern}_'+difficulty+'_'+method+'_'+'.json'):
                with open(f'{prompt}_response/{difficulty}/{deployment_name}/{deployment_name}_{pattern}_'+difficulty+'_'+method+'_'+'.json','r') as f:
                    ori_data=json.load(f)

            response_file=[]
            for idx,input_text in tqdm(enumerate(input_txt),total=len(input_txt)):
                responses_dicts={}
                responses_dicts['response']=get_response(deployment_name,input_text)
                responses_dicts['text']=input_text
                responses_dicts['idx']=idx
                response_file.append(responses_dicts)

            if os.path.exists(f'{prompt}_response/{difficulty}/{deployment_name}')==False:
                os.makedirs(f'{prompt}_response/{difficulty}/{deployment_name}')
            with open(f'{prompt}_response/{difficulty}/{deployment_name}/{deployment_name}_{pattern}_'+difficulty+'_'+method+'_'+'.json','w') as f:
                json.dump(response_file,f)
