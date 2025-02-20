import networkx as nx
import pickle
import sys
from tqdm import tqdm
import json
import os
sys.path.append('../../')
from graph_nlp import graph_txt
from response import get_response


deployment_name='GPT-4'
for difficulty in ['easy','mid','hard']:
    for method in ['node','edge']:

        path=f'../../Dataset/Basic_tasks/{difficulty}.pkl'
        with open(path,'rb') as f:
            graphs=pickle.load(f)

        if difficulty!='easy':
            new_graphs=[]
            with open('../../Dataset/sub_idx.pkl','rb')as f:
                sub_idx_list=pickle.load(f)
            for i in sub_idx_list:
                new_graphs.append(graphs[i])
            graphs=new_graphs
        
        input_txt=[]
        prompt='zero'
        for g in graphs:
            
            txt=graph_txt(g,method)
            claim='Determine the 3-core subgraphs in the graph.\n'
            txt=claim+txt
            states='Please identify the 3-core subgraphs and list the nodes in the subgraphs as: "The nodes in the subgraphs are: [#1, #2].'
            txt=txt+states
            
            input_txt.append(txt)
        response_file=[]
        for idx,input_text in tqdm(enumerate(input_txt),total=len(input_txt)):
            system_prompt=''
            responses=[[['Unknow error. No']]]
            responses_dicts={}
            # print(responses[0][0][0])
            responses_dicts['response']=get_response(model_name=deployment_name,input_text=input_text)
            responses_dicts['text']=input_text
            responses_dicts['idx']=idx
            response_file.append(responses_dicts)
            # break
            # with open(f'{prompt}_response/{deployment_name}_'+difficulty+'_'+method+'_'+'tmp_degrees.json','w') as f:
            #     json.dump(response_file,f)
        if os.path.exists(f'{prompt}_response/{deployment_name}')==False:
            os.makedirs(f'{prompt}_response/{deployment_name}')
        with open(f'{prompt}_response/{deployment_name}/{deployment_name}_'+difficulty+'_'+method+'_2'+'.json','w') as f:
            json.dump(response_file,f)
