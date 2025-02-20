import pickle as pkl
import networkx as nx
import numpy as np
import random
import json
import numpy as np
import sys
from tqdm import tqdm
sys.path.append('../')
from graph_nlp import graph_txt
from response import get_response

train_samples=15
with open('../Dataset/Summary/BA-2motif.pkl','rb') as fin:
    adjs,features,labels = pkl.load(fin)
train_positive=[idx for idx in range(450)]
train_negative=[idx for idx in range(500,950)]
test_list=[idx for idx in range(450,500)]+[idx for idx in range(950,1000)]


all_train_list=[]
for _ in range(100):
    train_list = random.sample(train_positive, train_samples)+random.sample(train_negative, train_samples)
    all_train_list.append(train_list)
with open('../Dataset/Summary/BA-train.pkl','rb') as fin:
    all_train_list=pkl.load(fin)



deployment_name='GPT-4'
for method in ['edge','node']:
    for gold_set in ['Set1','Set2']:
        input_txt=[]
        for train_list in all_train_list:
            graph_train_lists=[]
            centern_node=[]

            
            for i in train_list:
                graph_train_lists.append(nx.from_numpy_array(adjs[i,:,:]))
            txt='You are provided two sets of graphs. The first sets are:\n'
            counts=0
            for idx, (g,train_idx) in enumerate(zip(graph_train_lists,train_list)):
                if np.argmax(labels[train_idx],-1)==0:
                    counts+=1
                    txt+= f'\nSet1: No. {counts}:\n'+graph_txt(g,method)# +f'Label:{np.argmax(labels[train_idx],-1)}'
                    txt+='The second sets are:\n'    
            counts=0
            for idx, (g,train_idx) in enumerate(zip(graph_train_lists,train_list)):
                if np.argmax(labels[train_idx],-1)==1:
                    counts+=1
                    txt+= f'\nSet2: No. {counts}:\n'+graph_txt(g,method)


            txt+=f'What are the differences between the two sets? Show the special pattern in {gold_set}, shown as list as: [The pattern graph is: (Node #1, Node #2), ...] '
            input_txt.append(txt)
            
            

        response_file=[]
        for idx,input_text in tqdm(enumerate(input_txt),total=len(input_txt)):
            system_prompt=''
            responses_dicts={}
            # print(responses[0][0][0])
            responses_dicts['response']=get_response(deployment_name,input_text)
            responses_dicts['text']=input_text
            responses_dicts['idx']=idx
            response_file.append(responses_dicts)
        with open(f'zero_response/{gold_set}_{method}_{deployment_name}.json','w') as f:
            json.dump(response_file,f)
