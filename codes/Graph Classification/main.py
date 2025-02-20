import pickle as pkl
import networkx as nx
import numpy as np
import random
import json
import numpy as np
import sys
from tqdm import tqdm
sys.path.append('../')
from response import get_response

from graph_nlp import graph_txt

dataset='BA2'
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
    # print(np.sum(labels[test_list]),len(test_list))
with open('../Dataset/Summary/BA-train.pkl','wb') as fin:
    # adjs,features,labels = pkl.load(fin)
    pkl.dump(all_train_list,fin)
with open('../Dataset/Summary/BA-train.pkl','rb') as fin:
    all_train_list=pkl.load(fin)

deployment_name='GPT-4'
graph_test_lists=[]
for i in test_list:
    graph_test_lists.append(nx.from_numpy_array(adjs[i,:,:]))

for method in ['edge','node']:
    set_1_path=f'../Summary/patterns/{deployment_name}_{method}_Set1_patterns.pkl'
    set_2_path=f'../Summary/patterns/{deployment_name}_{method}_Set2_patterns.pkl'
    with open(set_1_path,'rb') as f:
        set1_list=pkl.load(f)
    with open(set_2_path,'rb') as f:
        set2_list=pkl.load(f)

    # for gold_set in ['Set1','Set2']:
    input_txt=[]
    for g_test in graph_test_lists:
        txt='You are an expert at classifying different types of graphs based on whether they contain specific patterns. The first type of graph includes patterns such as:'
        for idx,g in enumerate(set1_list):
            txt+= f'\npattern 1: No. {idx}:\n'+graph_txt(g,method)# +f'Label:{np.argmax(labels[train_idx],-1)}'
        txt+='The secod type of patterns like'
        for idx,g in enumerate(set2_list):
            txt+= f'\npattern 2: No. {idx}:\n'+graph_txt(g,method)# +f'Label:{np.argmax(labels[train_idx],-1)}'
        txt+='Now, please identify which type the given graph is most likely to belong to. The graph is shown as: '+graph_txt(g_test,method)
        txt+='Chosing the answers from "It should be in Type 1" or "It should be in Type 2" '
        input_txt.append(txt)
    print(input_txt[0])

    response_file=[]
    for idx,input_text in tqdm(enumerate(input_txt),total=len(input_txt)):
        responses_dicts={}
        responses_dicts['response']=get_response(deployment_name,input_text)
        responses_dicts['text']=input_text
        responses_dicts['idx']=idx
        response_file.append(responses_dicts)
    with open(f'zero_response/{method}_{deployment_name}.json','w') as f:
        json.dump(response_file,f)
