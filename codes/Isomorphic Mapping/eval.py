import pickle
import json
from sklearn.metrics import accuracy_score
import re
import os
import networkx as nx
path='../Dataset/Iso'
prompt='zero'




def extracts(input_txt,idx):
    
    input_txt = input_txt.replace( '↔','->')
    input_txt = input_txt.replace( '→','->')
    input_txt = input_txt.replace('**', '')
    input_txt = input_txt.replace('  ', ' ')
    input_txt = input_txt.replace('\t', ' ')
    
    cleaned_input = input_txt.replace('&', '')
    cleaned_input = cleaned_input.replace('\\\\', '')
    cleaned_input = re.sub(r'\\text\{', '', cleaned_input)
    cleaned_input = cleaned_input.replace('}', '')
    cleaned_input = cleaned_input.replace('\\rightarrow', '->')
    
    input_txt=cleaned_input
    input_txt = input_txt.replace('\u202F', ' ').replace('\u00A0', ' ')

    patterns=[r"- Node (\d+) -> Node (\d+)(?:.*)?",
              r"- Node (\d+) -> (\d+)(?:.*)?",
              r"- Graph 1: Node (\d+) -> Graph 2: Node (\d+)(?:.*)?",
              r"- Node (\d+) \(Graph 1\) -> Node (\d+) \(Graph 2\)(?:.*)?",
              r"- (\d+) -> (\d+)(?:.*)?",
              r"- Graph 1: (\d+) -> Graph 2: (\d+)(?:.*)?",
              r"Graph 1 (\d+) -> Graph 2 (\d+)(?:.*)?",
              r'- Node (\d+) (G1) -> Node (\d+) (G2)(?:.*)?',
              r"- Graph 1 Node (\d+) -> Graph 2 Node (\d+)(?:.*)?",
              r"- Node (\d+) \(G1\) -> Node (\d+) \(G2\)",
              r"Graph 1: Node (\d+) -> Graph 2: Node (\d+)",
              r"Node (\d+)\s*[-–>]+\s*Node (\d+)",
              r"- Node (\d+) \(Degree \d+\) -> Node (\d+) \(Degree \d+\)",
              r"- G1: Node (\d+) -> G2: Node (\d+)(?: \((.*)\))?",
              r"\\text\{Graph 1: Node (\d+)\} \\rightarrow \\text\{Graph 2: Node (\d+)\}",
              r'(\d+)\s*->\s*(\d+)',
              r'Graph 1: Node #(\d+) -> Graph 2: Node #(\d+)',
              r'G1: Node (\d+) -> G2: Node (\d+)',
              r'Graph 1: Node (\d+) -> Graph 2: Node (\d+)',
              r'Graph 1: Node (\d+) \([A-Za-z]+\) -> Graph 2: Node (\d+) \([A-Za-z]+\)',
              r'G1:\s*node\s*(\d+)\s*->\s*G2:\s*node\s*(\d+)',
              r'Graph 1: Node (\w+) \((\d+)\) -> Graph 2: Node (\w+) \((\d+)\)',
              r'Node (\d+)\s*->\s*Graph 2: Node (\d+)',
              r'G1\((\d+)\)\s*->\s*G2\((\d+)\)',
              r'G1:Node (\d+)\s*->\s*G2:Node (\d+)(?: \(degree \d+\))?',
              r'\|\s*Node\s*(\d+)\s*\|\s*Node\s*(\d+)\s*\|',
              r'Graph 1: Node(\d+)\s*->\s*Graph 2: Node(\d+)',
              r'Graph 1: Node (\d+)\s*->\s*Graph 2: Node (\d+)',
              r'-\s*Node\s*(\d+)\s*->\s*Graph\s*2:\s*Node\s*(\d+)',
              r'Graph 1: Node #(\d+)\s*->\s*Graph 2: Node (\d+)',
              r'Node (\d+)\s*->\s*Graph 2:\s*(\d+)',
              r'Graph 1: Node\s+(\d+)\s*->\s*Graph 2: Node\s+(\d+)',
              r'Graph G_1: Node (\d+)\s*->\s*Graph G_2: Node (\d+)',
              r'Graph 1: Node\s*(\d+)\s*->\s*Graph\s*2: Node\s*(\d+)',
              r'\|\s*(\d+)\s*\|\s*(\d+)\s*\|',
              r'Graph1: Node\s*(\d+)\s*->\s*Graph2: Node\s*(\d+)(?:.*)?',
              r'Graph\s*1\s*:\s*(\d+)\s*->\s*Graph\s*2\s*:\s*(\d+)(?:.*)?']

    # Find all matches in the input string
    matches=[]
    for p in patterns:
        matches.extend(re.findall(p,input_txt))
    if '\\boxed' in input_txt:
        pattern=r'(\d+)\s*:\s*(\d+)'
        matches.extend(re.findall(pattern,input_txt))
    mapping_dicts={}
    for i in matches:
        counts_list=[]
        for ids in i:
            try:
                nums=int(ids)
                counts_list.append(nums)
            except:pass

        mapping_dicts[int(counts_list[0])]=int(counts_list[1])
    # print(mapping_dicts)
    # if len(matches)==0:
    #     print(idx)
    #     print('extraction empty')
    #     print(input_txt)
    return mapping_dicts


def edge_detect(g1,g2):
    counts=0
    edge1=g1.edges
    edge2=g2.edges
    for e in edge1:
        if (e[0],e[1]) in edge2 or (e[1],e[0]) in edge2:
            counts+=1
    if counts==len(edge2):
        return True
    else:return False

method='edge'
difficulty='hard'
model_name='o1m'
for difficulty in ['easy','mid','hard']:
    over_all_score=[]
    lens=[]
    for model_name in ['GPT-4','GPT-4o','mix','llama','gemini','claude','o1m']:
        for method in ['node','edge']:
    # print('Iso/zero_response/GPT-4/GPT-4_easy_node.json')
            with open(os.path.join(path,f'{difficulty}.pkl'),'rb') as f:
                graph_pairs=pickle.load(f)


            with open(f'./zero_response/{model_name}/{model_name}_{difficulty}_{method}.json','r') as f:
                data=json.load(f)

            if difficulty!='easy':
                new_graphs=[]
                new_data=[]
                with open('../Dataset/sub_idx.pkl','rb')as f:
                    sub_idx_list=pickle.load(f)
                for i in sub_idx_list:
                    new_graphs.append(graph_pairs[i])
                graph_pairs=new_graphs
                if len(data)>250:
                    data=data[-250:]
                # print(len(data))
                if len(data)>50:
                    for idx in sub_idx_list:
                        new_data.append(data[idx])
                    data=new_data
                # print(len(data))
                

            preds=[]
            f_preds=[]
            f_grounds=[]


            # print(len(data))
            for idx,(d,g) in enumerate(zip(data,graph_pairs)):
                response=d['response']
                if 'degree' not in response.lower() :continue
                
                g1,g2=g
                mapping_dicts=extracts(response,idx)
                g1=nx.relabel_nodes(g1,mapping_dicts)
                flag=edge_detect(g1,g2)
                f_preds.append(flag)
            score=sum(f_preds)/len(f_preds)
            lens.append(len(f_preds))
            over_all_score.append(score)
    print(difficulty, end='\t')
    for score in over_all_score:
        print(score,end='\t')
    # for l in lens:
    #     print(l,end='\t')
    print()
    
