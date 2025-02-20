import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score
deployment_name='llama'
method='node'
with open(f'zero_response/{method}_{deployment_name}.json','r') as f:
    response_data=json.load(f)

def extract_ans(text):
    text=text.lower()
    if 'it should be in type 1' in text:
        ans=0
    else:
        ans=1
    return ans

ans_list=[]
ground_truth=[]
for idx,data in tqdm(enumerate(response_data),total=len(response_data)):
    try:
        response=data['response']# [0][0][0]
    except:
        print(idx)
        continue
    # print(response)
    ans_list.append(extract_ans(response))
    if idx<50:
        ground_truth.append(0)
    else:
        ground_truth.append(1)
    # break
acc=accuracy_score(ans_list,ground_truth)
print(acc)