"""
Name : pick_new_data.py
Author  : 北在哪
Contect : 1439187192@qq.com
Time    : 2022/11/11 23:59
Desc:
"""
import json
from collections import defaultdict

with open("results/intent_errors.json", "r") as f:
    data = json.load(f)

data_dict = defaultdict(list)

for i in data:
    label = i['intent_prediction']['name']
    confidence = i['intent_prediction']['confidence']
    if confidence > 0.9:
        data_dict[label].append('- ' + i['text'])
    else:
        data_dict['other'].append(i)
for j in data_dict.keys():
    if j == 'other':
        f = open(f"results/{j}.json", "w")
        json.dump(data_dict[j], f, ensure_ascii=False, indent=4)
    else:
        f = open(f"results/{j.replace('/', '_')}.txt", "w")
        for k in data_dict[j]:
            f.write(k)
            f.write('\n')
