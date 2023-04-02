"""
Name : generate_all_faq.py
Author  : 北在哪
Contact : 1439187192@qq.com
Time    : 2022/11/14 15:05
Desc:
"""
import pandas as pd
import yaml
import re
from collections import defaultdict
import json

if __name__ == '__main__':
    intent_mapping = pd.read_csv('../intent_description_mapping.csv')
    intent_query = {}
    for intent, query in zip(list(intent_mapping['intent']), list(intent_mapping['button'])):
        if '/' not in intent:
            continue
        intent_query[intent] = query

    with open('../../data/nlu/responses/responses.yml', 'r') as f:
        responses = yaml.load(f, yaml.FullLoader)

    intent_answer = {}
    for i, j in responses['responses'].items():
        intent = i
        answer = j[0]['text']
        answer = re.sub(r"<br>", "", answer)
        answer = re.sub(r"</div>", "", answer)
        answer = re.sub(r"</a>", "", answer)
        answer = re.sub(r"<a.*target='_blank'>", "", answer)
        answer = re.sub(r"<div class='msg-text'>", "", answer)
        intent_answer[intent.replace('utter_', '')] = answer

    infotype_query_answer = defaultdict(list)
    for i in intent_query.keys():
        if i[-3:] == '/其他':
            continue
        infotype = i.split('/')[0]
        temp = {
            'query': intent_query[i],
            'answer': intent_answer[i],
        }
        infotype_query_answer[infotype].append(temp)

    with open('./infotype_query_answer.json', 'w', encoding='utf-8') as f:
        json.dump(infotype_query_answer, f, ensure_ascii=False, indent=4)
