"""
Name : generate_mapping_csv.py
Author  : 北在哪
Contect : 1439187192@qq.com
Time    : 2022/11/13 15:44
Desc:
"""
from glob import glob
import yaml
import pandas as pd

yml_list = glob("../data/nlu/*.yml")
intent, button, entities = [], [], []
for yml_path in yml_list:
    with open(yml_path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    nlu = data['nlu']
    for n in nlu:
        if 'intent' not in n:
            continue
        intent.append(n['intent'])
        first_example = n['examples'].split('\n')[0].replace('- ', '')
        button.append(first_example)
        entities.append('')

pd.DataFrame({
    'intent': intent,
    'button': button,
    'entities': entities
}).to_csv('./intent_description_mapping.csv', index=False)
