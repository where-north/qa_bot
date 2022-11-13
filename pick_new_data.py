"""
Name : pick_new_data.py
Author  : 北在哪
Contect : 1439187192@qq.com
Time    : 2022/11/11 23:59
Desc:
"""
import json
from collections import defaultdict
import requests

url = "http://localhost:5005/model/parse"
data = {"text": "校园卡"}
data = json.dumps(data, ensure_ascii=False)
data = data.encode(encoding="utf-8")
r = requests.post(url=url, data=data)
res = json.loads(r.text)

print(res['text'])
print(res['intent'])
print(res['intent_ranking'])
print(res['response_selector'])

