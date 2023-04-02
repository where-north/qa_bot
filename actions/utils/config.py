"""
Name : config.py
Author  : 北在哪
Contect : 1439187192@qq.com
Time    : 2022/11/17 19:46
Desc:
"""

# ES docker容器内部的IP
ES_DOCKER_IP = "172.17.0.3"
# CQA DQA model docker容器内部的IP
QA_DOCKER_IP = "172.17.0.2"
# Chat Api docker容器内部的IP
CHAT_DOCKER_IP = "172.17.0.7"
# chatgpt url
CHAT_URL = f'http://{CHAT_DOCKER_IP}:8888/chat'
# CQA召回url
CQA_URL = f'http://{QA_DOCKER_IP}:7070/cqa'
# 模型推理url
QA_URL = f'http://{QA_DOCKER_IP}:8080/qa'
# DQA召回url
DQA_URL = f'http://{QA_DOCKER_IP}:9090/dqa'
# 抽取式阅读器模型路径
QA_MODEL_PATH = '../MODEL/luhua-chinese_pretrain_mrc_macbert_large'
# intent对应的文字描述文件路径
INTENT_DESCRIPTION_MAPPING_PATH = "actions/intent_description_mapping.csv"
# 天气预报API相关配置
QUERY_KEY = "82510add8a7340caa9afcabfab78a639"
CITY_LOOKUP_URL = "https://geoapi.qweather.com/v2/city/lookup"
WEATHER_URL = "https://devapi.qweather.com/v7/weather/now"
# CQA问答数据路径
CQA_CORPUS_PATH = '../official_document_crawler/data/cqa_data1.csv'
# CQA ES索引名称
CQA_INDEX_NAME = 'cqa'
# DQA ES索引名称
DQA_INDEX_NAME = 'dqa'
