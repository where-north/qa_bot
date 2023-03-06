"""
Name : qa_model_server.py
Author  : 北在哪
Contect : 1439187192@qq.com
Time    : 2023/3/6 9:31
Desc:
"""
import sys
import os

sys.path.append(os.path.abspath('.'))
from actions.utils.config import *
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from collections import OrderedDict
from actions.utils.dqa_es import ElasticSearchBM25 as DQA_ElasticSearchBM25
import asyncio

app = FastAPI()


class InputData(BaseModel):
    user_query: str


# 定义响应的数据模型
class ResponseData(BaseModel):
    documents_ranked: OrderedDict
    scores_ranked: OrderedDict
    ok: bool


@app.post('/dqa', response_model=ResponseData)
async def dqa_api(data: InputData):
    try:
        documents_ranked, scores_ranked = DQA_ES.query(topk=10, query=data.user_query, return_scores=True)
        response = ResponseData(documents_ranked=documents_ranked, scores_ranked=scores_ranked, ok=True)
    except Exception as e:
        response = ResponseData(documents_ranked=[], scores_ranked=[], ok=False)

    return response


if __name__ == "__main__":
    DQA_ES = DQA_ElasticSearchBM25(index_name=DQA_INDEX_NAME, reindexing=True)
    uvicorn.run(app, host="0.0.0.0", port=9090)
