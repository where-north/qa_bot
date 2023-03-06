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
from actions.utils.document_qa_utils import predict
from actions.utils.config import *
from transformers import BertForQuestionAnswering, BertTokenizer
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import asyncio

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=5)


# class ONNXModel():
#     def __init__(self, onnx_path):
#         """
#         :param onnx_path:
#         """
#         self.onnx_session = onnxruntime.InferenceSession(onnx_path)
#         self.input_name = self.get_input_name(self.onnx_session)
#         self.output_name = self.get_output_name(self.onnx_session)
#         print("input_name:{}".format(self.input_name))
#         print("output_name:{}".format(self.output_name))
#
#     def get_output_name(self, onnx_session):
#         """
#         output_name = onnx_session.get_outputs()[0].name
#         :param onnx_session:
#         :return:
#         """
#         output_name = []
#         for node in onnx_session.get_outputs():
#             output_name.append(node.name)
#         return output_name
#
#     def get_input_name(self, onnx_session):
#         """
#         input_name = onnx_session.get_inputs()[0].name
#         :param onnx_session:
#         :return:
#         """
#         input_name = []
#         for node in onnx_session.get_inputs():
#             input_name.append(node.name)
#         return input_name
#
#     def get_input_feed(self, input_name, input_numpy):
#         """
#         input_feed={self.input_name: input_numpy}
#         :param input_name:
#         :param input_numpy:
#         :return:
#         """
#         input_feed = {}
#         for i, name in enumerate(input_name):
#             input_feed[name] = input_numpy[i]
#         return input_feed
#
#     def forward(self, input_numpy):
#         # 输入数据的类型必须与模型一致
#         input_feed = self.get_input_feed(self.input_name, input_numpy)
#         scores = self.onnx_session.run(self.output_name, input_feed=input_feed)
#         return scores


# 正式测试，batch_size固定为1

# 定义请求的数据模型
class InputData(BaseModel):
    title: str
    document: str
    document_id: str
    question: str


# 定义响应的数据模型
class ResponseData(BaseModel):
    predict: OrderedDict
    ok: bool


@app.post('/qa', response_model=ResponseData)
async def qa_api(input_datas: List[InputData]):
    # 将处理请求的函数提交到线程池中
    future = executor.submit(predict, qa_model, tokenizer, input_datas)

    try:
        # 等待处理请求的函数执行完毕并返回结果
        pre = await asyncio.wrap_future(future)
        response = ResponseData(predict=pre, ok=True)
    except Exception as e:
        response = ResponseData(predict=[], ok=False)

    return response


# 此处示例，需要根据模型类型重写
def init_model():
    model = BertForQuestionAnswering.from_pretrained(QA_MODEL_PATH)
    model.cuda()
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(QA_MODEL_PATH, do_lower_case=True)
    return model, tokenizer


if __name__ == "__main__":
    qa_model, tokenizer = init_model()
    uvicorn.run(app, host="0.0.0.0", port=8080)
