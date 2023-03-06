"""
Name : model_to_onnx.py
Author  : 北在哪
Contect : 1439187192@qq.com
Time    : 2023/3/6 9:35
Desc:
"""
import torch
from transformers import BertForQuestionAnswering, BertTokenizer
# from onnxruntime_tools import optimizer
from config import *


def main():
    # model = BertForQuestionAnswering.from_pretrained(QA_MODEL_PATH)
    # model.to('cuda')
    # model.eval()
    model_save_to = './actions/qa_onnx_model/qa_model.onnx'
    optimized_model_save_to = './actions/qa_onnx_model/optimized_qa_model.onnx'

    # seg_length = 512
    # batch_size = 10
    # dummy_input0 = torch.ones(batch_size, seg_length).long()
    # dummy_input1 = torch.ones(batch_size, seg_length).long()
    # dummy_input2 = torch.ones(batch_size, seg_length).long()
    # inputs = {
    #     'input_ids': dummy_input0.to('cuda'),
    #     'attention_mask': dummy_input1.to('cuda'),
    #     'token_type_ids': dummy_input2.to('cuda')
    # }
    # torch.onnx.export(model, tuple(inputs.values()), model_save_to, opset_version=11)
    # optimized_model = optimizer.optimize_model(model_save_to, model_type='bert', num_heads=16, hidden_size=1024)
    # optimized_model.save_model_to_file(optimized_model_save_to)


if __name__ == '__main__':
    main()
