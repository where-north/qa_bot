"""
Name : model_to_onnx.py
Author  : 北在哪
Contect : 1439187192@qq.com
Time    : 2023/3/6 9:35
Desc:
"""
import torch
from transformers import BertForQuestionAnswering, BertTokenizer
from onnxruntime_tools import optimizer
from config import *


def main():
    model = BertForQuestionAnswering.from_pretrained(QA_MODEL_PATH)
    model.cuda()
    model.eval()
    model_save_to = '../qa_onnx_model/qa_model.onnx'

    seg_length = 512
    dummy_input0 = torch.Tensor([[1 for _ in range(seg_length)]]).long()
    dummy_input1 = torch.Tensor([[1 for _ in range(seg_length)]]).long()
    dummy_input2 = torch.Tensor([[1 for _ in range(seg_length)]]).long()
    inputs = {
        'input_ids': dummy_input0.cuda(),
        'attention_mask': dummy_input1.cuda(),
        'token_type_ids': dummy_input2.cuda()
    }
    torch.onnx.export(model, inputs, model_save_to, opset_version=11)
    optimized_model = optimizer.optimize_model(model_save_to, model_type='bert', num_heads=16, hidden_size=1024)
    optimized_model.save_model_to_file(model_save_to)


if __name__ == '__main__':
    main()
