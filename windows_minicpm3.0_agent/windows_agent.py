#!/usr/bin/env python
# encoding: utf-8
import re
from transformers import AutoTokenizer,AutoModelForCausalLM
from cli_demo import get_relevant_image
import requests
import json
import base64
import os
import gc
import torch
import copy
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")
use_cuda_tools = ['text_to_image_search']
model_path = r"D:\model_best\minicpm\pretrain_model\minicpm3"

def text_to_image_search(text_description,images_path):
    if text_description != None and images_path != None:
        return get_relevant_image(text_description,images_path)
def image_anwer_question(image_path,query):
    with open(image_path, 'rb') as image_file:
        # 将图片文件转换为 base64 编码
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    data = {
    "model": "minicpm-v",
    "prompt": query,
    "stream": False,
    "images": [encoded_string]
    }

    # 设置请求 URL
    url = "http://localhost:11434/api/generate"
    response = requests.post(url, json=data)

    return response.json()["response"]
    
def fake_tool_execute(toolcalls):
    tool_name = toolcalls['function']['name']
    arguments = toolcalls['function']["arguments"]
    if tool_name == "text_to_image_search":
        
        return text_to_image_search(arguments["text_description"],arguments["images_path"].strip())
    elif tool_name == "image_anwer_question":
        return image_anwer_question(arguments["image_path"],arguments["query"])
    else:
        return "没有这个工具"
    
tools0 = [
    {
        "type": "function",
        "function": {
            "name": "text_to_image_search",
            "description": "According to the input language description, find the most relevant picture with the language description under the specified path",
            "parameters": {
                "type": "object",
                "properties": {
                    "text_description": {
                        "type": "string",
                        "description": "input language description of the target image",
                    },
                    "images_path": {
                        "type": "string",
                        "description": "Contains the path to where the target image is",
                    },
                },
                "required": ["text_description","images_path"],
                "additionalProperties": False,
            },
        },
    }

]
tools1=[{
        "type": "function",
        "function": {
            "name": "image_anwer_question",
            "description": "Based on the provided image, output the most relevant answer to the Questions",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "The path to the target image",
                    },
                    "query": {
                        "type": "string",
                        "description": "Questions based on the target image",
                    },
                },
                "required": ["image_path","query"],
                "additionalProperties": False,
            },
        },
    }]
query="你帮我找出D:\model_best\minicpm\infer\images下的股票走势图，图中今日开盘指数是多少？"
messages = [
    {
        "role": "system",
        "content": "You are a helpful customer support assistant. Use the supplied tools to assist the user.",
    }
    
    
]
tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True
)

tools=tools0
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).cuda()

def clear_cuda_variables():
    # 删除所有局部变量
    local_vars = [var for var in locals().keys()]
    for var in local_vars:
        if isinstance(locals()[var], torch.Tensor) or var == 'model':
            del locals()[var]

    # 删除所有全局变量
    global_vars = [var for var in globals().keys()]
    for var in global_vars:
        if isinstance(globals()[var], torch.Tensor) or var == 'model':
            del globals()[var]

while True:
    if model.device == torch.device("cpu"):
        model = model.cuda()
        tools=copy.deepcopy(tools1)
    query = input("我有什么能够帮你：\n")
    messages.append({"role": "user", "content": query})
    #temp = copy.deepcopy(messages[-1])
    prompt = tokenizer.apply_chat_template(
    messages, tools=tools, tokenize=False, add_generation_prompt=True
    )
    prompt=tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():  # 明确指定不计算梯度
        response=model.generate(prompt.input_ids.cuda(),attention_mask=prompt["attention_mask"].cuda(),max_length=2048,do_sample=False)
    response = response[:, prompt.input_ids.shape[1]:]
    response=tokenizer.batch_decode(response, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(response)
    #messages[-1]=temp
    #response = outputs[0].outputs[0].text
    msg = tokenizer.decode_function_call(response)
    if (
        "tool_calls" in msg
        and msg["tool_calls"] is not None
        and len(msg["tool_calls"]) > 0
    ):
        messages.append(msg)
        #print(msg)
        for toolcall in msg["tool_calls"]:
            if toolcall['function']["name"] in use_cuda_tools:
                model=model.to("cpu")
                #clear_cuda_variables()
                torch.cuda.empty_cache()
                # 手动触发垃圾回收
                gc.collect()
            tool_response = fake_tool_execute(toolcall)
            tool_msg = {
                "role": "tool",
                "content": tool_response,
                "tool_call_id": toolcall["id"],
            }
            print('<tool_response>',tool_response)
            messages.append(tool_msg)
            #print(tool_msg)
    else:
        messages.append(msg)
        #print(msg)
        break

