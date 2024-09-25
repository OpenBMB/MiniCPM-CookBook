# Swift 脚本推理

## 设备要求
- 多卡显存总共不超过24GB

## 步骤1：安装Swift

按照[swift安装教程](./swift_commandline.md)安装Swift。

## 步骤2：执行脚本

根据注释，修改以下脚本参数，并执行。

```python
import os
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything
import torch

# 设置可见的GPU设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

model_type = ModelType.minicpm_v_v2_5_chat  #获取模板类型，主要是用于特殊token的构造和图像的处理流程
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')

model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16,
                                       model_id_or_path='/root/ld/ld_model_pretrain/MiniCPM-Llama3-V-2_5',# 改成你的模型路径
                                       model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 256
template = get_template(template_type, tokenizer)
seed_everything(42)

images = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png']
query = '距离各城市多远？'
response, history = inference(model, template, query, images=images)
print(f'query: {query}')
print(f'response: {response}')

# 以下是流式输出结果
query = '距离最远的城市是哪？'
gen = inference_stream(model, template, query, history, images=images)
print_idx = 0
print(f'query: {query}\nresponse: ', end='')
for response, history in gen:
    delta = response[print_idx:]
    print(delta, end='', flush=True)
    print_idx = len(response)
print()
print(f'history: {history}')
```