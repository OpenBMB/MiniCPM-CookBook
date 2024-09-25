# Swift Script Inference

## System Requirements
- Total GPU memory across multiple cards does not exceed 24GB

## Step 1: Install Swift

Follow the [Swift installation guide](./swift_commandline.md) to install Swift.

## Step 2: Execute the Script

Modify the following script parameters according to the comments and execute it.

```python
import os
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything
import torch

# Set the visible GPU devices
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

model_type = ModelType.minicpm_v_v2_5_chat  # Get the template type, mainly for constructing special tokens and image processing
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')

model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16,
                                       model_id_or_path='/root/ld/ld_model_pretrain/MiniCPM-Llama3-V-2_5',  # Change to your model path
                                       model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 256
template = get_template(template_type, tokenizer)
seed_everything(42)

images = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png']
query = 'How far is it from each city?'
response, history = inference(model, template, query, images=images)
print(f'query: {query}')
print(f'response: {response}')

# The following is for streaming output
query = 'Which city is the farthest?'
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
