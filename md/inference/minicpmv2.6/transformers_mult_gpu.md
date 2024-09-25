# transformers_mult_gpu
```python
#!/usr/bin/env python
# encoding: utf-8
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
from decord import VideoReader, cpu


device = 'cuda'
multi_gpus = True

# Load model
model_path = 'openbmb/MiniCPM-V-2_6' # model path

if multi_gpus:
    from accelerate import load_checkpoint_and_dispatch, init_empty_weights, infer_auto_device_map
    with init_empty_weights():
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, attn_implementation='sdpa', torch_dtype=torch.bfloat16)
    device_map = infer_auto_device_map(model, max_memory={0: "10GB", 1: "10GB"},
        no_split_module_classes=['SiglipVisionTransformer', 'Qwen2DecoderLayer'])
    device_id = device_map["llm.model.embed_tokens"]
    device_map["llm.lm_head"] = device_id # firtt and last layer should be in same device
    device_map["vpm"] = device_id
    device_map["resampler"] = device_id
    device_id2 = device_map["llm.model.layers.26"]
    device_map["llm.model.layers.8"] = device_id2
    device_map["llm.model.layers.9"] = device_id2
    device_map["llm.model.layers.10"] = device_id2
    device_map["llm.model.layers.11"] = device_id2
    device_map["llm.model.layers.12"] = device_id2
    device_map["llm.model.layers.13"] = device_id2
    device_map["llm.model.layers.14"] = device_id2
    device_map["llm.model.layers.15"] = device_id2
    device_map["llm.model.layers.16"] = device_id2
    #print(device_map)

    model = load_checkpoint_and_dispatch(model, model_path, dtype=torch.bfloat16, device_map=device_map)
else:
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = model.to(device=device)
    
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.eval()
```