```python
ffrom PIL import Image
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model, dispatch_model

MODEL_PATH = '/root/ld/ld_model_pretrain/MiniCPM-Llama3-V-2_5' # 写入模型地址或者huggingface id
max_memory_each_gpu = '5GiB' # 每张卡准备用多少内存
gpu_device_ids = [0,1,2,3,4,5,6,7] # gpu的index 列表，这个示例代表单机八卡
no_split_module_classes = ["LlamaDecoderLayer"] # 哪些模块是不切分到不同的卡里
max_memory = {
    device_id: max_memory_each_gpu for device_id in gpu_device_ids
} # 每张卡的gpu最大占用量

config = AutoConfig.from_pretrained(
    MODEL_PATH, 
    trust_remote_code=True
) # 加载模型config文件

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, 
    trust_remote_code=True
) # 加载分词文件

with init_empty_weights():
    model = AutoModel.from_config(
        config, 
        torch_dtype=torch.float16, 
        trust_remote_code=True
    ) # 加载模型

device_map = infer_auto_device_map(
    model,
    max_memory=max_memory,
    no_split_module_classes=no_split_module_classes
) # 初始化每个模型群众分配到哪个gpu的字典

 # 由于模型结构特殊，modeling代码需要以下resampler，embed_tokens，lm_head在一个卡上
device_map['resampler']=device_map['llm.model.embed_tokens']
device_map['llm.lm_head']=device_map['llm.model.embed_tokens']
device_map["llm.model.layers.0"]=device_map['llm.model.embed_tokens']

print("auto determined device_map", device_map)

# Here we want to make sure the input and output layer are all on the first gpu to avoid any modifications to original inference script.device_map["llm.model.embed_tokens"] = 0device_map["llm.model.layers.0"] = 0device_map["llm.lm_head"] = 0device_map["vpm"] = 0device_map["resampler"] = 0print("modified device_map", device_map)

load_checkpoint_in_model(
    model, 
    MODEL_PATH, 
    device_map=device_map)

model = dispatch_model(
    model, 
    device_map=device_map
)

torch.set_grad_enabled(False) #不计算梯度

model.eval()

image_path = '/root/ld/ld_project/MiniCPM-V/assets/airplane.jpeg' #图片地址
response = model.chat(
    image=Image.open(image_path).convert("RGB"),
    msgs=[
        {
            "role": "user",
            "content": "guess what I am doing?"
        }
    ],
    tokenizer=tokenizer
) # 模型推理
print(response)
```