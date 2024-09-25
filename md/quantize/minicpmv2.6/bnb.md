# bitsandbytes量化脚本
修改以下model_path和save_path，以及需要能够加载未量化模型的显卡，大约17G显存
```python
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from PIL import Image
import time
import torch
import GPUtil
import os

model_path = '/root/ld/ld_model_pretrain/minicpm-v-2_6' # 模型下载地址
device = 'cuda'
save_path = '/root/ld/ld_model_pretrain/minicpm-v-2_6_bnb_int4' # 量化模型保存地址
image_path = '/root/ld/ld_project/MiniCPM-V/assets/airplane.jpeg'
# 创建一个配置对象来指定量化参数
quantization_config = BitsAndBytesConfig(
    load_in_4bit= True, # 是否进行4bit量化
    load_in_8bit=False, # 是否进行8bit量化
    bnb_4bit_compute_dtype=torch.float16, # 计算精度设置
    bnb_4bit_quant_storage=torch.uint8, # 量化权重的储存格式
    bnb_4bit_quant_type="nf4", # 量化格式，这里用的是正太分布的int4
    bnb_4bit_use_double_quant= True, # 是否采用双量化，即对zeropoint和scaling参数进行量化
    llm_int8_enable_fp32_cpu_offload=False, # 是否llm使用int8，cpu上保存的参数使用fp32
    llm_int8_has_fp16_weight=False, # 是否启用混合精度
    llm_int8_skip_modules=[ "out_proj", "kv_proj", "lm_head" ], # 不进行量化的模块
    llm_int8_threshold= 6.0 # llm.int8()算法中的离群值，根据这个值区分是否进行量化
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_path,
    device_map="cuda:0",  # 分配模型到GPU0
    quantization_config=quantization_config,
    trust_remote_code=True
)
gpu_usage = GPUtil.getGPUs()[0].memoryUsed 
        
start=time.time()
response = model.chat(
    image=Image.open(image_path).convert("RGB"),
    msgs=[
        {
            "role": "user",
            "content": "这张图片中有什么?"
        }
    ],
    tokenizer=tokenizer
) # 模型推理
print('量化后输出',response)
print('量化后用时',time.time()-start)
print(f"量化后显存占用: {round(gpu_usage/1024,2)}GB")

# 保存模型和分词器
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path, safe_serialization=True)
tokenizer.save_pretrained(save_path)
```
