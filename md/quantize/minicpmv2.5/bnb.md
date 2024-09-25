# bnb量化
## 量化脚本
```python
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from PIL import Image
import time
import torch
import GPUtil
import os

model_path = '/root/ld/ld_model_pretrain/MiniCPM-Llama3-V-2_5' # 模型下载地址
device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_path = '/root/ld/ld_model_pretrain/MiniCPM-Llama3-V-2_5_int4' # 量化模型保存地址
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
## 量化前后简单比较
下面是按照您的要求制作的Markdown格式的表格，展示了量化前后模型性能对比的情况：
[输入图片](../../../asset/airplane.jpeg)
| 项目 | 量化前 | 量化后 |
| --- | --- | --- |
| prompt | 这张图片中有什么? | 这张图片中有什么? |
| 输出 | 这张图片中显示的是一架大型商用客机，具体来说是空客A380-800型号。这架飞机是一架双发、四发动机的宽体飞机，尾部有独特的双垂尾。飞机的涂装主要为白色，带有蓝色和红色的标志，这些颜色通常与中国航空公司相关联，表明这可能是该航空公司的机队之一。清澈的蓝天作为背景，突显了飞机的细节和设计。 | 这张图片中，飞机的尾部有一个标志，由一个红色图案组成，看起来像一朵花或火焰。这个标志是飞机上最显著的特征之一，通常代表航空公司的品牌或身份。它位于飞机的垂直稳定器后面，垂直稳定器在尾部可见。这种标志性设计有助于飞机在空中被识别，并且可能对熟悉该航空公司的人具有重要意义。 |
| 10次用时 | 34.89081048965454 | 51.239391803741455 |
| 开启混合精度10次用时 | 36.42154074 | 65.37664222717285 |
| 10次生成总token | 1279 | 1261 |
| 显存占用 | 16.49GB | 6.45GB |
| 初步结论 | 对于模型占用显存下降巨大，bnb库使用的是llm.int8()量化算法，为混合精度算法，这导致硬件效率低，量化前后生成token数基本相同。这也充分说明llm.int8()算法的对速度带来的影响是巨大的。 

