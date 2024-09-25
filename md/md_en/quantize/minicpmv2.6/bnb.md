# bitsandbytes Quantization Script
Modify the following `model_path`, `save_path`, and ensure you have a GPU capable of loading the unquantized model, approximately 17GB of VRAM is required.

```python
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from PIL import Image
import time
import torch
import GPUtil
import os

model_path = '/root/ld/ld_model_pretrain/minicpm-v-2_6' # Model download path
device = 'cuda'
save_path = '/root/ld/ld_model_pretrain/minicpm-v-2_6_bnb_int4' # Path to save the quantized model
image_path = '/root/ld/ld_project/MiniCPM-V/assets/airplane.jpeg'

# Create a configuration object to specify quantization parameters
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, # Whether to perform 4-bit quantization
    load_in_8bit=False, # Whether to perform 8-bit quantization
    bnb_4bit_compute_dtype=torch.float16, # Computation precision setting
    bnb_4bit_quant_storage=torch.uint8, # Storage format for quantized weights
    bnb_4bit_quant_type="nf4", # Quantization format, using normal distribution int4 here
    bnb_4bit_use_double_quant=True, # Whether to use double quantization, i.e., quantizing zeropoint and scaling parameters
    llm_int8_enable_fp32_cpu_offload=False, # Whether to use int8 for LLM, and fp32 for parameters stored on CPU
    llm_int8_has_fp32_weight=False, # Whether to enable mixed precision
    llm_int8_skip_modules=["out_proj", "kv_proj", "lm_head"], # Modules to skip quantization
    llm_int8_threshold=6.0 # Outliers in the LLM.int8() algorithm, distinguishing whether to perform quantization based on this value
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_path,
    device_map="cuda:0",  # Allocate the model to GPU0
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
            "content": "What is in this picture?"
        }
    ],
    tokenizer=tokenizer
) # Model inference
print('Output after quantization:', response)
print('Time taken after quantization:', time.time()-start)
print(f"VRAM usage after quantization: {round(gpu_usage/1024,2)}GB")

# Save the model and tokenizer
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path, safe_serialization=True)
tokenizer.save_pretrained(save_path)
```
