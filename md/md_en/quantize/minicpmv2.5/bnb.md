# BNB Quantization
## Quantization Script
```python
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from PIL import Image
import time
import GPUtil
import os

model_path = '/root/ld/ld_model_pretrain/MiniCPM-Llama3-V-2_5' # Model download path
device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_path = '/root/ld/ld_model_pretrain/MiniCPM-Llama3-V-2_5_int4' # Path to save the quantized model
image_path = '/root/ld/ld_project/MiniCPM-V/assets/airplane.jpeg'

# Create a configuration object to specify quantization parameters
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, # Whether to perform 4-bit quantization
    load_in_8bit=False, # Whether to perform 8-bit quantization
    bnb_4bit_compute_dtype=torch.float16, # Setting the computation precision
    bnb_4bit_quant_storage=torch.uint8, # Storage format for quantized weights
    bnb_4bit_quant_type="nf4", # Quantization type, here using normal distribution int4
    bnb_4bit_use_double_quant=True, # Whether to use double quantization, i.e., quantizing zeropoint and scaling parameters
    llm_int8_enable_fp32_cpu_offload=False, # Whether LLM uses int8, and CPU-stored parameters use fp32
    llm_int8_has_fp16_weight=False, # Whether to enable mixed precision
    llm_int8_skip_modules=["out_proj", "kv_proj", "lm_head"], # Modules not to be quantized
    llm_int8_threshold=6.0 # Outliers in the LLM.int8() algorithm, used to distinguish whether to quantize based on this value
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
            "content": "What's in this picture?"
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

## Simple Comparison Before and After Quantization
Below is a Markdown-formatted table showing the comparison of model performance before and after quantization:
![Input Image](../../../../asset/airplane.jpeg)
| Item | Before Quantization | After Quantization |
| --- | --- | --- |
| Prompt | What's in this picture? | What's in this picture? |
| Output | This picture shows a large commercial airliner, specifically an Airbus A380-800 model. This aircraft is a twin-fuselage, four-engine wide-body aircraft, with a unique dual vertical stabilizer at the tail. The aircraft's livery is primarily white, with blue and red markings, colors often associated with Chinese airlines, indicating that this might be part of such an airline's fleet. Against a clear blue sky background, the details and design of the aircraft stand out. | In this picture, there is a logo on the tail of the aircraft, composed of a red pattern that looks like a flower or flame. This logo is one of the most prominent features of the aircraft, typically representing the brand or identity of the airline. It is located behind the vertical stabilizer, which is visible at the rear of the aircraft. Such a distinctive design helps identify the aircraft in flight and may hold significant meaning for those familiar with the airline. |
| Time for 10 runs | 34.89081048965454 seconds | 51.239391803741455 seconds |
| Time for 10 runs with mixed precision | 36.42154074 seconds | 65.37664222717285 seconds |
| Total tokens generated over 10 runs | 1279 | 1261 |
| VRAM Usage | 16.49GB | 6.45GB |
| Preliminary Conclusion | There is a significant reduction in VRAM usage for the model after quantization. The BNB library uses the llm.int8() quantization algorithm, which is a mixed precision algorithm, leading to lower hardware efficiency. The number of tokens generated remains largely the same before and after quantization. This clearly demonstrates the significant impact of the llm.int8() algorithm on speed.
