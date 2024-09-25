# VLLM Inference

## Author's pip list (supports awq, fp16, and vllm)

```plaintext
vllm 0.5.4
transformers 4.44.0
torchvision 0.19.0
torch 2.4.0
triton 3.0.0
trl 0.9.6
autoawq_kernels 0.0.6
```

## Deploying Int4 Models with VLLM

Refer to the [MiniCPM-V 2.6 Quantization Tutorial](../../../md_en/quantize/minicpmv2.6/awq.md) for a speed improvement of up to twice and over 50% reduction in VRAM usage.

### Method 1: Using Python to Call VLLM for Inference

#### 1. Download Model Weights

Download the model weights from Hugging Face:

```sh
git clone https://huggingface.co/openbmb/MiniCPM-V-2_6
```

Alternatively, download the quantized AWQ model for faster speed and reduced VRAM usage (only 7GB required):

```sh
git clone https://www.modelscope.cn/models/linglingdan/MiniCPM-V_2_6_awq_int4
```

Install the AutoAWQ branch:

```sh
git clone https://github.com/LDLINGLINGLING/AutoAWQ.git
cd AutoAWQ
git checkout minicpmv2.6
pip install -e .
```

#### FP16 vs. AWQ Comparison (Single 4090 Card)

|         | Awq int4 | fp16 |
|---------|----------|------|
| Speed   |          |      |
| Input   | 172.64 toks/s | 113.63 toks/s |
| Output  | 28.21 toks/s | 32.49 toks/s |
| Time use| 00:07 | 00:12 |
| Output  | This image shows a commercial airliner flying in a clear blue sky... | This image shows a commercial airliner flying in a clear blue sky... |
| Memory use of model | 7Gb | 16Gb |
| Max length of 24g memory | 2048*3 | 2048 |
| Max batch size of 24g memory | 52 | 2 |

#### 2. Install VLLM

The latest version of VLLM now supports our models:

```sh
pip install vllm==0.5.4
```

#### 3. Create Python Code to Call VLLM

##### Single Image Inference

```python
from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# List of image file paths
IMAGES = [
    "/root/ld/ld_project/MiniCPM-V/assets/airplane.jpeg",  # Local image path
]

# Model name or path
MODEL_NAME = "/root/ld/ld_model_pretrained/Minicpmv2_6"  # Local model path or Hugging Face model name

# Open and convert the image
image = Image.open(IMAGES[0]).convert("RGB")

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Initialize the language model
llm = LLM(
    model=MODEL_NAME,
    gpu_memory_utilization=1,  # Use full GPU memory
    trust_remote_code=True,
    max_model_len=2048  # Adjust this value based on memory conditions
)

# Build the conversation message
messages = [{'role': 'user', 'content': '(<image>./</image>)\n' + 'Please describe this image'}]

# Apply the conversation template to the message
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Set the stop token IDs
# 2.0
# stop_token_ids = [tokenizer.eos_id]
# 2.5
# stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]
# 2.6 
stop_tokens = ['<|im_end|>', '<|endoftext|>']
stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

# 设置生成参数
sampling_params = SamplingParams(
    stop_token_ids=stop_token_ids,
    # temperature=0.7,
    # top_p=0.8,
    # top_k=100,
    # seed=3472,
    max_tokens=1024,
    # min_tokens=150,
    temperature=0,
    use_beam_search=True,
    # length_penalty=1.2,
    best_of=3)

# 获取模型输出
outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": {
        "image": image
    }
}, sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
```
##### Multi-Image Inference

Note: For multi-image and video inference, you currently need to compile and install our custom VLLM branch:

```sh
git clone https://github.com/OpenBMB/vllm
cd vllm
git checkout minicpmv
pip install -e .
```

Inference script:

```python
from transformers import AutoTokenizer
from PIL import Image
from vllm import LLM, SamplingParams

MODEL_NAME = "openbmb/MiniCPM-V-2_6"

image = Image.open("xxx.png").convert("RGB")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
llm = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
    gpu_memory_utilization=1,
    max_model_len=2048
)

messages = [{
    "role": "user",
    "content": "(<image>./</image>)\nWhat is the content of this image?"
}]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Single Inference
inputs = {
    "prompt": prompt,
    "multi_modal_data": {
        "image": image
        # For multiple images, the number of images should match the number of `(<image>./</image>)` tokens
        # "image": [image, image] 
    },
}

# 2.6
stop_tokens = ['<|im_end|>', '<|endoftext|>']
stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

sampling_params = SamplingParams(
    stop_token_ids=stop_token_ids, 
    use_beam_search=True,
    temperature=0, 
    best_of=3,
    max_tokens=64
)

outputs = llm.generate(inputs, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)
```

##### Video Inference

Note: For multi-image and video inference, you currently need to compile and install our custom VLLM branch:

```sh
git clone https://github.com/OpenBMB/vllm
cd vllm
git checkout minicpmv
pip install -e .
```

Inference script:

```python
from transformers import AutoTokenizer
from decord import VideoReader, cpu
from PIL import Image
from vllm import LLM, SamplingParams

# Function to encode video into a list of frames
MAX_NUM_FRAMES = 16
def encode_video(filepath):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]
    
    vr = VideoReader(filepath, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    
    video = vr.get_batch(frame_idx).asnumpy()
    video = [Image.fromarray(v.astype('uint8')) for v in video]
    return video

MODEL_NAME = "openbmb/MiniCPM-V-2_6"  # or local model path
llm = LLM(
    model=MODEL_NAME,
    gpu_memory_utilization=0.95,
    max_model_len=4096
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
stop_tokens = ['<|im_end|>', '<|endoftext|>']
stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

frames = encode_video("xxx.mp4")
messages = [{
    "role": "user",
    "content": "".join(["(<image>./</image>)"] * len(frames)) + "\nPlease describe this video."
}]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

sampling_params = SamplingParams(
    stop_token_ids=stop_token_ids, 
    use_beam_search=False,
    temperature=0.7,
    top_p=0.8,
    top_k=100, 
    max_tokens=512
)

outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": {
        "image": {
            "images": frames,
            "use_image_id": False,
            "max_slice_nums": 1 if len(frames) > 16 else 2
        }
    }
}, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)
```
