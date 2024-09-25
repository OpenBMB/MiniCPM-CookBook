
# VLLM 推理

## 笔者的pip list（awq，fp16，vllm都能跑）

```plaintext
vllm 0.5.4
transformers 4.44.0
torchvision 0.19.0
torch 2.4.0
triton 3.0.0
trl 0.9.6
autoawq_kernels 0.0.6
```

## VLLM部署int4模型

请参考[MiniCPM-V 2.6量化教程](../../../md/quantize/minicpmv2.6/awq.md)，速度提升一倍、显存减少一半以上。

### 方法1: 使用Python调用VLLM推理

#### 1. 下载模型权重

前往Hugging Face下载模型权重：

```sh
git clone https://huggingface.co/openbmb/MiniCPM-V-2_6
```

也可以下载量化后的AWQ模型，速度快一倍，显存只需7GB：

```sh
git clone https://www.modelscope.cn/models/linglingdan/MiniCPM-V_2_6_awq_int4
```

安装AutoAWQ的分支：

```sh
git clone https://github.com/LDLINGLINGLING/AutoAWQ.git
cd AutoAWQ
git checkout minicpmv2.6
pip install e .
```

#### FP16和AWQ使用VLLM简单对比（4090单卡）

|         | Awq int4 | fp16 |
|---------|----------|------|
| Speed   |          |      |
| Input   | 172.64 toks/s | 113.63 toks/s |
| Output  | 28.21 toks/s | 32.49 toks/s |
| Time use| 00:07 | 00:12 |
| Output  | 这幅图片展示了一架商用客机在晴朗的蓝天下飞行... | 这幅图片展示了一架商用客机在晴朗的蓝天下飞行... |
| Memory use of model | 7Gb | 16Gb |
| Max length of 24g memory | 2048*3 | 2048 |
| Max batch size of 24g memory | 52 | 2 |

#### 2. 安装VLLM

最新版VLLM目前已经支持我们的模型：

```sh
pip install vllm==0.5.4
```

#### 3. 创建Python代码调用VLLM

##### 单图推理

```python
from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# 图像文件路径列表
IMAGES = [
    "/root/ld/ld_project/MiniCPM-V/assets/airplane.jpeg",  # 本地图片路径
]

# 模型名称或路径
MODEL_NAME = "/root/ld/ld_model_pretrained/Minicpmv2_6"  # 本地模型路径或Hugging Face模型名称

# 打开并转换图像
image = Image.open(IMAGES[0]).convert("RGB")

# 初始化分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# 初始化语言模型
llm = LLM(model=MODEL_NAME,
           gpu_memory_utilization=1,  # 使用全部GPU内存
           trust_remote_code=True,
           max_model_len=2048)  # 根据内存状况可调整此值

# 构建对话消息
messages = [{'role': 'user', 'content': '(<image>./</image>)\n' + '请描述这张图片'}]

# 应用对话模板到消息
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 设置停止符ID
# 2.0
# stop_token_ids = [tokenizer.eos_id]
# 2.5
#stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]
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

##### 多图推理
注意：多图和视频推理暂时需要用我们自己的vllm分支编译安装：
```sh
git clone https://github.com/OpenBMB/vllm
cd vllm
git checkout minicpmv
pip install e .
```
推理脚本：
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
    "role":
    "user",
    "content":
    # Number of images
    "(<image>./</image>)" + \
    "\nWhat is the content of this image?" 
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
        # Multi images, the number of images should be equal to that of `(<image>./</image>)`
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
##### 视频推理
注意：多图和视频推理暂时需要用我们自己的vllm分支编译安装：
```sh
git clone https://github.com/OpenBMB/vllm
cd vllm
git checkout minicpmv
pip install e .
```
推理脚本：
```python
from transformers import AutoTokenizer
from decord import VideoReader, cpu
from PIL import Image
from vllm import LLM, SamplingParams

# 进行图片推理
MAX_NUM_FRAMES = 16
def encode_video(filepath):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]
    vr = VideoReader(filepath, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx)>MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    video = vr.get_batch(frame_idx).asnumpy()
    video = [Image.fromarray(v.astype('uint8')) for v in video]
    return video

MODEL_NAME = "openbmb/MiniCPM-V-2_6" # or local model path
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
    "role":
    "user",
    "content":
    "".join(["(<image>./</image>)"] * len(frames)) + "\nPlease describe this video."
}]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)



sampling_params = SamplingParams(
    stop_token_ids=stop_token_ids, 
    use_beam_search=False
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
