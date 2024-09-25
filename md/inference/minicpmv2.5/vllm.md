
# vLLM 部署（并发推荐）

## 步骤1：获取vLLM代码

通过Git克隆vLLM仓库：

```sh
git clone https://github.com/vllm-project/vllm.git
```

## 步骤2：编译安装vLLM

进入vLLM目录并安装：

```sh
cd vllm
pip install -e .
```

## 步骤3：安装依赖库timm

安装`timm`库：

```sh
pip install timm==0.9.10
```

## 步骤4：复制以下文件进行推理

```python
from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

IMAGES = [
    "./examples/images/375.jpg",
]

MODEL_NAME = "openbmb/MiniCPM-Llama3-V-2_5"  # 如果使用本地模型，请将模型代码更新到最新

image = Image.open(IMAGES[0]).convert("RGB")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
llm = LLM(model=MODEL_NAME,
          gpu_memory_utilization=1,
          trust_remote_code=True,
          max_model_len=4096)

messages = [{'role': 'user', 'content': '(<image>./</image>)\n' + 'what kind of wine is this?'}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 2.5
stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]

sampling_params = SamplingParams(
    stop_token_ids=stop_token_ids,
    max_tokens=1024,
    temperature=0,
    use_beam_search=True,
    best_of=3)

outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": {
        "image": image
    }
}, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)
```

## 步骤5：实现并发推理

编辑并发推理脚本：

```python
from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# 所有待输入的图像
IMAGES = [
    "./examples/images/375.jpg",
    "./examples/images/376.jpg",
    "./examples/images/377.jpg",
    "./examples/images/378.jpg"
]

MODEL_NAME = "openbmb/MiniCPM-Llama3-V-2_5"  # 如果使用本地模型，请将模型代码更新到最新

images = [Image.open(i).convert("RGB") for i in IMAGES]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
llm = LLM(model=MODEL_NAME,
          gpu_memory_utilization=1,
          trust_remote_code=True,
          max_model_len=4096)

messages = [{'role': 'user', 'content': '(<image>./</image>)\n' + '帮我识别图中的内容?'}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 构建多个输入，本示例共用prompt，也可以不共用prompt
inputs = [{"prompt": prompt, "multi_modal_data": {"image": i}} for i in images]

# 2.5
stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]

sampling_params = SamplingParams(
    stop_token_ids=stop_token_ids,
    max_tokens=1024,
    temperature=0,
    use_beam_search=True,
    best_of=3)

outputs = llm.generate(inputs, sampling_params=sampling_params)

for i in range(len(inputs)):
    print(outputs[i].outputs[0].text)
```

现在您可以开始使用vLLM进行高效的并发推理了！
```