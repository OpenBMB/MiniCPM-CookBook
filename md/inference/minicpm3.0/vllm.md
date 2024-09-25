
# VLLM 安装与使用指南

## 安装 VLLM

首先通过 Git 克隆 VLLM 仓库:

```bash
git clone https://github.com/LDLINGLINGLING/vllm.git
cd vllm
pip install -e .
```

注意这里的 `-e` 参数表示以开发模式安装，这样您可以直接对代码进行修改而不需要重新安装。

## Python 运行示例

下面是一个使用 VLLM 生成文本的 Python 示例脚本:

```python
from vllm import LLM, SamplingParams
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="/root/ld/ld_model_pretrained/minicpm3")
parser.add_argument("--prompt_path", type=str, default="")
parser.add_argument("--batch", type=int, default=2)  # 可以修改这个 batch 达到并发
args = parser.parse_args()

# 提示列表
prompts = ["你好啊", "吃饭了没有", "你好，今天天气怎么样？", "孙悟空是谁？"]
prompt_template = "<|im_start|> user\n{} <|im_end|>"

# 格式化提示
prompts = [prompt_template.format(prompt.strip()) for prompt in prompts]

# 采样参数字典
params_dict = {
    "n": 1,
    "best_of": 1,
    "presence_penalty": 1.0,
    "frequency_penalty": 0.0,
    "temperature": 0.5,
    "top_p": 0.8,
    "top_k": -1,
    "use_beam_search": False,
    "length_penalty": 1,
    "early_stopping": False,
    "stop": None,
    "stop_token_ids": None,
    "ignore_eos": False,
    "max_tokens": 1000,
    "logprobs": None,
    "prompt_logprobs": None,
    "skip_special_tokens": True,
}

# 创建一个采样参数对象
sampling_params = SamplingParams(**params_dict)

# 创建一个 LLM 模型实例
llm = LLM(model=args.model_path, tensor_parallel_size=1, dtype='bfloat16',
          trust_remote_code=True, max_model_len=2048, gpu_memory_utilization=0.5)

# 从提示生成文本
batch_input = []
for prompt in prompts:
    batch_input.append(prompt)
    if len(batch_input) == args.batch:
        outputs = llm.generate(batch_input, sampling_params)
        # 打印输出结果
        for output in outputs:
            prompt = output.prompt
            print("用户：{}".format(prompt))
            generated_text = output.outputs[0].text
            print("AI助手：{}".format(generated_text))
        batch_input = []

```

## 启动 VLLM Server 并使用 OpenAPI 接口

### 启动 VLLM Server

启动 VLLM server 并指定 API 密钥:

```bash
vllm serve /root/ld/ld_model_pretrained/minicpm3 --dtype auto --api-key token-abc123 --trust-remote-code --max_model_len 2048 --gpu_memory_utilization 0.7
```

### 使用 Python 调用 API 接口

使用 `openai` 库来调用 VLLM server 提供的 API 接口:

```python
from openai import OpenAI

# 创建一个客户端实例
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

# 创建一个聊天完成请求
completion = client.chat.completions.create(
  model="/root/ld/ld_model_pretrained/minicpm3",
  messages=[
    {"role": "user", "content": "hello, nice to meet you."}
  ]
)

# 输出结果
print(completion.choices[0].message)
```
