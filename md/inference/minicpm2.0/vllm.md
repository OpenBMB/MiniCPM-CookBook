
### 安装vLLM

首先确保安装了`vllm`库：

```bash
pip install vllm
```

### Python 脚本示例

接下来，在Python脚本中使用`vllm`进行文本生成：

```python
from vllm import LLM, SamplingParams
import argparse

# 创建命令行参数解析器
parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument("--model_path", type=str, default="")

# 解析命令行参数
args = parser.parse_args()


prompts = ["你吃饭了没？"，"世界上最高的山是什么山"]  # prompts是一个列表，其中每一个元素都是一个要输入的prompt文本

# 格式化提示模板
prompt_template = "<用户>{}<AI>"

# 应用模板到每个提示
prompts = [prompt_template.format(prompt.strip()) for prompt in prompts]

# 设置采样参数
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

# 创建一个LLM对象
llm = LLM(model=args.model_path, tensor_parallel_size=1, dtype='bfloat16')

# 从提示生成文本。输出是一个包含RequestOutput对象的列表，
# 其中包含提示、生成的文本和其他信息。
for prompt in prompts:
    outputs = llm.generate(prompt, sampling_params)
    # 打印输出
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print("================")
        # 找到第一个<用户>并移除之前的文本。
        clean_prompt = prompt[prompt.find("<用户>") + len("<用户>"):]
        print(f"""<用户>: {clean_prompt.replace("<AI>", "")}""")
        print(f"<AI>:")
        print(generated_text)
```

请确保在运行此脚本之前正确设置了`--model_path`参数。
