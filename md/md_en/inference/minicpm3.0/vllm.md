# VLLM Installation and Usage Guide

## Installing VLLM

First, clone the VLLM repository via Git:

```bash
git clone https://github.com/LDLINGLINGLING/vllm.git
cd vllm
pip install -e .
```

Note that the `-e` parameter signifies an editable installation, which allows you to make changes directly to the codebase without needing to reinstall the package.

## Python Execution Example

Below is a Python script example that demonstrates how to generate text using VLLM:

```python
from vllm import LLM, SamplingParams
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="/root/ld/ld_model_pretrained/minicpm3")
parser.add_argument("--prompt_path", type=str, default="")
parser.add_argument("--batch", type=int, default=2)  # You can adjust this batch size for concurrency
args = parser.parse_args()

# List of prompts
prompts = ["你好啊", "吃饭了没有", "你好，今天天气怎么样？", "孙悟空是谁？"]
prompt_template = "<|im_start|> user\n{} <|im_end|>"

# Format prompts
prompts = [prompt_template.format(prompt.strip()) for prompt in prompts]

# Sampling parameters dictionary
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

# Create a sampling parameters object
sampling_params = SamplingParams(**params_dict)

# Create an LLM model instance
llm = LLM(model=args.model_path, tensor_parallel_size=1, dtype='bfloat16',
          trust_remote_code=True, max_model_len=2048, gpu_memory_utilization=0.5)

# Generate text from prompts
batch_input = []
for prompt in prompts:
    batch_input.append(prompt)
    if len(batch_input) == args.batch:
        outputs = llm.generate(batch_input, sampling_params)
        # Print output results
        for output in outputs:
            prompt = output.prompt
            print("User: {}".format(prompt))
            generated_text = output.outputs[0].text
            print("AI Assistant: {}".format(generated_text))
        batch_input = []
```

## Launching the VLLM Server and Using the OpenAPI Interface

### Launching the VLLM Server

To start the VLLM server and specify an API key:

```bash
vllm serve /root/ld/ld_model_pretrained/minicpm3 --dtype auto --api-key token-abc123 --trust-remote-code --max_model_len 2048 --gpu_memory_utilization 0.7
```

### Using Python to Call the API Interface

Use the `openai` library to call the API interface provided by the VLLM server:

```python
from openai import OpenAI

# Create a client instance
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

# Create a chat completion request
completion = client.chat.completions.create(
  model="/root/ld/ld_model_pretrained/minicpm3",
  messages=[
    {"role": "user", "content": "hello, nice to meet you."}
  ]
)

# Print the result
print(completion.choices[0].message)
```