# VLLM API Server

## 步骤1：使用Git下载并安装VLLM

通过Git克隆VLLM仓库并安装依赖：

```sh
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install e .
```

## 步骤2：命令行启动VLLM服务

启动VLLM服务时，可以指定计算精度、模型处理最大长度、API密钥和GPU使用率等参数：

```sh
vllm serve /root/ld/ld_model_pretrained/Minicpmv2_6 --dtype auto --max-model-len 2048 --api-key token-abc123 --gpu_memory_utilization 1 --trust-remote-code
```

更多参数请访问[vllm_argument](https://docs.vllm.ai/en/latest/models/engine_args.html)。

## 步骤3：使用Python代码调用VLLM的HTTP服务

### 1. 传入网络图片

```python
from openai import OpenAI

# 设置API密钥和基础URL
openai_api_key = "token-abc123"  # 在启动服务时设置的API密钥
openai_api_base = "http://localhost:8000/v1"  # HTTP接口地址

# 创建客户端实例
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# 创建聊天完成请求
chat_response = client.chat.completions.create(
    model="/root/ld/ld_model_pretrained/Minicpmv2_6",  # 模型本地路径或Hugging Face ID
    messages=[
        {
            "role": "user",
            "content": [
                # NOTE: 使用图像令牌 <image> 的提示格式是不必要的，因为提示将由API服务器自动处理。
                # 由于提示将由API服务器自动处理，因此不需要使用包含 <image> 图像令牌的提示格式。
                {"type": "text", "text": "请描述这张图片"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://air-example-data-2.s3.us-west-2.amazonaws.com/vllm_opensource_llava/stop_sign.jpg",
                    },
                },
            ],
        }
    ],
    extra_body={
        "stop_token_ids": [151645, 151643]
    }
)

print("Chat response:", chat_response)
print("Chat response content:", chat_response.choices[0].message.content)
```

### 2. 传入本地图片

```python
from openai import OpenAI
import base64

# 设置API密钥和基础URL
openai_api_key = "token-abc123"  # 在启动服务时设置的API密钥
openai_api_base = "http://localhost:8000/v1"  # HTTP接口地址

# 创建客户端实例
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# 用于传本地图片
with open('your/local/pic/path', 'rb') as file:
    image = "data:image/jpeg;base64," + base64.b64encode(file.read()).decode('utf-8')

# 创建聊天完成请求
chat_response = client.chat.completions.create(
    model="/root/ld/ld_model_pretrained/Minicpmv2_6",  # 模型本地路径或Hugging Face ID
    messages=[
        {
            "role": "user",
            "content": [
                # NOTE: 使用图像令牌 <image> 的提示格式是不必要的，因为提示将由API服务器自动处理。
                # 由于提示将由API服务器自动处理，因此不需要使用包含 <image> 图像令牌的提示格式。
                {"type": "text", "text": "请描述这张图片"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image,
                    },
                },
            ],
        }
    ],
    extra_body={
        "stop_token_ids": [151645, 151643]
    }
)

print("Chat response:", chat_response)
print("Chat response content:", chat_response.choices[0].message.content)
```
