# VLLM API Server

## Step 1: Download and Install VLLM Using Git

Clone the VLLM repository and install dependencies:

```sh
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

## Step 2: Start the VLLM Service via Command Line

When starting the VLLM service, you can specify parameters such as computation precision, maximum model length, API key, and GPU utilization:

```sh
vllm serve /root/ld/ld_model_pretrained/Minicpmv2_6 --dtype auto --max-model-len 2048 --api-key token-abc123 --gpu_memory_utilization 1 --trust-remote-code
```

For more parameters, refer to [vllm_argument](https://docs.vllm.ai/en/latest/models/engine_args.html).

## Step 3: Call the VLLM HTTP Service Using Python Code

### 1. Pass a Web Image

```python
from openai import OpenAI

# Set the API key and base URL
openai_api_key = "token-abc123"  # API key set when starting the service
openai_api_base = "http://localhost:8000/v1"  # HTTP interface address

# Create a client instance
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# Create a chat completion request
chat_response = client.chat.completions.create(
    model="/root/ld/ld_model_pretrained/Minicpmv2_6",  # Local model path or Hugging Face ID
    messages=[
        {
            "role": "user",
            "content": [
                # NOTE: The prompt format using the image token <image> is unnecessary because the prompt will be automatically handled by the API server.
                # Since the prompt will be automatically handled by the API server, there is no need to use a prompt format containing the <image> image token.
                {"type": "text", "text": "Please describe this image"},
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

### 2. Pass a Local Image

```python
from openai import OpenAI
import base64

# Set the API key and base URL
openai_api_key = "token-abc123"  # API key set when starting the service
openai_api_base = "http://localhost:8000/v1"  # HTTP interface address

# Create a client instance
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# To pass a local image
with open('your/local/pic/path', 'rb') as file:
    image = "data:image/jpeg;base64," + base64.b64encode(file.read()).decode('utf-8')

# Create a chat completion request
chat_response = client.chat.completions.create(
    model="/root/ld/ld_model_pretrained/Minicpmv2_6",  # Local model path or Hugging Face ID
    messages=[
        {
            "role": "user",
            "content": [
                # NOTE: The prompt format using the image token <image> is unnecessary because the prompt will be automatically handled by the API server.
                # Since the prompt will be automatically handled by the API server, there is no need to use a prompt format containing the <image> image token.
                {"type": "text", "text": "Please describe this image"},
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
