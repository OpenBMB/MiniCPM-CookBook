# Sglang Installation and Usage Guide

## Source Code Installation of Sglang

1. First, clone the Sglang project repository from GitHub:

   ```bash
   git clone https://github.com/sgl-project/sglang.git
   cd sglang
   ```

2. Upgrade `pip` and install Sglang along with all its dependencies:

   ```bash
   pip install --upgrade pip
   pip install -e "python[all]"
   ```

## Installing FlashInfer Dependencies

### Method 1: Install Using pip (May Fail Due to Network Speed)

```bash
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
```

### Method 2: Install Using a .whl File

1. Visit the webpage: https://flashinfer.ai/whl/cu121/torch2.4/flashinfer/
2. Download the `.whl` file suitable for your server configuration.
3. Install the `.whl` file using `pip`. For example:

   ```bash
   pip install your/path/flashinfer-0.1.6+cu121torch2.4-cp310-cp310-linux_x86_64.whl
   ```

## Launching the Sglang Inference Service

Before starting the service, ensure the model path is correct. If you encounter out-of-memory issues, try adding the `--disable-cuda-graph` parameter.

```bash
python -m sglang.launch_server --model openbmb/MiniCPM3-4B --trust-remote-code --port 30000 --chat-template chatml
```

## Calling the Service API

### Bash Call Example

```bash
curl http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Once upon a time,",
    "sampling_params": {
      "max_new_tokens": 16,
      "temperature": 0
    }
  }'
```

### Python Call Example

```python
import openai
client = openai.Client(
    base_url="http://127.0.0.1:30000/v1", api_key="EMPTY"
)

# Text Completion
response = client.completions.create(
    model="default",
    prompt="The capital of France is",
    temperature=0,
    max_tokens=32,
)
print(response)

# Chat Completion
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": "What is your name?"},
    ],
    temperature=0,
    max_tokens=64,
)
print(response.choices[0].message.content)
```
