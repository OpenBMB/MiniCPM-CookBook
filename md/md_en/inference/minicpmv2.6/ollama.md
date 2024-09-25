# Ollama Inference

## System Requirements

- Non-quantized version requires more than 19GB of memory
- Quantized version requires more than 8GB of memory

## Official Ollama Support

1. The official repository has merged our branch, so you can directly use the new version of Ollama:

```bash
ollama run minicpm-v
# Output log
pulling manifest 
pulling 262843d4806a... 100% ▕████████████████▏ 4.4 GB                         
pulling f8a805e9e620... 100% ▕████████████████▏ 1.0 GB                         
pulling 60ed67c565f8... 100% ▕████████████████▏  506 B                         
pulling 43070e2d4e53... 100% ▕████████████████▏  11 KB                         
pulling f02dd72bb242... 100% ▕████████████████▏   59 B                         
pulling 175e3bb367ab... 100% ▕████████████████▏  566 B                         
verifying sha256 digest 
writing manifest
```

### Command Line Method

Separate the input question and image path with a space:

```bash
What is described in this picture? /Users/liudan/Desktop/WechatIMG70.jpg

# Output
This picture shows a young adult male standing in front of a white background. He has short hair, wears metal-rimmed glasses, and is wearing a light blue shirt. His expression is neutral, with his lips closed and looking straight at the camera. The lighting in the photo is bright and even, indicating that it is a professionally taken photograph. The man is not visibly wearing any tattoos, jewelry, or other accessories that might influence perceptions of his profession or identity.
```

### API Method

```python
import base64
import requests

with open(image_path, 'rb') as image_file:
    # Convert the image file to base64 encoding
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

data = {
    "model": "minicpm-v",
    "prompt": query,
    "stream": False,
    "images": [encoded_string]  # List can contain multiple images, each converted to base64 format as shown above
}

# Set the request URL
url = "http://localhost:11434/api/generate"
response = requests.post(url, json=data)

return response
```

## Step 1: Obtain the gguf Model

Note:If the official tutorial works, you can skip this section. 

Otherwise, follow the Llama.cpp tutorial to obtain the gguf model. It is recommended to use a quantized language model.

## Step 2: Install Dependencies

Use Homebrew to install dependencies:

```sh
brew install ffmpeg
brew install pkg-config
```

## Step 3: Get the Official OpenBMB Ollama Branch

```sh
git clone -b minicpm-v2.6 https://github.com/OpenBMB/ollama.git
cd ollama/llm
git clone -b minicpmv-main https://github.com/OpenBMB/llama.cpp.git
cd ../
```

## Step 4: Environment Requirements

- cmake version 3.24 or above
- go version 1.22 or above
- gcc version 11.4.0 or above

Install the required tools using Homebrew:

```sh
brew install go cmake gcc
```

## Step 5: Install Large Model Dependencies

```sh
go generate ./...
```

## Step 6: Compile Ollama

```sh
go build .
```

## Step 7: Start the Ollama Service

After successful compilation, start Ollama from the main Ollama directory:

```sh
./ollama serve
```

## Step 8: Create a ModelFile

Edit the ModelFile:

```sh
vim minicpmv2_6.Modelfile
```

The content of the ModelFile should be:

```plaintext
FROM ./MiniCPM-V-2_6/model/ggml-model-Q4_K_M.gguf
FROM ./MiniCPM-V-2_6/mmproj-model-f16.gguf

TEMPLATE """{{ if .System }}
system

{{ .System }}{{ end }}

{{ if .Prompt }}
user

{{ .Prompt }}{{ end }}

assistant

{{ .Response }}"""
"""

PARAMETER stop ""
PARAMETER stop "assistant"
PARAMETER num_ctx 2048
```

Parameter Explanation:

| Parameter         | Description                                   |
|-------------------|-----------------------------------------------|
| first from        | Path to your language gguf model              |
| second from       | Path to your vision gguf model                |
| num_keep          | Maximum connection limit                      |
| num_ctx           | Maximum model length                          |

## Step 9: Create an Ollama Model Instance

```bash
ollama create minicpm2.6 -f minicpmv2_6.Modelfile
```

## Step 10: Run the Ollama Model Instance

In another command line window, run the Ollama model instance:

```bash
ollama run minicpm2.6
```

## Step 11: Input Question and Image URL

Separate the input question and image URL with a space:

```bash
What is described in this picture? /Users/liudan/Desktop/11.jpg
```
