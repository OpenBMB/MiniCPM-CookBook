
# Ollama 推理
## 设备要求
- 运行非量化版本需要19GB以上内存
- 运行量化版本需要8GB以上内存
## ollama官方支持
1. 官方已经合并我们的分支，可以直接用新版ollama
```bash
ollama run minicpm-v
#以下是输出日志
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
### 命令行方式
用空格分割输入问题，图片路径
```bash
这张图片描述了什么？ /Users/liudan/Desktop/WechatIMG70.jpg

#以下是输出
这张图片展示了一个年轻成年男性站在白色背景前。他有着短发，戴着金属边眼镜，并
穿着浅蓝色的衬衫。他的表情中立，嘴唇紧闭，目光直视镜头。照片中的光线明亮均匀
，表明这是一张专业拍摄的照片。这个男人身上没有可见的纹身、珠宝或其他配饰，这
些可能会影响他对职业或身份的感知
```
### 接口方式
```python
with open(image_path, 'rb') as image_file:
        # 将图片文件转换为 base64 编码
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    data = {
    "model": "minicpm-v",
    "prompt": query,
    "stream": False,
    "images": [encoded_string]# 列表可以放多张图，每张图用上面的方式转化为base64的格式
    }

    # 设置请求 URL
    url = "http://localhost:11434/api/generate"
    response = requests.post(url, json=data)

    return response
```
## 步骤1：获取gguf模型
Note:上述官方教程跑通，则无需看以下教程

按照上述Llama.cpp教程获取gguf模型。语言模型最好是量化模型。

## 步骤2：安装依赖包

使用Homebrew安装依赖包：

```sh
brew install ffmpeg
brew install pkg-config
```

## 步骤3：获取OpenBMB官方Ollama分支

```sh
git clone -b minicpm-v2.6 https://github.com/OpenBMB/ollama.git
cd ollama/llm
git clone -b minicpmv-main https://github.com/OpenBMB/llama.cpp.git
cd ../
```

## 步骤4：环境需求

- cmake version 3.24 or above
- go version 1.22 or above
- gcc version 11.4.0 or above

使用Homebrew安装所需工具：

```sh
brew install go cmake gcc
```

## 步骤5：安装大模型依赖项

```sh
go generate ./...
```

## 步骤6：编译Ollama

```sh
go build .
```

## 步骤7：启动Ollama服务

编译成功后，在Ollama主路径启动Ollama：

```sh
./ollama serve
```

## 步骤8：创建一个ModelFile

编辑ModelFile：

```sh
vim minicpmv2_6.Modelfile
```

ModelFile的内容如下：

```plaintext
FROM ./MiniCPM-V-2_6/model/ggml-model-Q4_K_M.gguf
FROM ./MiniCPM-V-2_6/mmproj-model-f16.gguf

TEMPLATE """{{ if .System }}<|im_start|>system

{{ .System }}<|im_end|>{{ end }}

{{ if .Prompt }}<|im_start|>user

{{ .Prompt }}<|im_end|>{{ end }}

<|im_start|>assistant<|im_end|>

{{ .Response }}<|im_end|>"""

PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|im_end|>"
PARAMETER num_ctx 2048
```
参数说明:

| first from | second from | num_keep | num_ctx |
|-----|-----|-----|-----|
| Your language gguf Model path | Your visionGguf modelpath | max connection limit| Max Model length |

9. 创建ollama模型实例：
```bash
ollama create minicpm2.6 -f minicpmv2_6.Modelfile
```

10. 另起一个命令行窗口，运行ollama模型实例：
```bash
ollama run minicpm2.6
```

11. 输入问题和图片 URL，以空格分隔
```bash
What is described in this picture? /Users/liudan/Desktop/11.jpg
```
