
# Ollama 部署

## 设备要求
- 运行非量化版：内存超过19GB
- 运行量化版：内存超过8GB

## 步骤1：获取gguf模型

按照[llama.cpp的教程](llamacpp_pc.md)获取gguf模型文件，语言模型最好进行量化处理。

## 步骤2：获取OpenBMB官方的Ollama分支

通过Git克隆指定分支：

```sh
git clone -b minicpm-v2.5 https://github.com/OpenBMB/ollama.git
cd ollama/llm
```

## 步骤3：确保环境依赖

确保满足以下依赖条件：
- CMake版本3.24以上
- Go版本1.22以上
- GCC版本11.4.0以上

使用Homebrew安装这些依赖：

```sh
brew install go cmake gcc
```

## 步骤4：安装大模型依赖

安装Ollama的大模型依赖：

```sh
go generate ./...
```

## 步骤5：编译Ollama

编译Ollama：

```sh
go build .
```

## 步骤6：启动Ollama服务

编译成功后，在Ollama主路径下启动服务：

```sh
./ollama serve
```

## 步骤7：创建Model文件

创建一个名为`minicpmv2_5.Modelfile`的文件：

```sh
vim minicpmv2_5.Modelfile
```

文件内容如下：

```plaintext
# 第一个和第二个 FROM 空格后面分别写上量化后的语言模型地址和图像投影模型地址

FROM ./MiniCPM-V-2_5/model/ggml-model-Q4_K_M.gguf
FROM ./MiniCPM-V-2_5/mmproj-model-f16.gguf

TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""

PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER num_keep 4
PARAMETER num_ctx 2048
```

## 步骤8：创建Ollama模型

使用以下命令创建Ollama模型：

```sh
ollama create minicpm2.5 -f minicpmv2_5.Modelfile
```

## 步骤9：运行Ollama模型

运行创建的Ollama模型：

```sh
ollama run minicpm2.5
```

## 步骤10：输入问题和图片地址

输入问题和图片地址时，请使用空格进行分割。

现在您可以开始使用Ollama进行高效推理了！
