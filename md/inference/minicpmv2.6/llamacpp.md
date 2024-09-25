# Llama.cpp 推理

## 设备要求

- 运行非量化版本需要超过19GB内存
- 运行量化版本需要超过8GB内存

## 步骤1：下载依赖包

使用Homebrew安装依赖包：

```sh
brew install ffmpeg
brew install pkg-config
```

## 步骤2：获取Llama.cpp

通过Git克隆Llama.cpp仓库：

```sh
git clone https://github.com/ggerganov/llama.cpp
```

如果需要使用视频模式，则需要使用OpenBMB的特定分支：

```sh
git clone -b minicpmv-main https://github.com/OpenBMB/llama.cpp.git
```

## 步骤3：编译Llama.cpp

进入Llama.cpp目录并编译：

```sh
cd llama.cpp
make
```

## 步骤4：获取MiniCPM-V 2.6的gguf权重

### 方法一：

1. 首先前往HuggingFace或ModelScope下载PyTorch权重：

   ```sh
   git clone https://huggingface.co/openbmb/MiniCPM-V-2_6
   ```

2. 使用上述Llama.cpp将模型权重转化为gguf文件：

   ```sh
   # 第一行为获得模型中间输出，为转换为gguf作准备
   python ./examples/llava/minicpmv-convert/minicpmv2_6-surgery.py -m ../MiniCPM-V-2_6
   
   # 将Siglip模型转换为gguf
   python ./examples/llava/minicpmv-convert/minicpmv2_6-convert-image-encoder-to-gguf.py -m ../MiniCPM-V-2_6 --minicpmv-projector ../MiniCPM-V-2_6/minicpmv.projector --output-dir ../MiniCPM-V-2_6/ --image-mean 0.5 0.5 0.5 --image-std 0.5 0.5 0.5
   
   # 将语言模型转换为gguf
   python ./convert-hf-to-gguf.py ../MiniCPM-V-2_6/model
   ```

3. 如果需要的话，对语言模块进行量化：

   ```sh
   # 量化为int4版本
   ./llama-quantize ../MiniCPM-V-2_6/model/ggml-model-f16.gguf ../MiniCPM-V-2_6/model/ggml-model-Q4_K_M.gguf Q4_K_M
   ```

### 方法二：

1. 直接前往MiniCPM-V 2.6-gguf下载模型，选择`ggml-model-Q4_K_M.gguf`（量化版）或`ggml-model-f16.gguf`之一。

## 步骤5：开始推理

### 5.1 图片推理指令

```sh
./llama-minicpmv-cli -m ./Minicpmv2_6gguf/ggml-model-Q4_K_M.gguf --mmproj ./Minicpmv2_6gguf/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image ./Minicpmv2_6gguf/42.jpg -p "这张图片中有什么？"
```

### 5.2 视频推理指令（需要使用我们fork的llamacpp）

```sh
./llama-minicpmv-cli -m /Users/liudan/Downloads/Minicpmv2_6gguf/ggml-model-Q4_K_M.gguf --mmproj /Users/liudan/Downloads/Minicpmv2_6gguf/mmproj-model-f16.gguf -c 8192 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --video ./Minicpmv2_6gguf/test_vedieo.mp4 -p "我接下来会给你一个视频，请告诉我视频中描述了什么"
```
### 5.3 参数说明

| 参数名 | 含义                           |
| ------ | ------------------------------ |
| `-m`   | 语言模型地址                   |
| `--mmproj` | 图像模型地址                   |
| `--image` | 输入图片地址                   |
| `-p`   | prompt                         |
| `--video` | mp4视频地址                    |
| `-c`   | 输入最大长度                   |