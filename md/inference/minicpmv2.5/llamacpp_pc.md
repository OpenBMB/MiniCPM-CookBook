
# llama.cpp 在Linux或macOS上的部署

## 配套视频
- [llamacpp](https://www.bilibili.com/video/BV1tS42197NL/?spm_id_from=333.337.search-card.all.click&vd_source=1534be4f756204643265d5f6aaa38c7b)

## 设备要求
- 运行非量化版：内存超过19GB
- 运行量化版：内存超过8GB

## 步骤1：获取OpenBMB的llama.cpp分支

通过Git克隆指定分支：

```sh
git clone -b minicpm-v2.5 https://github.com/OpenBMB/llama.cpp.git
```

## 步骤2：编译llama.cpp

进入llama.cpp目录并编译：

```sh
cd llama.cpp
make
make minicpmv-cli
```

## 步骤3：获取MiniCPMv2.5的gguf权重

### 3.1 下载pytorch权重

前往Hugging Face或ModelScope下载所需模型：

```sh
git clone https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5
# 或者
git clone https://modelscope.cn/models/OpenBMB/MiniCPM-Llama3-V-2_5.git
```

### 3.2 转换模型权重为gguf文件

1. **获取模型中间输出，为转换为gguf作准备**
   ```sh
   python ./examples/minicpmv/minicpmv-surgery.py -m ../MiniCPM-Llama3-V-2_5
   ```

2. **将Siglip模型转换为gguf**
   ```sh
   python ./examples/minicpmv/minicpmv-convert-image-encoder-to-gguf.py -m ../MiniCPM-Llama3-V-2_5 --minicpmv-projector ../MiniCPM-Llama3-V-2_5/minicpmv.projector --output-dir ../MiniCPM-Llama3-V-2_5/ --image-mean 0.5 0.5 0.5 --image-std 0.5 0.5 0.5
   ```

3. **将语言模型转换为gguf**
   ```sh
   python ./convert.py ../MiniCPM-Llama3-V-2_5/model --outtype f16 --vocab-type bpe
   ```

### 3.3 对语言模型进行量化（可选）

如果需要，对语言模型进行量化：

```sh
./quantize ../MiniCPM-Llama3-V-2_5/model/model-8B-F16.gguf ../MiniCPM-Llama3-V-2_5/model/ggml-model-Q4_K_M.gguf Q4_K_M
```

## 步骤4：开始推理

使用以下命令进行推理：

- **使用非量化模型推理**
  ```sh
  ./minicpmv-cli -m ../MiniCPM-Llama3-V-2_5/model/model-8B-F16.gguf --mmproj ../MiniCPM-Llama3-V-2_5/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image xx.jpg -p "What is in the image?"
  ```

- **使用量化模型推理**
  ```sh
  ./minicpmv-cli -m ../MiniCPM-Llama3-V-2_5/model/ggml-model-Q4_K_M.gguf --mmproj ../MiniCPM-Llama3-V-2_5/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image xx.jpg -i
  ```

现在您可以开始使用llama.cpp进行高效推理了！