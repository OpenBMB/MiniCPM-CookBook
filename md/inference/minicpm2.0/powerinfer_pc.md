
# PowerInfer 简介

PowerInfer是由上海交通大学开发的一个推理引擎，它可以在CPU/GPU上基于稀疏模型进行加速，据称能够获得最高达llama.cpp 11倍的推理性能。然而，PowerInfer目前仅适配了包括MiniCPM-S-2B-SFT在内的少数模型，并非适配所有模型。

# 安装 PowerInfer

## 准备工作

确保您的系统满足以下条件：
- CMake版本在3.17以上
- Python版本3.8+

### 查看CMake版本

```sh
cmake --version
```

如果出现以下提示，则说明未安装CMake：
```
cmake: command not found
```

### 安装CMake 3.17+

1. **下载安装包**
   ```sh
   sudo wget https://cmake.org/files/v3.23/cmake-3.23.0.tar.gz
   ```

2. **解压安装包**
   ```sh
   sudo tar -zxvf cmake-3.23.0.tar.gz
   ```

3. **配置安装环境**
   ```sh
   sudo ./configure
   sudo make -j8
   ```

4. **编译安装**
   ```sh
   sudo make install
   ```

5. **查看安装后的版本**
   ```sh
   cmake --version
   ```

   如果返回版本号`cmake version 3.23.0`，则表示安装成功。

### 安装PowerInfer

1. **克隆PowerInfer仓库**
   ```sh
   git clone https://github.com/SJTU-IPADS/PowerInfer
   cd PowerInfer
   pip install -r requirements.txt  # 安装Python辅助工具的依赖项
   ```

2. **编译CPU推理版的PowerInfer**
   ```sh
   cmake -S . -B build
   cmake --build build --config Release
   ```

3. **编译GPU推理版的PowerInfer**
   ```sh
   cmake -S . -B build -DLLAMA_CUBLAS=ON
   cmake --build build --config Release
   ```

## 获取模型

克隆MiniCPM-S-1B-sft-gguf模型：
```sh
git clone https://huggingface.co/openbmb/MiniCPM-S-1B-sft-gguf/tree/main
```

## 开始推理

进入PowerInfer目录：
```sh
cd PowerInfer
```

以下是命令模板：
- `output_token_count` 为最大输出tokens数量
- `thread_num` 为线程数
- `prompt` 为输入的prompt字符

```sh
./build/bin/main -m /PATH/TO/MODEL -n $output_token_count -t $thread_num -p $prompt
```

示例：
```sh
./build/bin/main -m /root/ld/ld_model_pretrain/1b-s-minicpm/MiniCPM-S-1B-sft.gguf -n 2048 -t 8 -p '<用户>hello,tell me a story please.<AI>'
```

## 推理速度展示

在配备NVIDIA 4090 GPU的环境中：
- **Prefilling阶段**：221 token/s
- **Decode阶段**：45 token/s

![alt text](../../../asset/powerinfer.png)


这样，您就可以开始使用PowerInfer进行高效推理了！
