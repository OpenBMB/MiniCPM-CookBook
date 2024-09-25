
# Introduction to PowerInfer

PowerInfer is an inference engine developed by Shanghai Jiao Tong University that accelerates sparse models on CPU/GPU, reportedly achieving up to 11 times the inference performance of llama.cpp. However, PowerInfer currently supports only a few models, including MiniCPM-S-2B-SFT, and is not compatible with all models.

# Installing PowerInfer

## Prerequisites

Ensure your system meets the following conditions:
- CMake version 3.17 or later
- Python version 3.8 or later

### Check CMake Version

```sh
cmake --version
```

If you see the following message, it means CMake is not installed:
```
cmake: command not found
```

### Install CMake 3.17+

1. **Download the Installation Package**
   ```sh
   sudo wget https://cmake.org/files/v3.23/cmake-3.23.0.tar.gz
   ```

2. **Extract the Installation Package**
   ```sh
   sudo tar -zxvf cmake-3.23.0.tar.gz
   ```

3. **Configure the Installation Environment**
   ```sh
   cd cmake-3.23.0
   sudo ./configure
   sudo make -j8
   ```

4. **Compile and Install**
   ```sh
   sudo make install
   ```

5. **Check the Installed Version**
   ```sh
   cmake --version
   ```

   If it returns the version number `cmake version 3.23.0`, the installation was successful.

### Install PowerInfer

1. **Clone the PowerInfer Repository**
   ```sh
   git clone https://github.com/SJTU-IPADS/PowerInfer
   cd PowerInfer
   pip install -r requirements.txt  # Install dependencies for Python auxiliary tools
   ```

2. **Compile the CPU Inference Version of PowerInfer**
   ```sh
   cmake -S . -B build
   cmake --build build --config Release
   ```

3. **Compile the GPU Inference Version of PowerInfer**
   ```sh
   cmake -S . -B build -DLLAMA_CUBLAS=ON
   cmake --build build --config Release
   ```

## Obtain the Model

Clone the MiniCPM-S-1B-sft-gguf model:
```sh
git clone https://huggingface.co/openbmb/MiniCPM-S-1B-sft-gguf/tree/main
```

## Start Inference

Navigate to the PowerInfer directory:
```sh
cd PowerInfer
```

Here is the command template:
- `output_token_count` is the maximum number of output tokens
- `thread_num` is the number of threads
- `prompt` is the input prompt string

```sh
./build/bin/main -m /PATH/TO/MODEL -n $output_token_count -t $thread_num -p $prompt
```

Example:
```sh
./build/bin/main -m /root/ld/ld_model_pretrain/1b-s-minicpm/MiniCPM-S-1B-sft.gguf -n 2048 -t 8 -p '<User>hello, tell me a story please.<AI>'
```

## Inference Speed Demonstration

In an environment equipped with an NVIDIA 4090 GPU:
- **Prefilling Phase**: 221 token/s
- **Decode Phase**: 45 token/s

![PowerInfer Performance](../../../../asset/powerinfer.png)
