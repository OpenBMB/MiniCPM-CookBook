# Deploying llama.cpp on Linux or macOS

## Accompanying Video
- [llamacpp](https://www.bilibili.com/video/BV1tS42197NL/?spm_id_from=333.337.search-card.all.click&vd_source=1534be4f756204643265d5f6aaa38c7b)

## System Requirements
- Running non-quantized version: More than 19GB of memory
- Running quantized version: More than 8GB of memory

## Step 1: Obtain the OpenBMB llama.cpp Branch

Clone the specified branch using Git:

```sh
git clone -b minicpm-v2.5 https://github.com/OpenBMB/llama.cpp.git
```

## Step 2: Compile llama.cpp

Navigate to the llama.cpp directory and compile:

```sh
cd llama.cpp
make
make minicpmv-cli
```

## Step 3: Obtain the MiniCPMv2.5 gguf Weights

### 3.1 Download PyTorch Weights

Download the required model from Hugging Face or ModelScope:

```sh
git clone https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5
# Or
git clone https://modelscope.cn/models/OpenBMB/MiniCPM-Llama3-V-2_5.git
```

### 3.2 Convert Model Weights to gguf File

1. **Get intermediate model outputs for converting to gguf**
   ```sh
   python ./examples/minicpmv/minicpmv-surgery.py -m ../MiniCPM-Llama3-V-2_5
   ```

2. **Convert the Siglip model to gguf**
   ```sh
   python ./examples/minicpmv/minicpmv-convert-image-encoder-to-gguf.py -m ../MiniCPM-Llama3-V-2_5 --minicpmv-projector ../MiniCPM-Llama3-V-2_5/minicpmv.projector --output-dir ../MiniCPM-Llama3-V-2_5/ --image-mean 0.5 0.5 0.5 --image-std 0.5 0.5 0.5
   ```

3. **Convert the language model to gguf**
   ```sh
   python ./convert.py ../MiniCPM-Llama3-V-2_5/model --outtype f16 --vocab-type bpe
   ```

### 3.3 Quantize the Language Model (Optional)

If needed, quantize the language model:

```sh
./quantize ../MiniCPM-Llama3-V-2_5/model/model-8B-F16.gguf ../MiniCPM-Llama3-V-2_5/model/ggml-model-Q4_K_M.gguf Q4_K_M
```

## Step 4: Start Inference

Use the following commands for inference:

- **Infer with the Non-Quantized Model**
  ```sh
  ./minicpmv-cli -m ../MiniCPM-Llama3-V-2_5/model/model-8B-F16.gguf --mmproj ../MiniCPM-Llama3-V-2_5/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image xx.jpg -p "What is in the image?"
  ```

- **Infer with the Quantized Model**
  ```sh
  ./minicpmv-cli -m ../MiniCPM-Llama3-V-2_5/model/ggml-model-Q4_K_M.gguf --mmproj ../MiniCPM-Llama3-V-2_5/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image xx.jpg -i
  ```

Now you are ready to start efficient inference with llama.cpp!