# Llama.cpp Inference

## System Requirements

- Non-quantized version requires more than 19GB of memory
- Quantized version requires more than 8GB of memory

## Step 1: Download Dependencies

Install dependencies using Homebrew:

```sh
brew install ffmpeg
brew install pkg-config
```

## Step 2: Get Llama.cpp

Clone the Llama.cpp repository using Git:

```sh
git clone https://github.com/ggerganov/llama.cpp
```

If you need to use video mode, use the specific branch from OpenBMB:

```sh
git clone -b minicpmv-main https://github.com/OpenBMB/llama.cpp.git
```

## Step 3: Compile Llama.cpp

Navigate to the Llama.cpp directory and compile:

```sh
cd llama.cpp
make
```

## Step 4: Obtain MiniCPM-V 2.6 gguf Weights

### Method 1:

1. First, download PyTorch weights from HuggingFace or ModelScope:

   ```sh
   git clone https://huggingface.co/openbmb/MiniCPM-V-2_6
   ```

2. Use the provided Llama.cpp scripts to convert the model weights to gguf format:

   ```sh
   # The first command generates intermediate outputs for converting to gguf
   python ./examples/llava/minicpmv-convert/minicpmv2_6-surgery.py -m ../MiniCPM-V-2_6
   
   # Convert the Siglip model to gguf
   python ./examples/llava/minicpmv-convert/minicpmv2_6-convert-image-encoder-to-gguf.py -m ../MiniCPM-V-2_6 --minicpmv-projector ../MiniCPM-V-2_6/minicpmv.projector --output-dir ../MiniCPM-V-2_6/ --image-mean 0.5 0.5 0.5 --image-std 0.5 0.5 0.5
   
   # Convert the language model to gguf
   python ./convert-hf-to-gguf.py ../MiniCPM-V-2_6/model
   ```

3. Optionally, quantize the language model:

   ```sh
   # Quantize to int4 version
   ./llama-quantize ../MiniCPM-V-2_6/model/ggml-model-f16.gguf ../MiniCPM-V-2_6/model/ggml-model-Q4_K_M.gguf Q4_K_M
   ```

### Method 2:

1. Directly download the MiniCPM-V 2.6-gguf model from the official source, choosing either `ggml-model-Q4_K_M.gguf` (quantized version) or `ggml-model-f16.gguf`.

## Step 5: Start Inference

### 5.1 Image Inference Command

```sh
./llama-minicpmv-cli -m ./Minicpmv2_6gguf/ggml-model-Q4_K_M.gguf --mmproj ./Minicpmv2_6gguf/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image ./Minicpmv2_6gguf/42.jpg -p "What is in this picture?"
```

### 5.2 Video Inference Command (Requires the forked llamacpp)

```sh
./llama-minicpmv-cli -m /Users/liudan/Downloads/Minicpmv2_6gguf/ggml-model-Q4_K_M.gguf --mmproj /Users/liudan/Downloads/Minicpmv2_6gguf/mmproj-model-f16.gguf -c 8192 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --video ./Minicpmv2_6gguf/test_video.mp4 -p "I will give you a video next, please tell me what is described in the video."
```

### 5.3 Parameter Explanation

| Parameter Name | Meaning                           |
| -------------- | --------------------------------- |
| `-m`           | Path to the language model        |
| `--mmproj`     | Path to the image model           |
| `--image`      | Path to the input image           |
| `-p`           | Prompt                            |
| `--video`      | Path to the mp4 video             |
| `-c`           | Maximum input length              |
