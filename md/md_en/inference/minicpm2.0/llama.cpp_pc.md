# Deploying llama.cpp on PC

## Supported Devices
- Linux
- macOS

### Step 1: Download llama.cpp

Clone the llama.cpp repository via Git:
```sh
git clone https://github.com/ggerganov/llama.cpp
```

### Step 2: Compile llama.cpp

Navigate into the llama.cpp directory and compile it:
```sh
cd llama.cpp
make
```

### Step 3: Obtain the MiniCPM gguf Model

#### Method 1: Direct Download
- [Download Link - fp16 Format](https://huggingface.co/runfuture/MiniCPM-2B-dpo-fp16-gguf)
- [Download Link - q4km Format](https://huggingface.co/runfuture/MiniCPM-2B-dpo-q4km-gguf)

#### Method 2: Convert the MiniCPM Model to gguf Format Yourself

1. **Create the Model Storage Path**
   ```sh
   cd llama.cpp/models
   mkdir Minicpm
   ```

2. **Download the MiniCPM PyTorch Model**
   Download all files from the [MiniCPM PyTorch model](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16) and save them to the `llama.cpp/models/Minicpm` directory.

3. **Modify the Conversion Script**
   Check the `_reverse_hf_permute` function in the `llama.cpp/convert-hf-to-gguf.py` file. If you find the following code:
   ```python
   def _reverse_hf_permute(self, weights: Tensor, n_head: int, n_kv_head: int | None = None) -> Tensor:
       if n_kv_head is not None and n_head != n_kv_head:
           n_head //= n_kv_head
   ```
   Replace it with:
   ```python
   @staticmethod
   def permute(weights: Tensor, n_head: int, n_head_kv: int | None):
       if n_head_kv is not None and n_head != n_head_kv:
           n_head = n_head_kv
       return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
               .swapaxes(1, 2)
               .reshape(weights.shape))

   def _reverse_hf_permute(self, weights: Tensor, n_head: int, n_kv_head: int | None = None) -> Tensor:
       if n_kv_head is not None and n_head != n_kv_head:
           n_head //= n_kv_head
   ```

4. **Install Dependencies and Convert the Model**
   ```sh
   python3 -m pip install -r requirements.txt
   python3 convert-hf-to-gguf.py models/Minicpm/
   ```

   After completing these steps, there will be a model file named `ggml-model-f16.gguf` in the `llama.cpp/models/Minicpm` directory.

### Step 4: Quantize the fp16 gguf File

Skip this step if the downloaded model is already in quantized format.

```sh
./llama-quantize ./models/Minicpm/ggml-model-f16.gguf ./models/Minicpm/ggml-model-Q4_K_M.gguf Q4_K_M
```

If you cannot find `llama-quantize`, try recompiling:
```sh
cd llama.cpp
make llama-quantize
```

### Step 5: Start Inference

Perform inference using the quantized model:
```sh
./llama-cli -m ./models/Minicpm/ggml-model-Q4_K_M.gguf -n 128 --prompt "<User>Do you know openmbmb?<AI>"
```

Please note that the download links for the models are placeholders and should be replaced with actual URLs where the models can be downloaded.