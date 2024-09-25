# 部署llama.cpp到PC端

## 支持设备
- Linux
- macOS

### 步骤1：下载llama.cpp

通过Git克隆llama.cpp仓库：
```sh
git clone https://github.com/ggerganov/llama.cpp
```

### 步骤2：编译llama.cpp

进入llama.cpp目录并编译：
```sh
cd llama.cpp
make
```

### 步骤3：获取MiniCPM的gguf模型

#### 方法1：直接下载
- [下载链接 - fp16格式](https://huggingface.co/runfuture/MiniCPM-2B-dpo-fp16-gguf)
- [下载链接 - q4km格式](https://huggingface.co/runfuture/MiniCPM-2B-dpo-q4km-gguf)

#### 方法2：自行转换MiniCPM模型为gguf格式

1. **创建模型存储路径**
   ```sh
   cd llama.cpp/models
   mkdir Minicpm
   ```

2. **下载MiniCPM pytorch模型**
   下载[MiniCPM pytorch模型](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16)的所有文件，并保存到`llama.cpp/models/Minicpm`目录下。

3. **修改转换脚本**
   检查`llama.cpp/convert-hf-to-gguf.py`文件中的`_reverse_hf_permute`函数，如果发现如下代码：
   ```python
   def _reverse_hf_permute(self, weights: Tensor, n_head: int, n_kv_head: int | None = None) -> Tensor:
       if n_kv_head is not None and n_head != n_kv_head:
           n_head //= n_kv_head
   ```
   替换为：
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

4. **安装依赖并转换模型**
   ```sh
   python3 -m pip install -r requirements.txt
   python3 convert-hf-to-gguf.py models/Minicpm/
   ```

   完成以上步骤后，`llama.cpp/models/Minicpm`目录下将会有一个名为`ggml-model-f16.gguf`的模型文件。

### 步骤4：量化fp16的gguf文件

若下载的模型已经是量化格式，则跳过此步骤。

```sh
./llama-quantize ./models/Minicpm/ggml-model-f16.gguf ./models/Minicpm/ggml-model-Q4_K_M.gguf Q4_K_M
```

如果找不到`llama-quantize`，可以尝试重新编译：
```sh
cd llama.cpp
make llama-quantize
```

### 步骤5：开始推理

使用量化后的模型进行推理：
```sh
./llama-cli -m ./models/Minicpm/ggml-model-Q4_K_M.gguf -n 128 --prompt "<用户>你知道openmbmb么<AI>"
```

