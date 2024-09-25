
## Llamacpp
**设备：Linux，Mac**

### 1. 下载llama.cpp的minicpm3分支
```bash
git clone https://github.com/OpenBMB/llama.cpp.git
git checkout minicpm3
```

### 2. 编译llama.cpp
```bash
cd llama.cpp
make
```

### 3. 获取MiniCPM的gguf模型

#### 3.1 创建llama.cpp/models/Minicpm路径
```bash
cd llama.cpp/models
mkdir Minicpm3
```

#### 3.2 下载[MiniCPM3模型]所有文件(也可以是训练后的模型)并保存到llama.cpp/models/Minicpm
#### 3.3 将模型转换为gguf格式
```bash
python3 -m pip install -r requirements.txt
# 将pytorch模型转化为fp16的gguf
python3 convert_hf_to_gguf.py models/Minicpm3/ --outfile /your/path/llama.cpp/models/Minicpm3/CPM-4B-F16.gguf
# 完成以上步骤后，llama.cpp/models/Minicpm3目录下将存在一个CPM-4B-F16.gguf的模型文件
```

### 4. 将fp16的gguf文件进行量化
```bash
# 使用本行命令执行成功后，/models/Minicpm/下将存在 ggml-model-Q4_K_M.gguf的4bit量化文件
./llama-quantize ./models/Minicpm3/CPM-4B-F16.gguf ./models/Minicpm3/ggml-model-Q4_K_M.gguf Q4_K_M
# 如果找不到llama-quantize，可以尝试以下方法
cd llama.cpp
make llama-quantize
```

### 5. 开始推理

#### 命令行推理
```bash
./llama-cli -c 1024 -m ./models/Minicpm/ggml-model-Q4_K_M.gguf -n 1024 --top-p 0.7 --temp 0.7 --prompt 
```
#### server服务
##### 发起服务
```bash
./llama-server  -m ./models/Minicpm3/CPM-2B-F16.gguf -c 2048
```
##### 调用接口
```python
.import requests

url = "http://localhost:8080/completion"
headers = {
    "Content-Type": "application/json"
}
data = {
    "prompt": "MiniCPM3 是哪家公司发布的？",
    "n_predict": 128
}

response = requests.post(url, json=data, headers=headers)

if response.status_code == 200:
    result = response.json()
    print(result["content"])
else:
    print(f"Request failed with status code {response.status_code}: {response.text}")
```