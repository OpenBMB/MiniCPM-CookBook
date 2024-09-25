## Llamacpp
**Device: Linux, Mac**

### 1. Download the minicpm3 branch of llama.cpp
```bash
git clone https://github.com/OpenBMB/llama.cpp.git
git checkout minicpm3
```

### 2. Compile llama.cpp
```bash
cd llama.cpp
make
```

### 3. Obtain the MiniCPM gguf Model

#### 3.1 Create the `llama.cpp/models/Minicpm3` Path
```bash
cd llama.cpp/models
mkdir Minicpm3
```

#### 3.2 Download All Files of the [MiniCPM3 Model] (or a trained model) and Save Them to `llama.cpp/models/Minicpm3`
#### 3.3 Convert the Model to gguf Format
```bash
python3 -m pip install -r requirements.txt
# Convert the PyTorch model to fp16 gguf
python3 convert_hf_to_gguf.py models/Minicpm3/ --outfile /your/path/llama.cpp/models/Minicpm3/CPM-4B-F16.gguf
# After completing these steps, there will be a CPM-4B-F16.gguf model file in the `llama.cpp/models/Minicpm3` directory
```

### 4. Quantize the fp16 gguf File
```bash
# After executing this command successfully, a 4-bit quantized file named `ggml-model-Q4_K_M.gguf` will exist in the `/models/Minicpm3/` directory
./llama-quantize ./models/Minicpm3/CPM-4B-F16.gguf ./models/Minicpm3/ggml-model-Q4_K_M.gguf Q4_K_M
# If `llama-quantize` is not found, you can try the following method
cd llama.cpp
make llama-quantize
```

### 5. Start Inference

#### Command Line Inference
```bash
./llama-cli -c 1024 -m ./models/Minicpm3/ggml-model-Q4_K_M.gguf -n 1024 --top-p 0.7 --temp 0.7 --prompt 
```

#### Server Service
##### Start the Service
```bash
./llama-server -m ./models/Minicpm3/CPM-4B-F16.gguf -c 2048
```

##### Call the API
```python
import requests

url = "http://localhost:8080/completion"
headers = {
    "Content-Type": "application/json"
}
data = {
    "prompt": "Who released MiniCPM3?",
    "n_predict": 128
}

response = requests.post(url, json=data, headers=headers)

if response.status_code == 200:
    result = response.json()
    print(result["content"])
else:
    print(f"Request failed with status code {response.status_code}: {response.text}")
```
