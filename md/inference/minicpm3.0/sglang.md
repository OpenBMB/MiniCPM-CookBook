```markdown
# Sglang 安装与使用指南

## 源码安装 Sglang

1. 首先，从 GitHub 克隆 Sglang 项目仓库：

   ```bash
   git clone https://github.com/sgl-project/sglang.git
   cd sglang
   ```

2. 升级 `pip` 并安装 Sglang 及其所有依赖项：

   ```bash
   pip install --upgrade pip
   pip install -e "python[all]"
   ```

## 安装 FlashInfer 依赖

### 方法 1: 使用 pip 安装 (可能因网络速度而失败)

```bash
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
```

### 方法 2: 使用 whl 文件安装

1. 访问网页：https://flashinfer.ai/whl/cu121/torch2.4/flashinfer/
2. 下载适合您服务器配置的 `.whl` 文件。
3. 使用 `pip` 安装 `.whl` 文件。例如：

   ```bash
   pip install your/path/flashinfer-0.1.6+cu121torch2.4-cp310-cp310-linux_x86_64.whl
   ```

## 发起 Sglang 推理服务

启动服务前，请确保模型路径正确。如果遇到内存不足的问题，尝试添加 `--disable-cuda-graph` 参数。

```bash
python -m sglang.launch_server --model openbmb/MiniCPM3-4B --trust-remote-code --port 30000 --chat-template chatml
```

## 调用服务接口

### Bash 调用示例

```bash
curl http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Once upon a time,",
    "sampling_params": {
      "max_new_tokens": 16,
      "temperature": 0
    }
  }'
```

### Python 调用示例

```python
import openai
client = openai.Client(
    base_url="http://127.0.0.1:30000/v1", api_key="EMPTY"
)

# 文本补全
response = client.completions.create(
    model="default",
    prompt="The capital of France is",
    temperature=0,
    max_tokens=32,
)
print(response)

# 对话补全
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": "你叫什么名字？"},
    ],
    temperature=0,
    max_tokens=64,
)
print(response.choices[0].message.content)
```
