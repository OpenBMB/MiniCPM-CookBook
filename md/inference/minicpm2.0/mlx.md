
# 更新macOS至13.5及以上版本

为了使用`mlx-lm`进行推理，您需要将Mac设备的操作系统升级到至少13.5版本。可以通过以下步骤检查并安装更新：

1. 打开mac的“设置”。
2. 选择“通用”选项。
3. 在“软件更新”部分点击“自动更新”。
4. 启用“安装macOS更新”。

# 安装 mlx-lm

通过命令行安装`mlx-lm`库：

```sh
pip install mlx-lm
```

---

# 使用 MLX 快速推理 MiniCPM

如果您使用的是Mac设备进行推理，可以直接使用MLX进行推理。由于MiniCPM暂时不支持mlx格式转换，您可以下载由MLX社区转换好的模型：

- [MiniCPM-2B-sft-bf16-llama-format-mlx](https://huggingface.co/mlx-community/MiniCPM-2B-sft-bf16-llama-format-mlx)

安装对应的依赖包：

```bash
pip install mlx-lm
```

下面是一个简单的推理代码，使用Mac设备推理MiniCPM-2：

```python
python -m mlx_lm.generate --model mlx-community/MiniCPM-2B-sft-bf16-llama-format-mlx --prompt "hello, tell me a joke." --trust-remote-code
```

---

# 实现与模型交互

以下是一个Python脚本，用于与MiniCPM模型进行交互：

```python
from mlx_lm import load, generate
from jinja2 import Template

def chat_with_model():
    model, tokenizer = load("mlx-community/MiniCPM-2B-sft-bf16-llama-format-mlx")
    print("Model loaded. Start chatting! (Type 'quit' to stop)")

    messages = []
    chat_template = Template(
        "{% for message in messages %}{% if message['role'] == 'user' %}{{'<用户>' + message['content'].strip() + '<AI>'}}{% else %}{{message['content'].strip()}}{% endif %}{% endfor %}")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        messages.append({"role": "user", "content": user_input})
        response = generate(model, tokenizer, prompt=chat_template.render(messages=messages), verbose=True)
        print("Model:", response)
        messages.append({"role": "ai", "content": response})

chat_with_model()
```

# 开始体验

运行上述Python脚本，开始与MiniCPM模型的互动吧！