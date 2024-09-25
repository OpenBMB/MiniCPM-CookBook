# Update macOS to Version 13.5 or Later

To use `mlx-lm` for inference, you need to upgrade your Mac's operating system to at least version 13.5. You can check and install updates through the following steps:

1. Open "System Settings" on your Mac.
2. Select the "General" tab.
3. In the "Software Update" section, click on "Automatic Updates".
4. Enable "Install macOS updates".

# Install mlx-lm

Install the `mlx-lm` library via the command line:

```sh
pip install mlx-lm
```

---

# Quick Inference of MiniCPM Using MLX

If you are performing inference on a Mac, you can directly use MLX for inference. Since MiniCPM does not currently support conversion to the mlx format, you can download a model that has been converted by the MLX community:

- [MiniCPM-2B-sft-bf16-llama-format-mlx](https://huggingface.co/mlx-community/MiniCPM-2B-sft-bf16-llama-format-mlx)

Install the necessary dependencies:

```bash
pip install mlx-lm
```

Below is a simple script for inference using the MiniCPM-2 model on a Mac:

```python
python -m mlx_lm.generate --model mlx-community/MiniCPM-2B-sft-bf16-llama-format-mlx --prompt "hello, tell me a joke." --trust-remote-code
```

---

# Implementing Interaction with the Model

The following is a Python script designed for interacting with the MiniCPM model:

```python
from mlx_lm import load, generate
from jinja2 import Template

def chat_with_model():
    model, tokenizer = load("mlx-community/MiniCPM-2B-sft-bf16-llama-format-mlx")
    print("Model loaded. Start chatting! (Type 'quit' to stop)")

    messages = []
    chat_template = Template(
        "{% for message in messages %}{% if message['role'] == 'user' %}{{'<User>' + message['content'].strip() + '<AI>'}}{% else %}{{message['content'].strip()}}{% endif %}{% endfor %}")

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

# Get Started

Run the above Python script to begin interacting with the MiniCPM model!