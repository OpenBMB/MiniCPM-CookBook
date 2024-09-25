
# MiniCPM 3.0 使用示例

## Chat 方法

下面的代码示例展示了如何使用 `transformers` 库来实现与 MiniCPM 3.0 模型的聊天功能：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载预训练的模型和分词器
tokenizer = AutoTokenizer.from_pretrained("/root/ld/ld_model_pretrained/minicpm3")
model = AutoModelForCausalLM.from_pretrained("/root/ld/ld_model_pretrained/minicpm3", trust_remote_code=True).cuda()

# 初始化历史记录列表
history = []

while True:
    # 获取用户输入
    query = input("user:")
    
    # 生成回复，并更新历史记录
    response, history = model.chat(tokenizer, query=query, history=history)
    
    # 打印模型回复
    print("model:", response)
    
# 注意：history 是一个列表，其中包含历史对话记录，格式如下：
# history = [{"role": "assistant", "content": answer1}, {"role": "assistant", "content": response}]
```

## Generate 方法

接下来是一个使用 `transformers` 库生成文本的简单示例：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载预训练的模型和分词器
model = AutoModelForCausalLM.from_pretrained("/root/ld/ld_model_pretrained/minicpm3", trust_remote_code=True).cuda()
tokenizer = AutoTokenizer.from_pretrained("/root/ld/ld_model_pretrained/minicpm3", trust_remote_code=True)

# 定义提示
prompt = "Hey, are you conscious? Can you tell me "

# 对提示进行编码
inputs = tokenizer(prompt, return_tensors="pt")

# 生成文本
generate_ids = model.generate(inputs.input_ids.cuda(), max_length=300, do_sample=False)

# 解码生成的ID序列
output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# 打印输出结果
print(output)
```

上述代码片段提供了两种不同的方式来与 MiniCPM 3.0 模型进行交互，一种是基于对话历史的聊天功能，另一种是简单的文本生成。希望这些示例能帮助您更好地利用 MiniCPM 3.0 模型的功能。