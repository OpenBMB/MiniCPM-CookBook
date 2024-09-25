
# Function Call简易实现(Minicpm3.0)

下面是一个简单的Python脚本，用于演示如何通过调用特定的函数来获取订单的配送日期。此脚本包含了一个函数`get_delivery_date`以及一个处理函数调用的函数`get_response_call`。此外，还定义了一个`tools`列表，其中包含了函数`get_delivery_date`的相关描述和参数。

```python
#!/usr/bin/env python
# encoding: utf-8
import re
from transformers import AutoTokenizer,AutoModelForCausalLM

# 定义函数用来获取订单的配送日期
def get_delivery_date(order_id=None):
    """
    获取客户的订单配送日期。
    
    :param order_id: 客户的订单ID。
    :return: 配送日期字符串或者错误消息。
    """
    if order_id == None:
        return "没有订单号无法查询"
    else:
        print("get_delivery_date:这里应该替换查询方法函数，结果用return返回")
        return "2024-09-02"

# 从给定的字符串中提取函数调用部分
def get_response_call(tool_call_str):
    """
    从提供的字符串中使用正则表达式提取函数调用部分。
    
    :param tool_call_str: 包含函数调用的字符串。
    :return: 提取的函数调用字符串或None。
    """
    # 正则表达式
    pattern = r'(?<=```python\n)(.*?)(?=\n```\n)'
    
    # 使用正则表达式匹配
    match = re.search(pattern, tool_call_str)
    
    if match:
        function_call = match.group(1)
        return function_call
    else:
        return None

# 定义工具列表
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_delivery_date",  # 函数名，需要定义一个一样的python函数
            "description": "Get the delivery date for a customer's order. Call this whenever you need to know the delivery date, for example when a customer asks 'Where is my package'",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {  # 参数名
                        "type": "string",  # 参数类型
                        "description": "The customer's order ID.",  # 参数描述
                    },
                },
                "required": ["order_id"],  # 哪些是必须的
                "additionalProperties": False,
            },
        },
    }
]

# 消息列表初始化
messages = [
    {
        "role": "system",
        "content": "You are a helpful customer support assistant. Use the supplied tools to assist the user.",
    }
]

# 用户查询
query="Hi, can you tell me the delivery date for my order，my order id is 123456。"

# 加载预训练模型的分词器
tokenizer = AutoTokenizer.from_pretrained(
    "/root/ld/ld_model_pretrained/minicpm3", trust_remote_code=True
)

# 构建提示模板
prompt = tokenizer.apply_chat_template(
    messages, tools=tools, tokenize=False, add_generation_prompt=True
)

# 加载预训练模型
model=AutoModelForCausalLM.from_pretrained("/root/ld/ld_model_pretrained/minicpm3", trust_remote_code=True).cuda()

# 聊天交互，获取响应
response,history=model.chat(tokenizer, query=query,history=messages,do_sample=False)#由于functioncall的精确性，这里建议将do_sample设置为False

# 获取函数调用字符串
call_str=get_response_call(response)

# 执行函数调用并打印结果
print(eval(call_str))
# 输出: 2024-09-02
```