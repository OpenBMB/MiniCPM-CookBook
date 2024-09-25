Below is a simple Python script translated into English and formatted as Markdown. This script demonstrates how to call a specific function to retrieve an order's delivery date. It includes a function `get_delivery_date`, a processing function `get_response_call`, and a defined list `tools` containing a description of the `get_delivery_date` function and its parameters.


# Simple Function Call Implementation (Minicpm3.0)

The following is a simple Python script to demonstrate how to call a specific function to get an order's delivery date. This script contains a function `get_delivery_date` and a function `get_response_call` that processes the function calls. Additionally, it defines a `tools` list that includes a description of the `get_delivery_date` function and its parameters.

```python
#!/usr/bin/env python
# encoding: utf-8
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define a function to get the delivery date of an order
def get_delivery_date(order_id=None):
    """
    Get the delivery date for a customer's order.
    
    :param order_id: The customer's order ID.
    :return: A string of the delivery date or an error message.
    """
    if order_id is None:
        return "Cannot query without an order number"
    else:
        print("get_delivery_date: This should be replaced with the actual query method, the result should be returned using return")
        return "2024-09-02"

# Extract the function call part from the given string
def get_response_call(tool_call_str):
    """
    Use regular expressions to extract the function call part from the provided string.
    
    :param tool_call_str: A string containing the function call.
    :return: The extracted function call string or None.
    """
    # Regular expression
    pattern = r'(?<=```python\n)(.*?)(?=\n```\n)'
    
    # Match using regular expression
    match = re.search(pattern, tool_call_str, re.DOTALL)
    
    if match:
        function_call = match.group(1)
        return function_call
    else:
        return None

# Define the tools list
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_delivery_date",  # Function name, a matching Python function must be defined
            "description": "Get the delivery date for a customer's order. Call this whenever you need to know the delivery date, for example when a customer asks 'Where is my package'",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {  # Parameter name
                        "type": "string",  # Parameter type
                        "description": "The customer's order ID.",  # Parameter description
                    },
                },
                "required": ["order_id"],  # Which ones are required
                "additionalProperties": False,
            },
        },
    }
]

# Initialize the messages list
messages = [
    {
        "role": "system",
        "content": "You are a helpful customer support assistant. Use the supplied tools to assist the user.",
    }
]

# User query
query = "Hi, can you tell me the delivery date for my order, my order id is 123456."

# Load the tokenizer of the pre-trained model
tokenizer = AutoTokenizer.from_pretrained(
    "/root/ld/ld_model_pretrained/minicpm3", trust_remote_code=True
)

# Build the prompt template
prompt = tokenizer.apply_chat_template(
    messages, tools=tools, tokenize=False, add_generation_prompt=True
)

# Load the pre-trained model
model = AutoModelForCausalLM.from_pretrained("/root/ld/ld_model_pretrained/minicpm3", trust_remote_code=True).cuda()

# Chat interaction, get response
response, history = model.chat(tokenizer, query=query, history=messages, do_sample=False)  # For the accuracy of function calls, it is recommended to set do_sample to False here

# Get the function call string
call_str = get_response_call(response)

# Execute the function call and print the result
print(eval(call_str))
# Output: 2024-09-02
```

This script is designed to simulate a customer service scenario where a user inquires about the delivery status of their order. By utilizing the `get_delivery_date` function and processing the response through `get_response_call`, the script demonstrates how to integrate function calls within a conversational AI application.
