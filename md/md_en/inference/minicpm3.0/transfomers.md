# MiniCPM 3.0 Usage Examples

## Chat Method

The following code example demonstrates how to implement a chat feature with the MiniCPM 3.0 model using the `transformers` library:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("/root/ld/ld_model_pretrained/minicpm3")
model = AutoModelForCausalLM.from_pretrained("/root/ld/ld_model_pretrained/minicpm3", trust_remote_code=True).cuda()

# Initialize the history list
history = []

while True:
    # Get user input
    query = input("user:")
    
    # Generate a response and update the history
    response, history = model.chat(tokenizer, query=query, history=history)
    
    # Print the model's response
    print("model:", response)
    
# Note: `history` is a list containing the history of the conversation, formatted as follows:
# history = [{"role": "assistant", "content": answer1}, {"role": "assistant", "content": response}]
```

## Generate Method

The following is a simple example of generating text using the `transformers` library:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the pre-trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained("/root/ld/ld_model_pretrained/minicpm3", trust_remote_code=True).cuda()
tokenizer = AutoTokenizer.from_pretrained("/root/ld/ld_model_pretrained/minicpm3", trust_remote_code=True)

# Define the prompt
prompt = "Hey, are you conscious? Can you tell me "

# Encode the prompt
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text
generate_ids = model.generate(inputs.input_ids.cuda(), max_length=300, do_sample=False)

# Decode the generated ID sequence
output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# Print the output
print(output)
```

The code snippets above provide two different ways to interact with the MiniCPM 3.0 model: one is a chat feature based on conversation history, and the other is simple text generation. We hope these examples help you better utilize the capabilities of the MiniCPM 3.0 model.
