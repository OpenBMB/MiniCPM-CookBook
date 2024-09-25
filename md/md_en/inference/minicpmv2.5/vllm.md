# vLLM Deployment (Recommended for Concurrency)

## Step 1: Get the vLLM Code

Clone the vLLM repository using Git:

```sh
git clone https://github.com/vllm-project/vllm.git
```

## Step 2: Compile and Install vLLM

Navigate to the vLLM directory and install it:

```sh
cd vllm
pip install -e .
```

## Step 3: Install the Dependency Library `timm`

Install the `timm` library:

```sh
pip install timm==0.9.10
```

## Step 4: Copy the Following Script for Inference

```python
from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

IMAGES = [
    "./examples/images/375.jpg",
]

MODEL_NAME = "openbmb/MiniCPM-Llama3-V-2_5"  # Update to the latest model code if using a local model

image = Image.open(IMAGES[0]).convert("RGB")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
llm = LLM(model=MODEL_NAME,
          gpu_memory_utilization=1,
          trust_remote_code=True,
          max_model_len=4096)

messages = [{'role': 'user', 'content': '(<image>./</image>)\n' + 'What kind of wine is this?'}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# For version 2.5
stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]

sampling_params = SamplingParams(
    stop_token_ids=stop_token_ids,
    max_tokens=1024,
    temperature=0,
    use_beam_search=True,
    best_of=3)

outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": {
        "image": image
    }
}, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)
```

## Step 5: Implement Concurrent Inference

Edit the concurrent inference script:

```python
from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# All input images
IMAGES = [
    "./examples/images/375.jpg",
    "./examples/images/376.jpg",
    "./examples/images/377.jpg",
    "./examples/images/378.jpg"
]

MODEL_NAME = "openbmb/MiniCPM-Llama3-V-2_5"  # Update to the latest model code if using a local model

images = [Image.open(i).convert("RGB") for i in IMAGES]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
llm = LLM(model=MODEL_NAME,
          gpu_memory_utilization=1,
          trust_remote_code=True,
          max_model_len=4096)

messages = [{'role': 'user', 'content': '(<image>./</image>)\n' + 'Help me identify the content in the image?'}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Build multiple inputs; this example shares the same prompt, but you can also use different prompts
inputs = [{"prompt": prompt, "multi_modal_data": {"image": i}} for i in images]

# For version 2.5
stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]

sampling_params = SamplingParams(
    stop_token_ids=stop_token_ids,
    max_tokens=1024,
    temperature=0,
    use_beam_search=True,
    best_of=3)

outputs = llm.generate(inputs, sampling_params=sampling_params)

for i in range(len(inputs)):
    print(outputs[i].outputs[0].text)
```

Now you are ready to start efficient concurrent inference with vLLM!
