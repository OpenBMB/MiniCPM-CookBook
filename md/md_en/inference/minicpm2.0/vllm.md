
### Install vLLM

First, ensure that the `vllm` library is installed:

```bash
pip install vllm
```

### Python Script Example

Next, use `vllm` in a Python script for text generation:

```python
from vllm import LLM, SamplingParams
import argparse

# Create a command-line argument parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("--model_path", type=str, default="")

# Parse command-line arguments
args = parser.parse_args()

prompts = ["What's your name.", "Which is the highest mountain in the world?"]  # prompts is a list where each element is a prompt text to input

# Format the prompt template
prompt_template = "<User>{}<AI>"

# Apply the template to each prompt
prompts = [prompt_template.format(prompt.strip()) for prompt in prompts]

# Set sampling parameters
params_dict = {
    "n": 1,
    "best_of": 1,
    "presence_penalty": 1.0,
    "frequency_penalty": 0.0,
    "temperature": 0.5,
    "top_p": 0.8,
    "top_k": -1,
    "use_beam_search": False,
    "length_penalty": 1,
    "early_stopping": False,
    "stop": None,
    "stop_token_ids": None,
    "ignore_eos": False,
    "max_tokens": 1000,
    "logprobs": None,
    "prompt_logprobs": None,
    "skip_special_tokens": True,
}

# Create a sampling parameters object
sampling_params = SamplingParams(**params_dict)

# Create an LLM object
llm = LLM(model=args.model_path, tensor_parallel_size=1, dtype='bfloat16')

# Generate text from prompts. The output is a list of RequestOutput objects,
# which contain the prompt, generated text, and other information.
for prompt in prompts:
    outputs = llm.generate(prompt, sampling_params)
    # Print the output
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print("================")
        # Find the first <User> and remove the text before it.
        clean_prompt = prompt[prompt.find("<User>") + len("<User>"):]
        print(f"""<User>: {clean_prompt.replace("<AI>", "")}""")
        print(f"<AI>:")
        print(generated_text)
```

Make sure to set the `--model_path` parameter correctly before running this script.
