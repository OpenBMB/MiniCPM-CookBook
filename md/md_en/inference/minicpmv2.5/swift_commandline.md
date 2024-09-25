# Swift Command Line Inference

## System Requirements
- Total GPU memory across all cards must be at least 24GB

## Step 1: Install Swift

Clone the Swift repository and install dependencies:

```sh
git clone https://github.com/modelscope/swift.git
cd swift
pip install -r requirements.txt
pip install -e '.[llm]'
```

## Step 2: Quick Start Code

The following command will automatically download the `minicpm-v-v2_5` model from the ModelScope community and load the default generation parameters:

```sh
CUDA_VISIBLE_DEVICES=0 swift infer --model_type minicpm-v-v2_5-chat
```

## Common Parameters

- `model_id_or_path`: Can be a Hugging Face model ID or a local model path.
- `infer_backend`: Inference backend, options are `['AUTO', 'vllm', 'pt']`, default is `AUTO`.
- `dtype`: Computation precision, options are `['bf16', 'fp16', 'fp32', 'AUTO']`.
- `max_length`: Maximum length.
- `max_new_tokens`: Maximum number of tokens to generate, default is 2048.
- `do_sample`: Whether to sample, default is `True`.
- `temperature`: Temperature coefficient for generation, default is 0.3.
- `top_k`: Default is 20.
- `top_p`: Default is 0.7.
- `repetition_penalty`: Default is 1.0.
- `num_beams`: Default is 1.
- `stop_words`: List of stop words, default is `None`.
- `quant_method`: Quantization method for the model, options are `['bnb', 'hqq', 'eetq', 'awq', 'gptq', 'aqlm']`.
- `quantization_bit`: Number of bits for quantization, default is 0 (no quantization).

## Example

```sh
CUDA_VISIBLE_DEVICES=0,1 swift infer --model_type minicpm-v-v2_5-chat --model_id_or_path /root/ld/ld_model_pretrain/MiniCPM-Llama3-V-2_5 --dtype bf16
```
