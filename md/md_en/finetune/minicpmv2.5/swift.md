# Swift Installation and Training Guide

## Installing Swift

First, clone the Swift repository:

```sh
git clone https://github.com/modelscope/swift.git
cd swift
pip install -r requirements.txt
pip install -e '.[llm]'
```

## Training Swift

### 1. Custom Dataset

The dataset should contain the following fields:
- `query`: Corresponds to the human's question in the current round of dialogue.
- `response`: The expected response to the human's question.
- `images`: Can be a local image path or a URL of an online image.

Example dataset entries:

```json
{
    "query": "What does this picture describe?",
    "response": "This picture has a giant panda.",
    "images": ["local_image_path"]
}
{
    "query": "What does this picture describe?",
    "response": "This picture has a giant panda.",
    "history": [],
    "images": ["image_path"]
}
{
    "query": "Is bamboo tasty?",
    "response": "It seems pretty good from the look of the panda.",
    "history": [["What's in this picture?", "This picture has a giant panda."], ["What is the panda doing?", "Eating bamboo."]],
    "images": ["image_url"]
}
```

### 2. LoRA Fine-Tuning

By default, Swift performs fine-tuning on the Q, K, V matrices of the language model. The `CUDA_VISIBLE_DEVICES` environment variable specifies the GPU devices to use. When fine-tuning with multiple GPUs, it is recommended to set `eval_steps` to a very high value to avoid out-of-memory issues.

```sh
CUDA_VISIBLE_DEVICES=0,1 swift sft \
    --model_type minicpm-v-v2_5-chat \
    --dataset local_data.jsonl \
    --eval_steps 200000
```

### 3. Full Fine-Tuning

To enable full fine-tuning, set `--lora_target_modules` to `ALL`.

```sh
CUDA_VISIBLE_DEVICES=0,1 swift sft \
    --model_type minicpm-v-v2_5-chat \
    --dataset coco-en-2-mini \
    --lora_target_modules ALL \
    --eval_steps 200000
```

## Inference and Testing

### 1. LoRA Inference

To perform inference using the trained model, specify the directory containing the model checkpoint.

```sh
CUDA_VISIBLE_DEVICES=0 swift infer    \
 --ckpt_dir /root/ld/ld_project/swift/output/minicpm-v-v2_5-chat/v9-20240705-171455/checkpoint-10
```

### 2. Merging LoRA Inference

This method merges the LoRA weights into the backbone network and saves them to the specified path.

```sh
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir /root/ld/ld_project/swift/output/minicpm-v-v2_5-chat/v9-20240705-171455/checkpoint-10 \
    --merge_lora true
```

### 3. Testing the Merged Model

Load the dataset from the configuration file to conduct testing.

```sh
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/minicpm-v-v2_5-chat/vx-xxx/checkpoint-xxx-merged \
    --load_dataset_config true
```
