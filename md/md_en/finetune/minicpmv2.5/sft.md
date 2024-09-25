# Official Code Model Training Guide

## 1. Install Dependencies

First, enter the project directory and install the dependencies:
```sh
cd MiniCPM-V
pip install -r requirements.txt
```
**Note**: It is recommended to download the latest version from the DeepSpeed official website and install it using `pip install e .`.

## 2. Preprocess the Dataset

Preprocess the dataset to match the following format:
- `id` values must be unique.
- `<image>\n` should appear at the beginning of each conversation in the dataset.
- The address specified by `"image"` must correspond to an existing image.
- Each `conversations` list represents a multi-turn dialogue, where `content` is the dialogue content, `role` with `user` represents user input, and `role` with `assistant` represents the model's output.
- Each data entry contains only one image.

Example data format:
```json
[
    {
        "id": "0",
        "conversations": [
            {
                "content": "<image>\nWho are they?",
                "role": "user"
            },
            {
                "content": "They're Kane and Gretzka from Bayern Munich.",
                "role": "assistant"
            },
            {
                "content": "What are they doing?",
                "role": "user"
            },
            {
                "content": "They are celebrating on the soccer field.",
                "role": "assistant"
            }
        ],
        "image": "/root/ld/ld_project/LLaMA-Factory/data/mllm_demo_data/1.jpg"
    },
    // Multiple entries...
]
```

## 3. LoRA Fine-Tuning

### 3.1 Modify `MiniCPM-V/finetune/finetune_lora.sh`

```sh
#!/bin/bash
GPUS_PER_NODE=8 # Change to the number of GPUs per node on your machine, 8 for a single 8-GPU machine
NNODES=1 # Change to the number of nodes, 1 for a single server
NODE_RANK=0 # Rank of the server being used
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL="/root/ld/ld_model_pretrained/MiniCPM-Llama3-V-2_5" # Local model path or openbmb/MiniCPM-V-2.5
# ATTENTION: Specify the path to your training data, which should be a JSON file consisting of a list of conversations.
# Refer to the finetuning section in the README for more information.
DATA="/root/ld/ld_project/MiniCPM-V/finetune/mllm_demo.json" # Path to the training data file
EVAL_DATA="/root/ld/ld_project/MiniCPM-V/finetune/mllm_demo.json" # Path to the evaluation data file
LLM_TYPE="llama3" # If using openbmb/MiniCPM-V-2, set LLM_TYPE=minicpm

export NCCL_P2P_DISABLE=1 # Remove this line for GPUs like A100 that support nccl_p2p
export NCCL_IB_DISABLE=1 # Remove this line for GPUs like A100

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
torchrun $DISTRIBUTED_ARGS finetune.py  \
    --model_name_or_path $MODEL \
    --llm_type $LLM_TYPE \
    --data_path $DATA \
    --eval_data_path $EVAL_DATA \
    --remove_unused_columns false \ 
    --label_names "labels" \ # Data construction, do not modify
    --prediction_loss_only false \ 
    --bf16 false \ # Use bf16 precision for training, enable for GPUs like 4090, A100, H100
    --bf16_full_eval false \ # Use bf16 precision for evaluation
    --fp16 true \ # Use fp16 precision for training
    --fp16_full_eval true \ # Use fp16 precision for evaluation
    --do_train \ # Whether to train
    --do_eval \ # Whether to evaluate during training
    --tune_vision true \ # Whether to fine-tune the SigLIP (ViT) module
    --tune_llm false \ # Whether to fine-tune the large language model module
    --use_lora true \ # Whether to use LoRA fine-tuning
    --lora_target_modules "llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)" \ # Layers for LoRA insertion, regular expression, suggest not to modify
    --model_max_length 2048 \ # Maximum length for model training
    --max_slice_nums 9 \ # Maximum number of slices for the model
    --max_steps 10000 \ # Maximum number of training steps
    --eval_steps 1000 \ # Evaluate every 1000 steps
    --output_dir output/output_minicpmv2_lora \ # Directory to save the LoRA model
    --logging_dir output/output_minicpmv2_lora \ # Directory to save logs
    --logging_strategy "steps" \ # Logging strategy (can be 'epoch')
    --per_device_train_batch_size 2 \ # Batch size per device for training
    --per_device_eval_batch_size 1 \ # Batch size per device for evaluation
    --gradient_accumulation_steps 8 \ # Gradient accumulation, increase this parameter to reduce `per_device_train_batch_size` when GPU memory is limited
    --evaluation_strategy "steps" \ # Evaluation strategy (can be 'epoch')
    --save_strategy "steps" \ # Saving strategy (can be 'epoch') works with `save_steps`
    --save_steps 10 \ # Save every 10 steps
    --save_total_limit 10 \ # Maximum number of checkpoints to keep
    --learning_rate 1e-6 \ # Learning rate
    --weight_decay 0.1 \ # Weight decay parameter
    --adam_beta2 0.95 \ 
    --warmup_ratio 0.01 \ # Warm-up ratio, i.e., total training steps * warmup_ratio = warm-up steps
    --lr_scheduler_type "cosine" \ # Learning rate scheduler type
    --logging_steps 1 \
    --gradient_checkpointing true \ # Gradient checkpointing, recommended to enable to significantly reduce GPU memory usage
    --deepspeed ds_config_zero3.json \ # Use Zero3, recommend `ds_config_zero2.json` if GPU memory is sufficient
    --report_to "tensorboard" # wandb # Record loss using tensorboard or wandb
```

### 3.2 Key Parameters to Focus On

- `MODEL`: Local model path or remote model path.
- `DATA`: Path to the training data file.
- `EVAL_DATA`: Path to the evaluation data file.
- `--tune_vision true`: Whether to fine-tune the SigLIP (ViT) module.
- `--lora_target_modules "llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"`: Layers for LoRA insertion, regular expression, suggest not to modify.
- `--tune_llm false`: Whether to fine-tune the large language model module.
- `--use_lora true`: Whether to use LoRA fine-tuning.
- `--model_max_length 2048`: Maximum length for model training (1000 + text length / 1.5).
- `--per_device_train_batch_size 2`: Batch size per device for training.
- `--per_device_eval_batch_size 1`: Batch size per device for evaluation.
- `--gradient_accumulation_steps 1`: Gradient accumulation, increase this parameter to reduce `per_device_train_batch_size` when GPU memory is limited.
- `--learning_rate 1e-6`: Learning rate.
- `--gradient_checkpointing true`: Gradient checkpointing, recommended to enable to significantly reduce GPU memory usage.
- `--deepspeed ds_config_zero3.json`: Use Zero3, recommend `ds_config_zero2.json` if GPU memory is sufficient.

### 3.3 Start Training

Navigate to the `MiniCPM-V/finetune` directory and execute the training script:
```sh
cd MiniCPM-V/finetune
bash finetune_lora.sh
```

### 3.4 Merge LoRA with the Base Model

```python
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

model_type="/root/ld/ld_model_pretrained/MiniCPM-Llama3-V-2_5" # Local model path or openbmb/MiniCPM-V-2.5
path_to_adapter="/root/ld/ld_project/MiniCPM-V/finetune/output/output_minicpmv2_lora/checkpoint-400" # Path to the saved LoRA adapter
merge_path="/root/ld/ld_project/MiniCPM-V/finetune/output/merge_model_path" # Path to save the merged model

model = AutoModel.from_pretrained(
    model_type,
    trust_remote_code=True
)
# Load the LoRA adapter
lora_model = PeftModel.from_pretrained(
    model,
    path_to_adapter,
    device_map="auto",
    trust_remote_code=True
).eval().cuda()

# Merge the LoRA adapter into the base model, resulting in a model with the same shape as the original MiniCPM-Llama3-V-2_5
merge_model = lora_model.merge_and_unload()
# Save the new model, which has the same shape as the original MiniCPM-Llama3-V-2_5
merge_model.save_pretrained(merge_path, safe_serialization=False)

# Load the tokenizer and save it to the merged model directory
tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)
tokenizer.save_pretrained(merge_path)
```
# QLoRA Fine-Tuning Guide (For Low-Memory Devices)

## 4. Obtain Quantized Models

### Method One: Direct Download

- **Hugging Face**
  ```sh
  git clone https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-int4
  ```
- **ModelScope**
  ```sh
  git clone https://modelscope.cn/models/OpenBMB/MiniCPM-Llama3-V-2_5-int4
  ```

### Method Two: Quantize the Model Yourself

## 4.1 Quantize the Model According to the [bnb Quantization Tutorial](../../quantize/minicpmv2.5/bnb.md)
## 4.2 Modify `MiniCPM-V/finetune/finetune_lora.sh`

```sh
#!/bin/bash
GPUS_PER_NODE=8 # Change to the number of GPUs per node on your machine, 8 for a single 8-GPU machine
NNODES=1 # Change to the number of nodes, 1 for a single server
NODE_RANK=0 # Rank of the server being used
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL="/root/ld/ld_model_pretrained/MiniCPM-Llama3-V-2_5" # Local model path or openbmb/MiniCPM-V-2.5
# ATTENTION: Specify the path to your training data, which should be a JSON file consisting of a list of conversations.
# Refer to the finetuning section in the README for more information.
DATA="/root/ld/ld_project/MiniCPM-V/finetune/mllm_demo.json" # Path to the training data file
EVAL_DATA="/root/ld/ld_project/MiniCPM-V/finetune/mllm_demo.json" # Path to the evaluation data file
LLM_TYPE="llama3" # If using openbmb/MiniCPM-V-2, set LLM_TYPE=minicpm

export NCCL_P2P_DISABLE=1 # Remove this line for GPUs like A100 that support nccl_p2p
export NCCL_IB_DISABLE=1 # Remove this line for GPUs like A100

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
torchrun $DISTRIBUTED_ARGS finetune.py  \
    --model_name_or_path $MODEL \
    --llm_type $LLM_TYPE \
    --data_path $DATA \
    --eval_data_path $EVAL_DATA \
    --remove_unused_columns false \ 
    --label_names "labels" \ # Data construction, do not modify
    --prediction_loss_only false \ 
    --bf16 false \ # Use bf16 precision for training, enable for GPUs like 4090, A100, H100
    --bf16_full_eval false \ # Use bf16 precision for evaluation
    --fp16 true \ # Use fp16 precision for training
    --fp16_full_eval true \ # Use fp16 precision for evaluation
    --do_train \ # Whether to train
    --do_eval \ # Whether to evaluate during training
    --tune_vision false \ # Whether to fine-tune the SigLIP (ViT) module
    --tune_llm false \ # Whether to fine-tune the large language model module
    --use_lora true \ # Whether to use LoRA fine-tuning
    --q_lora true \
    --lora_target_modules "llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)" \ # Layers for LoRA insertion, regular expression, suggest not to modify
    --model_max_length 2048 \ # Maximum length for model training
    --max_slice_nums 9 \ # Maximum number of slices for the model
    --max_steps 10000 \ # Maximum number of training steps
    --eval_steps 1000 \ # Evaluate every 1000 steps
    --output_dir output/output_minicpmv2_lora \ # Directory to save the LoRA model
    --logging_dir output/output_minicpmv2_lora \ # Directory to save logs
    --logging_strategy "steps" \ # Logging strategy (can be 'epoch')
    --per_device_train_batch_size 2 \ # Batch size per device for training
    --per_device_eval_batch_size 1 \ # Batch size per device for evaluation
    --gradient_accumulation_steps 8 \ # Gradient accumulation, increase this parameter to reduce `per_device_train_batch_size` when GPU memory is limited
    --evaluation_strategy "steps" \ # Evaluation strategy (can be 'epoch')
    --save_strategy "steps" \ # Saving strategy (can be 'epoch') works with `save_steps`
    --save_steps 10 \ # Save every 10 steps
    --save_total_limit 10 \ # Maximum number of checkpoints to keep
    --learning_rate 1e-6 \ # Learning rate
    --weight_decay 0.1 \ # Weight decay parameter
    --adam_beta2 0.95 \ 
    --warmup_ratio 0.01 \ # Warm-up ratio, i.e., total training steps * warmup_ratio = warm-up steps
    --lr_scheduler_type "cosine" \ # Learning rate scheduler type
    --logging_steps 1 \
    --gradient_checkpointing true \ # Gradient checkpointing, recommended to enable to significantly reduce GPU memory usage
    --deepspeed ds_config_zero3.json \ # Use Zero3, recommend `ds_config_zero2.json` if GPU memory is sufficient
    --report_to "tensorboard" # wandb # Record loss using tensorboard or wandb
```

## 5. Full Fine-Tuning

### 5.1 Modify `MiniCPM-V/finetune/finetune_ds.sh` Parameters

```bash
#!/bin/bash

GPUS_PER_NODE=8 # Change to the number of GPUs per node on your machine, 8 for a single 8-GPU machine
NNODES=1 # Change to the number of nodes, 1 for a single server
NODE_RANK=0 # Rank of the server being used
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL="/root/ld/ld_model_pretrained/MiniCPM-Llama3-V-2_5" # Local model path or openbmb/MiniCPM-V-2
# ATTENTION: Specify the path to your training data, which should be a JSON file consisting of a list of conversations.
# Refer to the finetuning section in the README for more information.
DATA="/root/ld/ld_project/MiniCPM-V/finetune/mllm_demo.json" # Path to the training data file
EVAL_DATA="/root/ld/ld_project/MiniCPM-V/finetune/mllm_demo.json" # Path to the evaluation data file
LLM_TYPE="llama3" # If using openbmb/MiniCPM-V-2, set LLM_TYPE=minicpm

export NCCL_P2P_DISABLE=1 # Remove this line for GPUs like A100 that support nccl_p2p
export NCCL_IB_DISABLE=1 # Remove this line for GPUs like A100

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
torchrun $DISTRIBUTED_ARGS finetune.py  \
    --model_name_or_path $MODEL \
    --llm_type $LLM_TYPE \
    --data_path $DATA \
    --eval_data_path $EVAL_DATA \
    --remove_unused_columns false \
    --label_names "labels" \ # Data construction, do not modify
    --prediction_loss_only false \ 
    --bf16 false \ # Use bf16 precision for training, enable for GPUs like 4090, A100, H100
    --bf16_full_eval false \ # Use bf16 precision for evaluation
    --fp16 true \ # Use fp16 precision for training
    --fp16_full_eval true \ # Use fp16 precision for evaluation
    --do_train \ # Whether to train
    --do_eval \ # Whether to evaluate during training
    --tune_llm true \ # Whether to fine-tune the large language model module
    --tune_vision true \ # Whether to fine-tune the vision module
    --model_max_length 2048 \ # Maximum length for model training
    --max_slice_nums 9 \ # Maximum number of slices for the model
    --max_steps 10000 \ # Maximum number of training steps
    --eval_steps 1000 \ # Evaluate every 1000 steps
    --output_dir output/output_minicpmv2_lora \ # Directory to save the LoRA model
    --logging_dir output/output_minicpmv2_lora \ # Directory to save logs
    --logging_strategy "steps" \ # Logging strategy (can be 'epoch')
    --per_device_train_batch_size 2 \ # Batch size per device for training
    --per_device_eval_batch_size 1 \ # Batch size per device for evaluation
    --gradient_accumulation_steps 1 \ # Gradient accumulation, increase this parameter to reduce `per_device_train_batch_size` when GPU memory is limited
    --evaluation_strategy "steps" \ # Evaluation strategy (can be 'epoch')
    --save_strategy "steps" \ # Saving strategy (can be 'epoch') works with `save_steps`
    --save_steps 10 \ # Save every 10 steps
    --save_total_limit 10 \ # Maximum number of checkpoints to keep
    --learning_rate 1e-6 \ # Learning rate
    --weight_decay 0.1 \ # Weight decay parameter
    --adam_beta2 0.95 \ 
    --warmup_ratio 0.01 \ # Warm-up ratio, i.e., total training steps * warmup_ratio = warm-up steps
    --lr_scheduler_type "cosine" \ # Learning rate scheduler type
    --logging_steps 1 \
    --gradient_checkpointing true \  # Gradient checkpointing, recommended to enable to significantly reduce GPU memory usage
    --deepspeed ds_config_zero3.json \ # Use Zero3, recommend `ds_config_zero3.json` if GPU memory is sufficient
    --report_to "tensorboard" # wandb # Record loss using tensorboard or wandb
```

### 5.2 Start Training

Navigate to the `MiniCPM-V/finetune` directory and execute the training script:
```sh
cd MiniCPM-V/finetune
bash finetune_ds.sh
```
