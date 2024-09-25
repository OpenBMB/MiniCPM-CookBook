# MiniCPMV Training Environment Introduction and Steps

## 1. Training Environment Overview
The training environment setup can be found in: [pip list](./pip_list.md)

## 2. Obtaining MiniCPMV GitHub Code
Clone the MiniCPMV project to your local machine via Git:
```bash
git clone https://github.com/OpenBMB/MiniCPM-V.git
```

## 3. Installing Dependencies
Enter the project directory and install the required Python dependencies:
```bash
cd MiniCPM-V
pip install -r requirements.txt
```

## 4. Preparing the Dataset
Process the dataset to meet the following format requirements:

### Single Image (One Image Per Conversation)
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
    }
    ...
]
```

### Multiple Images (Multiple Images in One Conversation)
```json
[
    {
        "id": "0",
        "image": {
            "<image_00>": "path/to/image_0.jpg",
            "<image_01>": "path/to/image_1.jpg",
            "<image_02>": "path/to/image_2.jpg",
            "<image_03>": "path/to/image_3.jpg"
        },
        "conversations": [
            {
                "role": "user",
                "content": "How to create such text-only videos using CapCut?\n<image_00>\n<image_01>\n<image_02>\n<image_03>\n"
            },
            {
                "role": "assistant",
                "content": "To create a text-only video as shown in the images, follow these steps in CapCut..."
            }
        ]
    }
]
```

## 5. LoRA Fine-Tuning

### 5.1 Modifying `finetune_lora.sh`
Modify the `MiniCPM-V/finetune/finetune_lora.sh` script to suit the LoRA fine-tuning requirements. If you need to fine-tune an int4 model, make the following adjustments to the script:

```bash
#!/bin/bash
GPUS_PER_NODE=8 # Change to the number of GPUs per node on your machine, 8 for a single 8-GPU machine
NNODES=1 # Change to the number of nodes, 1 for a single server
NODE_RANK=0 # Rank of the server being used
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL="/root/ld/ld_model_pretrained/Minicpmv2_6" # Local model path or Hugging Face ID
DATA="/root/ld/ld_project/MiniCPM-V/finetune/mllm_demo.json" # Path to the training data file
EVAL_DATA="/root/ld/ld_project/MiniCPM-V/finetune/mllm_demo.json" # Path to the evaluation data file
LLM_TYPE="qwen2" # If using openbmb/MiniCPM-V-2, set LLM_TYPE=minicpm

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
    --label_names "labels" \ 
    --prediction_loss_only false \ 
    --bf16 false \ 
    --bf16_full_eval false \ 
    --fp16 true \ 
    --fp16_full_eval true \ 
    --do_train \ 
    --do_eval \ 
    --tune_vision true \ 
    --tune_llm false \ 
    --use_lora true \ 
    --lora_target_modules "llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)" \ 
    --model_max_length 2048 \ 
    --max_slice_nums 9 \ 
    --max_steps 10000 \ 
    --eval_steps 1000 \ 
    --output_dir output/output_minicpmv2_lora \ 
    --logging_dir output/output_minicpmv2_lora \ 
    --logging_strategy "steps" \ 
    --per_device_train_batch_size 2 \ 
    --per_device_eval_batch_size 1 \ 
    --gradient_accumulation_steps 8 \ 
    --evaluation_strategy "steps" \ 
    --save_strategy "steps" \ 
    --save_steps 10 \ 
    --save_total_limit 10 \ 
    --learning_rate 1e-6 \ 
    --weight_decay 0.1 \ 
    --adam_beta2 0.95 \ 
    --warmup_ratio 0.01 \ 
    --lr_scheduler_type "cosine" \ 
    --logging_steps 1 \
    --gradient_checkpointing true \ 
    --deepspeed ds_config_zero3.json \ 
    --report_to "tensorboard"
```

### 5.2 Key Parameters to Focus On
- `MODEL`: Path to the local model or Hugging Face ID.
- `DATA`: Path to the training data file.
- `EVAL_DATA`: Path to the evaluation data file.

- `--tune_vision true`: Whether to fine-tune the SigLIP (ViT) module.
- `--lora_target_modules`: Layers for LoRA insertion, specified using a regular expression.
- `--tune_llm false`: Whether to fine-tune the large language model module.
- `--use_lora true`: Whether to perform LoRA fine-tuning.

- `--model_max_length 2048`: Maximum length for model training.
- `--per_device_train_batch_size 2`: Batch size per device for training.
- `--per_device_eval_batch_size 1`: Batch size per device for evaluation.
- `--gradient_accumulation_steps 1`: Gradient accumulation; increase this parameter to reduce `per_device_train_batch_size` when GPU memory is limited.
- `--learning_rate 1e-6`: Learning rate.
- `--gradient_checkpointing true`: Gradient checkpointing, recommended to enable to significantly reduce GPU memory usage.
- `--deepspeed ds_config_zero3.json`: Use Zero3; recommend `ds_config_zero2.json` if GPU memory is sufficient.

### 5.3 Starting the Training
Navigate to the directory containing the fine-tuning script and run the script to start training:
```bash
cd MiniCPM-V/finetune
bash finetune_lora.sh
```

## 5.4 Merging LoRA with the Base Model
Use the following script to merge the LoRA model into the base model and save the merged model:

```python
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer
import os
import shutil

# Specify the base model path
model_type = "/root/ld/ld_model_pretrained/Minicpmv2_6"  
# Path to the saved LoRA adapter
path_to_adapter = "/root/ld/ld_project/minicpmv2_6/MiniCPM-V/finetune/output/output_minicpmv2_lora/checkpoint-30"  
# Path to save the merged model
merge_path = "/root/ld/ld_project/minicpmv2_6/MiniCPM-V/finetune/output/merge_minicpmv"  

# Ensure all files from the original model are copied to merge_path
def copy_files_not_in_B(A_path, B_path):
    """
    Copies files from directory A to directory B if they exist in A but not in B.

    :param A_path: Path to the source directory (A).
    :param B_path: Path to the destination directory (B).
    """
    if not os.path.exists(A_path):
        raise FileNotFoundError(f"The directory {A_path} does not exist.")
    if not os.path.exists(B_path):
        os.makedirs(B_path)

    files_in_A = os.listdir(A_path)
    files_in_A = set([file for file in files_in_A if not (".bin" in file or "safetensors" in file)])
    files_in_B = set(os.listdir(B_path))

    files_to_copy = files_in_A - files_in_B

    for file in files_to_copy:
        src_file = os.path.join(A_path, file)
        dst_file = os.path.join(B_path, file)
        shutil.copy2(src_file, dst_file)

# Load the base model
model = AutoModel.from_pretrained(
    model_type,
    trust_remote_code=True
)

# Load the LoRA module into the base model
lora_model = PeftModel.from_pretrained(
    model,
    path_to_adapter,
    device_map="auto",
    trust_remote_code=True
).eval()

# Merge the loaded LoRA module into the base model
merge_model = lora_model.merge_and_unload()

# Save the newly merged model
merge_model.save_pretrained(merge_path, safe_serialization=False)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)
tokenizer.save_pretrained(merge_path)

# Copy other files from the base model to the merged path
copy_files_not_in_B(model_type, merge_path)
```

## 6. Full Fine-Tuning

### 6.1 Modifying `finetune_ds.sh` Parameters
Modify the `MiniCPM-V/finetune/finetune_ds.sh` script to suit the full fine-tuning requirements:

```bash
#!/bin/bash

GPUS_PER_NODE=8 # Change to the number of GPUs per node on your machine, 8 for a single 8-GPU machine
NNODES=1 # Change to the number of nodes, 1 for a single server
NODE_RANK=0 # Rank of the server being used
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL="/root/ld/ld_model_pretrained/Minicpmv2_6" # Local model path or Hugging Face ID
# ATTENTION: Specify the path to your training data, which should be a JSON file consisting of a list of conversations.
# Refer to the finetuning section in the README for more information.
DATA="/root/ld/ld_project/MiniCPM-V/finetune/mllm_demo.json" # Path to the training data file
EVAL_DATA="/root/ld/ld_project/MiniCPM-V/finetune/mllm_demo.json" # Path to the evaluation data file
LLM_TYPE="qwen2" # If using openbmb/MiniCPM-V-2, set LLM_TYPE=minicpm

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
    --gradient_checkpointing true \ # Gradient checkpointing, recommended to enable to significantly reduce GPU memory usage
    --deepspeed ds_config_zero3.json \ # Use Zero3, recommend `ds_config_zero3.json` if GPU memory is sufficient
    --report_to "tensorboard" # wandb # Record loss using tensorboard or wandb
```

### 6.2 Starting the Training
Navigate to the directory containing the fine-tuning script and run the script to start training:
```bash
cd MiniCPM-V/finetune
bash finetune_ds.sh
```

Make sure to correctly configure your environment and verify that all paths point to the correct files and directories before running the scripts.
