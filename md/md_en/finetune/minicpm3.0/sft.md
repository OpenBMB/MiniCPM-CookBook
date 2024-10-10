# Official Code Fine-Tuning (SFT Recommended)

## Device Requirements
- At least one GPU with 24GB of VRAM, Series 20 or above
- For QLoRA, a GPU with 12GB of VRAM can be attempted

## Steps

### 1. Obtain the Official Code Using Git
```bash
git clone https://github.com/OpenBMB/MiniCPM
```

### 2. Prepare the Dataset

#### 2.1 Dialogue Dataset
Process it into the following JSON format, where each entry represents a `messages` field. The system field marked in red is not mandatory.
```json
[
  {
    "messages": [
      {
        "role": "system",
        "content": "<system prompt text>"
      },
      {
        "role": "user",
        "content": "<user prompt text>"
      },
      {
        "role": "assistant",
        "content": "<assistant response text>"
      },
      // ... Multi-turn dialogues
      {
        "role": "user",
        "content": "<user prompt text>"
      },
      {
        "role": "assistant",
        "content": "<assistant response text>"
      }
    ]
  }
]
```

#### 2.2 Tool Invocation Dataset
Refer to cold start for obtaining function call data.

### 3. Modify the `MiniCPM/finetune/lora_finetune_ocnli.sh` File
```bash
formatted_time=$(date +"%Y%m%d%H%M%S")
echo $formatted_time

# Add these two lines for 4090
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1 

deepspeed --include localhost:1 --master_port 19888 finetune.py \
    --model_name_or_path openbmb/MiniCPM3-4B \ # Can be modified to the local model directory or the 1B model address
    --output_dir output/OCNLILoRA/$formatted_time/ \ # Can be modified to another directory for saving the output model
    --train_data_path data/ocnli_public_chatml/train.json \ # Specify the path to the training set processed as per step 2
    --eval_data_path data/ocnli_public_chatml/dev.json \ # Specify the path to the validation set processed as per step 2
    --learning_rate 5e-5 \ # Learning rate
    --per_device_train_batch_size 16 \ # Batch size for training per device
    --per_device_eval_batch_size 128 \ # Batch size for evaluation per device
    --model_max_length 1024 \ # Maximum token length for training, will truncate if exceeded
    --bf16 \ # Whether to use bf16 data format, change to false if not applicable
    --use_lora \ # Whether to use LoRA
    --gradient_accumulation_steps 1 \ # Gradient accumulation steps
    --warmup_steps 100 \ # Warm-up steps
    --max_steps 1000 \ # Maximum number of training steps, stop training when reached
    --weight_decay 0.01 \ # Weight decay value
    --evaluation_strategy steps \ # Evaluation method, can be changed to epoch
    --eval_steps 500 \ # Works with evaluation_strategy steps, evaluate every 500 steps
    --save_strategy steps \ # Model saving strategy, can be changed to epoch for saving once per epoch
    --save_steps 500 \ # Works with save_strategy steps, save every 500 steps
    --seed 42 \ # Random seed
    --log_level info --logging_strategy steps --logging_steps 10 \ # Logging settings
    --deepspeed configs/ds_config_zero2_offload.json # DeepSpeed configuration file setting, can be changed to configs/ds_config_zero2_offload.json if there's sufficient VRAM
```

### 4. (Optional) Train Using LoRA/QLoRA

#### 4.1 LoRA Usage
In the `MiniCPM/finetune/lora_finetune_ocnli.sh` file, add the `use_lora` parameter,
if unsure, add the following code before `--deepspeed configs/ds_config_zero2_offload.json`
```bash
use_lora \
```

#### 4.2 QLoRA Usage
In the `MiniCPM/finetune/lora_finetune_ocnli.sh` file, add both `use_lora` and `qlora` parameters,
if unsure, add the following code before `--deepspeed configs/ds_config_zero2_offload.json`
```bash
use_lora \
qlora \
```

### 5. Start Training
After modifying the bash file, run the following commands to start training:
```bash
cd MiniCPM/finetune
bash lora_finetune_ocnli.sh
```
