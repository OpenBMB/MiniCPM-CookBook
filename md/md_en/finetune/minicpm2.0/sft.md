# MiniCPM Model Fine-Tuning Guide

## System Requirements
- At least one GPU with 12GB VRAM, NVIDIA 20 series or better
- When using QLoRA, try with 6-8GB GPUs

## Steps
1. **Clone the Official Repository Using Git**
   ```sh
   git clone https://github.com/OpenBMB/MiniCPM
   ```

2. **Prepare Your Dataset and Format It as JSON**
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
         // ... multiple rounds of conversation
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

3. **Modify the `MiniCPM/finetune/lora_finetune_ocnli.sh` File**
   ```sh
   formatted_time=$(date +"%Y%m%d%H%M%S")
   echo $formatted_time

   # Add these two lines for RTX 4090 GPUs
   export NCCL_P2P_DISABLE=1
   export NCCL_IB_DISABLE=1 

   deepspeed --include localhost:1 --master_port 19888 finetune.py \
       --model_name_or_path MiniCPM-2B-sft-bf16 \ # Can be modified to the local model directory or 1B model path
       --output_dir output/OCNLILoRA/$formatted_time/ \ # Can be modified to another directory for saving the output model
       --train_data_path data/ocnli_public_chatml/train.json \ # Path to the processed training dataset from step 2
       --eval_data_path data/ocnli_public_chatml/dev.json \ # Path to the processed validation dataset from step 2
       --learning_rate 5e-5 \ # Learning rate
       --per_device_train_batch_size 16 \ # Batch size per device during training
       --per_device_eval_batch_size 128 \ # Batch size per device during evaluation
       --model_max_length 1024 \ # Maximum token length for the model, sequences exceeding this will be truncated
       --bf16 \ # Whether to use bf16 data format, change to false if not applicable
       --use_lora \ # Whether to use LoRA
       --gradient_accumulation_steps 1 \ # Number of gradient accumulation steps
       --warmup_steps 100 \ # Number of warm-up steps
       --max_steps 1000 \ # Maximum number of training steps, training will stop once reached
       --weight_decay 0.01 \ # Weight decay value
       --evaluation_strategy steps \ # Evaluation strategy, can be changed to epoch
       --eval_steps 500 \ # Works with evaluation_strategy steps, evaluates every 500 steps
       --save_strategy steps \ # Model saving strategy, can be changed to epoch to save after each epoch
       --save_steps 500 \ # Works with save_strategy steps, saves the model every 500 steps
       --seed 42 \ # Random seed
       --log_level info --logging_strategy steps --logging_steps 10 \ # Logging settings
       --deepspeed configs/ds_config_zero2_offload.json # DeepSpeed configuration file, if sufficient VRAM, use configs/ds_config_zero2_offload.json
   ```

4. **(Optional) Train with LoRA/QLoRA**

   - **Using LoRA**:
     Add the `use_lora` parameter in the `MiniCPM/finetune/lora_finetune_ocnli.sh` file.
     If unsure, add the following code right before `--deepspeed configs/ds_config_zero2_offload.json`:
     ```sh
     use_lora \
     ```
   - **Using QLoRA**:
     Add both `use_lora` and `qlora` parameters in the `MiniCPM/finetune/lora_finetune_ocnli.sh` file.
     If unsure, add the following code right before `--deepspeed configs/ds_config_zero2_offload.json`:
     ```sh
     use_lora \
     qlora \
     ```

5. **Start Training**
   After modifying the bash file, navigate to the directory and run the following command to start training:
   ```sh
   cd MiniCPM/finetune
   bash lora_finetune_ocnli.sh
   ```
