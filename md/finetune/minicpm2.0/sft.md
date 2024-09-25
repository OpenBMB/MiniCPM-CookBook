
# MiniCPM 模型微调指南

## 设备需求
- 最少一张12GB显存，20系列以上显卡
- 使用QLoRA时可尝试6-8GB显卡

## 步骤
1. **使用Git获取官方代码**
   ```sh
   git clone https://github.com/OpenBMB/MiniCPM
   ```

2. **准备数据集，并处理成以下JSON格式**
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
         // ... 多轮对话
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

3. **修改 `MiniCPM/finetune/lora_finetune_ocnli.sh` 文件**
   ```sh
   formatted_time=$(date +"%Y%m%d%H%M%S")
   echo $formatted_time

   # 对于4090显卡添加这两行代码
   export NCCL_P2P_DISABLE=1
   export NCCL_IB_DISABLE=1 

   deepspeed --include localhost:1 --master_port 19888 finetune.py \
       --model_name_or_path MiniCPM-2B-sft-bf16 \ # 可以修改为本地模型目录和1B模型地址
       --output_dir output/OCNLILoRA/$formatted_time/ \ # 可以修改为其他用来保存输出模型的地址
       --train_data_path data/ocnli_public_chatml/train.json \ # 这里写按照第二步处理好的训练集地址
       --eval_data_path data/ocnli_public_chatml/dev.json \ # 这里写按照第二步处理好的验证集
       --learning_rate 5e-5 \ # 学习率
       --per_device_train_batch_size 16 \ # 每张卡训练时的batch_size
       --per_device_eval_batch_size 128 \ # 每张卡测试时的batch_size
       --model_max_length 1024 \ # 模型训练时最大token数，超出将截断
       --bf16 \ # 是否使用bf16数据格式，如果不是改为false
       --use_lora \ # 是否使用lora
       --gradient_accumulation_steps 1 \ # 梯度累计次数
       --warmup_steps 100 \ # 预热步数
       --max_steps 1000 \ # 最大训练步数，到达后停止训练
       --weight_decay 0.01 \ # 权重正则化值
       --evaluation_strategy steps \ # 测试方法，可以改为epoch
       --eval_steps 500 \ # 与evaluation_strategy steps一起起作用，500个step测试一次
       --save_strategy steps \ # 模型保存策略，可以改为epoch即每个epoch保存一次
       --save_steps 500 \ # save_strategy steps一起起作用，代表500步保存一次
       --seed 42 \ # 随机种子
       --log_level info --logging_strategy steps --logging_steps 10 \ # logging的设置
       --deepspeed configs/ds_config_zero2_offload.json # deepspeed配置文件设置，如果显存充足可以改为configs/ds_config_zero2_offload.json
   ```

4. **(可选)使用LoRA/QLoRA训练**

   - **LoRA使用**:
     在 `MiniCPM/finetune/lora_finetune_ocnli.sh` 文件中增加 `use_lora` 参数。
     如果不确定，则将以下代码加到 `--deepspeed configs/ds_config_zero2_offload.json` 的前一行:
     ```sh
     use_lora \
     ```
   - **QLoRA使用**:
     在 `MiniCPM/finetune/lora_finetune_ocnli.sh` 文件中增加 `use_lora` 和 `qlora` 参数。
     如果不确定，则将以下代码加到 `--deepspeed configs/ds_config_zero2_offload.json` 的前一行:
     ```sh
     use_lora \
     qlora \
     ```

5. **开始训练**
   修改完上述bash文件后，进入目录并执行以下命令开始训练：
   ```sh
   cd MiniCPM/finetune
   bash lora_finetune_ocnli.sh
   ```