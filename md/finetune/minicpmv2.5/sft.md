
# 官方代码模型训练指南

## 1. 安装依赖包

首先，进入项目目录并安装依赖包：
```sh
cd MiniCPM-V
pip install -r requirements.txt
```
**注意**：推荐从DeepSpeed官网下载最新版本并使用 `pip install e .` 安装。

## 2. 处理数据集

处理数据集使其符合以下形式：
- `id` 值不可重复。
- `<image>\n` 应该出现在每个数据集对话数据的开头。
- `"image"` 对应的地址需要存在图片。
- 每个 `conversations` 对应的列表中是一个多轮对话，`content` 代表对话内容，`role` 对应 `user` 代表用户输入，`role` 对应 `assistant` 代表模型输出。
- 每条数据仅包含一张图片。

示例数据格式如下：
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
    //多条...
]
```

## 3. LoRA 微调

### 3.1 修改 `MiniCPM-V/finetune/finetune_lora.sh`

```sh
#!/bin/bash
GPUS_PER_NODE=8 # 改成你的机器每个节点共有多少张显卡，如果是单机八卡就是8
NNODES=1 # 改成你的机器有多少个节点，如果就是一台服务器就是1
NODE_RANK=0 # 使用第几个服务器训练
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL="/root/ld/ld_model_pretrained/MiniCPM-Llama3-V-2_5" # 本地模型路径 or openbmb/MiniCPM-V-2.5
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="/root/ld/ld_project/MiniCPM-V/finetune/mllm_demo.json" # 训练数据文件地址
EVAL_DATA="/root/ld/ld_project/MiniCPM-V/finetune/mllm_demo.json" # 验证集数据文件地址
LLM_TYPE="llama3" # if use openbmb/MiniCPM-V-2, please set LLM_TYPE=minicpm

export NCCL_P2P_DISABLE=1 # a100等支持nccl_p2p的显卡去掉此行
export NCCL_IB_DISABLE=1 # a100等显卡去掉此行

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
    --label_names "labels" \ # 数据构造，不要动
    --prediction_loss_only false \ 
    --bf16 false \ # 使用bf16精度训练，4090，a100，h100等可以开启
    --bf16_full_eval false \ # 使用bf16精度测试
    --fp16 true \ # 使用fp16精度训练
    --fp16_full_eval true \ # 使用pf16精度测试
    --do_train \ # 是否训练
    --do_eval \ # 训练过程中是否做验证
    --tune_vision true \ # 是否微调siglip(vit)模块
    --tune_llm false \ # 是否微调大语言模型模块
    --use_lora true \ # 是否lora微调
    --lora_target_modules "llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj｜v_proj|o_proj)" \ #lora插入的层，这里写的是正则表达式，建议不改
    --model_max_length 2048 \ # 模型训练的最大长度
    --max_slice_nums 9 \ # 模型最大切分次数
    --max_steps 10000 \ # 最多训练步数
    --eval_steps 1000 \ # 每多少步验证一次
    --output_dir output/output_minicpmv2_lora \ # 模型lora保存地址
    --logging_dir output/output_minicpmv2_lora \ # 日志保存地址
    --logging_strategy "steps" \ # 日志输出策略（可选epoch）
    --per_device_train_batch_size 2 \ # 每张卡训练的batch_size
    --per_device_eval_batch_size 1 \ # 每张卡验证的batch_size
    --gradient_accumulation_steps 8 \ # 梯度累积，当显存少时可以增大这个参数从而减少per_device_train_batch_size
    --evaluation_strategy "steps" \ # 验证策略(可选epoch)
    --save_strategy "steps" \ # 保存策略(可选epoch)与save_steps同时起作用
    --save_steps 10 \ # 10个step保存一次
    --save_total_limit 10 \ # 最大储存总数
    --learning_rate 1e-6 \ # 学习率
    --weight_decay 0.1 \ # 权重正则化参数
    --adam_beta2 0.95 \ # 
    --warmup_ratio 0.01 \ # 总步数的预热率，即：总训练步数*warmup_ratio=预热步数
    --lr_scheduler_type "cosine" \ # 学习率调整器
    --logging_steps 1 \
    --gradient_checkpointing true \ # 梯度检查点，建议开启，极大减少显存使用
    --deepspeed ds_config_zero3.json \ # 使用zero3，显存充足建议使用ds_config_zero2.json
    --report_to "tensorboard" # wandb # tensorboard或者wandb记录损失
```

### 3.2 需要重点关注的参数

- `MODEL`：本地模型路径或指定的远程模型路径。
- `DATA`：训练数据文件。
- `EVAL_DATA`：验证集数据文件。
- `--tune_vision true`：是否微调SigLIP(ViT)模块。
- `--lora_target_modules "llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"`：LoRA插入的层，这里写的是正则表达式，建议不改。
- `--tune_llm false`：是否微调大语言模型模块。
- `--use_lora true`：是否LoRA微调。
- `--model_max_length 2048`：模型训练的最大长度（#1000+文字数/1.5）。
- `--per_device_train_batch_size 2`：每张卡训练的batch size。
- `--per_device_eval_batch_size 1`：每张卡验证的batch size。
- `--gradient_accumulation_steps 1`：梯度累积，当显存少时可以增大这个参数从而减少`per_device_train_batch_size`。
- `--learning_rate 1e-6`：学习率。
- `--gradient_checkpointing true`：梯度检查点，建议开启，极大减少显存使用。
- `--deepspeed ds_config_zero3.json`：使用Zero3，显存充足建议使用`ds_config_zero2.json`。

### 3.3 开始训练

进入 `MiniCPM-V/finetune` 目录并执行训练脚本：
```sh
cd MiniCPM-V/finetune
bash finetune_lora.sh
```

### 3.4 LoRA 与模型合并保存

```python
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

model_type="/root/ld/ld_model_pretrained/MiniCPM-Llama3-V-2_5" # local_model_path or openbmb/MiniCPM-V-2.5
path_to_adapter="/root/ld/ld_project/MiniCPM-V/finetune/output/output_minicpmv2_lora/checkpoint-400" # LoRA保存的地址
merge_path="/root/ld/ld_project/MiniCPM-V/finetune/output/merge_model_path" # 希望将LoRA合并到主模型后的保存地址

model = AutoModel.from_pretrained(
    model_type,
    trust_remote_code=True
)
# 挂载LoRA模块
lora_model = PeftModel.from_pretrained(
    model,
    path_to_adapter,
    device_map="auto",
    trust_remote_code=True
).eval().cuda()

# 合并LoRA模块到原模型，模型shape与原始MiniCPM-Llama3-V-2_5相同
merge_model = lora_model.merge_and_unload()
# 保存新的模型，与原始MiniCPM-Llama3-V-2_5的shape相同
merge_model.save_pretrained(merge_path, safe_serialization=False)

# 加载分词文件与模型保存到merge后的模型地址
tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)
tokenizer.save_pretrained(merge_path)
```
下面是将上述步骤整理成的Markdown文件内容：

# QLoRA 微调指南 (适用于低显存设备)

## 4. 获取量化版模型

### 方法一：直接下载

- **Hugging Face**
  ```sh
  git clone https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-int4
  ```
- **ModelScope**
  ```sh
  git clone https://modelscope.cn/models/OpenBMB/MiniCPM-Llama3-V-2_5-int4
  ```

### 方法二：自行量化模型

## 4.1 按照[bnb量化教程](../../quantize/minicpmv2.5/bnb.md)对模型进行量化。 
## 4.2 修改 `MiniCPM-V/finetune/finetune_lora.sh`

```sh
#!/bin/bash
GPUS_PER_NODE=8 # 改成你的机器每个节点共有多少张显卡，如果是单机八卡就是8
NNODES=1 # 改成你的机器有多少个节点，如果就是一台服务器就是1
NODE_RANK=0 # 使用第几个服务器训练
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL="/root/ld/ld_model_pretrained/MiniCPM-Llama3-V-2_5" # 本地模型路径 or openbmb/MiniCPM-V-2.5
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="/root/ld/ld_project/MiniCPM-V/finetune/mllm_demo.json" # 训练数据文件地址
EVAL_DATA="/root/ld/ld_project/MiniCPM-V/finetune/mllm_demo.json" # 验证集数据文件地址
LLM_TYPE="llama3" # if use openbmb/MiniCPM-V-2, please set LLM_TYPE=minicpm

export NCCL_P2P_DISABLE=1 # a100等支持nccl_p2p的显卡去掉此行
export NCCL_IB_DISABLE=1 # a100等显卡去掉此行

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
    --label_names "labels" \ # 数据构造，不要动
    --prediction_loss_only false \ 
    --bf16 false \ # 使用bf16精度训练，4090，a100，h100等可以开启
    --bf16_full_eval false \ # 使用bf16精度测试
    --fp16 true \ # 使用fp16精度训练
    --fp16_full_eval true \ # 使用pf16精度测试
    --do_train \ # 是否训练
    --do_eval \ # 训练过程中是否做验证
    --tune_vision false \ # 是否微调siglip(vit)模块
    --tune_llm false \ # 是否微调大语言模型模块
    --use_lora true \ # 是否lora微调
    --q_lora true \
    --lora_target_modules "llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj｜v_proj|o_proj)" \ #lora插入的层，这里写的是正则表达式，建议不改
    --model_max_length 2048 \ # 模型训练的最大长度
    --max_slice_nums 9 \ # 模型最大切分次数
    --max_steps 10000 \ # 最多训练步数
    --eval_steps 1000 \ # 每多少步验证一次
    --output_dir output/output_minicpmv2_lora \ # 模型lora保存地址
    --logging_dir output/output_minicpmv2_lora \ # 日志保存地址
    --logging_strategy "steps" \ # 日志输出策略（可选epoch）
    --per_device_train_batch_size 2 \ # 每张卡训练的batch_size
    --per_device_eval_batch_size 1 \ # 每张卡验证的batch_size
    --gradient_accumulation_steps 8 \ # 梯度累积，当显存少时可以增大这个参数从而减少per_device_train_batch_size
    --evaluation_strategy "steps" \ # 验证策略(可选epoch)
    --save_strategy "steps" \ # 保存策略(可选epoch)与save_steps同时起作用
    --save_steps 10 \ # 10个step保存一次
    --save_total_limit 10 \ # 最大储存总数
    --learning_rate 1e-6 \ # 学习率
    --weight_decay 0.1 \ # 权重正则化参数
    --adam_beta2 0.95 \ # 
    --warmup_ratio 0.01 \ # 总步数的预热率，即：总训练步数*warmup_ratio=预热步数
    --lr_scheduler_type "cosine" \ # 学习率调整器
    --logging_steps 1 \
    --gradient_checkpointing true \ # 梯度检查点，建议开启，极大减少显存使用
    --deepspeed ds_config_zero3.json \ # 使用zero3，显存充足建议使用ds_config_zero2.json
    --report_to "tensorboard" # wandb # tensorboard或者wandb记录损失
```

## 5. 全量微调

### 5.1 修改 `MiniCPM-V/finetune/finetune_ds.sh` 参数

```bash
#!/bin/bash

GPUS_PER_NODE=8 # 改成你的机器每个节点共有多少张显卡，如果是单机八卡就是8
NNODES=1 # 改成你的机器有多少个节点，如果就是一台服务器就是1
NODE_RANK=0 # 使用第几个服务器训练
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL="/root/ld/ld_model_pretrained/MiniCPM-Llama3-V-2_5" # 模型本地路径 or openbmb/MiniCPM-V-2
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="/root/ld/ld_project/MiniCPM-V/finetune/mllm_demo.json" # 训练数据文件
EVAL_DATA="/root/ld/ld_project/MiniCPM-V/finetune/mllm_demo.json" # 验证集数据文件
LLM_TYPE="llama3" # if use openbmb/MiniCPM-V-2, please set LLM_TYPE=minicpm

export NCCL_P2P_DISABLE=1 # a100等支持nccl_p2p的显卡去掉此行
export NCCL_IB_DISABLE=1 # a100等显卡去掉此行

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
    --label_names "labels" \ # 数据构造，不要动
    --prediction_loss_only false \ 
    --bf16 false \ # 使用bf16精度训练，4090，a100，h100等可以开启
    --bf16_full_eval false \ # 使用bf16精度测试
    --fp16 true \ # 使用fp16精度训练
    --fp16_full_eval true \ # 使用pf16精度测试
    --do_train \ # 是否训练
    --do_eval \ # 训练过程中是否做验证
    --tune_llm true \ # 是否微调大语言模型模块
    --tune_vision true \ # 是否微调视觉模块
    --model_max_length 2048 \ # 模型训练的最大长度
    --max_slice_nums 9 \ # 模型最大切分次数
    --max_steps 10000 \ # 最多训练部署
    --eval_steps 1000 \ # 每多少步验证一次
    --output_dir output/output_minicpmv2_lora \ # 模型lora保存地址
    --logging_dir output/output_minicpmv2_lora \ # 日志保存地址
    --logging_strategy "steps" \ # 日志输出策略（可选epoch）
    --per_device_train_batch_size 2 \ # 每张卡训练的batch_size
    --per_device_eval_batch_size 1 \ # 每张卡验证的batch_size
    --gradient_accumulation_steps 1 \ # 梯度累积，当显存少时可以增大这个参数从而减少per_device_train_batch_size
    --evaluation_strategy "steps" \ # 验证策略(可选epoch)
    --save_strategy "steps" \ # 保存策略(可选epoch)与save_steps同时起作用
    --save_steps 10 \ # 10个step保存一次
    --save_total_limit 10 \ # 最大储存总数
    --learning_rate 1e-6 \ # 学习率
    --weight_decay 0.1 \ # 权重正则化参数
    --adam_beta2 0.95 \ # 
    --warmup_ratio 0.01 \ # 总步数的预热率，即：总训练步数*warmup_ratio=预热步数
    --lr_scheduler_type "cosine" \ # 学习率调整器
    --logging_steps 1 \
    --gradient_checkpointing true \ # 梯度检查点，建议开启，极大减少显存使用
    --deepspeed ds_config_zero3.json \ # 使用zero3，显存充足建议使用ds_config_zero3.json
    --report_to "tensorboard" # wandb # tensorboard或者wandb记录损失
```

### 5.2 开始训练

进入 `MiniCPM-V/finetune` 目录并执行训练脚本：
```sh
cd MiniCPM-V/finetune
bash finetune_ds.sh
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

## 4.1 Modify `MiniCPM-V/finetune/finetune_lora.sh`

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
    --gradient_checkpointing true \ # Gradient checkpointing, recommended to enable to significantly reduce GPU memory usage
    --deepspeed ds_config_zero3.json \ # Use Zero3, recommend `ds_config_zero3.json` if GPU memory is sufficient
    --report_to "tensorboard" # wandb # Record loss using tensorboard or wandb
```

### 5.2 Start Training

Navigate to the `MiniCPM-V/finetune` directory and execute the training script:
```sh
cd MiniCPM-V/finetune
bash finetune_ds.sh
```
