
# MiniCPMV训练环境介绍及步骤

## 1. 训练环境介绍
笔者的训练环境为：[pip list](./pip_list.md)
## 2. 获取MiniCPMV的GitHub代码
通过Git克隆MiniCPMV项目到本地：
```bash
git clone https://github.com/OpenBMB/MiniCPM-V.git
```

## 3. 安装依赖包
进入项目目录并安装所需的Python依赖包：
```bash
cd MiniCPM-V
pip install -r requirements.txt
```

## 4. 处理数据集
处理数据集使其符合以下格式要求：

### 单图（一轮对话中仅一张图）
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

### 多图（一次对话包含多张图片）
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

## 5. LoRA微调

### 5.1 修改`finetune_lora.sh`
修改`MiniCPM-V/finetune/finetune_lora.sh`脚本以适应LoRA微调需求。如果需要微调int4模型，请按照以下说明修改脚本：

```bash
#!/bin/bash
GPUS_PER_NODE=8 # 改成你的机器每个节点共有多少张显卡，如果是单机八卡就是8
NNODES=1 # 改成你的机器有多少个节点，如果就是一台服务器就是1
NODE_RANK=0 # 使用第几个服务器训练
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL="/root/ld/ld_model_pretrained/Minicpmv2_6" # 本地模型路径 or openbmb/MiniCPM-V-2.5
DATA="/root/ld/ld_project/MiniCPM-V/finetune/mllm_demo.json" # 训练数据文件地址
EVAL_DATA="/root/ld/ld_project/MiniCPM-V/finetune/mllm_demo.json" # 验证集数据文件地址
LLM_TYPE="qwen2" # if use openbmb/MiniCPM-V-2, please set LLM_TYPE=minicpm

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

### 5.2 需要重点关注的参数
- `MODEL`：本地模型路径或Hugging Face ID。
- `DATA`：训练数据文件。
- `EVAL_DATA`：验证集数据文件。

- `--tune_vision true`：是否微调siglip(vit)模块。
- `--lora_target_modules`：LoRA插入的层，这里写的是正则表达式。
- `--tune_llm false`：是否微调大语言模型模块。
- `--use_lora true`：是否进行LoRA微调。

- `--model_max_length 2048`：模型训练的最大长度。
- `--per_device_train_batch_size 2`：每张卡训练的batch size。
- `--per_device_eval_batch_size 1`：每张卡验证的batch size。
- `--gradient_accumulation_steps 1`：梯度累积，当显存少时可以增大这个参数从而减少`per_device_train_batch_size`。
- `--learning_rate 1e-6`：学习率。
- `--gradient_checkpointing true`：梯度检查点，建议开启，极大减少显存使用。
- `--deepspeed ds_config_zero3.json`：使用zero3，显存充足建议使用`ds_config_zero2.json`。

### 5.3 开始训练
进入微调脚本所在的目录并执行脚本开始训练：
```bash
cd MiniCPM-V/finetune
bash finetune_lora.sh
```
以下是根据您提供的信息整理的Markdown文档：

# MiniCPMV LoRA与模型合并保存及全量微调

## 5.4 LoRA与模型合并保存
使用以下脚本将LoRA模型合并到基础模型中，并保存合并后的模型：

```python
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer
import os
import shutil

# 指定基础模型路径
model_type = "/root/ld/ld_model_pretrained/Minicpmv2_6"  
# LoRA适配器保存路径
path_to_adapter = "/root/ld/ld_project/minicpmv2_6/MiniCPM-V/finetune/output/output_minicpmv2_lora/checkpoint-30"  
# 合并后模型保存路径
merge_path = "/root/ld/ld_project/minicpmv2_6/MiniCPM-V/finetune/output/merge_minicpmv"  

# 保证原始模型的各个文件不遗漏保存到merge_path中
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

# 加载原始模型
model = AutoModel.from_pretrained(
    model_type,
    trust_remote_code=True
)

# 加载LoRA模块到原始模型中
lora_model = PeftModel.from_pretrained(
    model,
    path_to_adapter,
    device_map="auto",
    trust_remote_code=True
).eval()

# 将加载的LoRA模块合并到原始模型中
merge_model = lora_model.merge_and_unload()

# 将新合并的模型进行保存
merge_model.save_pretrained(merge_path, safe_serialization=False)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)
tokenizer.save_pretrained(merge_path)

# 复制基础模型的其他文件到合并后的路径
copy_files_not_in_B(model_type, merge_path)
```

## 6. 全量微调

### 6.1 修改`finetune_ds.sh`参数
修改`MiniCPM-V/finetune/finetune_ds.sh`脚本以适应全量微调的需求：

```bash
#!/bin/bash

GPUS_PER_NODE=8 # 改成你的机器每个节点共有多少张显卡，如果是单机八卡就是8
NNODES=1 # 改成你的机器有多少个节点，如果就是一台服务器就是1
NODE_RANK=0 # 使用第几个服务器训练
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL="/root/ld/ld_model_pretrained/Minicpmv2_6" # 模型本地路径 or huggingface id
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="/root/ld/ld_project/MiniCPM-V/finetune/mllm_demo.json" # 训练数据文件
EVAL_DATA="/root/ld/ld_project/MiniCPM-V/finetune/mllm_demo.json" # 验证集数据文件
LLM_TYPE="qwen2" # if use openbmb/MiniCPM-V-2, please set LLM_TYPE=minicpm

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

### 6.2 开始训练
进入微调脚本所在的目录并执行脚本开始训练：
```bash
cd MiniCPM-V/finetune
bash finetune_sh.sh
```
```

请确保在运行上述脚本前已经正确配置了环境，并且所有路径都指向正确的文件和目录。