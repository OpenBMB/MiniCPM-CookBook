# 官方代码微调(SFT推荐)

## 设备需求
- 最少一张24G显存，20系列以上显卡
- qlora可尝试12G显卡

## 步骤

### 1. 使用git获取官方代码
```bash
git clone https://github.com/OpenBMB/MiniCPM
```

### 2. 准备数据集

#### 2.1 对话数据集
处理成以下JSON格式，每一条数据就是一个messages。标红的system字段非必需。
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

#### 2.2 工具调用数据集
制作参考[冷启动获取function call数据](https://modelbest.feishu.cn/wiki/MB7HwpHEWiu0pakafkBcrfX2nNe?from=from_copylink)

### 3. 修改`MiniCPM/finetune/lora_finetune_ocnli.sh`文件
```bash
formatted_time=$(date +"%Y%m%d%H%M%S")
echo $formatted_time

# 4090添加这两行代码
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1 

deepspeed --include localhost:1 --master_port 19888 finetune.py \
    --model_name_or_path openbmb/MiniCPM3-4B \ # 可以修改为本地模型目录和1b模型地址
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

### 4. (可选) 使用LORA/QLORA训练

#### 4.1 LORA使用
在`MiniCPM/finetune/lora_finetune_ocnli.sh`文件中增加`use_lora`参数，
如不确定就把以下代码加到`--deepspeed configs/ds_config_zero2_offload.json`的前一行
```bash
use_lora \
```

#### 4.2 QLORA使用
在`MiniCPM/finetune/lora_finetune_ocnli.sh`文件中增加`use_lora`，`qlora`参数，
如不确定就把以下代码加到`--deepspeed configs/ds_config_zero2_offload.json`的前一行
```bash
use_lora \
qlora \
```

### 5. 开始训练
修改以上bash文件后, 输入以下指令开始训练：
```bash
cd MiniCPM/finetune
bash lora_finetune_ocnli.sh
```
