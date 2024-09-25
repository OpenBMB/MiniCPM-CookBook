

# 安装LLaMA-Factory依赖

首先，克隆LLaMA-Factory仓库，并安装依赖项：
```sh
git clone https://github.com/hiyouga/LLaMA-Factory
cd LLaMA-Factory
pip install -r requirements.txt
```

# 数据处理

将数据集处理成 `Minicpm/finetune/llama_factory_example/llama_factory_data` 文件夹中的格式，并放置到 `llama_factory/data` 目录下。示例包括DPO, KTO, SFT三种微调方式。

### 2.1 DPO 数据格式
```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "Hi! I'd like to create a new language game simulating the first person perspective of a character named Angela."
      }
    ],
    "chosen": {
      "from": "gpt",
      "value": "That sounds like a fun and engaging idea! Here are some tips to help you create the game:\n1. Start with the character's name and background: "
    },
    "rejected": {
      "from": "gpt",
      "value": "Hello! 😊"
    }
  }
]
```

### 2.2 KTO 数据格式
每个 `messages` 代表一条数据，`label` 代表该条数据的回复是否合理。
```json
{
  "messages": [
    {
      "content": "The Federal Trade Commission is going after spyware, bits of computer code that install themselves on the computers of Internet users to track or disrupt their online activities.\nQ: Which is the best summary of this article?\nPick your answer from:\n(A). World\n(B). Sports\n(C). Business\n(D). Science/Tech\nI think the answer is",
      "role": "user"
    },
    {
      "content": "Ooh ooh ah ah! Science/Tech! Yay! *confetti* The Federal Trade Commission is talking about spyware, which is like a little computer bug that can watch what you do on your computer without you knowing. It's like a secret agent, but not a nice one! *gasp* The FTC wants to stop the spyware from doing its thing, so they're going after it! *cheer* Woohoo!",
      "role": "assistant"
    }
  ],
  "label": true // true or false， true is good response  ，false is bad response
}
```

### 2.3 SFT 数据格式
每个字典组成一条数据。
```json
[
  {
    "instruction": "识别并解释给定列表中的两个科学理论：细胞理论和日心说。",
    "input": "",
    "output": "细胞理论是生物科学的一个理论，它认为所有生命体都是由微小的基本单元——细胞所构成。这是生物学的一个基础理论，认为细胞是所有生物的基本结构和功能单位，所有的生物都是由一个或多个细胞组成，细胞只能通过细胞分裂产生新的细胞。这一理论由薛定谔、施瓦内和雪莱于1839年首次提出。\n\n日心说是指太阳是太阳系的中心，也就是说，行星围绕太阳旋转的理论。这个理论打破了传统的地心说观点，认为地球并不是宇宙的中心。日心说的提出者是尼古拉·哥白尼，他在16世纪初发表了他的著作《天体运行论》，阐述了太阳系行星围绕太阳运行的模型，为天文学的发展做出了巨大贡献。"
  }
]
```

# 将数据信息添加到 `dataset_info.json`

在 `llama_factory/data/dataset_info.json` 中添加数据集信息，确保 `dataset_info.json` 中能找到你的数据集。
```json
{
  "identity": {
    "file_name": "identity.json"
  },
  "sft_zh_demo": {
    "file_name": "alpaca_zh_demo.json"
  },
  "kto_en_demo": {
    "file_name": "kto_en_demo.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "kto_tag": "label"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  },
  "dpo_en_demo": {
    "file_name": "dpo_en_demo.json",
    "ranking": true,
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "chosen": "chosen",
      "rejected": "rejected"
    }
  }
}
```

# 设置训练脚本

### 4.1 复制示例文件
将 `MiniCPM/finetune/llama_factory_example` 中的文件复制到 `LLaMA-Factory/examples/minicpm` 目录下。
```sh
cd LLaMA-Factory/examples
mkdir minicpm
cp -r /your/path/MiniCPM/finetune/llama_factory_example/* /your/path/LLaMA-Factory/examples/minicpm
```

### 4.2 修改配置文件
根据需要微调的方式，以DPO为例。必须修改 `LLaMA-Factory/examples/minicpm/minicpm_dpo.yaml` 中的配置参数如下：
```yaml
model_name_or_path: openbmb/MiniCPM-2B-sft-bf16 # 或者你本地保存的地址
dataset: dpo_en_demo # 这里写dataset_info.json中的键名
output_dir: your/finetune_minicpm/save/path # 你微调后模型的保存地址
bf16: true # 如果你的设备支持bf16，否则false
deepspeed: examples/deepspeed/ds_z2_config.json # 如果显存不够可以改成ds_z3_config.json
```

### 4.3 修改 `single_node.sh` 文件
修改 `LLaMA-Factory/examples/minicpm/single_node.sh` 文件中的以下配置：
```sh
NPROC_PER_NODE=8
NNODES=1
RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500

# 以下两行如果是A100，H100等以上的高端显卡可以删除
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1 

# 以下数字设置为你机器中参与训练的显卡，这里是0-7号卡都参与训练
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \ 

# 以下这行需要修改成配置文件地址
    src/train.py /your/path/LLaMA-Factory/examples/minicpm/minicpm_dpo.yaml
```

# 开始训练

最后，在 `LLaMA-Factory` 目录下执行训练脚本：
```sh
cd LLaMA-Factory
bash /your/path/LLaMA-Factory/examples/minicpm/single_node.sh
```