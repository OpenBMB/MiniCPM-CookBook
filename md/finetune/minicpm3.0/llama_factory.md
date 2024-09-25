
## Llama_factory

使用方法与[【微调教程】MiniCPM2.0](../minicpm2.0/llama_factory.md)基本相同，仅需修改`template: cpm3`

1. **首先安装Llama_factory依赖**
   我的个人环境在这MiniCPM3 [pip_list](pip_list.md)
   ```bash
   git clone https://github.com/hiyouga/LLaMA-Factory.git
   cd LLaMA-Factory
   pip install -e ".[torch,metrics]"
   ```

2. **处理数据**
   将数据集处理成`Minicpm/finetune/llama_factory_example/llama_factory_data`文件夹中的格式, 示例包括DPO,KTO,SFT三种微调方式并放置到`llama_factory/data`目录下.

   #### 2.1 DPO数据格式
     对于一个"human"的answer，需要给一个chosen（好的答案），一个rejected（坏的答案）
     ```json
     [
       {
         "conversations": [
           {
             "from": "human",
             "value": "你吃饭了没有，吃的什么啊？"
           },
           {
             "from": "gpt",
             "value": "刚吃了啊，你去哪里玩？"
           },
           {
             "from": "gpt",
             "value": "吃得茄子，西红柿，红烧肉。"
           }
         ],
         "chosen": {
           "from": "gpt",
           "value": "刚吃了啊，你去哪里玩？"
         },
         "rejected": {
           "from": "gpt",
           "value": "吃得茄子，西红柿，红烧肉。"
         }
       }
     ]
     ```

   #### 2.2 KTO数据格式
     1. 其中每个messages代表一条数据，label代表是该条数据的回复是否合理
     2. 不需要和DPO那样相同问题不同答案，对于每个问题和答案对给予一个评价
     3. 每个问题和回复都可以有一个True（好的），或者false（不好的评价）
     ```json
     [
       {
         "messages": [
           {
             "content": "minicpm3.0哪里最强？",
             "role": "user"
           },
           {
             "content": "哪里都强，六边形战士好吗",
             "role": "assistant"
           }
         ],
         "label": true
       },
       {
         "messages": [
           {
             "content": "minicpm3.0 有哪些优点。",
             "role": "user"
           },
           {
             "content": "1.长文本能力。2.function call能力 3.其他所有能力",
             "role": "assistant"
           },
           {
             "content": "细说长文本能力",
             "role": "user"
           },
           {
             "content": "反正很强，细说就去我们github吧。https://github.com/OpenBMB/MiniCPM.git",
             "role": "assistant"
           }
         ],
         "label": false
       }
     ]
     ```

   #### 2.3 SFT数据格式如下, 每个字典组成一条数据
     ```json
     [
       {
         "instruction": "你现在比牛顿还牛的科学家",
         "input": "识别并解释给定列表中的两个科学理论：细胞理论和日心说。",
         "output": "细胞理论是生物科学的一个理论，它认为所有生命体都是由微小的基本单元——细胞所构成。这是生物学的一个基础理论，认为细胞是所有生物的基本结构和功能单位，所有的生物都是由一个或多个细胞组成，细胞只能通过细胞分裂产生新的细胞。这一理论由薛定谔、施瓦内和雪莱于1839年首次提出。\n\n日心说是指太阳是太阳系的中心，也就是说，行星围绕太阳旋转的理论。这个理论打破了传统的地心说观点，认为地球并不是宇宙的中心。日心说的提出者是尼古拉·哥白尼，他在16世纪初发表了他的著作《天体运行论》，阐述了太阳系行星围绕太阳运行的模型，为天文学的发展做出了巨大贡献。"
       }
     ]
     ```

3. **将数据信息添加到dataset_info.json**
   在`llama_factory/data/dataset_info.json`中添加数据集信息, 保证`dataset_info.json`中能找到你的数据集，如下例：
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

4. **设置训练脚本**

   #### 4.1 将`MiniCPM/finetune/llama_factory_example`中文件复制到`LLaMA-Factory/examples/minicpm`目录下
   ```bash
   cd LLaMA-Factory/examples
   mkdir minicpm
   # 以下代码中的/your/path要改成你的MiniCPM代码和LLaMA-Factory路径
   cp -r /your/path/MiniCPM/finetune/llama_factory_example/*  /your/path/LLaMA-Factory/examples/minicpm
   ```

   #### 4.2 根据你需要微调的方式，以DPO为例。`LLaMA-Factory/examples/minicpm/minicpm_dpo.yaml`必须修改的配置参数如下
   ```yaml
   model_name_or_path: openbmb/MiniCPM3-4B # 或者你本地保存的地址
   template: cpm3 # 注意如果微调MiniCPM3就写cpm3，否则写cpm
   dataset: dpo_en_demo # 这里写dataset_info.json中的键名
   output_dir: your/finetune_minicpm/save/path # 你微调后模型的保存地址
   bf16: true # 如果你的设备支持bf16，否则false
   deepspeed: examples/deepspeed/ds_z2_config.json # 如果显存不够可以改成ds_z3_config.json
   ```

   #### 4.3 修改`LLaMA-Factory/examples/minicpm/single_node.sh`文件中以下配置
   ```bash
   # 以下这行修改为你机器有多少张GPU
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

   # 以下这行需要修改成配置文件地址，如/root/ld/ld_project/LLaMAFactory/examples/minicpm/minicpm_dpo.yaml
     src/train.py /your/path/LLaMA-Factory/examples/minicpm/minicpm_dpo.yaml
   ```

5. **开始训练**
   ```bash
   cd LLaMA-Factory
   bash /your/path/LLaMA-Factory/examples/minicpm/single_node.sh
   ```
