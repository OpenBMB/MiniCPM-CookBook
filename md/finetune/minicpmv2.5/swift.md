# Swift 安装与训练指南

## 安装Swift

首先克隆Swift仓库：

```sh
git clone https://github.com/modelscope/swift.git
cd swift
pip install -r requirements.txt
pip install -e '.[llm]'
```

## 训练Swift

### 1. 自定义数据集

数据集应该包含如下字段：
- `query`: 对应本轮对话人类的提问。
- `response`: 对应本轮人类提问的期望回复。
- `images`: 可以是本地图片路径或者网络图片的URL。

示例数据集条目：

```json
{
    "query": "这张图片描述了什么",
    "response": "这张图片有一个大熊猫",
    "images": ["local_image_path"]
}
{
    "query": "这张图片描述了什么",
    "response": "这张图片有一个大熊猫",
    "history": [],
    "images": ["image_path"]
}
{
    "query": "竹子好吃么",
    "response": "看大熊猫的样子挺好吃呢",
    "history": [["这张图有什么", "这张图片有大熊猫"], ["大熊猫在干嘛", "吃竹子"]],
    "images": ["image_url"]
}
```

### 2. LoRA微调

默认情况下，Swift针对语言模型的Q、K、V矩阵进行微调。`CUDA_VISIBLE_DEVICES`用于指定使用的GPU设备。当使用多GPU微调时，建议将`eval_steps`设为一个非常大的值以避免内存溢出问题。

```sh
CUDA_VISIBLE_DEVICES=0,1 swift sft \
    --model_type minicpm-v-v2_5-chat \
    --dataset local_data.jsonl \
    --eval_steps 200000
```

### 3. 全量微调

通过设置`--lora_target_modules`为`ALL`来启用全量微调。

```sh
CUDA_VISIBLE_DEVICES=0,1 swift sft \
    --model_type minicpm-v-v2_5-chat \
    --dataset coco-en-2-mini \
    --lora_target_modules ALL \
    --eval_steps 200000
```

## 推理与测试

### 1. 挂载LoRA推理

使用训练好的模型进行推理，需要指定模型检查点目录。

```sh
CUDA_VISIBLE_DEVICES=0 swift infer    \
 --ckpt_dir /root/ld/ld_project/swift/output/minicpm-v-v2_5-chat/v9-20240705-171455/checkpoint-10
```

### 2. 合并LoRA推理

此方法会将LoRA权重合并到主干网络，并保存到指定路径下。

```sh
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir /root/ld/ld_project/swift/output/minicpm-v-v2_5-chat/v9-20240705-171455/checkpoint-10 \
    --merge_lora true
```

### 3. 测试合并后的模型

加载配置文件中的数据集来进行测试。

```sh
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/minicpm-v-v2_5-chat/vx-xxx/checkpoint-xxx-merged \
    --load_dataset_config true
```
