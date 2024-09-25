
# Swift 命令行推理

## 设备要求
- 所有显卡内存总共不低于24GB

## 步骤1：安装Swift

通过Git克隆Swift仓库，并安装依赖：

```sh
git clone https://github.com/modelscope/swift.git
cd swift
pip install -r requirements.txt
pip install -e '.[llm]'
```

## 步骤2：快速启动代码

以下命令将自动从ModelScope社区下载`minicpm-v-v2_5`模型，并加载默认生成参数：

```sh
CUDA_VISIBLE_DEVICES=0 swift infer --model_type minicpm-v-v2_5-chat
```

## 常用参数

- `model_id_or_path`: 可以写Hugging Face的模型ID或者本地模型地址
- `infer_backend`: 推理后端，可选值为`['AUTO', 'vllm', 'pt']`，默认为`AUTO`
- `dtype`: 计算精度，可选值为`['bf16', 'fp16', 'fp32', 'AUTO']`
- `max_length`: 最大长度
- `max_new_tokens`: 最多生成多少token，默认为2048
- `do_sample`: 是否采样，默认为`True`
- `temperature`: 生成时的温度系数，默认为0.3
- `top_k`: 默认为20
- `top_p`: 默认为0.7
- `repetition_penalty`: 默认为1.0
- `num_beams`: 默认为1
- `stop_words`: 停止词列表，默认为`None`
- `quant_method`: 模型的量化方式，可选值为`['bnb', 'hqq', 'eetq', 'awq', 'gptq', 'aqlm']`
- `quantization_bit`: 量化位数，默认为0（不使用量化）

## 示例

```sh
CUDA_VISIBLE_DEVICES=0,1 swift infer --model_type minicpm-v-v2_5-chat --model_id_or_path /root/ld/ld_model_pretrain/MiniCPM-Llama3-V-2_5 --dtype bf16
```

---
