
# MiniCPM 模型量化指南 - 使用AutoAWQ

为了执行MiniCPM模型的量化，您需要遵循以下步骤，并确保您的设备满足以下要求：
- 至少存在一张Nvidia 20系以上的显卡；
- 量化2b需要6GB显存；
- 量化1b需要4GB显存。

## 1. 获取MiniCPM开源代码

```bash
git clone https://github.com/OpenBMB/MiniCPM
```

## 2. 获取MiniCPM模型权重

以MiniCPM-2b-sft为例：

```bash
git clone https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16
```

## 3. 修改脚本配置参数

在`MiniCPM/quantize/awq_quantize.py`文件中根据注释修改配置参数：

```python
# 量化前模型保存路径或者huggingface id
model_path = '/root/ld/ld_model_pretrained/MiniCPM-1B-sft-bf16'
# 量化后模型保存路径
quant_path = '/root/ld/ld_project/pull_request/MiniCPM/quantize/awq_cpm_1b_4bit'
# 量化数据集路径，在MiniCPM/quantize/quantize_data下有wikitext和alpaca，可以二选一
quant_data_path='/root/ld/ld_project/pull_request/MiniCPM/quantize/quantize_data/wikitext'
# 量化配置参数，建议仅修改w_bit,即量化bit数
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,  # 可选4 or 8
    "version": "GEMM"
}
# 最多使用多少条量化校准数据
quant_samples=512  # how many samples to use for calibration
# 如果要使用自己的量化数据集，按以下格式进行配置数据
custom_data=[
    {'question': '你叫什么名字。', 'answer': '我是openmbmb开源的小钢炮minicpm。'},
    {'question': '你有什么特色。', 'answer': '我很小，但是我很强。'}
]
```

## 4. 制作量化校准集

根据选择的数据集，选择以下某一行代码替换`quantize/awq_quantize.py`中的相应行（例如第38行）：

```python
# 使用wikitext进行量化
model.quantize(tokenizer, quant_config=quant_config, calib_data=load_wikitext(quant_data_path=quant_data_path))

# 使用alpaca进行量化
model.quantize(tokenizer, quant_config=quant_config, calib_data=load_alpaca(quant_data_path=quant_data_path))

# 使用自定义数据集进行量化
model.quantize(tokenizer, quant_config=quant_config, calib_data=load_cust_data(quant_data_path=quant_data_path))
```

## 5. 开始量化

运行`MiniCPM/quantize/awq_quantize.py`，开始量化过程。完成后，您将获得量化的模型权重。

请根据实际情况调整路径和配置参数。希望这份指南能够帮助您顺利完成MiniCPM模型的量化工作。