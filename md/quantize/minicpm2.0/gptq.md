
# MiniCPM 模型量化指南 - 使用AutoGPTQ

为了执行MiniCPM模型的量化，您需要遵循以下步骤，并确保您的设备满足以下要求：
- 至少存在一张Nvidia 20系以上的显卡；
- 量化2b需要6GB显存；
- 量化1b需要4GB显存。

## 1. 获取MiniCPM模型权重

以MiniCPM-2b-sft为例：

```bash
git clone https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16
```

## 2. 获取量化脚本

由于AutoGPTQ不再更新，这里获取分支代码：

```bash
git clone -b minicpm_gptq https://github.com/LDLINGLINGLING/AutoGPTQ
```

## 3. 安装AutoGPTQ分支

进入AutoGPTQ目录并安装依赖：

```bash
cd AutoGPTQ
git checkout minicpm_autogptq
pip install -e .
```

## 4. 开始量化

进入MiniCPM量化目录，并修改量化脚本中的路径参数：

```bash
cd MiniCPM/quantize
```

运行量化脚本，将`no_quant_model_path`修改为未量化的MiniCPM模型权重的保存地址，将`quant_save_path`修改为量化后的模型保存地址：

```bash
python gptq_quantize.py --pretrained_model_dir no_quant_model_path --quantized_model_dir quant_save_path --bits 4
```

以上步骤完成后，您将获得量化的MiniCPM模型权重。

