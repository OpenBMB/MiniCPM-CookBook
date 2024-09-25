
# AutoAWQ量化模型部署指南

## 方法1（推荐）

### 1. 直接下载量化好的模型

通过Git克隆已量化的模型仓库：

```bash
git clone https://www.modelscope.cn/models/linglingdan/MiniCPM-V_2_6_awq_int4
```

### 2. 下载编译笔者fork的autoawq

安装`AutoAWQ`的一个分支，该分支已提交PR，等待官方合并：

```bash
git clone https://github.com/LDLINGLINGLING/AutoAWQ.git
cd AutoAWQ
pip install -e .
```
### 3. 以上模型可以直接使用[vllm](../../inference/minicpmv2.6/vllm.md)进行推理，


## 方法2（自行量化，对训练后模型进行量化推荐这种方法）

### 1. 下载非模型

#### Hugging Face

通过Git克隆模型仓库，并确保已安装`git-lfs`：

```bash
git clone https://huggingface.co/openbmb/MiniCPM-V-2_6
```

#### ModelScope

也可以通过ModelScope克隆模型仓库：

```bash
git clone https://www.modelscope.cn/models/openbmb/minicpm-v-2_6
```

### 2. 下载安装我的autoawq的分支

安装`AutoAWQ`的一个分支，该分支已提交PR，等待官方合并：

```bash
git clone https://github.com/LDLINGLINGLING/AutoAWQ.git
cd AutoAWQ
pip install -e .
```


### 3. 开始量化

#### 修改量化脚本参数

修改`AutoAWQ/examples/minicpmv2.6_quantize.py`中的参数：

```python
 parser.add_argument('--model-path', type=str, default="/root/ld/ld_model_pretrained/Minicpmv2_6",
                        help='Path to the model directory.')
parser.add_argument('--quant-path', type=str, default="/root/ld/ld_model_pretrained/Minicpmv2_6_awq_new",
                        help='Path to save the quantized model.')
# 修改以上模型地址和量化后保存地址
```

#### 运行量化脚本

运行量化脚本(需要访问huggingface)：

```bash
cd  AutoAWQ/examples
python minicpmv2.6_quantize.py
```

量化完成后，在`quant_path`下将会得到您的AWQ量化模型。

#### 显存占用

量化过程中显存占用仅为7.3GB。

以上模型可以直接使用[vllm](../../inference/minicpmv2.6/vllm.md)进行推理，


