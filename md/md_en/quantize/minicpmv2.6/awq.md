# AutoAWQ Model Quantization Deployment Guide

## Method 1 (Recommended)

### 1. Directly Download the Pre-Quantized Model

Clone the pre-quantized model repository using Git:

```bash
git clone https://www.modelscope.cn/models/linglingdan/MiniCPM-V_2_6_awq_int4
```

### 2. Download and Build the Forked autoawq
Install a fork of AutoAWQ, which has a PR submitted and is awaiting official merge:
```bash
git clone https://github.com/LDLINGLINGLING/AutoAWQ.git
cd AutoAWQ
git checkout minicpmv2.6
pip install -e .
```

### 3. The Above Model Can Be Used Directly with [vllm](../../inference/minicpmv2.6/vllm.md) for Inference

## Method 2 (For Quantizing Trained Models, Recommended This Method)

### 1. Download the Non-Quantized Model
Hugging Face
Clone the model repository via Git and ensure git-lfs is installed:
```bash
git clone https://huggingface.co/openbmb/MiniCPM-V-2_6
```

ModelScope
Alternatively, clone the model repository via ModelScope:
```bash
git clone https://www.modelscope.cn/models/openbmb/minicpm-v-2_6
```

### 2. Download and Install My Fork of autoawq
Install a fork of AutoAWQ, which has a PR submitted and is awaiting official merge:
```bash
git clone https://github.com/LDLINGLINGLING/AutoAWQ.git
cd AutoAWQ
git checkout minicpmv2.6
pip install -e .
```

### 3. Quantize the Model
Modify Quantization Script Parameters
Modify the parameters in AutoAWQ/examples/minicpmv2.6_quantize.py:
```python
parser.add_argument('--model-path', type=str, default="/root/ld/ld_model_pretrained/Minicpmv2_6",
                    help='Path to the model directory.')
parser.add_argument('--quant-path', type=str, default="/root/ld/ld_model_pretrained/Minicpmv2_6_awq_new",
                    help='Path to save the quantized model.')
# Update the model path and the path to save the quantized model
```

Run the Quantization Script
Run the quantization script (requires access to Hugging Face):

```bash
cd AutoAWQ/examples
python minicpmv2.6_quantize.py
```

### 4. GPU Memory Usage
The GPU memory usage during quantization is only 7.3GB.

The above model can be used directly with [vllm](../../inference/minicpmv2.6/vllm.md) for inference.