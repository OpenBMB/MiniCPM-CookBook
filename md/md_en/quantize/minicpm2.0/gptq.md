# MiniCPM Model Quantization Guide - Using AutoGPTQ

To perform quantization of the MiniCPM model, you need to follow the steps below and ensure that your device meets the following requirements:
- At least one Nvidia 20-series or higher GPU;
- 6GB of VRAM for quantizing 2-bit models;
- 4GB of VRAM for quantizing 1-bit models.

## 1. Obtain MiniCPM Model Weights

For example, using MiniCPM-2B-sft:

```bash
git clone https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16
```

## 2. Obtain the Quantization Script

Since AutoGPTQ is no longer being updated, we will use a branch version:

```bash
git clone -b minicpm_gptq https://github.com/LDLINGLINGLING/AutoGPTQ
```

## 3. Install the AutoGPTQ Branch

Navigate to the AutoGPTQ directory and install the dependencies:

```bash
cd AutoGPTQ
git checkout minicpm_autogptq
pip install -e .
```

## 4. Start Quantization

Navigate to the MiniCPM quantization directory and modify the path parameters in the quantization script:

```bash
cd MiniCPM/quantize
```

Run the quantization script, modifying `no_quant_model_path` to the path where the unquantized MiniCPM model weights are saved, and `quant_save_path` to the path where the quantized model will be saved:

```bash
python gptq_quantize.py --pretrained_model_dir no_quant_model_path --quantized_model_dir quant_save_path --bits 4
```

After completing the above steps, you will have obtained the quantized MiniCPM model weights.
