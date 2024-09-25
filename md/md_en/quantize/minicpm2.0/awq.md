# MiniCPM Model Quantization Guide - Using AutoAWQ

To perform quantization of the MiniCPM model, you need to follow the steps below and ensure that your device meets the following requirements:
- At least one Nvidia 20-series or higher GPU;
- 6GB of VRAM for quantizing 2-bit models;
- 4GB of VRAM for quantizing 1-bit models.

## 1. Obtain MiniCPM Open Source Code

```bash
git clone https://github.com/OpenBMB/MiniCPM
```

## 2. Obtain MiniCPM Model Weights

For example, using MiniCPM-2B-sft:

```bash
git clone https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16
```

## 3. Modify Script Configuration Parameters

In the `MiniCPM/quantize/awq_quantize.py` file, modify the configuration parameters according to the comments:

```python
# Path to the pre-quantized model or Hugging Face ID
model_path = '/root/ld/ld_model_pretrained/MiniCPM-1B-sft-bf16'
# Path to save the post-quantized model
quant_path = '/root/ld/ld_project/pull_request/MiniCPM/quantize/awq_cpm_1b_4bit'
# Path to the quantization dataset; choose between wikitext and alpaca in MiniCPM/quantize/quantize_data
quant_data_path = '/root/ld/ld_project/pull_request/MiniCPM/quantize/quantize_data/wikitext'
# Quantization configuration parameters; it is recommended to only modify `w_bit`, the quantization bit width
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,  # Choose 4 or 8
    "version": "GEMM"
}
# Maximum number of samples to use for quantization calibration
quant_samples = 512  # Number of samples to use for calibration
# If using a custom quantization dataset, configure the data in the following format
custom_data = [
    {'question': 'What is your name?', 'answer': 'I am the open-source MiniCPM by OpenBMB.'},
    {'question': 'What are your features?', 'answer': 'I am small but powerful.'}
]
```

## 4. Prepare the Quantization Calibration Set

Based on the chosen dataset, replace the corresponding line in `quantize/awq_quantize.py` (e.g., line 38) with one of the following lines of code:

```python
# Use wikitext for quantization
model.quantize(tokenizer, quant_config=quant_config, calib_data=load_wikitext(quant_data_path=quant_data_path))

# Use alpaca for quantization
model.quantize(tokenizer, quant_config=quant_config, calib_data=load_alpaca(quant_data_path=quant_data_path))

# Use a custom dataset for quantization
model.quantize(tokenizer, quant_config=quant_config, calib_data=load_cust_data(quant_data_path=quant_data_path))
```

## 5. Start Quantization

Run `MiniCPM/quantize/awq_quantize.py` to begin the quantization process. Upon completion, you will obtain the quantized model weights.

Please adjust the paths and configuration parameters according to your actual situation. This guide aims to help you successfully complete the quantization of the MiniCPM model.
