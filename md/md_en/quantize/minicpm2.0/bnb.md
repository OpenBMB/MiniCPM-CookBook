# MiniCPM Model Quantization Guide - Using bitsandbytes (BNB)

To perform quantization of the MiniCPM model, you need to follow the steps below and ensure that your device meets the following requirements:
- At least one Nvidia 20-series or higher GPU;
- Sufficient VRAM to load the model.

## 1. Install bitsandbytes

Install the `bitsandbytes` library:

```bash
pip install bitsandbytes
```

## 2. Modify Quantization Script Parameters

In the `MiniCPM/quantize/bnb_quantize.py` file, modify the following parameters:

```python
model_path = "/root/ld/ld_model_pretrain/MiniCPM-1B-sft-bf16"  # Path to the model
save_path = "/root/ld/ld_model_pretrain/MiniCPM-1B-sft-bf16_int4"  # Path to save the quantized model
```

## 3. Additional Quantization Parameters

Adjust the parameters based on the comments and the `llm.int8()` algorithm:

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Whether to perform 4-bit quantization
    load_in_8bit=False,  # Whether to perform 8-bit quantization
    bnb_4bit_compute_dtype=torch.float16,  # Computation precision
    bnb_4bit_quant_storage=torch.uint8,  # Storage format for quantized weights
    bnb_4bit_quant_type="nf4",  # Quantization type, here using normalized float 4-bit
    bnb_4bit_use_double_quant=True,  # Whether to use double quantization, i.e., quantize zeropoint and scaling parameters
    llm_int8_enable_fp32_cpu_offload=False,  # Whether to use int8 for the model and offload parameters to FP32 on CPU
    llm_int8_has_fp16_weight=False,  # Whether to enable mixed precision
    # llm_int8_skip_modules=["out_proj", "kv_proj", "lm_head"],  # Modules to skip quantization
    llm_int8_threshold=6.0,  # Outlier threshold in the llm.int8() algorithm, used to determine whether to quantize
)
```

## 4. Run the Quantization Script

Navigate to the MiniCPM quantization directory and run the quantization script:

```bash
cd MiniCPM/quantize
python bnb_quantize.py
```

After completing the above steps, you will have a quantized MiniCPM model.
