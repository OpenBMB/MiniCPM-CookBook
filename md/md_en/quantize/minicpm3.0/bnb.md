## BNB Quantization

### Device Requirements
- At least one Nvidia 20-series or higher GPU;
- Sufficient VRAM to load the model.

1. **Install bitsandbytes**
   ```bash
   pip install bitsandbytes
   ```

2. **Modify Quantization Script Parameters**
   Modify the following parameters in `MiniCPM/quantize/bnb_quantize.py`:
   ```python
   model_path = "/root/ld/ld_model_pretrain/MiniCPM3-4B"  # Path to the model
   save_path = "/root/ld/ld_model_pretrain/MiniCPM3-4B-int4"  # Path to save the quantized model
   ```

3. **Additional Quantization Parameters Can Be Modified According to Comments and the llm.int8() Algorithm**
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

4. **Run the Following Code to Perform Quantization**
   ```bash
   cd MiniCPM/quantize
   python bnb_quantize.py
   ```
