## AutoGPTQ

### Device Requirements
At least one NVIDIA 20-series or higher GPU with more than 12GB of VRAM is required.

### Method 1: Directly Obtain Quantized GPTQ Weights (Recommended)
```bash
git clone https://huggingface.co/openbmb/MiniCPM3-4B-GPTQ-Int4
```

### Method 2: Quantize by Yourself (Recommended after Fine-Tuning)

1. **Acquire MiniCPM Model Weights**
   ```bash
   git clone https://huggingface.co/openbmb/MiniCPM3-4B
   ```

2. **Acquire Quantization Script**
   ```bash
   git clone https://github.com/OpenBMB/MiniCPM
   ```

3. **Install the AutoGPTQ Branch**
   Here, you will get the code from my forked branch. (A PR has been submitted)
   ```bash
   git clone https://github.com/LDLINGLINGLING/AutoGPTQ.git
   cd AutoGPTQ
   git checkout minicpm3
   pip install -e .
   ```

4. **Start Quantization**
   ```bash
   cd MiniCPM/quantize
   # In the following command, modify no_quant_model_path to the location where the MiniCPM3 weights are saved, and quant_save_path to the directory where the quantized model will be saved.
   python gptq_quantize.py --pretrained_model_dir no_quant_model_path --quantized_model_dir quant_save_path --bits 4
   ```
