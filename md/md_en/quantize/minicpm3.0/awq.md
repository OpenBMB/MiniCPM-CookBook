## AutoAWQ (Slightly Lower Speed)

### Device Requirements
At least one NVIDIA 20-series or higher GPU is required, with 12GB of VRAM for quantizing 4-bit models.

1. **Obtain MiniCPM Open Source Code**
   ```bash
   git clone https://github.com/OpenBMB/MiniCPM
   ```

2. **Install the AutoAWQ Branch**
   ```bash
   git clone https://github.com/LDLINGLINGLING/AutoAWQ.git
   cd AutoAWQ
   git checkout minicpm3
   pip install -e .
   ```

3. **Obtain MiniCPM Model Weights, Example: MiniCPM3-4B**
   ```bash
   git clone https://huggingface.co/openbmb/MiniCPM3-4B
   ```

4. **Modify Script Configuration Parameters**
   In the `MiniCPM/quantize/awq_quantize.py` file, modify the configuration parameters according to the comments:
   1. Provides three options for the quantization dataset: Alpaca, Wikitext, and Custom Dataset.
   2. If using Alpaca, modify `quant_data_path` to the absolute path of `MiniCPM/quantize/quantize_data/alpaca`.
   3. If using Wikitext, modify `quant_data_path` to the absolute path of `MiniCPM/quantize/quantize_data/wikitext`.
   4. For a custom dataset, fill in `custom_data` as shown in the example.
   ```python
   model_path = '/root/ld/ld_model_pretrained/minicpm3'  # model_path or model_id
   quant_path = '/root/ld/ld_model_pretrained/minicpm3_awq'  # quant_save_path
   quant_data_path = '/root/ld/ld_project/pull_request/MiniCPM/quantize/quantize_data/alpaca'  # Enter the path to the built-in Wikitext or Alpaca dataset
   quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }  # "w_bit": 4 or 8
   quant_samples = 512  # Number of samples to use for calibration
   custom_data = [  # First custom data set; if using a custom dataset, modify according to the following format
       [
           {"role": "system", "content": "You are a helpful assistant."},
           {"role": "assistant", "content": "Hello, what can I assist you with?"},
           {"role": "user", "content": "I want to learn how to write Python code."},
       ],  # Second custom data set
       [
           {"role": "system", "content": "You are a helpful assistant."},
           {"role": "assistant", "content": "Hello, what can I assist you with?"},
           {"role": "user", "content": "I want to learn how to write Python code."},
       ]
       # ... More custom data
   ]
   ```

5. **Prepare the Quantization Calibration Set**
   Based on the chosen dataset, there are three options provided: Wikitext, Alpaca, and Custom Dataset. Replace the 69th line in `quantize/awq_quantize.py` with one of the following lines of code:
   ```python
   # Use Wikitext for quantization
   model.quantize(tokenizer, quant_config=quant_config, calib_data=load_wikitext(quant_data_path=quant_data_path))
   # Use Alpaca for quantization
   model.quantize(tokenizer, quant_config=quant_config, calib_data=load_alpaca(quant_data_path=quant_data_path))
   # Use a custom dataset for quantization
   model.quantize(tokenizer, quant_config=quant_config, calib_data=load_cust_data(custom_data=custom_data))
   ```
