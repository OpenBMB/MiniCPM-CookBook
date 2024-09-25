
## BNB量化

### 设备要求
20以上显卡，显存能够加载模型。

1. **安装bitsandbytes**
   ```bash
   pip install bitsandbytes
   ```

2. **修改量化脚本参数**
   修改`MiniCPM/quantize/bnb_quantize.py` 中以下参数
   ```python
   model_path = "/root/ld/ld_model_pretrain/MiniCPM3-4B"  # 模型下载地址
   save_path = "/root/ld/ld_model_pretrain/MiniCPM3-4B-int4"  # 量化模型保存地址
   ```

3. **更多量化参数可根据注释以及llm.int8()算法进行修改**
   ```python
   quantization_config = BitsAndBytesConfig(
       load_in_4bit=True,  # 是否进行4bit量化
       load_in_8bit=False,  # 是否进行8bit量化
       bnb_4bit_compute_dtype=torch.float16,  # 计算精度设置
       bnb_4bit_quant_storage=torch.uint8,  # 量化权重的储存格式
       bnb_4bit_quant_type="nf4",  # 量化格式，这里用的是正态分布的int4
       bnb_4bit_use_double_quant=True,  # 是否采用双量化，即对zeropoint和scaling参数进行量化
       llm_int8_enable_fp32_cpu_offload=False,  # 是否llm使用int8，cpu上保存的参数使用fp32
       llm_int8_has_fp16_weight=False,  # 是否启用混合精度
       # llm_int8_skip_modules=["out_proj", "kv_proj", "lm_head"],  # 不进行量化的模块
       llm_int8_threshold=6.0,  # llm.int8()算法中的离群值，根据这个值区分是否进行量化
   )
   ```

4. **运行以下代码，执行量化**
   ```bash
   cd MiniCPM/quantize
   python bnb_quantize.py
   ```
