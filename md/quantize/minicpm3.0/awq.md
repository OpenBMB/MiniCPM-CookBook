
## AutoAWQ（速度略低）

### 设备要求
至少存在一张NVIDIA 20系以上显卡，量化4B需要12GB显存。

1. **获取MiniCPM开源代码**
   ```bash
   git clone https://github.com/OpenBMB/MiniCPM
   ```

2. **安装AutoAWQ分支**
   ```bash
   git clone https://github.com/LDLINGLINGLING/AutoAWQ.git
   cd AutoAWQ
   git checkout minicpm3
   pip install -e .
   ```

3. **获取MiniCPM模型权重，以MiniCPM3-4B为例**
   ```bash
   git clone https://huggingface.co/openbmb/MiniCPM3-4B
   ```

4. **修改脚本配置参数**
   在`MiniCPM/quantize/awq_quantize.py` 文件中根据注释修改配置参数：
   1. 提供Alpaca、Wikitext、自定义数据集三种方式
   2. 如果使用Alpaca，`quant_data_path`修改成`MiniCPM/quantize/quantize_data/alpaca`的绝对路径
   3. 如果使用Wikitext，`quant_data_path`修改成`MiniCPM/quantize/quantize_data/wikitext`的绝对路径
   4. 使用自定义数据集，按照示例补充`custom_data`即可
   ```python
   model_path = '/root/ld/ld_model_pretrained/minicpm3'  # model_path or model_id   
   quant_path = '/root/ld/ld_model_pretrained/minicpm3_awq'  # quant_save_path  
   quant_data_path = '/root/ld/ld_project/pull_request/MiniCPM/quantize/quantize_data/alpaca'  # 写入自带Wikitext或者Alpaca地址
   quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }  # "w_bit":4 or 8  
   quant_samples = 512  # how many samples to use for calibration 
   custom_data = [  # first custom data 如果要使用自定义数据集，根据以下格式修改
       [
           {"role": "system", "content": "You are a helpful assistant."},
           {"role": "assistant", "content": "你好，有什么我可以帮助你的吗？"},
           {"role": "user", "content": "我想了解如何编写Python代码。"},
       ],  # second custom data
       [
           {"role": "system", "content": "You are a helpful assistant."},
           {"role": "assistant", "content": "你好，有什么我可以帮助你的吗？"},
           {"role": "user", "content": "我想了解如何编写Python代码。"},
       ]
       # .... more custom data
   ]
   ```

5. **制作量化校准集**
   根据选择的数据集，提供了Wikitext、Alpaca以及自定义数据集三种，选择以下某一行代码替换 `quantize/awq_quantize.py` 中第69行：
   ```python
   # 使用Wikitext进行量化
   model.quantize(tokenizer, quant_config=quant_config, calib_data=load_wikitext(quant_data_path=quant_data_path))
   # 使用Alpaca进行量化
   model.quantize(tokenizer, quant_config=quant_config, calib_data=load_alpaca(quant_data_path=quant_data_path))
   # 使用自定义数据集进行量化
   model.quantize(tokenizer, quant_config=quant_config, calib_data=load_cust_data(custom_data=custom_data))
   ```

