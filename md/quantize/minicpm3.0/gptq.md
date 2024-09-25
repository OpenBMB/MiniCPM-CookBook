## AutoGPTQ

### 设备要求
需要至少一个NVIDIA 20系列或更高版本的GPU，并且具有超过12GB的显存。

### 方法1：直接获取量化后的GPTQ权重（推荐）
```bash
git clone https://huggingface.co/openbmb/MiniCPM3-4B-GPTQ-Int4
```

### 方法2：自行量化（推荐在进行微调后使用）

1. **获取MiniCPM模型权重**
   ```bash
   git clone https://huggingface.co/openbmb/MiniCPM3-4B
   ```

2. **获取量化脚本**
   ```bash
   git clone https://github.com/OpenBMB/MiniCPM
   ```

3. **安装AutoGPTQ分支**
   在这里，您将从我的fork分支获取代码。（已提交PR）
   ```bash
   git clone https://github.com/LDLINGLINGLING/AutoGPTQ.git
   cd AutoGPTQ
   git checkout minicpm3
   pip install -e .
   ```

4. **开始量化**
   ```bash
   cd MiniCPM/quantize
   # 在以下命令中，将no_quant_model_path修改为保存MiniCPM3权重的位置，将quant_save_path修改为保存量化后模型的目录。
   python gptq_quantize.py --pretrained_model_dir no_quant_model_path --quantized_model_dir quant_save_path --bits 4
   ```
