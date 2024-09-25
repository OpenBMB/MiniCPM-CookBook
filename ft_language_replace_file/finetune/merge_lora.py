from peft import PeftModel
import torch
from transformers import AutoModel,AutoTokenizer
model_type="/root/ld/ld_model_pretrained/MiniCPM-Llama3-V-2_5_int4" # or openbmb/MiniCPM-V-2
path_to_adapter="/root/ld/ld_project/MiniCPM-V/finetune/output/output_minicpmv2_lora/checkpoint-500"
merge_path="/root/ld/ld_project/MiniCPM-V/finetune/output/merge_model_path"

model =  AutoModel.from_pretrained(
        model_type,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cuda:0"
        )

merge_model = PeftModel.from_pretrained(
    model,
    path_to_adapter,
    device_map="auto",
    trust_remote_code=True
).eval()
#保存新的模型，与原始MiniCPM-Llama3-V-2_5架构相同
merge_model=merge_model.merge_and_unload()
merge_model.save_pretrained(merge_path,safe_serialization=False)

# 加载分词文件与模型保存到merge后的模型地址
tokenizer=AutoTokenizer.from_pretrained(model_type,trust_remote_code=True)
tokenizer.save_pretrained(merge_path)