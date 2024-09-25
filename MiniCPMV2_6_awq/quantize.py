

from datasets import load_dataset
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import os
import shutil


model_path = '/root/ld/ld_model_pretrain/MiniCPM-V-2_6'
quant_path = '/root/ld/ld_model_pretrain/MiniCPM-V-2_6_awq'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,device_map={"": "cuda:0"})
# 保证原始模型的各个文件不遗漏保存到merge_path中
def copy_files_not_in_B(A_path, B_path):
    """
    Copies files from directory A to directory B if they exist in A but not in B.

    :param A_path: Path to the source directory (A).
    :param B_path: Path to the destination directory (B).
    """
    # 保证路径存在
    if not os.path.exists(A_path):
        raise FileNotFoundError(f"The directory {A_path} does not exist.")
    if not os.path.exists(B_path):
        os.makedirs(B_path)

    # 获取路径A中所有非权重文件
    files_in_A = os.listdir(A_path)
    files_in_A = set([file for file in files_in_A if not (".bin" in file or "safetensors" in file )])
    # List all files in directory B
    files_in_B = set(os.listdir(B_path))

    # 找到所有A中存在但B中不存在的文件
    files_to_copy = files_in_A - files_in_B

    # 将这些文件复制到B路径下
    for file in files_to_copy:
        src_file = os.path.join(A_path, file)
        dst_file = os.path.join(B_path, file)
        shutil.copy2(src_file, dst_file)
# Define data loading methods
def load_alpaca():
    data = load_dataset('/root/ld/pull_request/MiniCPM/quantize/quantize_data/alpaca', split="train")

    # concatenate data
    def concatenate_data(x):
        msgs=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": x['input']},{"role": "system", "content": x['output']}]
        data=tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return {"text": data}
    
    concatenated = data.map(concatenate_data)
    return [text for text in concatenated["text"]][:1000]

def load_wikitext():
    data = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
    return [text for text in data["text"] if text.strip() != '' and len(text.split(' ')) > 20]

# Quantize
model.quantize(tokenizer, quant_config=quant_config, calib_data=load_alpaca())

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')

copy_files_not_in_B(model_path,quant_path)