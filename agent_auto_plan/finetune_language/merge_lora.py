from peft import PeftModel
from transformers import AutoModel, AutoTokenizer
import os
import shutil

model_type = "/root/ld/ld_model_pretrained/Minicpmv2_6"  # Local model path or huggingface id
path_to_adapter = "/root/ld/ld_project/pull_request/MiniCPM-V/finetune/output/output__lora/checkpoint-200"  # Path to the saved LoRA adapter
merge_path = "/root/ld/ld_project/pull_request/MiniCPM-V/finetune/output/merge"  # Path to save the merged model

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
    files_in_A = set([file for file in files_in_A if not (".bin" in file or "safetensors" in file)])
    # List all files in directory B
    files_in_B = set(os.listdir(B_path))

    # 找到所有A中存在但B中不存在的文件
    files_to_copy = files_in_A - files_in_B

    # 将这些文件复制到B路径下
    for file in files_to_copy:
        src_file = os.path.join(A_path, file)
        dst_file = os.path.join(B_path, file)
        shutil.copy2(src_file, dst_file)

# 加载原始模型
model = AutoModel.from_pretrained(
    model_type,
    trust_remote_code=True
)

# 加载lora模块到原始模型中
lora_model = PeftModel.from_pretrained(
    model,
    path_to_adapter,
    device_map="auto",
    trust_remote_code=True
).eval()

# 将加载的lora模块合并到原始模型中
merge_model = lora_model.merge_and_unload()

# 将新合并的模型进行保存
merge_model.save_pretrained(merge_path, safe_serialization=False)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)
tokenizer.save_pretrained(merge_path)

copy_files_not_in_B(model_type,merge_path)