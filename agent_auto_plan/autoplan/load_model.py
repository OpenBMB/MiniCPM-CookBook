from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig,AutoModel
import torch
from sentence_transformers import SentenceTransformer
from peft import PeftModel

def get_model(args):
    embeding_model=SentenceTransformer(args.embeding_model_path)
    for _ in range(10):  # 网络不稳定，多试几次
        try:#加载执行模块基座模型
            # merge_model,model,tokenizer=get_merge_model('/ai/ld/pretrain/Qwen-14B-Chat/','/ai/ld/remote/Qwen-main/output_qwen/checkpoint-400')
            tokenizer = AutoTokenizer.from_pretrained(args.execute_model_path, trust_remote_code=True)
            generation_config = GenerationConfig.from_pretrained(args.execute_model_path, trust_remote_code=True)
            max_memory = f"{int(torch.cuda.mem_get_info()[0] / 1024 ** 3) - 2}GB"
            model = AutoModel.from_pretrained(args.execute_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
            model = model.to(device="cuda:0")
            generation_config.max_length=4096
            model.generation_config = generation_config
            
            model.generation_config.do_sample = False

            #加载任务分解全量微调模型
            if args.allparams_split_task_chain:
                merge_model=AutoModelForCausalLM.from_pretrained(
                    args.allparams_split_task_chain,
                    device_map="cuda:0",
                    max_memory=max_memory,
                    trust_remote_code=True,
                    #use_safetensors=True,
                    bf16=True
                ).eval()
                merge_generation_config = GenerationConfig.from_pretrained(args.allparams_split_task_chain, trust_remote_code=True)
                merge_model.generation_config = merge_generation_config
                merge_model.generation_config.do_sample = False
                merge_model.generation_config.eos_token_id=[2512,19357,151643]
                merge_tokenizer = AutoTokenizer.from_pretrained(args.allparams_split_task_chain, trust_remote_code=True)
            
            #加载任务分解loar微调模型，已执行模型为基座模型
            elif args.lora_split_task_chain:
                merge_model = PeftModel.from_pretrained(model, args.lora_split_task_chain)
                merge_tokenizer = AutoTokenizer.from_pretrained(args.execute_model_path, trust_remote_code=True)
                merge_model.generation_config.eos_token_id=[2512,19357,151643]
                merge_model.generation_config.do_sample = False
            
            #没有任务分解模型
            else:
                merge_model=None
                merge_tokenizer=None
            break
            
        except Exception:
            print('加载错误')
            pass
    return model,merge_model,embeding_model,tokenizer,merge_tokenizer