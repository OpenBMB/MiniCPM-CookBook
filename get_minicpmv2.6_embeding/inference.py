import torch
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel,AutoTokenizer
from dataset import ImageDataset,QueryDataset,data_collator,data_collator_query
from torchvision import transforms

def build_transform():
    IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5) # timm.data.IMAGENET_INCEPTION_MEAN
    IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)  # timm.data.IMAGENET_INCEPTION_STD
    return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
                ),
            ]
        )
def main() -> None:
    images = ['/root/ld/ld_dataset/30k_data/60938244/42.jpg'] # images path list,exmaple:[/ld/image_path/1.jpg,/ld/image_path/2.jpg]
    queries = ['你好'] # text list ,exmaple;["图中有一只黑白相间的狗"，"一个小孩子在吃棒棒糖"]
    model_name = "/root/ld/ld_model_pretrain/MiniCPM-V-2_6" #


    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cuda",trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    
    if hasattr(model.config, "slice_config"):
        slice_config = model.config.slice_config.to_dict()
    else:
        slice_config = model.config.to_dict()
    transform_func = build_transform()
    images_dataset=ImageDataset(images,transform_func,tokenizer,slice_config,llm_type='qwen2',batch_vision=True)
    queries_dataset=QueryDataset(queries,tokenizer,llm_type='qwen2')

    image_dataloader = DataLoader(
        images_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: data_collator(x),
    )
    for batch_img in tqdm(image_dataloader):
        batch_img = {k: (v.to("cuda") if isinstance(v, torch.Tensor) else v) for k, v in batch_img.items()}
        with torch.no_grad():
            embeddings_img = model.get_vllm_embedding(batch_img) # 这里是

            
    dataloader = DataLoader(
        queries_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: data_collator_query(x),
    )

    for batch_text in tqdm(dataloader):
        with torch.no_grad():
            batch_text=batch_text.to("cuda")
            embeddings_query = model(data = batch_text, use_cache=False).logits
        
if __name__ == "__main__":
    typer.run(main)
