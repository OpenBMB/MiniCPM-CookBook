import torch
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor
from PIL import Image
import sys,os,json
from transformers import AutoModel,AutoTokenizer
from peft import PeftModel
from utils import build_transform,evaluate_colbert
from dataset import ImageDataset,QueryDataset,load_from_pdf,data_collator,data_collator_query,load_from_json
def main() -> None:
    # Load model and lora
    model_name = "/root/ld/ld_model_pretrained/minicpm-v"
    lora_path = "/root/ld/ld_project/pull_request/MiniCPM-V/finetune/output/output__lora/checkpoint-2000"
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cuda",trust_remote_code=True).eval()
    model = PeftModel.from_pretrained(model, lora_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    # 加载权重文件
    text_proj_weights = torch.load(os.path.join(lora_path,"text_proj.pth"))

    # 获取 text_proj 层
    text_proj_layer = model.text_proj

    # 更新 text_proj 层的权重
    text_proj_layer.load_state_dict(text_proj_weights)

    # select images -> load_from_pdf(<pdf_path>),  load_from_image_urls(["<url_1>"]), load_from_dataset(<path>)
    
    images = load_from_json("/root/ld/ld_dataset/pdf_cn_30k_search_eval1.json")[0][:30]
    queries = load_from_json('/root/ld/ld_dataset/pdf_cn_30k_search_eval1.json')[1][:30]
    if hasattr(model.config, "slice_config"):
        slice_config = model.config.slice_config.to_dict()
    else:
        slice_config = model.config.to_dict()
    transform_func = build_transform()
    images_dataset=ImageDataset(images,transform_func,tokenizer,slice_config)
    queries_dataset=QueryDataset(queries,tokenizer)
    # run inference - docs
    image_dataloader = DataLoader(
        images_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=lambda x: data_collator(x),
    )
    ds = []
    for batch_doc in tqdm(image_dataloader):
        batch_doc = {k: (v.to("cuda") if isinstance(v, torch.Tensor) else v) for k, v in batch_doc.items()}
        with torch.no_grad():
            embeddings_doc = model.base_model(data = batch_doc, use_cache=False).half()
            embeddings_doc = model.text_proj(embeddings_doc)
        ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
    del embeddings_doc
    # run inference - queries
    dataloader = DataLoader(
        queries_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=lambda x: data_collator_query(x),
    )

    qs = []
    for batch_query in tqdm(dataloader):
        with torch.no_grad():
            batch_query=batch_query.to("cuda")
            embeddings_query = model.base_model(data = batch_query, use_cache=False).half()
            embeddings_query = model.text_proj(embeddings_query)
        qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

    # run evaluation
    scores = evaluate_colbert(qs, ds)
    print(scores)
    print(scores.argmax(axis=1))

if __name__ == "__main__":
    typer.run(main)
