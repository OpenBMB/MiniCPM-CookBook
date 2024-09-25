import os

import gradio as gr
import torch
from pdf2image import convert_from_path
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor
from transformers import AutoModel,AutoTokenizer
from peft import PeftModel
from dataset import ImageDataset,QueryDataset,load_from_pdf,data_collator,data_collator_query,load_from_json
from utils import build_transform,evaluate_colbert
import json


def search(query: str, ds, images):
    qs = []
    queries_dataset=QueryDataset([query],tokenizer)
    dataloader = DataLoader(
        queries_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: data_collator_query(x),
    )
    for batch_query in tqdm(dataloader):
        with torch.no_grad():
            batch_query=batch_query.to("cuda")
            embeddings_query = model.base_model(data = batch_query, use_cache=False).half()
            embeddings_query = model.text_proj(embeddings_query)
        qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

    # run evaluation
    scores = evaluate_colbert(qs, ds)
    best_page = int(scores.argmax(axis=1).item())
    return f"The most relevant page is {best_page}", images[best_page]


def index(file, ds):
    """Example script to run inference with ColPali"""
    images = []
    for f in file:
        if f.endswith(".json"):
            with open(f, 'r', encoding='utf-8') as fi:
                data = json.load(fi)
            images.extend([d['image'] for d in data])
        if f.endswith(".pdf"):
            images.extend(convert_from_path(f))
        if f.endswith('png') or f.endswith('jpg'):
            images.append(f)
    if hasattr(model.config, "slice_config"):
        slice_config = model.config.slice_config.to_dict()
    else:
        slice_config = model.config.to_dict()
    transform_func = build_transform()
    images_dataset=ImageDataset(images,transform_func,tokenizer,slice_config)
    image_dataloader = DataLoader(
        images_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=lambda x: data_collator(x,padding_value=0, max_length=2048,device=device),
    )
    ds = []
    for batch_doc in tqdm(image_dataloader):
        with torch.no_grad():
            embeddings_doc = model.base_model(data = batch_doc, use_cache=False).half()
            embeddings_doc = model.text_proj(embeddings_doc)
        ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
    del embeddings_doc
    return f"Uploaded and converted {len(images)} pages", ds, images


COLORS = ["#4285f4", "#db4437", "#f4b400", "#0f9d58", "#e48ef1"]
# Load model
# Load model and lora
model_name = "/root/ld/ld_model_pretrained/minicpm-v"
lora_path = "/root/ld/ld_model_pretrained/adapter_minicpmv_search"
model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cuda",trust_remote_code=True).eval()
model = PeftModel.from_pretrained(model, lora_path)
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
# 加载权重文件
text_proj_weights = torch.load(os.path.join(lora_path,"text_proj.pth"))

# 获取 text_proj 层
text_proj_layer = model.text_proj

# 更新 text_proj 层的权重
text_proj_layer.load_state_dict(text_proj_weights)

device = model.device
mock_image = Image.new("RGB", (448, 448), (255, 255, 255))

with gr.Blocks() as demo:
    gr.Markdown("# ColCPM-V: Cross-modal HD retrieval ")
    gr.Markdown("## 1️⃣ Upload PDFs")
    file = gr.File(file_types=["pdf","json",'jpg','png'], file_count="multiple")

    gr.Markdown("## 2️⃣ Convert the PDFs and upload")
    convert_button = gr.Button("🔄 Convert and upload")
    message = gr.Textbox("Files not yet uploaded")
    embeds = gr.State(value=[])
    imgs = gr.State(value=[])

    # Define the actions
    convert_button.click(index, inputs=[file, embeds], outputs=[message, embeds, imgs])

    gr.Markdown("## 3️⃣ Search")
    query = gr.Textbox(placeholder="Enter your query here")
    search_button = gr.Button("🔍 Search")
    message2 = gr.Textbox("Query not yet set")
    output_img = gr.Image()

    search_button.click(search, inputs=[query, embeds, imgs], outputs=[message2, output_img])


if __name__ == "__main__":
    demo.queue(max_size=10).launch(debug=True)
