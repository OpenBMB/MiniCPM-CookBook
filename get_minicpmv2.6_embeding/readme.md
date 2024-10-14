### MiniCPM-V embeding Project Operation Guide

#### 1. Download the Project Code

First, you need to clone the `MiniCPM-CookBook` project code from GitHub.

```sh
git clone https://github.com/OpenBMB/MiniCPM-CookBook
```

#### 2. Replace the Original Model Code

Next, replace the `modeling_minicpmv.py` file in your local `MiniCPMV2.6` project with the one from the downloaded project.

```sh
cp MiniCPM-CookBook/get_minicpmv2.6_embeding/modeling_minicpmv.py /path/to/MiniCPMV2.6/modeling_minicpmv.py
```

Make sure to replace `/path/to/MiniCPMV2.6` with the actual path of your `MiniCPMV2.6` project.

#### 3. Write Model Address and Other Parameters

Modify the `main` function in the `MiniCPM-CookBook/get_minicpmv2.6_embeding/inference.py` file to set the following parameters:

```python
def main() -> None:
    images = ['/root/ld/ld_dataset/30k_data/60938244/42.jpg']  # List of image paths, example: ['/ld/image_path/1.jpg', '/ld/image_path/2.jpg']
    queries = ['hello']  # List of text queries, example: ["There is a black and white dog in the picture", "A child is eating a lollipop"]
    model_name = "/root/ld/ld_model_pretrain/MiniCPM-V-2_6"  # Path to the model
```

#### 4. Run `inference.py` to Get Embedding Vectors

In the `inference.py` file, add the following code to obtain the embedding vectors for images and text:

```python
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
def main() -> None:
    images = ['/root/ld/ld_dataset/30k_data/60938244/42.jpg']  # List of image paths, example: ['/ld/image_path/1.jpg', '/ld/image_path/2.jpg']
    queries = ['hello']  # List of text queries, example: ["There is a black and white dog in the picture", "A child is eating a lollipop"]
    model_name = "/root/ld/ld_model_pretrain/MiniCPM-V-2_6"  # Path to the model

    # Load the model
    model = ...  # Load the model according to your model loading method
    model.to("cuda")

    # Image data loader
    image_dataloader = DataLoader(
        images,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: data_collator_image(x),
    )

    # Get image embedding vectors
    for batch_img in tqdm(image_dataloader):
        batch_img = {k: (v.to("cuda") if isinstance(v, torch.Tensor) else v) for k, v in batch_img.items()}
        with torch.no_grad():
            embeddings_img = model.get_vllm_embedding(batch_img)  # Here we get the image vectors
            print(embeddings_img)

    # Text data loader
    dataloader = DataLoader(
        queries_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: data_collator_query(x),
    )

    # Get text embedding vectors
    for batch_text in tqdm(dataloader):
        with torch.no_grad():
            batch_text = batch_text.to("cuda")
            embeddings_query = model(data=batch_text, use_cache=False).logits  # Here we get the text vectors
            print(embeddings_query)

if __name__ == '__main__':
    main()
```

### Explanation

1. **Download the Project Code**: Use the `git clone` command to clone the project from GitHub.
2. **Replace the Original Model Code**: Copy the `modeling_minicpmv.py` file from the downloaded project to the corresponding location in your local `MiniCPMV2.6` project.
3. **Write Model Address and Other Parameters**: Modify the `inference.py` file's `main` function to set the image paths, text queries, and model path.
4. **Run `inference.py` to Get Embedding Vectors**: In the `inference.py` file, add code to load the model, create data loaders, and obtain the embedding vectors for images and text.
