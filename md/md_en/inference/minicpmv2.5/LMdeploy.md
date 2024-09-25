# Concurrent Inference Deployment

## System Requirements
- A single 24GB NVIDIA 20-series or higher GPU, or multiple 12GB 20-series or higher GPUs

## Step 1: Install `deploy`

Install the `deploy` library, note that source compilation is not recommended:

```sh
pip install deploy
```

## Step 2: Concurrent Inference Code

The following is an example of concurrent inference code:

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

# You can replace MiniCPM-Llama3-V 2.5 with a local path
# session_len=2048 represents the context length
# tp=8 represents the number of GPUs used, which must be a power of 2, such as 1, 2, 4, 8
pipe = pipeline('MiniCPM-Llama3-V 2.5',
                backend_config=TurbomindEngineConfig(session_len=2048, tp=8))

# URLs or local paths of images
image_urls = [
    "/root/ld/ld_project/MiniCPM-V/assets/minicpmv2-cases.png",
    "/root/ld/ld_project/MiniCPM-V/assets/llavabench_compare_phi3.png",
    "/root/ld/ld_project/MiniCPM-V/assets/MiniCPM-Llama3-V-2.5-peformance.png",
    "/root/ld/ld_project/MiniCPM-V/assets/zhihu.webp",
    "/root/ld/ld_project/MiniCPM-V/assets/thunlp.png"
]

prompts = [('describe this image', load_image(img_url)) for img_url in image_urls]
response = pipe(prompts)
print([i.text for i in response])
```

Now you are ready to start efficient concurrent inference with the provided code!
