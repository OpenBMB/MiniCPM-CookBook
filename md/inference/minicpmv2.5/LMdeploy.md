
# LMdeploy并发推理

## 设备要求
- 单张24GB NVIDIA 20系以上显卡，或者两张以上12GB 20系以上显卡

## 步骤1：安装deploy

安装`deploy`库，注意不要使用源码编译：

```sh
pip install deploy
```

## 步骤2：并发推理代码

以下是一个并发推理的代码示例：

```python
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

# 可将MiniCPM-Llama3-V 2.5换成本地路径
# session_len=2048 代表上下文长度
# tp=8 代表使用显卡数，必须是2**n，比如1，2，4，8
pipe = pipeline('MiniCPM-Llama3-V 2.5',
                backend_config=TurbomindEngineConfig(session_len=2048, tp=8),)

# 图片的URL或者本地路径
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

现在您可以开始使用并发推理代码进行高效推理了！
