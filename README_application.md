# MiniCPM_Series_Tutorial

本人是openbmb负责开源社区的同学，modelbest(面壁智能)一直致力降低大模型使用门槛，提高模型知识密度，让大模型飞入千家万户。

为此我写了MiniCPM和MiniCPMV的教程，包括推理，量化，边端部署，微调，技术报告，应用六个主题，基于MiniCPM的应用我会上传到本仓库。
配套完整教程地址如下：

https://modelbest.feishu.cn/wiki/D2tFw8Pcsi5CIzkaHNacLK64npg?from=from_copylink

b站配套视频：

https://space.bilibili.com/669720247?spm_id_from=333.1007.0.0

b站up名称：

面壁的车辆工程师

# 有趣的项目
以下项目都是个人原创，如果需要可自取，但是注意保护我的个人知识产权，用了给个星星。

## OCR_VG
同时将OCR和定位任务融合，考虑排版问题，该项目在OCR_VG的文件夹下，在可以自取[文字识别与定位教程](https://modelbest.feishu.cn/wiki/HLRiwNgKEic6cckGyGucFvxQnJw?from=from_copylink)。
### 项目效果
![alt text](./OCR_VG/out/1.jpg)
![alt text](./OCR_VG/out/4.jpg)

## 基于MiniCPMV2.0的跨模态搜索
使用多向量和对比学习的方法，目标是训练一个跨模态端到端搜索的模型，可以理解密集文字、复杂表格。[模型地址](https://www.modelscope.cn/models/linglingdan/Minicpmv_embeding_multi_vector)

### 效果展示：
1. 输入20张待选图片：
![alt text](./OCR_Multimodal_Search/asset/image-2.png)
2. 输入query文字进行搜索:
(![alt text](./OCR_Multimodal_Search/asset/image-1.png))
3. 得到与query最相近的图片。
(![alt text](./OCR_Multimodal_Search/asset/image-3.png))

### 使用教程
见[飞书文档](https://modelbest.feishu.cn/docx/CGEzdu25MoXkoVx3Qoac0e25nvg?from=from_copylink)

## 复杂agent项目
使用minicpmv2.6完成了论文[AutoPlan](https://github.com/LDLINGLINGLING/AutoPlan)的项目，能够对复杂任务进行规划和执行。

### 效果展示：
1. 输入query:
![alt text](./agent_auto_plan/asset/image.png)
2. 获得任务分解
![alt text](./agent_auto_plan/asset/image-1.png)
3. 获得任务执行
![alt text](./agent_auto_plan/asset/image-2.png)
![alt text](./agent_auto_plan/asset/image-3.png)
4. 获得最终答案

![final_anser](./agent_auto_plan/asset/image-4.png)

### 使用教程：
见[飞书文档](https://modelbest.feishu.cn/wiki/IgF0wRGJYizj4LkMyZvc7e2Inoe?from=from_copylink)

## mbti角色扮演
与北大Chatlaw团队每个人格训练一个模型不同，仅使用一个2b模型完成了16种人格的无缝切换（可玩人格分裂）

### 使用教程：
[角色扮演](https://modelbest.feishu.cn/docx/EcNjdGwvwoLkDrxpVrQcLwlknCg?from=from_copylink)

### 效果展示：
![ESTP](./mbti_role_play/demo_img/ESTP.PNG)
![INTJ](./mbti_role_play/demo_img/INTJ.PNG)
![ESTP1](./mbti_role_play/demo_img/ESTP1.PNG)
![INTJ1](./mbti_role_play/demo_img/INTJ1.PNG)

## 混合模态微调
MiniCPMV的微调仅仅开放了图文双模态的训练，本项目修改了纯文本和图文对的混合训练模式，放在了MIniCPM_Series_Tutorial/ft_language_replace_file文件夹下，

### 使用教程：
可以自取[混合模态微调教程](https://modelbest.feishu.cn/wiki/Y1NbwYijHiuiqvkSf0jcUOvFnTe?from=from_copylink)
对于对齐训练导致的语言模态能力下降是指的对齐后的多模态模型mllm，对于纯语言输入的回复能力有所下降，俗称对齐税（本质上也许是另外一种灾难性遗忘）。
对于抑制灾难性遗忘一种比较简单的方法是混入原始数据，对于多模态的语言能力丢失，则是混入语言数据。这就迎来了另外一个问题，混入哪些语言数据，占比又是多少，这不是本文的重点，笔者亦无力解决这个问题。
但是对于应用来说，mllm并不需要十项全能的语言能力，更多的是在有优秀的多模态能力下保持基础问答以及某一个领域的专业的回复能力。

## 4g显存玩转rag
![alt text](./4G_memory_rag/image.png)
![alt text](./4G_memory_rag/image1.png)
这个没什么好解释的，可以在极低显存下运行rag，
### 使用教程：
教程自取[RAG](https://modelbest.feishu.cn/wiki/G5NlwYGGAiJWGmkCc4NcQ3sAnms?from=from_copylink)

## MiniCPMV2.6的awq量化
由于bnb量化的minicpmv2.6无法用vllm加载，因此适配了autoawq，目前已经向autoawq提了pr，等合并后可以直接使用。

### 使用教程：
使用方法如下：

1. 获取个人autoawq分支
```bash
git clone https://github.com/LDLINGLINGLING/AutoAWQ
cd AutoAWQ
pip install e .
```
2. 将MiniCPM_Series_Tutorial/MiniCPMV2_6_awq/modeling_minicpmv.py文件替换掉minicpmv2.6模型保存路径下的同名文件
3. 修改MiniCPM_Series_Tutorial/MiniCPMV2_6_awq/quantize.py中的model_path为你minicpmv2.6的保存路径。
4. 运行quantize.py

获得minicpmv2.6的awq模型后可以使用原来的vllm进行部署，部署方式完全相同,模型从16g显存将为7g显存
![alt text](./MiniCPMV2_6_awq/image.png)