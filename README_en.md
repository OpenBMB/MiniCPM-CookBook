# MiniCPM_Series Cookbook
<div align="center">
<img src="./asset/logo.png" width="500em" ></img> 

This repository is a guide for the MiniCPM series of edge-side models, covering inference, quantization, edge-end deployment, fine-tuning, applications, and technical reports.
</div>
<p align="center">
<a href="https://github.com/OpenBMB" target="_blank">MiniCPM Repository</a> |
<a href="https://github.com/OpenBMB/MiniCPM-V/" target="_blank">MiniCPM-V Repository</a> |
<a href="https://modelbest.feishu.cn/wiki/D2tFw8Pcsi5CIzkaHNacLK64npg" target="_blank">MiniCPM Series Knowledge Base</a> |
<a href="README.md" target="_blank">中文教程</a> |
Join our <a href="https://discord.gg/3cGQn9b3YM" target="_blank">Discord</a> and <a href="./asset/weixin.png" target="_blank">WeChat Group</a>
 
</p>

# Table of Contents
## About MiniCPM (✅)
The MiniCPM edge-side series is jointly open-sourced by ModelBest and OpenBMB, in collaboration with the Tsinghua NLP Lab. It comprises globally lightweight high-performance AI models，including the MiniCPM foundation model and the MiniCPM-V multimodal model. We have now ushered in the "Edge-Side ChatGPT Era" in terms of performance; in the multimodal direction, we have made GPT-4V level MLLMsfor Single Image, Multi Image and Video on Your Phone. It is currently being deployed in end devices such as smartphones, computers, cars, wearable devices, VR, and more. For more detailed information about the MiniCPM series, please visit the [OpenBMB](https://github.com/OpenBMB) page.

## Technical Reports (✅)
- [MiniCPM Language Model Technical Report](https://arxiv.org/abs/2404.06395)
- [MiniCPM-V Multimodal Model Technical Report](https://arxiv.org/abs/2408.01800)
- [Evolution of Attention Mechanisms in MiniCPM](https://modelbest.feishu.cn/docx/JwBMdtwQ2orB5KxxS94cdydenWf?from=from_copylink)
- [Architecture Principles of MiniCPM-V Multimodal Model](https://modelbest.feishu.cn/wiki/X15nwGzqpioxlikbi2RcXDpJnjd?from=from_copylink)
- [Principles of High-Definition Decoding in MiniCPM-V](https://modelbest.feishu.cn/wiki/L0ajwm8VAiiPY6kDZfJce3B7nRg?from=from_copylink)

## Supported Hardware (Cloud and Edge) (✅)
- GPU
- CPU
- NPU
- Android
- Mac
- Windows
- iOS

## Model Addresses and Downloads (Partial) (✅)
- [MiniCPM 2.4B](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16)
- [MiniCPM-V](https://huggingface.co/openbmb/MiniCPM-V)
- [MiniCPM-V 2.0](https://huggingface.co/openbmb/MiniCPM-V-2)
- [MiniCPM-Llama3-V 2.5](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5)
- [MiniCPM-V 2.6](https://huggingface.co/openbmb/MiniCPM-V-2_6)
- [MiniCPM 3.0 4B](https://huggingface.co/openbmb/MiniCPM3-4B)

## Inference Deployment (✅)
#### MiniCPM 2.4B
- [MiniCPM 2.4B_transformers_cuda](./md/md_en/inference/minicpm2.0/transformers.md)
- [MiniCPM 2.4B_vllm_cuda](./md/md_en/inference/minicpm2.0/vllm.md)
- [MiniCPM 2.4B__mlx_mac](./md/md_en/inference/minicpm2.0/mlx.md)
- [MiniCPM 2.4B_ollama_cuda_cpu_mac](./md/md_en/inference/minicpm2.0/ollama.md)
- [MiniCPM 2.4B_llamacpp_cuda_cpu](./md/md_en/inference/minicpm2.0/llama.cpp_pc.md)
- [MiniCPM 2.4B_llamacpp_android](./md/md_en/inference/minicpm2.0/llama.cpp_android.md)
- FAQ
#### MiniCPM-S 1.2B
- [MiniCPM-S 1.2B_powerinfer_cuda](./md/md_en/inference/minicpm2.0/powerinfer_pc.md)
- [MiniCPM-S 1.2B_powerinfer_android](./md/md_en/inference/minicpm2.0/powerinfer_android.md)
- FAQ
#### MiniCPM 3.0
- [MiniCPM 3.0_vllm_cuda](./md/md_en/inference/minicpm3.0/vllm.md)
- [MiniCPM 3.0_transformers_cuda_cpu](./md/md_en/inference/minicpm3.0/transformers.md)
- [MiniCPM 3.0_llamacpp_cuda_cpu](./md/md_en/inference/minicpm3.0/llamcpp.md)
- [MiniCPM 3.0_sglang_cuda](./md/md_en/inference/minicpm3.0/sglang.md)
#### MiniCPM-Llama3-V 2.5
- [MiniCPM-Llama3-V 2.5_vllm_cuda](./md/md_en/inference/minicpmv2.5/vllm.md)
- [MiniCPM-Llama3-V 2.5_LMdeploy_cuda](./md/md_en/inference/minicpmv2.5/LMdeploy.md)
- [MiniCPM-Llama3-V 2.5_llamacpp_cuda_cpu](./md/md_en/inference/minicpmv2.5/llamacpp_pc.md)
- [MiniCPM-Llama3-V 2.5_ollama_cuda_cpu](./md/md_en/inference/minicpmv2.5/ollama.md)
- [MiniCPM-Llama3-V 2.5_transformers_cuda](./md/md_en/inference/minicpmv2.5/transformers_multi_gpu.md)
- [MiniCPM-Llama3-V 2.5_xinference_cuda](./md/md_en/inference/minicpmv2.5/xinference.md)
- [MiniCPM-Llama3-V 2.5_swift_cuda](./md/md_en/inference/minicpmv2.5/swift_python.md)
#### MiniCPM-V 2.6
- [MiniCPM-V 2.6_vllm_cuda](./md/md_en/inference/minicpmv2.6/vllm.md)
- [MiniCPM-V 2.6_vllm_api_server_cuda](./md/md_en/inference/minicpmv2.6/vllm_api_server.md)
- [MiniCPM-V 2.6_llamacpp_cuda_cpu](./md/md_en/inference/minicpmv2.6/llamacpp.md)
- [MiniCPM-V 2.6_transformers_cuda](./md/md_en/inference/minicpmv2.6/transformers_mult_gpu.md)
- [MiniCPM-V 2.6_swift_cuda](https://github.com/modelscope/ms-swift/issues/1613)
- FAQ

## Fine-Tuning (✅)
#### MiniCPM 3.0
- [MiniCPM3_llamafactory_sft_RLHF_cuda](./md/md_en/finetune/minicpm3.0/llama_factory.md)
#### MiniCPM 2.4B
- [MiniCPM2.0_official_code_sft_cuda](./md/md_en/finetune/minicpm2.0/sft.md)
- [MiniCPM2.0_mlx_sft_lora_mac](./md/md_en/finetune/minicpm2.0/mlx_sft.md)
- [MiniCPM2.0_llamafactory_RLHF_cuda](./md/md_en/finetune/minicpm2.0/llama_factory.md)
- FAQ

#### MiniCPM-Llama3-V 2.5
- [MiniCPM-Llama3-V 2.5 Official Code CUDA](./md/md_en/finetune/minicpmv2.5/sft.md)
- [MiniCPM-Llama3-V-2_5 Swift CUDA](./md/md_en/finetune/minicpmv2.5/swift.md)
- [Hybrid Modality Training](https://modelbest.feishu.cn/wiki/Y1NbwYijHiuiqvkSf0jcUOvFnTe?from=from_copylink)

#### MiniCPM-V 2.6
- [MiniCPM-V 2.6 Official Code SFT CUDA](./md/md_en/finetune/minicpmv2.6/sft.md)
- [MiniCPM-V 2.6 Swift SFT CUDA](https://github.com/modelscope/ms-swift/issues/1613)
- [Hybrid Modality Training](https://modelbest.feishu.cn/wiki/As5Ow99z3i4hrCkooRIcz79Zn2f?from=from_copylink)
- FAQ

## Model Quantization (✅)
#### MiniCPM 2.4B
- [MiniCPM 2.4B AWQ Quantization](./md/md_en/quantize/minicpm2.0/awq.md)
- [MiniCPM 2.4B GGUF Quantization](./md/md_en/inference/minicpm2.0/llama.cpp_pc.md)
- [MiniCPM 2.4B GPTQ Quantization](./md/md_en/quantize/minicpm2.0/gptq.md)
- [MiniCPM 2.4B BNB Quantization](./md/md_en/quantize/minicpm2.0/bnb.md)

#### MiniCPM 3.0
- [MiniCPM 3.0 AWQ Quantization](./md/md_en/quantize/minicpm3.0/awq.md)
- [MiniCPM 3.0 GGUF Quantization](./md/md_en/inference/minicpm3.0/llamcpp.md)
- [MiniCPM 3.0 GPTQ Quantization](./md/md_en/quantize/minicpm3.0/gptq.md)
- [MiniCPM 3.0 BNB Quantization](./md/md_en/quantize/minicpm3.0/bnb.md)

#### MiniCPM-Llama3-V 2.5
- [MiniCPM-Llama3-V 2.5 BNB Quantization](./md/md_en/md_en/quantize/minicpmv2.5/bnb.md)
- [MiniCPM-Llama3-V 2.5 GGUF Quantization](./md/md_en/md_en/inference/minicpmv2.5/llamacpp_pc.md)

#### MiniCPM-V 2.6
- [MiniCPM-V 2.6 BNB Quantization](./md/md_en/quantize/minicpmv2.6/bnb.md)
- [MiniCPM-V 2.6 AWQ Quantization](./md/md_en/quantize/minicpmv2.6/awq.md)
- [MiniCPM-V 2.6 GGUF Quantization](./md/md_en/inference/minicpmv2.6/llamacpp.md)

## Integration (✅)
- [LangChain](./md/md_en/integrate/langchain.md)
- [OpenAI API](./md/md_en/integrate/openai_api.md)

## Applications (✅)
### Language Models
- [Playing with RAG LangChain on 4GB VRAM](https://modelbest.feishu.cn/wiki/G5NlwYGGAiJWGmkCc4NcQ3sAnms?from=from_copylink)
- [Controllable Text Generation with RLHF](https://modelbest.feishu.cn/wiki/ZEzGwgDgSi2Nk1kjAfFcrZn4nKd?from=from_copylink)
- [Function Call](https://modelbest.feishu.cn/wiki/ARJtwko3gisbw5kdPiDcDIOvnGg?from=from_copylink)
- [Building an Agent on AIPC-Windows](https://modelbest.feishu.cn/wiki/N0tswVXEqipuSUkWc96comFXnnd?from=from_copylink)

### Multimodal Models
- [Cross-Modality High-Definition Retrieval](https://modelbest.feishu.cn/wiki/NdEjwo0hxilCIikN6RycOKp0nYf?from=from_copylink)
- [Text Recognition and Localization](https://modelbest.feishu.cn/wiki/HLRiwNgKEic6cckGyGucFvxQnJw?from=from_copylink)
- [Getting Started with Agents](https://modelbest.feishu.cn/wiki/HKQdwbUUgiL0HNkSetjctaMcnrw?from=from_copylink)
- [Constructing Long-Chain Agents](https://modelbest.feishu.cn/wiki/IgF0wRGJYizj4LkMyZvc7e2Inoe?from=from_copylink)
- [Multimodal Document RAG](https://modelbest.feishu.cn/wiki/NwhIwkJZYiHOPSkzwPUcq6hanif?from=from_copylink)

## Open Source Community Collaboration (✅)
- [xtuner](https://github.com/InternLM/xtuner): [The Optimal Choice for Efficient Fine-Tuning of MiniCPM](https://modelbest.feishu.cn/wiki/AIU3wbREcirOm9kkvd7cxujFnMb#AMdXdzz8qoadZhxU4EucELWznzd)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory.git): [One-Click Fine-Tuning Solution for MiniCPM](https://modelbest.feishu.cn/wiki/AIU3wbREcirOm9kkvd7cxujFnMb#BAWrdSjXuoFvX4xuIuzc8Amln5E)
- [ChatLLM Framework](https://github.com/foldl/chatllm.cpp): [Running MiniCPM on CPU](https://huggingface.co/openbmb/MiniCPM-2B-dpo-bf16/discussions/2#65c59c4f27b8c11e43fc8796)

## Community Contributions
In the spirit of open source, we encourage contributions to this repository, including but not limited to adding new MiniCPM tutorials, sharing user experiences, providing ecosystem compatibility, and model applications. We look forward to contributions from developers to enhance our open-source repository.
