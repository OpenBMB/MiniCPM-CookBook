import json
import math
from datetime import datetime

import numpy as np
from PIL import Image

import torch
from decord import VideoReader, cpu
from transformers import AutoModel, AutoTokenizer

# 视频预处理
def extract_frames(video_path, sample_fps=5, max_frames=None):
    """从视频文件中提取帧"""
    # 使用decord读取视频
    vr = VideoReader(video_path, ctx=cpu(0))
    
    # 获取视频信息
    fps = vr.get_avg_fps()
    total_frames = len(vr)
    
    # 计算采样间隔
    sample_interval = int(fps / sample_fps)
    
    # 确定要提取的帧索引
    frame_indices = list(range(0, total_frames, sample_interval))
    
    # 限制最大帧数
    if max_frames and len(frame_indices) > max_frames:
        # 均匀采样以获取指定数量的帧
        step = len(frame_indices) / max_frames
        indices = [int(i * step) for i in range(max_frames)]
        frame_indices = [frame_indices[i] for i in indices]
    
    # 提取帧
    frames = vr.get_batch(frame_indices).asnumpy()
    
    # 转换为PIL图像
    pil_frames = [Image.fromarray(frame) for frame in frames]
    
    return pil_frames

# 长视频处理
class LongVideoProcessor:
    def __init__(self, 
                 model_path="openbmb/MiniCPM-o-2_6",
                 max_frames_per_chunk=64,  # 每个块的最大帧数
                 max_slice_nums=9,         # 每帧图像的最大切片数
                 scale_resolution=448,     # 每个切片的分辨率
                 memory_bank_size=32,      # 记忆库大小
                 overlap_frames=8,         # 块之间的重叠帧数
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        
        # 加载模型
        self.model = AutoModel.from_pretrained(
            model_path, 
            trust_remote_code=True,
            attn_implementation='sdpa', 
            torch_dtype=torch.bfloat16
        )
        self.model = self.model.eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        self.max_frames_per_chunk = max_frames_per_chunk
        self.max_slice_nums = max_slice_nums
        self.scale_resolution = scale_resolution
        self.memory_bank_size = memory_bank_size
        self.overlap_frames = overlap_frames
        self.device = device
        
        # 初始化记忆库
        self.memory_bank = []
    
    def preprocess_frame(self, frame):
        """预处理单个视频帧"""
        # 调整图像大小，保持宽高比
        width, height = frame.size
        max_size = max(width, height)
        if max_size > self.scale_resolution:
            scale = self.scale_resolution / max_size
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = frame.resize((new_width, new_height), Image.LANCZOS)
        
        return frame
    
    def update_memory_bank(self, frames):
        """更新记忆库"""
        # 将新的帧添加到记忆库
        self.memory_bank.extend(frames)
        
        # 如果记忆库超出大小限制，移除最旧的帧
        if len(self.memory_bank) > self.memory_bank_size:
            self.memory_bank = self.memory_bank[-self.memory_bank_size:]
    
    def process_long_video(self, video_frames, query):
        """处理长视频"""
        # 将视频分成多个块
        chunks = []
        for i in range(0, len(video_frames), self.max_frames_per_chunk - self.overlap_frames):
            end_idx = min(i + self.max_frames_per_chunk, len(video_frames))
            chunk = video_frames[i:end_idx]
            chunks.append(chunk)
        
        # 逐块处理视频
        all_results = []
        for i, chunk in enumerate(chunks):
            # 预处理当前块的帧
            processed_frames = [self.preprocess_frame(frame) for frame in chunk]
            
            # 使用记忆库和当前块进行推理
            result = self._inference_with_memory(processed_frames, query, i, len(chunks))
            all_results.append(result)
            
            # 更新记忆库
            self.update_memory_bank(processed_frames)
        
        # 合并所有块的结果
        final_result = self._merge_results(all_results, query)
        
        return final_result
    
    def _inference_with_memory(self, frames, query, chunk_idx, total_chunks):
        """使用记忆库和当前帧进行推理"""
        # 合并记忆库帧和当前帧，但需要控制总帧数
        combined_frames = []
        
        # 如果有记忆库，添加部分记忆库帧
        if self.memory_bank:
            # 从记忆库中均匀采样一些帧
            memory_sample_count = min(16, len(self.memory_bank))
            memory_indices = np.linspace(0, len(self.memory_bank)-1, memory_sample_count, dtype=int)
            for idx in memory_indices:
                combined_frames.append(self.memory_bank[idx])
        
        # 添加当前块的帧，也可能需要采样
        current_sample_count = min(self.max_frames_per_chunk - len(combined_frames), len(frames))
        if current_sample_count < len(frames):
            # 需要采样
            current_indices = np.linspace(0, len(frames)-1, current_sample_count, dtype=int)
            for idx in current_indices:
                combined_frames.append(frames[idx])
        else:
            # 不需要采样，直接添加所有帧
            combined_frames.extend(frames)
        
        # 构建带有视频帧的消息
        video_msg = [{
            "role": "user", 
            "content": combined_frames + [query]
        }]
        
        # 设置视频处理参数
        params = {
            "use_image_id": False,
            "max_slice_nums": 2,  # 如果CUDA内存不足，可以设为1
        }
        
        # 调用模型进行推理
        response = self.model.chat(
            msgs=video_msg,
            tokenizer=self.tokenizer,
            **params
        )
        
        # 构建结果
        result = {
            "chunk_idx": chunk_idx,
            "total_chunks": total_chunks,
            "frames_count": len(frames),
            "memory_bank_size": len(self.memory_bank),
            "query": query,
            "response": response
        }
        
        return result
    
    def _merge_results(self, all_results, query):
        """合并所有块的结果"""
        # 构建消息
        msgs = []
        
        # 添加系统消息
        msgs.append({
            "role": "system",
            "content": "你是一个视频理解助手，可以分析视频内容并回答问题。请基于所有视频块的分析结果，给出一个综合的回答。"
        })
        
        # 构建用户消息
        content = []
        content.append("以下是各个视频块的分析结果:")
        
        for i, result in enumerate(all_results):
            content.append(f"块 {i+1}/{len(all_results)} 的分析结果: {result['response']}")
        
        content.append(f"请基于以上所有块的分析结果，回答问题: {query}")
        
        msgs.append({
            "role": "user",
            "content": content
        })
        
        # 调用模型进行推理
        final_answer = self.model.chat(
            msgs=msgs,
            tokenizer=self.tokenizer
        )
        
        # 构建最终结果
        final_result = {
            "total_chunks": len(all_results),
            "memory_bank_size": len(self.memory_bank),
            "query": query,
            "chunk_responses": [result["response"] for result in all_results],
            "final_answer": final_answer
        }
        
        return final_result


def main():
    video_path = "long_video.mp4"  # 视频的路径
    video_frames = extract_frames(video_path, sample_fps=5, max_frames=None)
    
    processor = LongVideoProcessor(
        model_path="openbmb/MiniCPM-o-2_6",
        max_frames_per_chunk=64,  # 每个块的最大帧数
        max_slice_nums=9,         # 每帧图像的最大切片数
        scale_resolution=448,     # 每个切片的分辨率
        memory_bank_size=32,      # 记忆库大小
        overlap_frames=8,         # 块之间的重叠帧数
    )
    
    query = "视频中发生了什么事情？请详细描述。"
    result = processor.process_long_video(video_frames, query)

    with open("long_video_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(result["final_answer"])

if __name__ == "__main__":
    main()