import json
import math
import tempfile
from datetime import datetime

import numpy as np
import torch
import librosa
import soundfile as sf
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoModel, AutoTokenizer
from moviepy.editor import VideoFileClip

import librosa
import soundfile as sf
from moviepy.editor import VideoFileClip
import tempfile

# 视频预处理
def extract_frames_and_audio(video_path, sample_fps=5, max_frames=None, audio_processor=None):
    """从视频文件中提取帧和对应的音频"""
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
    
    # 提取音频
    audio_segments = None
    if audio_processor:
        # 提取完整音频
        audio, duration = audio_processor.extract_audio_from_video(video_path)
        
        # 将音频分段，与视频帧对齐
        frame_times = [idx / fps for idx in frame_indices]
        audio_segments = []
        
        for time in frame_times:
            start_sample = int(time * audio_processor.sample_rate)
            end_sample = start_sample + int(1.0 * audio_processor.sample_rate)  # 取1秒音频
            if end_sample <= len(audio):
                segment = audio[start_sample:end_sample]
            else:
                segment = np.pad(audio[start_sample:], (0, end_sample - len(audio)), 'constant')
            audio_segments.append(segment)
    
    return pil_frames, audio_segments

# 长视频处理
class LongVideoAudioProcessor:
    def __init__(self, 
                 model_path="openbmb/MiniCPM-o-2_6",
                 max_frames_per_chunk=64,  # 每个块的最大帧数
                 max_slice_nums=9,         # 每帧图像的最大切片数
                 scale_resolution=448,     # 每个切片的分辨率
                 memory_bank_size=32,      # 记忆库大小
                 overlap_frames=8,         # 块之间的重叠帧数
                 audio_sample_rate=16000,  # 音频采样率
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
        
        # 初始化记忆库 - 现在包含视频帧和音频
        self.visual_memory_bank = []
        self.audio_memory_bank = []
        
        # 初始化音频处理器
        self.audio_processor = AudioProcessor(sample_rate=audio_sample_rate)
    
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
    
    def update_memory_bank(self, frames, audio_segments):
        """更新记忆库 - 同时存储视频帧和音频"""
        # 将新的帧添加到视觉记忆库
        self.visual_memory_bank.extend(frames)
        
        # 如果视觉记忆库超出大小限制，移除最旧的帧
        if len(self.visual_memory_bank) > self.memory_bank_size:
            self.visual_memory_bank = self.visual_memory_bank[-self.memory_bank_size:]
        
        # 如果有音频，更新音频记忆库
        if audio_segments:
            self.audio_memory_bank.extend(audio_segments)
            
            # 如果音频记忆库超出大小限制，移除最旧的片段
            if len(self.audio_memory_bank) > self.memory_bank_size:
                self.audio_memory_bank = self.audio_memory_bank[-self.memory_bank_size:]
    
    def process_long_video(self, video_path, query):
        """处理长视频，包括音频"""
        # 提取视频帧和音频
        video_frames, audio_segments = extract_frames_and_audio(
            video_path, 
            sample_fps=5, 
            max_frames=None, 
            audio_processor=self.audio_processor
        )
        
        # 将视频分成多个块
        chunks = []
        audio_chunks = []
        
        for i in range(0, len(video_frames), self.max_frames_per_chunk - self.overlap_frames):
            end_idx = min(i + self.max_frames_per_chunk, len(video_frames))
            chunk = video_frames[i:end_idx]
            chunks.append(chunk)
            
            # 如果有音频，也分块
            if audio_segments:
                audio_chunk = audio_segments[i:end_idx]
                audio_chunks.append(audio_chunk)
        
        # 逐块处理视频
        all_results = []
        for i, chunk in enumerate(chunks):
            # 预处理当前块的帧
            processed_frames = [self.preprocess_frame(frame) for frame in chunk]
            
            # 获取对应的音频块
            audio_chunk = audio_chunks[i] if audio_segments else None
            
            # 使用记忆库和当前块进行推理
            result = self._inference_with_memory(processed_frames, audio_chunk, query, i, len(chunks))
            all_results.append(result)
            
            # 更新记忆库
            self.update_memory_bank(processed_frames, audio_chunk)
        
        # 合并所有块的结果
        final_result = self._merge_results(all_results, query)
        
        return final_result
    
    def _inference_with_memory(self, frames, audio_segments, query, chunk_idx, total_chunks):
        """使用记忆库和当前帧进行推理，包括音频"""
        # 合并记忆库帧和当前帧，但需要控制总帧数
        combined_frames = []
        combined_audio = []
        
        # 如果有视觉记忆库，添加部分记忆库帧
        if self.visual_memory_bank:
            # 从记忆库中均匀采样一些帧
            memory_sample_count = min(16, len(self.visual_memory_bank))
            memory_indices = np.linspace(0, len(self.visual_memory_bank)-1, memory_sample_count, dtype=int)
            for idx in memory_indices:
                combined_frames.append(self.visual_memory_bank[idx])
                
                # 如果有对应的音频记忆，也添加
                if idx < len(self.audio_memory_bank):
                    combined_audio.append(self.audio_memory_bank[idx])
        
        # 添加当前块的帧，也可能需要采样
        current_sample_count = min(self.max_frames_per_chunk - len(combined_frames), len(frames))
        if current_sample_count < len(frames):
            # 需要采样
            current_indices = np.linspace(0, len(frames)-1, current_sample_count, dtype=int)
            for idx in current_indices:
                combined_frames.append(frames[idx])
                
                # 如果有对应的音频，也添加
                if audio_segments and idx < len(audio_segments):
                    combined_audio.append(audio_segments[idx])
        else:
            # 不需要采样，直接添加所有帧
            combined_frames.extend(frames)
            
            # 如果有音频，也添加所有音频片段
            if audio_segments:
                combined_audio.extend(audio_segments)
        
        # 构建带有视频帧和音频的消息
        content = []
        
        # 交替添加视频帧和音频，使用<unit>标记分隔
        for i in range(len(combined_frames)):
            content.append("<unit>")
            content.append(combined_frames[i])
            if i < len(combined_audio):
                content.append(combined_audio[i])
        
        # 添加查询
        content.append(query)
        
        video_msg = [{
            "role": "user", 
            "content": content
        }]
        
        # 设置视频处理参数
        params = {
            "use_image_id": False,
            "max_slice_nums": 2,  # 如果CUDA内存不足，可以设为1
            "omni_input": True,   # 启用全模态输入处理
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
            "audio_count": len(audio_segments) if audio_segments else 0,
            "memory_bank_size": len(self.visual_memory_bank),
            "audio_memory_bank_size": len(self.audio_memory_bank),
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
            "content": "你是一个视频理解助手，可以分析视频内容（包括视觉和音频）并回答问题。请基于所有视频块的分析结果，给出一个综合的回答。"
        })
        
        # 构建用户消息
        content = []
        content.append("以下是各个视频块的分析结果（包含视觉和音频信息）:")
        
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
            "visual_memory_bank_size": len(self.visual_memory_bank),
            "audio_memory_bank_size": len(self.audio_memory_bank),
            "query": query,
            "chunk_responses": [result["response"] for result in all_results],
            "final_answer": final_answer
        }
        
        return final_result

# 音频处理
class AudioProcessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def extract_audio_from_video(self, video_path):
        """从视频文件中提取音频"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
            temp_audio_path = temp_audio_file.name
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(temp_audio_path, codec="pcm_s16le", fps=self.sample_rate, verbose=False, logger=None)
            audio, _ = librosa.load(temp_audio_path, sr=self.sample_rate, mono=True)
        return audio, video.duration
    
    def segment_audio(self, audio, duration, segment_duration=1.0):
        """将音频分割成固定时长的片段"""
        num_segments = int(np.ceil(duration / segment_duration))
        segments = []
        samples_per_segment = int(self.sample_rate * segment_duration)
        
        for i in range(num_segments):
            start_idx = i * samples_per_segment
            end_idx = min(start_idx + samples_per_segment, len(audio))
            segment = audio[start_idx:end_idx]
            # 如果最后一个片段长度不足，用0填充
            if len(segment) < samples_per_segment:
                segment = np.pad(segment, (0, samples_per_segment - len(segment)), 'constant')
            segments.append(segment)
        
        return segments
    
    def extract_audio_features(self, audio_segment):
        """提取音频特征（可选）"""
        # 这里可以添加更复杂的特征提取，如MFCC等
        return audio_segment
    

def main():

    video_path = "long_video.mp4" # 修改为您的视频路径

    processor = LongVideoAudioProcessor(
        model_path="openbmb/MiniCPM-o-2_6",  # 修改为您的模型路径
        max_frames_per_chunk=64,
        max_slice_nums=9,
        scale_resolution=448,
        memory_bank_size=32,
        overlap_frames=8,
        audio_sample_rate=16000
    )
    
    query = "视频中发生了什么事情？请详细描述视觉内容和音频内容。"
    result = processor.process_long_video(video_path, query)

    with open("long_video_audio_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(result["final_answer"])

if __name__ == "__main__":
    main()