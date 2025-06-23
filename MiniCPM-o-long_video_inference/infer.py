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
from tqdm import tqdm

import librosa
import soundfile as sf
from moviepy.editor import VideoFileClip
import tempfile

# 视频预处理
def extract_frames_and_audio(video_path, sample_fps=2, max_frames=None, audio_processor=None):
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
                 time_decay_factor=0.8,    # 时间衰减因子
                 sample_fps=2,            # 视频采样频率
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 high_res_mode=True        # 高清模式开关，默认开启
                 ):
        
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
        self.sample_fps = sample_fps
        
        # 初始化记忆库 - 包含视频帧、音频和时间戳
        self.visual_memory_bank = []
        self.audio_memory_bank = []
        self.timestamps = []  # 帧时间戳
        self.text_summaries = []  # 历史文本摘要
        
        self.time_decay_factor = time_decay_factor
        self.high_res_mode = high_res_mode  # 保存高清模式开关
        
        # 初始化音频处理器
        self.audio_processor = AudioProcessor(sample_rate=audio_sample_rate)
    
    def preprocess_frame(self, frame):
        """预处理单个视频帧"""
        width, height = frame.size
        # 如果关闭高清模式，先缩小为原分辨率1/2
        if not self.high_res_mode:
            frame = frame.resize((width // 2, height // 2), Image.LANCZOS)
            width, height = frame.size
        max_size = max(width, height)
        if max_size > self.scale_resolution:
            scale = self.scale_resolution / max_size
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = frame.resize((new_width, new_height), Image.LANCZOS)
        
        return frame
    
    def update_memory_bank(self, frames, audio_segments, current_time):
        """更新记忆库 - 同时存储视频帧、音频和时间戳"""
        # 将新的帧添加到视觉记忆库
        self.visual_memory_bank.extend(frames)
        
        # 更新时间戳
        new_timestamps = [current_time + i/self.sample_fps for i in range(len(frames))]  # 使用类属性中的采样频率
        self.timestamps.extend(new_timestamps)
        
        # 如果视觉记忆库超出大小限制，移除最旧的帧
        if len(self.visual_memory_bank) > self.memory_bank_size:
            self.visual_memory_bank = self.visual_memory_bank[-self.memory_bank_size:] 
            self.timestamps = self.timestamps[-self.memory_bank_size:]
        
        # 如果有音频，更新音频记忆库
        if audio_segments:
            self.audio_memory_bank.extend(audio_segments)
            if len(self.audio_memory_bank) > self.memory_bank_size:
                self.audio_memory_bank = self.audio_memory_bank[-self.memory_bank_size:]
    
    def calculate_time_weights(self, current_time):
        """计算基于时间的衰减权重"""
        weights = []
        for timestamp in self.timestamps:
            # 计算时间差（秒）
            time_diff = current_time - timestamp
            # 使用指数衰减函数计算权重
            weight = self.time_decay_factor ** time_diff
            weights.append(weight)
        return np.array(weights)
    
    def weighted_sampling(self, frames, weights, sample_count):
        """基于权重的采样"""
        # 归一化权重
        weights = weights / np.sum(weights)
        # 使用权重进行采样
        indices = np.random.choice(len(frames), size=sample_count, p=weights, replace=False)
        return sorted(indices)  # 返回排序后的索引以保持时间顺序
    
    def update_text_summary(self, new_summary):
        """更新历史文本摘要"""
        self.text_summaries.append(new_summary)
        # 保持摘要数量在合理范围内
        if len(self.text_summaries) > 5:  # 最多保留5个摘要
            self.text_summaries = self.text_summaries[-5:]
    
    def process_long_video(self, video_path, query):
        """处理长视频，包括音频"""
        # 提取视频帧和音频
        video_frames, audio_segments = extract_frames_and_audio(
            video_path, 
            sample_fps=self.sample_fps,  # 使用类属性中的采样频率
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
        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks", unit="chunks", ncols=80)):
            # 预处理当前块的帧
            processed_frames = [self.preprocess_frame(frame) for frame in chunk]
            
            # 获取对应的音频块
            audio_chunk = audio_chunks[i] if audio_segments else None
            
            # 使用记忆库和当前块进行推理
            result = self._inference_with_memory(processed_frames, audio_chunk, query, i, len(chunks), datetime.now().timestamp())
            all_results.append(result)
            
            # 更新记忆库
            self.update_memory_bank(processed_frames, audio_chunk, datetime.now().timestamp())
        
        # 合并所有块的结果
        final_result = self._merge_results(all_results, query)
        
        return final_result
    
    def _inference_with_memory(self, frames, audio_segments, query, chunk_idx, total_chunks, current_time):
        """使用记忆库和当前帧进行推理，包括音频和时间加权"""
        combined_frames = []
        combined_audio = []
        
        # 使用时间加权采样
        if self.visual_memory_bank:
            # 计算时间权重
            time_weights = self.calculate_time_weights(current_time)
            # 从记忆库中加权采样
            memory_sample_count = min(16, len(self.visual_memory_bank))
            memory_indices = self.weighted_sampling(self.visual_memory_bank, time_weights, memory_sample_count)
            
            for idx in memory_indices:
                combined_frames.append(self.visual_memory_bank[idx])
                if idx < len(self.audio_memory_bank):
                    combined_audio.append(self.audio_memory_bank[idx])
        
        # 添加当前块的帧
        current_sample_count = min(self.max_frames_per_chunk - len(combined_frames), len(frames))
        if current_sample_count < len(frames):
            current_indices = np.linspace(0, len(frames)-1, current_sample_count, dtype=int)
            for idx in current_indices:
                combined_frames.append(frames[idx])
                if audio_segments and idx < len(audio_segments):
                    combined_audio.append(audio_segments[idx])
        else:
            combined_frames.extend(frames)
            if audio_segments:
                combined_audio.extend(audio_segments)
        
        # 构建系统提示
        system_prompt = """
        你是一个专业的视频理解助手，具有以下能力：
        1. 视觉分析：能够准确识别视频中的场景、物体、人物和动作
        2. 音频理解：能够识别背景音乐、对话、环境声音等
        3. 时序理解：能够理解视频中的时间顺序和事件发展
        4. 上下文关联：能够将当前内容与历史信息关联起来
        5. 回答要简洁、准确、客观
        6. 优先描述视觉和音频的关键信息
        7. 保持时间顺序的连贯性
        8. 如果信息不足，明确说明
        9. 不要编造或推测不存在的信息
        """

        # 构建带有视频帧、音频和历史摘要的消息
        content = []
        
        # 添加历史摘要，使用更结构化的格式
        if self.text_summaries:
            content.append("=== 历史上下文 ===")
            for i, summary in enumerate(self.text_summaries):
                content.append(f"[时间点 {i+1}] {summary}")
            content.append("=== 当前内容 ===")
        
        # 添加当前时间信息
        content.append(f"当前处理第 {chunk_idx + 1}/{total_chunks} 个视频块")
        
        # 交替添加视频帧和音频，使用更清晰的标记
        for i in range(len(combined_frames)):
            content.append("<visual_audio_unit>")
            content.append(f"[帧 {i+1}]")
            content.append(combined_frames[i])
            if i < len(combined_audio):
                content.append(f"[音频 {i+1}]")
                content.append(combined_audio[i])
        
        # 添加查询，使用更结构化的格式
        content.append("\n=== 用户查询 ===")
        content.append(query)
        content.append("\n请基于以上信息，按照以下格式回答：")
        content.append("1. 视觉内容：[描述当前视频块中的主要视觉内容]")
        content.append("2. 音频内容：[描述当前视频块中的主要音频内容]")
        content.append("3. 上下文关联：[说明与历史内容的关联]")
        content.append("4. 总结：[简要总结当前块的关键信息]")
        
        video_msg = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": content
            }
        ]
        
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
        
        # 更新文本摘要
        self.update_text_summary(response)
        
        return {
            "chunk_idx": chunk_idx,
            "total_chunks": total_chunks,
            "frames_count": len(frames),
            "audio_count": len(audio_segments) if audio_segments else 0,
            "memory_bank_size": len(self.visual_memory_bank),
            "audio_memory_bank_size": len(self.audio_memory_bank),
            "query": query,
            "response": response
        }
    
    def _merge_results(self, all_results, query):
        """合并所有块的结果"""
        # 构建系统提示
        system_prompt = """
        你是一个专业的视频总结助手，负责整合多个视频块的分析结果。
        请遵循以下规则：
        1. 保持时间顺序的连贯性
        2. 突出重要事件和关键信息
        3. 确保信息的完整性和准确性
        4. 避免重复和冗余信息
        5. 如果信息有冲突，选择最可信的信息"""

        content = []
        content.append("=== 视频块分析结果 ===")
        
        for i, result in enumerate(all_results):
            content.append(f"\n[视频块 {i+1}/{len(all_results)}]")
            content.append(result['response'])
        
        content.append("\n=== 用户查询 ===")
        content.append(query)
        content.append("\n请按照以下格式提供最终答案：")
        content.append("1. 时间线：[按时间顺序总结主要事件]")
        content.append("2. 关键信息：[提取最重要的信息点]")
        content.append("3. 综合分析：[将各个块的信息整合分析]")
        content.append("4. 直接回答：[针对用户查询的具体回答]")
        
        msgs = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": content
            }
        ]
        
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
        model_path="openbmb/MiniCPM-o-2_6", # 修改为您的模型路径
        max_frames_per_chunk=64,  # 每个块的最大帧数
        max_slice_nums=9,         # 每帧图像的最大切片数
        scale_resolution=448,     # 每个切片的分辨率
        memory_bank_size=32,      # 记忆库大小
        overlap_frames=8,         # 块之间的重叠帧数
        audio_sample_rate=16000,  # 音频采样率
        time_decay_factor=0.8,    # 时间衰减因子
        sample_fps=2,             # 视频采样频率
        high_res_mode=True        # 高清模式开关，默认开启
    )
    
    query = "视频中发生了什么事情？请详细描述视觉内容和音频内容。"
    result = processor.process_long_video(video_path, query)

    with open("long_video_audio_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(result["final_answer"])

if __name__ == "__main__":
    main()