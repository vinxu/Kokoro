# timestamp_calculator.py
"""
时间戳计算模块

根据 Kokoro 模型输出的 durations 数组计算单词级时间戳
"""

from typing import List, Dict, Optional
import numpy as np

# Kokoro 帧率转换常数
# 模型输出的 duration 单位是帧，需要转换为秒
# 根据实际测试，Kokoro 的帧率约为 33.5 帧/秒
# 这对应 24000 Hz 采样率，hop_length ≈ 717
DEFAULT_FRAMES_PER_SECOND = 33.5


class TimestampCalculator:
    """
    根据 Kokoro 输出的 durations 计算单词级时间戳

    核心算法：
    1. 遍历每个 token（单词/标点）
    2. 根据 token 的音素数量，从 durations 数组中取对应数量的时长
    3. 累加时长得到每个单词的开始和结束时间
    """

    def __init__(self, frames_per_second: float = DEFAULT_FRAMES_PER_SECOND):
        """
        Args:
            frames_per_second: 帧率（帧/秒），用于将帧数转换为秒
        """
        self.fps = frames_per_second

    def calculate(
        self,
        tokens: List[Dict],
        durations: np.ndarray
    ) -> List[Dict]:
        """
        计算每个单词的开始和结束时间

        Args:
            tokens: G2P 返回的分词列表，每个元素需包含:
                - text: 单词文本
                - phoneme_count: 音素数量
                - is_word: 是否为单词
            durations: Kokoro 模型输出的时长数组 (numpy array)

        Returns:
            时间戳列表，每个元素包含:
                - word: 单词文本
                - start: 开始时间（秒）
                - end: 结束时间（秒）
        """
        if durations is None or len(durations) == 0:
            return []

        timestamps = []
        duration_idx = 0
        current_time = 0.0

        for token in tokens:
            if not token.get('is_word', False):
                # 非单词 token（标点等）
                # 如果是句末标点，可能有对应的停顿时长
                if token['text'] in '.!?':
                    if duration_idx < len(durations):
                        pause_duration = durations[duration_idx] / self.fps
                        current_time += pause_duration
                        duration_idx += 1
                elif token['text'] == ',':
                    # 逗号可能有短暂停顿
                    if duration_idx < len(durations):
                        pause_duration = durations[duration_idx] / self.fps
                        current_time += pause_duration
                        duration_idx += 1
                continue

            word = token['text']
            phoneme_count = token.get('phoneme_count', 1)

            # 确保至少有 1 个音素
            phoneme_count = max(phoneme_count, 1)

            # 记录单词开始时间
            word_start = current_time

            # 累加该单词所有音素的时长
            word_duration = 0.0
            for _ in range(phoneme_count):
                if duration_idx < len(durations):
                    frame_duration = durations[duration_idx]
                    word_duration += frame_duration / self.fps
                    duration_idx += 1
                else:
                    # durations 用完，使用默认值
                    word_duration += 0.08  # 80ms 默认值

            word_end = word_start + word_duration

            timestamps.append({
                'word': word,
                'start': round(word_start, 3),
                'end': round(word_end, 3)
            })

            current_time = word_end

        return timestamps

    def calculate_with_alignment_correction(
        self,
        tokens: List[Dict],
        durations: np.ndarray,
        total_audio_duration: float
    ) -> List[Dict]:
        """
        带对齐校正的时间戳计算

        如果计算出的总时长与实际音频时长不匹配，进行比例校正
        这可以补偿 FPS 估算不准确的问题

        Args:
            tokens: 分词列表
            durations: 时长数组
            total_audio_duration: 实际音频总时长（秒）

        Returns:
            校正后的时间戳列表
        """
        # 先计算原始时间戳
        raw_timestamps = self.calculate(tokens, durations)

        if not raw_timestamps:
            return []

        # 计算原始总时长
        raw_total = raw_timestamps[-1]['end']

        if raw_total <= 0:
            return raw_timestamps

        # 计算校正因子
        correction_factor = total_audio_duration / raw_total

        # 如果校正因子过大或过小，可能是计算错误，不校正
        if correction_factor < 0.5 or correction_factor > 2.0:
            print(f"Warning: Large correction factor {correction_factor:.2f}, skipping correction")
            return raw_timestamps

        # 应用校正
        corrected = []
        for ts in raw_timestamps:
            corrected.append({
                'word': ts['word'],
                'start': round(ts['start'] * correction_factor, 3),
                'end': round(ts['end'] * correction_factor, 3)
            })

        return corrected

    def merge_short_words(
        self,
        timestamps: List[Dict],
        min_duration: float = 0.05
    ) -> List[Dict]:
        """
        合并过短的时间戳

        有时某些短词（如 "a", "the"）的时间戳可能非常短，
        可以选择将它们合并到相邻的词

        Args:
            timestamps: 时间戳列表
            min_duration: 最小时长（秒）

        Returns:
            合并后的时间戳列表
        """
        if not timestamps:
            return []

        result = []
        pending = None

        for ts in timestamps:
            duration = ts['end'] - ts['start']

            if pending is not None:
                # 有待合并的短词，合并到当前词
                result.append({
                    'word': f"{pending['word']} {ts['word']}",
                    'start': pending['start'],
                    'end': ts['end']
                })
                pending = None
            elif duration < min_duration:
                # 当前词过短，标记待合并
                pending = ts
            else:
                result.append(ts)

        # 处理末尾可能剩余的短词
        if pending is not None:
            if result:
                # 合并到前一个词
                result[-1]['word'] += f" {pending['word']}"
                result[-1]['end'] = pending['end']
            else:
                result.append(pending)

        return result

    def add_gaps(
        self,
        timestamps: List[Dict],
        min_gap: float = 0.01
    ) -> List[Dict]:
        """
        在单词之间添加小间隙

        确保相邻单词之间有一定的时间间隔，避免完全无缝连接

        Args:
            timestamps: 时间戳列表
            min_gap: 最小间隙（秒）

        Returns:
            添加间隙后的时间戳列表
        """
        if not timestamps:
            return []

        result = [timestamps[0].copy()]

        for i in range(1, len(timestamps)):
            prev = result[-1]
            curr = timestamps[i].copy()

            # 检查是否有重叠或无间隙
            if curr['start'] <= prev['end']:
                # 添加间隙
                curr['start'] = prev['end'] + min_gap

            result.append(curr)

        return result


def frames_to_seconds(frames: int, sample_rate: int = 24000, hop_length: int = 256) -> float:
    """
    将帧数转换为秒

    Args:
        frames: 帧数
        sample_rate: 采样率
        hop_length: 帧移

    Returns:
        秒数
    """
    return frames * hop_length / sample_rate


def seconds_to_frames(seconds: float, sample_rate: int = 24000, hop_length: int = 256) -> int:
    """
    将秒转换为帧数

    Args:
        seconds: 秒数
        sample_rate: 采样率
        hop_length: 帧移

    Returns:
        帧数
    """
    return int(seconds * sample_rate / hop_length)


if __name__ == '__main__':
    # 测试
    calculator = TimestampCalculator()

    # 模拟数据
    tokens = [
        {'text': 'Hello', 'phoneme_count': 5, 'is_word': True},
        {'text': ',', 'phoneme_count': 1, 'is_word': False},
        {'text': 'world', 'phoneme_count': 5, 'is_word': True},
        {'text': '!', 'phoneme_count': 1, 'is_word': False},
    ]

    # 模拟 durations（每个音素约 8 帧，约 85ms）
    durations = np.array([
        8, 7, 9, 8, 8,  # Hello (5 phonemes)
        5,               # comma pause
        8, 8, 7, 9, 8,  # world (5 phonemes)
        10,              # exclamation pause
    ], dtype=np.float32)

    timestamps = calculator.calculate(tokens, durations)

    print("Timestamp Calculator Test")
    print("=" * 50)
    print(f"\nInput tokens: {[t['text'] for t in tokens]}")
    print(f"Durations: {durations}")
    print(f"\nTimestamps:")
    for ts in timestamps:
        duration = ts['end'] - ts['start']
        print(f"  '{ts['word']}': {ts['start']:.3f}s - {ts['end']:.3f}s (duration: {duration:.3f}s)")

    # 测试校正
    print("\n\nWith alignment correction (total audio = 1.5s):")
    corrected = calculator.calculate_with_alignment_correction(tokens, durations, 1.5)
    for ts in corrected:
        duration = ts['end'] - ts['start']
        print(f"  '{ts['word']}': {ts['start']:.3f}s - {ts['end']:.3f}s (duration: {duration:.3f}s)")
