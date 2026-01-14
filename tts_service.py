# tts_service.py
"""
完整的 TTS 服务

整合 G2P、模型推理、时间戳计算，提供统一的 API
"""

import base64
import io
import os
import numpy as np
from typing import Dict, List, Optional, Union

try:
    from scipy.io import wavfile
except ImportError:
    wavfile = None
    print("Warning: scipy 未安装，音频编码功能受限")

from multilingual_g2p import MultilingualG2P
from timestamp_calculator import TimestampCalculator
from kokoro_inference import KokoroInference, KokoroInferenceSimple


class TTSService:
    """
    完整的 TTS 服务

    提供多语言语音合成和词级时间戳功能
    """

    SAMPLE_RATE = 24000  # Kokoro 输出采样率

    def __init__(
        self,
        model_dir: Optional[str] = None,
        use_gpu: bool = False,
        mock_mode: bool = False
    ):
        """
        Args:
            model_dir: Kokoro 模型目录路径
            use_gpu: 是否使用 GPU 加速
            mock_mode: 模拟模式（用于测试，不需要实际模型）
        """
        self.mock_mode = mock_mode

        # 初始化 G2P
        self.g2p = MultilingualG2P()

        # 初始化时间戳计算器
        self.calculator = TimestampCalculator()

        # 初始化 Kokoro 推理
        if mock_mode:
            self.kokoro = KokoroInferenceSimple()
        else:
            if model_dir is None:
                raise ValueError("model_dir 不能为空（除非使用 mock_mode）")
            self.kokoro = KokoroInference(model_dir, use_gpu=use_gpu)

    def _normalize_audio(self, audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
        """
        音量归一化（RMS 归一化）

        Args:
            audio: 输入音频
            target_rms: 目标 RMS 值

        Returns:
            归一化后的音频
        """
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 1e-6:  # 避免除零
            return audio * (target_rms / rms)
        return audio

    def _apply_fade(
        self,
        audio: np.ndarray,
        fade_in_ms: int = 20,
        fade_out_ms: int = 30
    ) -> np.ndarray:
        """
        应用淡入淡出效果

        Args:
            audio: 输入音频
            fade_in_ms: 淡入时长（毫秒）
            fade_out_ms: 淡出时长（毫秒）

        Returns:
            处理后的音频
        """
        audio = audio.copy()  # 避免修改原数组

        fade_in_samples = int(fade_in_ms * self.SAMPLE_RATE / 1000)
        fade_out_samples = int(fade_out_ms * self.SAMPLE_RATE / 1000)

        # 淡入
        if fade_in_samples > 0 and len(audio) > fade_in_samples:
            fade_in = np.linspace(0, 1, fade_in_samples)
            audio[:fade_in_samples] *= fade_in

        # 淡出
        if fade_out_samples > 0 and len(audio) > fade_out_samples:
            fade_out = np.linspace(1, 0, fade_out_samples)
            audio[-fade_out_samples:] *= fade_out

        return audio

    def _create_silence(self, duration_sec: float) -> np.ndarray:
        """
        创建静音片段

        Args:
            duration_sec: 静音时长（秒）

        Returns:
            静音音频数组
        """
        return np.zeros(int(duration_sec * self.SAMPLE_RATE), dtype=np.float32)

    def synthesize_long(
        self,
        text: str,
        language: str = 'a',
        voice: str = 'af_heart',
        speed: float = 1.0,
        return_timestamps: bool = True,
        return_audio_array: bool = False
    ) -> Dict:
        """
        合成长文本（自动分句处理，带音频平滑）

        Args:
            text: 输入文本（可以很长）
            language: 语言代码
            voice: 语音名称
            speed: 语速倍率
            return_timestamps: 是否返回时间戳
            return_audio_array: 是否返回原始音频数组

        Returns:
            合并后的结果
        """
        import re

        # 按句子分割（保留标点）
        sentences = re.split(r'(?<=[.!?。！？])\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # 根据句末标点决定停顿时长
        def get_pause_duration(sentence: str) -> float:
            if sentence.endswith('!') or sentence.endswith('?'):
                return 0.4  # 感叹号/问号后停顿长一些
            elif sentence.endswith('.') or sentence.endswith('。'):
                return 0.35  # 句号停顿
            else:
                return 0.25  # 默认停顿

        all_audio_segments = []
        all_timestamps = []
        all_phonemes = []
        current_time = 0.0

        for i, sentence in enumerate(sentences):
            try:
                result = self.synthesize(
                    text=sentence,
                    language=language,
                    voice=voice,
                    speed=speed,
                    return_timestamps=return_timestamps,
                    return_audio_array=True
                )

                audio = result['audio']

                # 1. 音量归一化
                audio = self._normalize_audio(audio, target_rms=0.08)

                # 2. 应用淡入淡出
                audio = self._apply_fade(audio, fade_in_ms=15, fade_out_ms=25)

                # 累积音频
                all_audio_segments.append(audio)

                # 调整时间戳偏移
                if return_timestamps:
                    for ts in result['timestamps']:
                        all_timestamps.append({
                            'word': ts['word'],
                            'start': round(ts['start'] + current_time, 3),
                            'end': round(ts['end'] + current_time, 3)
                        })

                all_phonemes.append(result['phonemes'])
                current_time += len(audio) / self.SAMPLE_RATE

                # 3. 在句子之间添加静音（除了最后一句）
                if i < len(sentences) - 1:
                    pause_duration = get_pause_duration(sentence)
                    silence = self._create_silence(pause_duration)
                    all_audio_segments.append(silence)
                    current_time += pause_duration

            except Exception as e:
                print(f"Warning: Failed to synthesize '{sentence[:30]}...': {e}")
                continue

        if not all_audio_segments:
            raise ValueError("所有句子合成都失败了")

        # 合并所有音频段（包括静音）
        combined_audio = np.concatenate(all_audio_segments)

        result = {
            'duration': len(combined_audio) / self.SAMPLE_RATE,
            'sample_rate': self.SAMPLE_RATE,
            'phonemes': ' | '.join(all_phonemes),
        }

        if return_timestamps:
            result['timestamps'] = all_timestamps

        if return_audio_array:
            result['audio'] = combined_audio
        else:
            result['audio'] = self._encode_audio(combined_audio)

        return result

    def synthesize(
        self,
        text: str,
        language: str = 'a',
        voice: str = 'af_heart',
        speed: float = 1.0,
        return_timestamps: bool = True,
        return_audio_array: bool = False
    ) -> Dict:
        """
        合成语音并生成时间戳

        Args:
            text: 输入文本
            language: 语言代码 (参考 MultilingualG2P.LANG_MAP)
            voice: 语音名称
            speed: 语速倍率 (0.5-2.0)
            return_timestamps: 是否返回时间戳
            return_audio_array: 是否返回原始音频数组

        Returns:
            Dict containing:
            - audio: base64 编码的 WAV 音频（或 numpy array 如果 return_audio_array=True）
            - duration: 音频时长（秒）
            - timestamps: 时间戳列表（如果 return_timestamps=True）
            - sample_rate: 采样率
            - phonemes: 音素字符串（调试用）
        """
        # 1. G2P 转换
        phonemes, tokens = self.g2p.g2p(text, language)

        if not phonemes:
            raise ValueError(f"G2P 转换失败: {text}")

        # 2. 模型推理
        if self.mock_mode:
            audio, durations = self.kokoro.mock_inference(phonemes, speed)
        else:
            audio, durations = self.kokoro.inference(phonemes, voice, speed)

        # 3. 计算时间戳
        timestamps = []
        if return_timestamps and durations is not None:
            audio_duration = len(audio) / self.SAMPLE_RATE
            timestamps = self.calculator.calculate_with_alignment_correction(
                tokens, durations, audio_duration
            )

        # 4. 准备输出
        result = {
            'duration': len(audio) / self.SAMPLE_RATE,
            'sample_rate': self.SAMPLE_RATE,
            'phonemes': phonemes,
        }

        if return_timestamps:
            result['timestamps'] = timestamps

        if return_audio_array:
            result['audio'] = audio
        else:
            result['audio'] = self._encode_audio(audio)

        return result

    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        language: str = 'a',
        voice: str = 'af_heart',
        speed: float = 1.0
    ) -> Dict:
        """
        合成语音并保存到文件

        Args:
            text: 输入文本
            output_path: 输出文件路径 (.wav)
            language: 语言代码
            voice: 语音名称
            speed: 语速倍率

        Returns:
            包含时间戳和元数据的 Dict
        """
        result = self.synthesize(
            text=text,
            language=language,
            voice=voice,
            speed=speed,
            return_timestamps=True,
            return_audio_array=True
        )

        # 保存音频文件
        audio = result['audio']
        self._save_audio(audio, output_path)

        # 替换为文件路径
        result['audio'] = output_path

        return result

    def _encode_audio(self, audio: np.ndarray) -> str:
        """
        将音频波形编码为 base64 WAV

        Args:
            audio: 音频波形 (float32, -1 to 1)

        Returns:
            base64 编码的 WAV 数据
        """
        if wavfile is None:
            # 如果没有 scipy，返回原始数据的 base64
            audio_int16 = (audio * 32767).astype(np.int16)
            return base64.b64encode(audio_int16.tobytes()).decode('utf-8')

        # 转换为 16-bit PCM
        audio_clipped = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)

        # 写入 WAV 到内存
        buffer = io.BytesIO()
        wavfile.write(buffer, self.SAMPLE_RATE, audio_int16)

        # 编码为 base64
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    def _save_audio(self, audio: np.ndarray, path: str):
        """
        保存音频到文件

        Args:
            audio: 音频波形
            path: 输出路径
        """
        if wavfile is None:
            raise RuntimeError("需要 scipy 才能保存音频文件: pip install scipy")

        # 转换为 16-bit PCM
        audio_clipped = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)

        # 确保目录存在
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

        # 保存
        wavfile.write(path, self.SAMPLE_RATE, audio_int16)

    def get_supported_languages(self) -> List[str]:
        """返回支持的语言列表"""
        return self.g2p.get_supported_languages()

    def get_available_voices(self) -> List[str]:
        """返回可用的语音列表"""
        if self.mock_mode:
            return ['af_heart', 'af_bella', 'am_adam', 'bf_emma']
        return self.kokoro.get_available_voices()

    def get_language_info(self) -> Dict[str, str]:
        """返回语言代码和名称的映射"""
        return {
            code: self.g2p.get_language_name(code)
            for code in self.get_supported_languages()
        }


class TTSServiceAsync:
    """
    异步 TTS 服务封装

    用于 FastAPI 等异步框架
    """

    def __init__(self, tts_service: TTSService):
        self.tts = tts_service

    async def synthesize(self, *args, **kwargs) -> Dict:
        """异步合成（实际仍是同步，但符合异步接口）"""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.tts.synthesize(*args, **kwargs)
        )


def create_tts_service(
    model_dir: Optional[str] = None,
    use_gpu: bool = False,
    auto_mock: bool = True
) -> TTSService:
    """
    工厂函数：创建 TTS 服务

    Args:
        model_dir: 模型目录路径
        use_gpu: 是否使用 GPU
        auto_mock: 如果模型不存在，自动使用 mock 模式

    Returns:
        TTSService 实例
    """
    if model_dir and os.path.exists(model_dir):
        return TTSService(model_dir=model_dir, use_gpu=use_gpu, mock_mode=False)
    elif auto_mock:
        print("Warning: 模型目录不存在，使用 mock 模式")
        return TTSService(mock_mode=True)
    else:
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")


if __name__ == '__main__':
    # 测试（使用 mock 模式）
    print("TTS Service Test (Mock Mode)")
    print("=" * 50)

    tts = TTSService(mock_mode=True)

    # 测试多语言
    test_cases = [
        ("Hello, how are you?", "a", "American English"),
        ("Bonjour, comment allez-vous?", "f", "French"),
        ("Hola, cómo estás?", "e", "Spanish"),
        ("Guten Tag, wie geht es Ihnen?", "de", "German"),
        ("Ciao, come stai?", "i", "Italian"),
    ]

    for text, lang, lang_name in test_cases:
        print(f"\n{lang_name}: {text}")

        result = tts.synthesize(
            text=text,
            language=lang,
            return_timestamps=True,
            return_audio_array=True
        )

        print(f"  Phonemes: {result['phonemes']}")
        print(f"  Duration: {result['duration']:.2f}s")
        print(f"  Timestamps:")
        for ts in result.get('timestamps', []):
            print(f"    '{ts['word']}': {ts['start']:.3f}s - {ts['end']:.3f}s")

    # 显示支持的语言
    print("\n" + "=" * 50)
    print("Supported Languages:")
    for code, name in tts.get_language_info().items():
        print(f"  {code}: {name}")
