# kokoro_inference.py
"""
Kokoro ONNX 模型推理封装

封装 Kokoro TTS 模型的加载和推理逻辑
支持带时间戳的 ONNX 模型
"""

import os
import json
import numpy as np
from typing import Tuple, List, Dict, Optional

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    ort = None
    HAS_ONNX = False


class KokoroInference:
    """
    Kokoro ONNX 模型推理封装

    支持带时间戳的推理，返回音频和每个音素的时长
    """

    SAMPLE_RATE = 24000  # Kokoro 输出采样率

    def __init__(
        self,
        model_dir: str,
        use_gpu: bool = False,
        num_threads: int = 4,
        model_name: str = 'model.onnx'
    ):
        """
        Args:
            model_dir: 模型目录路径
            use_gpu: 是否使用 GPU 加速
            num_threads: CPU 推理线程数
            model_name: ONNX 模型文件名
        """
        if not HAS_ONNX:
            raise ImportError("请安装 onnxruntime: pip install onnxruntime")

        self.model_dir = model_dir

        # 查找模型文件
        model_path = self._find_model(model_dir, model_name)
        if not model_path:
            raise FileNotFoundError(f"未找到模型文件: {model_name}")

        # 配置执行提供者
        providers = []
        if use_gpu:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')

        # 配置会话选项
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # 加载模型
        print(f"Loading model: {model_path}")
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )

        # 获取模型输入输出信息
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        # 加载词汇表
        self.vocab = self._load_vocab()
        self.phoneme_to_id = self.vocab  # vocab 已经是 {phoneme: id} 格式
        self.id_to_phoneme = {i: p for p, i in self.vocab.items()}

        # 加载语音包
        self.voices = self._load_voices()

        print(f"Loaded {len(self.voices)} voices, vocab size: {len(self.vocab)}")

    def _find_model(self, model_dir: str, model_name: str) -> Optional[str]:
        """查找模型文件"""
        # 直接路径
        path = os.path.join(model_dir, model_name)
        if os.path.exists(path):
            return path

        # onnx 子目录
        path = os.path.join(model_dir, 'onnx', model_name)
        if os.path.exists(path):
            return path

        return None

    def _load_vocab(self) -> Dict[str, int]:
        """加载音素词汇表"""
        # 尝试 tokenizer.json
        tokenizer_path = os.path.join(self.model_dir, 'tokenizer.json')
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'r', encoding='utf-8') as f:
                tokenizer = json.load(f)
                if 'model' in tokenizer and 'vocab' in tokenizer['model']:
                    return tokenizer['model']['vocab']

        # 尝试 vocab.json
        vocab_path = os.path.join(self.model_dir, 'vocab.json')
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_list = json.load(f)
                return {p: i for i, p in enumerate(vocab_list)}

        # 默认词汇表
        print("Warning: 未找到词汇表文件，使用默认词汇表")
        default = "$;:,.!?—…\"()""  ̈dʒdztsʲꭧAIOQSTWYᴊabcdefhijklmnopqrstuvwxyzɑɐɒæβɔɕçɖðʤəɚɛɜɟɡɥɨɪʝɯɰŋɳɲɴøɸθœɹɾɻʁɽʂʃʈʧʊʋʌɣɤχʎʒʔˈˌːʰʲ↓→↗↘ᵻ"
        return {c: i for i, c in enumerate(default)}

    def _load_voices(self) -> Dict[str, np.ndarray]:
        """
        加载所有语音包

        每个 voice 文件包含 (N, 256) 的 style vectors，
        其中 N 对应不同的序列长度（通常是 510）
        """
        voices = {}
        voices_dir = os.path.join(self.model_dir, 'voices')

        if not os.path.exists(voices_dir):
            print(f"Warning: 语音目录不存在: {voices_dir}")
            return voices

        for voice_file in os.listdir(voices_dir):
            if voice_file.endswith('.bin'):
                voice_name = voice_file[:-4]
                voice_path = os.path.join(voices_dir, voice_file)
                try:
                    voice_data = np.fromfile(voice_path, dtype=np.float32)
                    # 重塑为 (N, 256) - 每个长度对应一个 style vector
                    num_styles = len(voice_data) // 256
                    if num_styles > 0:
                        voices[voice_name] = voice_data.reshape(num_styles, 256)
                except Exception as e:
                    print(f"Warning: 加载语音包失败 {voice_name}: {e}")

        return voices

    def phonemes_to_ids(self, phonemes: str) -> np.ndarray:
        """
        将音素字符串转换为 ID 序列

        Args:
            phonemes: 音素字符串

        Returns:
            音素 ID 数组（包含首尾的 $ 标记）
        """
        ids = [0]  # 开始标记 $
        for char in phonemes:
            if char in self.phoneme_to_id:
                ids.append(self.phoneme_to_id[char])
            # 跳过未知字符
        ids.append(0)  # 结束标记 $
        return np.array(ids, dtype=np.int64)

    def inference(
        self,
        phonemes: str,
        voice: str = 'af_heart',
        speed: float = 1.0
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        执行 TTS 推理

        Args:
            phonemes: 音素字符串
            voice: 语音名称
            speed: 语速倍率 (0.5 - 2.0)

        Returns:
            Tuple of:
            - audio: 音频波形 (24kHz, float32)
            - durations: 每个音素的时长数组
        """
        # 验证语速范围
        speed = max(0.5, min(2.0, speed))

        # 转换音素为 ID
        input_ids = self.phonemes_to_ids(phonemes)
        if len(input_ids) <= 2:  # 只有开始和结束标记
            raise ValueError("输入音素为空或全部未知")

        # 获取语音向量 - 根据 token 长度选择对应的 style vector
        if voice not in self.voices:
            available = list(self.voices.keys())[:5]
            raise ValueError(f"未知语音: {voice}。可用: {available}...")

        voice_styles = self.voices[voice]  # shape: (N, 256)
        num_tokens = len(input_ids)

        # 根据 token 长度选择 style（官方做法）
        # 确保索引不超出范围
        style_idx = min(num_tokens, len(voice_styles) - 1)
        style = voice_styles[style_idx]  # shape: (256,)

        # 准备输入
        inputs = {
            'input_ids': input_ids.reshape(1, -1),
            'style': style.reshape(1, -1),
            'speed': np.array([speed], dtype=np.float32)
        }

        # 运行推理
        try:
            outputs = self.session.run(None, inputs)
        except Exception as e:
            raise RuntimeError(f"推理失败: {e}")

        # 解析输出
        # outputs[0] = waveform, outputs[1] = durations
        audio = outputs[0].squeeze()
        durations = outputs[1].squeeze() if len(outputs) > 1 else None

        return audio, durations

    def get_available_voices(self) -> List[str]:
        """返回可用的语音列表"""
        return sorted(self.voices.keys())

    def get_vocab(self) -> Dict[str, int]:
        """返回词汇表"""
        return self.vocab.copy()


class KokoroInferenceSimple:
    """
    简化版 Kokoro 推理接口

    不需要实际模型文件，用于测试和开发
    """

    SAMPLE_RATE = 24000

    def __init__(self):
        # 默认词汇表
        default = "$;:,.!? abcdefghijklmnopqrstuvwxyzɑɐɒæɔəɛɜɪʊʌŋɹʃʒθðˈˌː"
        self.vocab = {c: i for i, c in enumerate(default)}
        self.phoneme_to_id = self.vocab

    def phonemes_to_ids(self, phonemes: str) -> np.ndarray:
        """将音素转换为 ID"""
        ids = [0]
        for char in phonemes:
            if char in self.phoneme_to_id:
                ids.append(self.phoneme_to_id[char])
        ids.append(0)
        return np.array(ids, dtype=np.int64)

    def mock_inference(
        self,
        phonemes: str,
        speed: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        模拟推理（用于测试）

        Args:
            phonemes: 音素字符串
            speed: 语速

        Returns:
            模拟的 audio 和 durations
        """
        ids = self.phonemes_to_ids(phonemes)
        num_phonemes = len(ids)

        # 模拟 durations：每个音素 6-12 帧
        durations = np.random.uniform(6, 12, num_phonemes).astype(np.float32)
        durations /= speed

        # 模拟 audio
        total_frames = int(np.sum(durations))
        samples_per_frame = 256
        audio_length = total_frames * samples_per_frame
        audio = np.zeros(audio_length, dtype=np.float32)

        return audio, durations


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
        print(f"Testing with model: {model_dir}")

        kokoro = KokoroInference(model_dir)
        print(f"Voices: {kokoro.get_available_voices()[:10]}...")

        # 测试推理
        test_phonemes = "həlˈoʊ wˈɜːld"
        print(f"\nTest phonemes: {test_phonemes}")

        ids = kokoro.phonemes_to_ids(test_phonemes)
        print(f"IDs: {ids}")

        audio, durations = kokoro.inference(test_phonemes, voice='af_heart')
        print(f"Audio samples: {len(audio)}")
        print(f"Durations: {durations}")
        print(f"Duration (seconds): {len(audio) / 24000:.2f}s")
    else:
        print("Usage: python kokoro_inference.py <model_dir>")
        print("\nTesting with mock inference...")

        kokoro = KokoroInferenceSimple()
        audio, durations = kokoro.mock_inference("həlˈoʊ")
        print(f"Mock audio: {len(audio)} samples")
        print(f"Mock durations: {durations}")
