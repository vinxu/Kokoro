# Kokoro TTS 多语言时间戳实现方案

## 背景

Kokoro TTS 是一个轻量级（82M 参数）开源语音合成模型，支持多种语言的语音合成。但其**词级时间戳功能目前仅支持英语**，原因是依赖 CMU 发音词典进行音素-单词对齐。

本文档详细描述如何通过 **espeak-ng** 实现多语言（拉丁语系、西里尔语系等）的时间戳支持。

---

## 一、技术原理

### 1.1 当前英语时间戳流程

```
输入文本
    ↓
misaki G2P (英语专用)
    ↓
phonemes (IPA 音素) + tokens (分词)
    ↓
Kokoro 模型推理
    ↓
audio (音频) + durations (每个音素的时长数组)
    ↓
时间戳计算 (durations → 单词时间边界)
    ↓
输出: {word, start_time, end_time}[]
```

### 1.2 问题分析

英语时间戳工作的关键依赖：
1. **CMU Pronouncing Dictionary**: 包含 134,000+ 英语单词的音素映射
2. **misaki G2P**: 将英语文本转换为 IPA 音素，同时保留单词边界信息
3. **durations 数组**: Kokoro 模型输出每个音素的时长（以帧为单位）

非英语语言缺少 CMU 词典等价物，misaki 的非英语支持有限。

### 1.3 解决方案：espeak-ng

**espeak-ng** 是一个开源语音合成器，支持 **127+ 种语言**的 G2P（文本转音素）功能，包括：
- 拉丁语系：法语、西班牙语、意大利语、葡萄牙语、德语、波兰语等
- 西里尔语系：俄语、乌克兰语、白俄罗斯语、保加利亚语等
- 其他：希腊语、希伯来语、阿拉伯语等

### 1.4 目标流程

```
输入文本 + 语言代码
    ↓
espeak-ng G2P (多语言)
    ↓
phonemes (IPA 音素) + tokens (分词)
    ↓
音素格式转换 (espeak IPA → Kokoro 格式)
    ↓
Kokoro 模型推理
    ↓
audio + durations
    ↓
单词-音素对齐 + 时间戳计算
    ↓
输出: {word, start_time, end_time}[]
```

---

## 二、环境准备

### 2.1 系统依赖

#### macOS
```bash
brew install espeak-ng
```

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install espeak-ng libespeak-ng-dev
```

#### CentOS/RHEL
```bash
sudo yum install espeak-ng espeak-ng-devel
```

#### Windows
从 [espeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases) 下载安装包。

### 2.2 Python 依赖

```bash
# 核心依赖
pip install py-espeak-ng          # espeak-ng Python 封装
pip install phonemizer            # 备选 G2P 库
pip install onnxruntime           # Kokoro ONNX 推理
pip install numpy

# 可选：更好的分词支持
pip install spacy
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
python -m spacy download es_core_news_sm
python -m spacy download de_core_news_sm
python -m spacy download ru_core_news_sm
```

### 2.3 Kokoro 模型文件

从 Hugging Face 下载带时间戳支持的 ONNX 模型：

```bash
# 下载模型
git lfs install
git clone https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX-timestamped

# 或使用 huggingface_hub
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('onnx-community/Kokoro-82M-v1.0-ONNX-timestamped', local_dir='./kokoro-model')"
```

模型文件结构：
```
kokoro-model/
├── model.onnx              # 主模型
├── config.json             # 配置
├── vocab.json              # 音素词汇表 (重要！)
└── voices/                 # 语音包
    ├── af_heart.bin
    ├── af_bella.bin
    └── ...
```

---

## 三、音素系统分析

### 3.1 Kokoro 音素集

Kokoro 使用基于 IPA 的音素系统，完整音素集定义在 `vocab.json` 中。

**关键音素类别：**

| 类别 | 示例 | 说明 |
|-----|------|------|
| 元音 | ə, ɑ, ɛ, ɪ, ɔ, ʊ, æ | 单元音 |
| 双元音 | aɪ, aʊ, eɪ, oʊ, ɔɪ | 组合元音 |
| 辅音 | p, b, t, d, k, g, m, n | 基本辅音 |
| 摩擦音 | f, v, s, z, ʃ, ʒ, θ, ð | 摩擦辅音 |
| 塞擦音 | tʃ, dʒ | 组合辅音 |
| 重音 | ˈ (主), ˌ (次) | 重音标记 |
| 长音 | ː | 延长标记 |
| 特殊 | ˈ, ˌ, ., ,, ?, ! | 韵律标记 |

### 3.2 espeak-ng 音素输出

espeak-ng 输出标准 IPA，但有些细微差异：

```python
from espeakng import ESpeakNG

esng = ESpeakNG()

# 英语
esng.voice = 'en-us'
print(esng.g2p('Hello world', ipa=2))
# 输出: həlˈoʊ wˈɜːld

# 法语
esng.voice = 'fr'
print(esng.g2p('Bonjour le monde', ipa=2))
# 输出: bɔ̃ʒˈuʁ lə mˈɔ̃d

# 俄语
esng.voice = 'ru'
print(esng.g2p('Привет мир', ipa=2))
# 输出: prʲɪvʲˈet mʲˈir
```

### 3.3 音素映射表

需要构建 espeak-ng → Kokoro 的音素映射表：

```python
# phoneme_mapping.py

# 通用 IPA 音素（直接映射）
DIRECT_MAP = {
    # 元音
    'ə': 'ə', 'ɑ': 'ɑ', 'æ': 'æ', 'ɛ': 'ɛ', 'ɪ': 'ɪ',
    'ɔ': 'ɔ', 'ʊ': 'ʊ', 'ʌ': 'ʌ', 'i': 'i', 'u': 'u',
    'e': 'e', 'o': 'o', 'a': 'a',

    # 辅音
    'p': 'p', 'b': 'b', 't': 't', 'd': 'd', 'k': 'k', 'g': 'g',
    'm': 'm', 'n': 'n', 'ŋ': 'ŋ',
    'f': 'f', 'v': 'v', 's': 's', 'z': 'z',
    'ʃ': 'ʃ', 'ʒ': 'ʒ', 'θ': 'θ', 'ð': 'ð',
    'h': 'h', 'w': 'w', 'j': 'j', 'l': 'l', 'r': 'r',

    # 重音和韵律
    'ˈ': 'ˈ', 'ˌ': 'ˌ', 'ː': 'ː',
    '.': '.', ',': ',', '?': '?', '!': '!',
    ' ': ' ',
}

# 需要转换的音素
CONVERT_MAP = {
    # espeak 特有 → Kokoro 等价
    'ɹ': 'r',      # 英语 r
    'ɾ': 'r',      # 弹舌 r
    'ʁ': 'r',      # 法语小舌 r
    'ɐ': 'ə',      # 近央元音
    'ɜ': 'ɜ',      #
    'ɝ': 'ɜr',     # 美式卷舌元音
    'ɚ': 'ər',     # 弱化卷舌元音

    # 鼻化元音（法语等）
    'ɑ̃': 'ɑn',    # on
    'ɛ̃': 'ɛn',    # in
    'ɔ̃': 'ɔn',    # on
    'œ̃': 'œn',    # un

    # 俄语特有
    'ʲ': 'j',      # 软化标记 → j
    'ɕ': 'ʃ',      # 软化 ш
    'ʑ': 'ʒ',      # 软化 ж
    'ɨ': 'ɪ',      # ы 音
    'ʂ': 'ʃ',      # 硬 ш
    'ʐ': 'ʒ',      # 硬 ж
    'ts': 'ts',    # ц

    # 德语特有
    'ç': 'h',      # ich-laut
    'x': 'h',      # ach-laut
    'ʏ': 'y',      # ü 短音
    'yː': 'y',     # ü 长音
    'øː': 'ø',     # ö 长音
    'œ': 'ø',      # ö 短音
}

# 组合音素（双元音等）
DIPHTHONG_MAP = {
    'aɪ': 'aɪ', 'aʊ': 'aʊ', 'eɪ': 'eɪ',
    'oʊ': 'oʊ', 'ɔɪ': 'ɔɪ', 'ɪə': 'ɪə',
    'eə': 'eə', 'ʊə': 'ʊə',
}

def convert_phonemes(espeak_output: str) -> str:
    """
    将 espeak-ng 的 IPA 输出转换为 Kokoro 格式

    Args:
        espeak_output: espeak-ng g2p 输出的 IPA 字符串

    Returns:
        Kokoro 兼容的音素字符串
    """
    result = espeak_output

    # 1. 先处理组合音素（双元音等）
    for src, dst in DIPHTHONG_MAP.items():
        result = result.replace(src, dst)

    # 2. 处理需要转换的音素
    for src, dst in CONVERT_MAP.items():
        result = result.replace(src, dst)

    # 3. 移除 Kokoro 不支持的字符
    # 保留 DIRECT_MAP 中的字符，移除其他
    cleaned = []
    i = 0
    while i < len(result):
        char = result[i]
        if char in DIRECT_MAP:
            cleaned.append(DIRECT_MAP[char])
            i += 1
        elif i + 1 < len(result) and result[i:i+2] in DIPHTHONG_MAP:
            cleaned.append(DIPHTHONG_MAP[result[i:i+2]])
            i += 2
        else:
            # 未知字符，跳过或记录警告
            print(f"Warning: Unknown phoneme '{char}' (U+{ord(char):04X})")
            i += 1

    return ''.join(cleaned)
```

---

## 四、核心实现

### 4.1 多语言 G2P 模块

```python
# multilingual_g2p.py

from espeakng import ESpeakNG
from typing import Tuple, List, Dict
import re
from phoneme_mapping import convert_phonemes

class MultilingualG2P:
    """
    多语言 G2P (Grapheme-to-Phoneme) 转换器
    使用 espeak-ng 作为后端，支持 100+ 种语言
    """

    # Kokoro 语言代码 → espeak-ng 语言代码
    LANG_MAP = {
        # Kokoro 原生支持
        'a': 'en-us',      # American English
        'b': 'en-gb',      # British English
        'e': 'es',         # Spanish (Spain)
        'f': 'fr',         # French
        'h': 'hi',         # Hindi
        'i': 'it',         # Italian
        'j': 'ja',         # Japanese
        'p': 'pt-br',      # Portuguese (Brazil)
        'z': 'zh',         # Mandarin Chinese

        # 扩展支持 - 拉丁语系
        'es-mx': 'es-mx',  # Spanish (Mexico)
        'fr-ca': 'fr',     # French (Canada)
        'pt-pt': 'pt',     # Portuguese (Portugal)
        'de': 'de',        # German
        'nl': 'nl',        # Dutch
        'pl': 'pl',        # Polish
        'cs': 'cs',        # Czech
        'ro': 'ro',        # Romanian
        'sv': 'sv',        # Swedish
        'da': 'da',        # Danish
        'no': 'nb',        # Norwegian
        'fi': 'fi',        # Finnish

        # 扩展支持 - 西里尔语系
        'ru': 'ru',        # Russian
        'uk': 'uk',        # Ukrainian
        'be': 'be',        # Belarusian
        'bg': 'bg',        # Bulgarian
        'sr': 'sr',        # Serbian

        # 其他
        'el': 'el',        # Greek
        'tr': 'tr',        # Turkish
        'ar': 'ar',        # Arabic
        'he': 'he',        # Hebrew
    }

    def __init__(self):
        self.esng = ESpeakNG()
        self._current_lang = None

    def g2p(self, text: str, lang: str = 'a') -> Tuple[str, List[Dict]]:
        """
        将文本转换为音素，同时返回分词信息

        Args:
            text: 输入文本
            lang: 语言代码 (Kokoro 格式)

        Returns:
            Tuple of:
            - phonemes: Kokoro 格式的音素字符串
            - tokens: 分词列表，每个元素包含:
                - text: 原始文本
                - phonemes: 该词的音素
                - phoneme_count: 音素数量
                - is_word: 是否为单词（非标点）
        """
        # 设置语言
        espeak_lang = self.LANG_MAP.get(lang, 'en-us')
        if self._current_lang != espeak_lang:
            self.esng.voice = espeak_lang
            self._current_lang = espeak_lang

        # 分词
        words = self._tokenize(text)

        # 对每个单词单独获取音素
        tokens = []
        all_phonemes = []

        for word_info in words:
            word = word_info['text']

            if word_info['is_word']:
                # 获取单词的音素
                try:
                    word_phonemes = self.esng.g2p(word, ipa=2)
                    # 转换为 Kokoro 格式
                    converted = convert_phonemes(word_phonemes)
                    phoneme_count = self._count_phonemes(converted)
                except Exception as e:
                    print(f"Warning: G2P failed for '{word}': {e}")
                    converted = ''
                    phoneme_count = 0

                tokens.append({
                    'text': word,
                    'phonemes': converted,
                    'phoneme_count': phoneme_count,
                    'is_word': True
                })
                all_phonemes.append(converted)
            else:
                # 标点符号
                tokens.append({
                    'text': word,
                    'phonemes': word,
                    'phoneme_count': 1 if word in '.!?,' else 0,
                    'is_word': False
                })
                if word in '.!?,':
                    all_phonemes.append(word)

        # 合并所有音素
        full_phonemes = ' '.join(all_phonemes)

        return full_phonemes, tokens

    def _tokenize(self, text: str) -> List[Dict]:
        """
        分词，保留单词和标点

        支持多种语言的分词策略
        """
        # 基本分词：匹配单词或单个标点
        pattern = r"[\w\u0400-\u04FF\u4e00-\u9fff]+|[^\w\s]"
        # \u0400-\u04FF: 西里尔字母
        # \u4e00-\u9fff: 中日韩字符

        matches = re.findall(pattern, text, re.UNICODE)

        tokens = []
        for match in matches:
            is_word = bool(re.match(r"[\w\u0400-\u04FF\u4e00-\u9fff]+", match, re.UNICODE))
            tokens.append({
                'text': match,
                'is_word': is_word
            })

        return tokens

    def _count_phonemes(self, phoneme_str: str) -> int:
        """
        统计音素数量

        排除重音标记、空格等非音素字符
        """
        if not phoneme_str:
            return 0

        # 非音素字符
        non_phonemes = set('ˈˌː .,?!-')

        count = 0
        i = 0
        while i < len(phoneme_str):
            char = phoneme_str[i]

            # 跳过非音素字符
            if char in non_phonemes:
                i += 1
                continue

            # 检查双字符音素（双元音等）
            if i + 1 < len(phoneme_str):
                two_char = phoneme_str[i:i+2]
                if two_char in ['aɪ', 'aʊ', 'eɪ', 'oʊ', 'ɔɪ', 'tʃ', 'dʒ', 'ɪə', 'eə', 'ʊə']:
                    count += 1
                    i += 2
                    continue

            count += 1
            i += 1

        return max(count, 1)

    def get_supported_languages(self) -> List[str]:
        """返回支持的语言列表"""
        return list(self.LANG_MAP.keys())
```

### 4.2 时间戳计算模块

```python
# timestamp_calculator.py

from typing import List, Dict
import numpy as np

# Kokoro 帧率转换常数
# 模型输出的 duration 单位是帧，需要转换为秒
# 24000 Hz 采样率，每帧对应 256 个采样点
# 所以 1 帧 = 256 / 24000 ≈ 0.0107 秒
# 实际使用的魔数是 80（经验值）
FRAMES_PER_SECOND = 80.0

class TimestampCalculator:
    """
    根据 Kokoro 输出的 durations 计算单词级时间戳
    """

    def __init__(self, frames_per_second: float = FRAMES_PER_SECOND):
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
        timestamps = []
        duration_idx = 0
        current_time = 0.0

        # 初始偏移（第一个音素的一半时长）
        if len(durations) > 0:
            initial_offset = durations[0] / (2 * self.fps)
            current_time = initial_offset

        for token in tokens:
            if not token.get('is_word', False):
                # 跳过标点，但可能需要分配时长
                if token['text'] in '.!?':
                    # 句末标点，分配一点静音时间
                    if duration_idx < len(durations):
                        pause_duration = durations[duration_idx] / self.fps
                        current_time += pause_duration
                        duration_idx += 1
                continue

            word = token['text']
            phoneme_count = token.get('phoneme_count', 1)

            # 记录单词开始时间
            word_start = current_time

            # 累加该单词所有音素的时长
            word_duration = 0.0
            for _ in range(phoneme_count):
                if duration_idx < len(durations):
                    word_duration += durations[duration_idx] / self.fps
                    duration_idx += 1
                else:
                    # 如果 durations 用完，使用平均值
                    avg_duration = 0.1  # 100ms 默认值
                    word_duration += avg_duration

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

        # 应用校正
        corrected = []
        for ts in raw_timestamps:
            corrected.append({
                'word': ts['word'],
                'start': round(ts['start'] * correction_factor, 3),
                'end': round(ts['end'] * correction_factor, 3)
            })

        return corrected
```

### 4.3 Kokoro 推理封装

```python
# kokoro_inference.py

import numpy as np
import onnxruntime as ort
from typing import Tuple, Optional
import json
import os

class KokoroInference:
    """
    Kokoro ONNX 模型推理封装
    """

    def __init__(self, model_dir: str):
        """
        Args:
            model_dir: 模型目录路径，包含 model.onnx, vocab.json 等
        """
        self.model_dir = model_dir

        # 加载模型
        model_path = os.path.join(model_dir, 'model.onnx')
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']  # 或 'CUDAExecutionProvider'
        )

        # 加载词汇表
        vocab_path = os.path.join(model_dir, 'vocab.json')
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

        # 构建音素到 ID 的映射
        self.phoneme_to_id = {p: i for i, p in enumerate(self.vocab)}

        # 加载语音包
        self.voices = {}
        voices_dir = os.path.join(model_dir, 'voices')
        if os.path.exists(voices_dir):
            for voice_file in os.listdir(voices_dir):
                if voice_file.endswith('.bin'):
                    voice_name = voice_file[:-4]
                    voice_path = os.path.join(voices_dir, voice_file)
                    self.voices[voice_name] = np.fromfile(voice_path, dtype=np.float32)

    def phonemes_to_ids(self, phonemes: str) -> np.ndarray:
        """
        将音素字符串转换为 ID 序列

        Args:
            phonemes: 音素字符串（空格分隔的单词）

        Returns:
            音素 ID 数组
        """
        ids = []
        for char in phonemes:
            if char in self.phoneme_to_id:
                ids.append(self.phoneme_to_id[char])
            elif char == ' ':
                # 空格使用特殊 token
                if ' ' in self.phoneme_to_id:
                    ids.append(self.phoneme_to_id[' '])
            else:
                # 未知音素，跳过或使用 UNK
                print(f"Warning: Unknown phoneme '{char}'")

        return np.array(ids, dtype=np.int64)

    def inference(
        self,
        phonemes: str,
        voice: str = 'af_heart',
        speed: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行 TTS 推理

        Args:
            phonemes: 音素字符串
            voice: 语音名称
            speed: 语速倍率

        Returns:
            Tuple of:
            - audio: 音频波形 (24kHz, float32)
            - durations: 每个音素的时长数组
        """
        # 转换音素为 ID
        input_ids = self.phonemes_to_ids(phonemes)

        # 获取语音向量
        if voice not in self.voices:
            raise ValueError(f"Unknown voice: {voice}. Available: {list(self.voices.keys())}")
        voice_embedding = self.voices[voice]

        # 准备输入
        inputs = {
            'input_ids': input_ids.reshape(1, -1),
            'voice': voice_embedding.reshape(1, -1),
            'speed': np.array([speed], dtype=np.float32)
        }

        # 运行推理
        outputs = self.session.run(None, inputs)

        # 解析输出
        # 输出顺序可能因模型版本而异，需要根据实际情况调整
        audio = outputs[0].squeeze()  # 音频波形
        durations = outputs[1].squeeze() if len(outputs) > 1 else None  # 时长数组

        return audio, durations

    def get_available_voices(self) -> list:
        """返回可用的语音列表"""
        return list(self.voices.keys())
```

### 4.4 完整 TTS 服务

```python
# tts_service.py

import base64
import io
import numpy as np
from scipy.io import wavfile
from typing import Dict, List, Optional

from multilingual_g2p import MultilingualG2P
from timestamp_calculator import TimestampCalculator
from kokoro_inference import KokoroInference

class TTSService:
    """
    完整的 TTS 服务，支持多语言时间戳
    """

    SAMPLE_RATE = 24000  # Kokoro 输出采样率

    def __init__(self, model_dir: str):
        """
        Args:
            model_dir: Kokoro 模型目录
        """
        self.g2p = MultilingualG2P()
        self.calculator = TimestampCalculator()
        self.kokoro = KokoroInference(model_dir)

    def synthesize(
        self,
        text: str,
        language: str = 'a',
        voice: str = 'af_heart',
        speed: float = 1.0,
        return_timestamps: bool = True
    ) -> Dict:
        """
        合成语音并生成时间戳

        Args:
            text: 输入文本
            language: 语言代码 (参考 MultilingualG2P.LANG_MAP)
            voice: 语音名称
            speed: 语速倍率 (0.5-2.0)
            return_timestamps: 是否返回时间戳

        Returns:
            {
                'audio': base64 编码的 WAV 音频,
                'duration': 音频时长（秒）,
                'timestamps': [{'word': str, 'start': float, 'end': float}, ...],
                'sample_rate': 采样率
            }
        """
        # 1. G2P 转换
        phonemes, tokens = self.g2p.g2p(text, language)

        if not phonemes:
            raise ValueError(f"G2P failed for text: {text}")

        # 2. Kokoro 推理
        audio, durations = self.kokoro.inference(phonemes, voice, speed)

        # 3. 计算时间戳
        timestamps = []
        if return_timestamps and durations is not None:
            audio_duration = len(audio) / self.SAMPLE_RATE
            timestamps = self.calculator.calculate_with_alignment_correction(
                tokens, durations, audio_duration
            )

        # 4. 编码音频为 base64
        audio_b64 = self._encode_audio(audio)

        return {
            'audio': audio_b64,
            'duration': len(audio) / self.SAMPLE_RATE,
            'timestamps': timestamps,
            'sample_rate': self.SAMPLE_RATE
        }

    def _encode_audio(self, audio: np.ndarray) -> str:
        """将音频波形编码为 base64 WAV"""
        # 转换为 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)

        # 写入 WAV
        buffer = io.BytesIO()
        wavfile.write(buffer, self.SAMPLE_RATE, audio_int16)

        # 编码为 base64
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    def get_supported_languages(self) -> List[str]:
        """返回支持的语言列表"""
        return self.g2p.get_supported_languages()

    def get_available_voices(self) -> List[str]:
        """返回可用的语音列表"""
        return self.kokoro.get_available_voices()
```

---

## 五、API 集成示例

### 5.1 FastAPI 服务

```python
# api_server.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from tts_service import TTSService

app = FastAPI(title="Kokoro TTS API")

# 初始化 TTS 服务
tts = TTSService(model_dir="./kokoro-model")

class TTSRequest(BaseModel):
    text: str
    language: str = 'a'
    voice: str = 'af_heart'
    speed: float = 1.0
    return_timestamps: bool = True

class TimestampItem(BaseModel):
    word: str
    start: float
    end: float

class TTSResponse(BaseModel):
    audio: str
    duration: float
    timestamps: List[TimestampItem]
    sample_rate: int

@app.post("/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest):
    """
    文本转语音 API

    支持的语言代码:
    - a: 美式英语
    - b: 英式英语
    - e: 西班牙语
    - f: 法语
    - i: 意大利语
    - de: 德语
    - ru: 俄语
    - ...
    """
    try:
        result = tts.synthesize(
            text=request.text,
            language=request.language,
            voice=request.voice,
            speed=request.speed,
            return_timestamps=request.return_timestamps
        )
        return TTSResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/languages")
async def get_languages():
    """获取支持的语言列表"""
    return {"languages": tts.get_supported_languages()}

@app.get("/voices")
async def get_voices():
    """获取可用的语音列表"""
    return {"voices": tts.get_available_voices()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 5.2 调用示例

```python
# client_example.py

import requests
import base64

# 法语示例
response = requests.post(
    "http://localhost:8000/tts",
    json={
        "text": "Bonjour, comment allez-vous?",
        "language": "f",
        "voice": "af_heart",
        "return_timestamps": True
    }
)

result = response.json()

# 保存音频
audio_data = base64.b64decode(result['audio'])
with open("output.wav", "wb") as f:
    f.write(audio_data)

# 打印时间戳
print(f"Duration: {result['duration']:.2f}s")
for ts in result['timestamps']:
    print(f"  {ts['word']}: {ts['start']:.3f}s - {ts['end']:.3f}s")
```

输出示例：
```
Duration: 2.15s
  Bonjour: 0.000s - 0.450s
  comment: 0.520s - 0.890s
  allez: 0.950s - 1.280s
  vous: 1.340s - 1.650s
```

---

## 六、测试与验证

### 6.1 单元测试

```python
# test_g2p.py

import pytest
from multilingual_g2p import MultilingualG2P

@pytest.fixture
def g2p():
    return MultilingualG2P()

class TestMultilingualG2P:

    def test_english(self, g2p):
        phonemes, tokens = g2p.g2p("Hello world", "a")
        assert len(tokens) == 2
        assert tokens[0]['text'] == "Hello"
        assert tokens[0]['is_word'] == True
        assert tokens[0]['phoneme_count'] > 0

    def test_french(self, g2p):
        phonemes, tokens = g2p.g2p("Bonjour le monde", "f")
        assert len(tokens) == 3
        assert "ʒ" in phonemes or "ʃ" in phonemes  # 法语特有

    def test_russian(self, g2p):
        phonemes, tokens = g2p.g2p("Привет мир", "ru")
        assert len(tokens) == 2
        assert tokens[0]['text'] == "Привет"

    def test_spanish(self, g2p):
        phonemes, tokens = g2p.g2p("Hola mundo", "e")
        assert len(tokens) == 2

    def test_german(self, g2p):
        phonemes, tokens = g2p.g2p("Guten Tag", "de")
        assert len(tokens) == 2

    def test_punctuation(self, g2p):
        phonemes, tokens = g2p.g2p("Hello, world!", "a")
        # 应该有 Hello, 逗号, world, 感叹号
        word_tokens = [t for t in tokens if t['is_word']]
        assert len(word_tokens) == 2

# test_timestamps.py

import pytest
import numpy as np
from timestamp_calculator import TimestampCalculator

@pytest.fixture
def calculator():
    return TimestampCalculator()

class TestTimestampCalculator:

    def test_basic_calculation(self, calculator):
        tokens = [
            {'text': 'Hello', 'phoneme_count': 5, 'is_word': True},
            {'text': 'world', 'phoneme_count': 5, 'is_word': True},
        ]
        durations = np.array([10, 8, 12, 9, 11, 10, 10, 8, 12, 10], dtype=np.float32)

        timestamps = calculator.calculate(tokens, durations)

        assert len(timestamps) == 2
        assert timestamps[0]['word'] == 'Hello'
        assert timestamps[1]['word'] == 'world'
        assert timestamps[0]['start'] < timestamps[0]['end']
        assert timestamps[0]['end'] <= timestamps[1]['start']

    def test_alignment_correction(self, calculator):
        tokens = [
            {'text': 'Test', 'phoneme_count': 4, 'is_word': True},
        ]
        durations = np.array([10, 10, 10, 10], dtype=np.float32)

        # 假设实际音频是 2 秒
        timestamps = calculator.calculate_with_alignment_correction(
            tokens, durations, total_audio_duration=2.0
        )

        assert len(timestamps) == 1
        assert timestamps[0]['end'] <= 2.0
```

### 6.2 集成测试

```python
# test_integration.py

import pytest
from tts_service import TTSService
import base64
import os

@pytest.fixture
def tts_service():
    model_dir = os.environ.get('KOKORO_MODEL_DIR', './kokoro-model')
    return TTSService(model_dir)

class TestTTSIntegration:

    @pytest.mark.parametrize("text,lang", [
        ("Hello world", "a"),
        ("Bonjour le monde", "f"),
        ("Hola mundo", "e"),
        ("Guten Tag", "de"),
        ("Привет мир", "ru"),
    ])
    def test_multilingual_synthesis(self, tts_service, text, lang):
        """测试多语言合成"""
        result = tts_service.synthesize(
            text=text,
            language=lang,
            return_timestamps=True
        )

        # 验证音频
        assert result['audio'] is not None
        assert result['duration'] > 0

        # 验证时间戳
        assert len(result['timestamps']) > 0

        # 验证时间戳顺序
        for i in range(1, len(result['timestamps'])):
            assert result['timestamps'][i]['start'] >= result['timestamps'][i-1]['end']

    def test_timestamp_accuracy(self, tts_service):
        """测试时间戳准确性"""
        result = tts_service.synthesize(
            text="One two three four five",
            language="a",
            return_timestamps=True
        )

        # 应该有 5 个单词
        assert len(result['timestamps']) == 5

        # 最后一个时间戳的结束时间应该接近总时长
        last_end = result['timestamps'][-1]['end']
        assert abs(last_end - result['duration']) < 0.5  # 允许 0.5 秒误差
```

### 6.3 运行测试

```bash
# 安装测试依赖
pip install pytest pytest-cov

# 运行所有测试
pytest tests/ -v

# 运行并生成覆盖率报告
pytest tests/ --cov=. --cov-report=html
```

---

## 七、部署注意事项

### 7.1 性能优化

1. **模型预加载**: 在服务启动时加载模型，避免每次请求加载
2. **批处理**: 如果有多个请求，可以批量处理
3. **GPU 加速**: 使用 CUDA provider 加速推理
4. **缓存**: 对常用短语缓存结果

### 7.2 资源需求

| 资源 | 最低配置 | 推荐配置 |
|-----|---------|---------|
| CPU | 2 核 | 4+ 核 |
| 内存 | 2 GB | 4+ GB |
| 磁盘 | 500 MB | 1 GB |
| GPU | 可选 | NVIDIA GPU (4GB+) |

### 7.3 错误处理

```python
# 常见错误及处理

class TTSError(Exception):
    """TTS 服务错误基类"""
    pass

class G2PError(TTSError):
    """G2P 转换失败"""
    pass

class InferenceError(TTSError):
    """模型推理失败"""
    pass

class UnsupportedLanguageError(TTSError):
    """不支持的语言"""
    pass

# 在服务中使用
def synthesize(self, text: str, language: str, ...):
    if language not in self.g2p.LANG_MAP:
        raise UnsupportedLanguageError(f"Language '{language}' is not supported")

    try:
        phonemes, tokens = self.g2p.g2p(text, language)
    except Exception as e:
        raise G2PError(f"G2P conversion failed: {e}")

    try:
        audio, durations = self.kokoro.inference(phonemes, voice, speed)
    except Exception as e:
        raise InferenceError(f"Model inference failed: {e}")
```

---

## 八、已知限制与未来改进

### 8.1 当前限制

1. **音素映射不完整**: 某些语言的特殊音素可能未正确映射
2. **时间戳精度**: 非英语语言的时间戳精度可能不如英语
3. **语音质量**: 非英语语言使用英语语音包，口音可能不自然
4. **分词准确性**: 某些语言（如中文、日文）的分词可能不准确

### 8.2 改进方向

1. **完善音素映射表**: 针对每种语言单独调优映射
2. **使用专用分词器**: 集成 spaCy 等 NLP 库提高分词准确性
3. **训练多语言语音包**: 使用目标语言数据微调模型
4. **添加强制对齐后处理**: 使用 Montreal Forced Aligner 提高时间戳精度

---

## 九、参考资源

### 官方文档
- [Kokoro GitHub](https://github.com/hexgrad/kokoro)
- [Kokoro Hugging Face](https://huggingface.co/hexgrad/Kokoro-82M)
- [misaki G2P](https://github.com/hexgrad/misaki)
- [espeak-ng](https://github.com/espeak-ng/espeak-ng)

### 相关实现
- [Kokoro 时间戳 Gist](https://gist.github.com/fagenorn/d4aa16704541370d9b9d5f91f1f07b34)
- [RealtimeTTS Kokoro 集成](https://github.com/KoljaB/RealtimeTTS/issues/278)
- [HeadTTS (浏览器端实现)](https://github.com/met4citizen/HeadTTS)

### 工具库
- [py-espeak-ng](https://pypi.org/project/py-espeak-ng/)
- [phonemizer](https://bootphon.github.io/phonemizer/)
- [ONNX Runtime](https://onnxruntime.ai/)

---

## 十、快速开始清单

执行此方案的步骤清单：

- [ ] 1. 安装系统依赖 (espeak-ng)
- [ ] 2. 安装 Python 依赖
- [ ] 3. 下载 Kokoro ONNX 模型
- [ ] 4. 实现 `phoneme_mapping.py` - 音素映射表
- [ ] 5. 实现 `multilingual_g2p.py` - 多语言 G2P
- [ ] 6. 实现 `timestamp_calculator.py` - 时间戳计算
- [ ] 7. 实现 `kokoro_inference.py` - 模型推理封装
- [ ] 8. 实现 `tts_service.py` - 完整服务
- [ ] 9. 编写测试用例
- [ ] 10. 集成到现有 API 服务
- [ ] 11. 测试各语言时间戳准确性
- [ ] 12. 根据测试结果调优音素映射表
