# Kokoro TTS 项目指南

## 项目概述

基于 Kokoro ONNX 模型的多语言 TTS 服务，使用 espeak-ng 实现 G2P（Grapheme-to-Phoneme）转换，支持 39 种语言和词级时间戳。

## 架构

```
用户文本 → MultilingualG2P (espeak-ng) → 音素序列 → KokoroInference (ONNX) → 音频 + 时长
                                                                              ↓
                                                      TimestampCalculator → 词级时间戳
```

## 核心模块

| 文件 | 职责 |
|------|------|
| `multilingual_g2p.py` | espeak-ng G2P 转换，支持多语言 |
| `phoneme_mapping.py` | IPA 音素到 Kokoro 词汇表映射 |
| `kokoro_inference.py` | ONNX 模型加载和推理 |
| `timestamp_calculator.py` | 从 duration 计算词级时间戳 |
| `tts_service.py` | 整合所有模块的完整 TTS 服务 |

## 关键经验教训

### 1. Voice Embedding 必须按 token 长度选择

**错误做法**：
```python
# 只取前 256 个值 - 会导致音频忽高忽低
voice_data = np.fromfile(voice_path, dtype=np.float32)[:256]
```

**正确做法**：
```python
# Voice 文件包含 (N, 256) 的 style vectors，N 通常是 510
voice_data = np.fromfile(voice_path, dtype=np.float32)
num_styles = len(voice_data) // 256
voices[voice_name] = voice_data.reshape(num_styles, 256)

# 推理时根据 token 长度选择对应的 style vector
style_idx = min(len(input_ids), len(voice_styles) - 1)
style = voice_styles[style_idx]
```

这是 Kokoro 官方的设计：每个 token 长度对应一个预计算的 style vector。

### 2. 长文本需要分句处理

Kokoro 模型有序列长度限制（约 500 tokens）。长文本需要：
1. 按句子分割（保留标点）
2. 分别合成每个句子
3. 添加句间静音（300-400ms）
4. 拼接时进行音频平滑处理

### 3. 音频质量优化三要素

```python
# 1. RMS 归一化 - 统一音量
def _normalize_audio(audio, target_rms=0.08):
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 1e-6:
        return audio * (target_rms / rms)
    return audio

# 2. 淡入淡出 - 避免能量突变
def _apply_fade(audio, fade_in_ms=15, fade_out_ms=25):
    # 在句子开头淡入，结尾淡出
    ...

# 3. 句间静音 - 自然停顿
silence = np.zeros(int(pause_sec * SAMPLE_RATE), dtype=np.float32)
```

### 4. 时间戳计算的帧率

Kokoro duration 输出的帧率是 **33.5 FPS**（不是常见的 93.75）：

```python
DEFAULT_FRAMES_PER_SECOND = 33.5  # 实测值
```

计算方式：`timestamp = frame_index / 33.5`

### 5. espeak-ng 音素映射

espeak-ng 输出的 IPA 需要映射到 Kokoro 词汇表：
- 直接映射：大部分音素一一对应
- 转换映射：如 `r` → `ɹ`（美式 R 音）
- 双元音拆分：如 `eɪ` → `e` + `ɪ`
- 跳过字符：重音标记等需要过滤

## 运行命令

```bash
# 安装依赖
brew install espeak-ng
pip install -r requirements.txt

# 下载模型
huggingface-cli download hexgrad/Kokoro-82M-v1.1-zh --local-dir ./kokoro-model

# 运行演示
python demo.py

# 单独测试
python tts_service.py
python kokoro_inference.py ./kokoro-model
```

## 模型文件结构

```
kokoro-model/
├── model.onnx          # 主模型（带 duration 输出）
├── tokenizer.json      # 音素词汇表
└── voices/             # 语音包
    ├── af_heart.bin    # (510, 256) float32
    ├── af_bella.bin
    └── ...
```

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 音频忽高忽低 | Voice embedding 使用错误 | 按 token 长度选择 style vector |
| 长文本报错 | 序列超长 | 使用 `synthesize_long()` 分句处理 |
| 时间戳不准 | 帧率设置错误 | 使用 33.5 FPS |
| 音素转换失败 | espeak-ng 未安装 | `brew install espeak-ng` |

## API 使用示例

```python
from tts_service import TTSService

tts = TTSService(model_dir='./kokoro-model')

# 单句合成
result = tts.synthesize(
    text="Hello, how are you?",
    language='a',  # American English
    voice='af_heart',
    return_timestamps=True,
    return_audio_array=True
)

# 长文本合成
result = tts.synthesize_long(
    text="Long paragraph...",
    language='a',
    voice='af_heart'
)

# 保存到文件
result = tts.synthesize_to_file(
    text="Hello",
    output_path="output.wav",
    language='a'
)
```

## 支持的语言代码（39 种）

### Kokoro 原生支持（9 种）

| 代码 | 语言 | 代码 | 语言 |
|------|------|------|------|
| a | American English | b | British English |
| e | Spanish | f | French |
| h | Hindi | i | Italian |
| j | Japanese | p | Portuguese (Brazil) |
| z | Chinese | | |

### 拉丁语系扩展（15 种）

| 代码 | 语言 | 代码 | 语言 |
|------|------|------|------|
| de | German | nl | Dutch |
| pl | Polish | cs | Czech |
| ro | Romanian | sv | Swedish |
| da | Danish | no | Norwegian |
| fi | Finnish | ca | Catalan |
| eu | Basque | gl | Galician |
| es-mx | Spanish (Mexico) | fr-ca | French (Canada) |
| pt-pt | Portuguese (Portugal) | | |

### 西里尔语系（6 种）

| 代码 | 语言 | 代码 | 语言 |
|------|------|------|------|
| ru | Russian | uk | Ukrainian |
| be | Belarusian | bg | Bulgarian |
| sr | Serbian | mk | Macedonian |

### 其他语言（9 种）

| 代码 | 语言 | 代码 | 语言 |
|------|------|------|------|
| el | Greek | tr | Turkish |
| ar | Arabic | he | Hebrew |
| ko | Korean | vi | Vietnamese |
| th | Thai | id | Indonesian |
| ms | Malay | | |

### 扩展语言支持

由于使用 espeak-ng 作为 G2P 后端，可轻松扩展到 100+ 种语言。只需在 `multilingual_g2p.py` 的 `LANG_MAP` 中添加映射：

```python
LANG_MAP = {
    'new_lang': 'espeak-lang-code',  # 添加新语言
    ...
}
```
