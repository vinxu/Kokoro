# phoneme_mapping.py
"""
espeak-ng 音素 -> Kokoro 音素映射表

将 espeak-ng 输出的 IPA 音素转换为 Kokoro 模型支持的格式
"""

from typing import Set

# 通用 IPA 音素（直接映射）
DIRECT_MAP = {
    # 元音
    'ə': 'ə', 'ɑ': 'ɑ', 'æ': 'æ', 'ɛ': 'ɛ', 'ɪ': 'ɪ',
    'ɔ': 'ɔ', 'ʊ': 'ʊ', 'ʌ': 'ʌ', 'i': 'i', 'u': 'u',
    'e': 'e', 'o': 'o', 'a': 'a', 'ɜ': 'ɜ',
    'y': 'y', 'ø': 'ø', 'œ': 'œ',

    # 辅音
    'p': 'p', 'b': 'b', 't': 't', 'd': 'd', 'k': 'k', 'g': 'g',
    'm': 'm', 'n': 'n', 'ŋ': 'ŋ', 'ɲ': 'ɲ',
    'f': 'f', 'v': 'v', 's': 's', 'z': 'z',
    'ʃ': 'ʃ', 'ʒ': 'ʒ', 'θ': 'θ', 'ð': 'ð',
    'h': 'h', 'w': 'w', 'j': 'j', 'l': 'l', 'r': 'r',
    'ɫ': 'l',  # dark l

    # 重音和韵律标记
    'ˈ': 'ˈ', 'ˌ': 'ˌ', 'ː': 'ː',
    '.': '.', ',': ',', '?': '?', '!': '!',
    ' ': ' ',
}

# 需要转换的音素 (espeak 特有 -> Kokoro 等价)
CONVERT_MAP = {
    # 英语 r 音变体
    'ɹ': 'r',      # 英语 r
    'ɾ': 'r',      # 弹舌 r (西班牙语等)
    'ʁ': 'r',      # 法语小舌 r
    'ʀ': 'r',      # 颤音 R

    # 元音变体
    'ɐ': 'ə',      # 近央元音
    'ɝ': 'ɜr',     # 美式卷舌元音
    'ɚ': 'ər',     # 弱化卷舌元音
    'ɵ': 'ə',      # 中央圆唇元音

    # 鼻化元音（法语等）- 分解为元音+n
    'ɑ̃': 'ɑn',    # on (法语 "bon")
    'ɛ̃': 'ɛn',    # in (法语 "vin")
    'ɔ̃': 'ɔn',    # on (法语 "mon")
    'œ̃': 'œn',    # un (法语 "un")
    'ã': 'an',     # 鼻化 a

    # 俄语特有
    'ʲ': 'j',      # 软化标记 -> j
    'ɕ': 'ʃ',      # 软化 ш (щ)
    'ʑ': 'ʒ',      # 软化 ж
    'ɨ': 'ɪ',      # ы 音
    'ʂ': 'ʃ',      # 硬 ш
    'ʐ': 'ʒ',      # 硬 ж
    'ɫ': 'l',      # 硬 л

    # 德语特有
    'ç': 'h',      # ich-laut (ich, nicht)
    'x': 'h',      # ach-laut (ach, Bach)
    'ʏ': 'y',      # ü 短音
    'yː': 'y',     # ü 长音
    'øː': 'ø',     # ö 长音

    # 波兰语/斯拉夫语
    'ʈ': 't',      # 卷舌 t
    'ɖ': 'd',      # 卷舌 d
    'ɳ': 'n',      # 卷舌 n
    'ɭ': 'l',      # 卷舌 l

    # 其他常见变体
    'ɤ': 'ə',      # 后不圆唇中元音
    'ɯ': 'u',      # 后不圆唇高元音
    'ʔ': '',       # 声门塞音 (通常可忽略)

    # IPA 字符 → ASCII 等价（espeak-ng 使用 IPA 字符）
    'ɡ': 'g',      # IPA g (U+0261) → ASCII g (U+0067)
    'ɑː': 'ɑ',     # 长元音简化
    'iː': 'i',
    'uː': 'u',
    'eː': 'e',
    'oː': 'o',
    'aː': 'a',
}

# 组合音素（双元音、塞擦音等）
DIPHTHONG_MAP = {
    # 英语双元音
    'aɪ': 'aɪ', 'aʊ': 'aʊ', 'eɪ': 'eɪ',
    'oʊ': 'oʊ', 'ɔɪ': 'ɔɪ',
    # 英式英语双元音
    'ɪə': 'ɪə', 'eə': 'eə', 'ʊə': 'ʊə',
    # 塞擦音
    'tʃ': 'tʃ', 'dʒ': 'dʒ',
    'ts': 'ts', 'dz': 'dz',
    # 其他组合
    'əʊ': 'oʊ',  # 英式英语 "go"
}

# 需要跳过的字符（如组合符号等）
SKIP_CHARS: Set[str] = {
    '\u0303',  # 组合波浪号 (鼻化标记)
    '\u0324',  # 组合下双点
    '\u0325',  # 组合下圈 (清音化)
    '\u032a',  # 组合下桥
    '\u0329',  # 组合下竖线 (成音节)
    '\u0361',  # 组合双弧 (用于连接音素)
    '\u02d0',  # 长音符 (ː 的变体)
    '̃',       # 组合波浪号
}


def convert_phonemes(espeak_output: str, verbose: bool = False) -> str:
    """
    将 espeak-ng 的 IPA 输出转换为 Kokoro 格式

    Args:
        espeak_output: espeak-ng g2p 输出的 IPA 字符串
        verbose: 是否打印未知字符警告

    Returns:
        Kokoro 兼容的音素字符串
    """
    if not espeak_output:
        return ''

    result = espeak_output

    # 1. 先处理组合音素（双元音、塞擦音等）
    # 按长度降序排列，避免短序列先匹配
    sorted_diphthongs = sorted(DIPHTHONG_MAP.keys(), key=len, reverse=True)
    for src in sorted_diphthongs:
        dst = DIPHTHONG_MAP[src]
        result = result.replace(src, dst)

    # 2. 处理需要转换的音素
    sorted_converts = sorted(CONVERT_MAP.keys(), key=len, reverse=True)
    for src in sorted_converts:
        dst = CONVERT_MAP[src]
        result = result.replace(src, dst)

    # 3. 逐字符过滤，只保留支持的音素
    cleaned = []
    i = 0
    while i < len(result):
        char = result[i]

        # 跳过组合字符
        if char in SKIP_CHARS:
            i += 1
            continue

        # 直接映射的字符
        if char in DIRECT_MAP:
            cleaned.append(DIRECT_MAP[char])
            i += 1
            continue

        # 检查两字符组合（双元音等）
        if i + 1 < len(result):
            two_char = result[i:i+2]
            if two_char in DIPHTHONG_MAP:
                cleaned.append(DIPHTHONG_MAP[two_char])
                i += 2
                continue

        # 未知字符
        if verbose:
            print(f"Warning: Unknown phoneme '{char}' (U+{ord(char):04X})")
        i += 1

    return ''.join(cleaned)


def count_phonemes(phoneme_str: str) -> int:
    """
    统计音素数量

    排除重音标记、空格等非音素字符
    双元音和塞擦音算作 1 个音素

    Args:
        phoneme_str: 音素字符串

    Returns:
        音素数量
    """
    if not phoneme_str:
        return 0

    # 非音素字符（韵律标记、空格等）
    non_phonemes = set('ˈˌː .,?!-')

    # 双字符音素
    two_char_phonemes = {
        'aɪ', 'aʊ', 'eɪ', 'oʊ', 'ɔɪ',  # 双元音
        'ɪə', 'eə', 'ʊə',               # 英式双元音
        'tʃ', 'dʒ', 'ts', 'dz',         # 塞擦音
        'ɜr', 'ər',                      # r 化元音
    }

    count = 0
    i = 0
    while i < len(phoneme_str):
        char = phoneme_str[i]

        # 跳过非音素字符
        if char in non_phonemes:
            i += 1
            continue

        # 检查双字符音素
        if i + 1 < len(phoneme_str):
            two_char = phoneme_str[i:i+2]
            if two_char in two_char_phonemes:
                count += 1
                i += 2
                continue

        count += 1
        i += 1

    return max(count, 1)  # 至少返回 1


if __name__ == '__main__':
    # 测试
    test_cases = [
        ("həlˈoʊ wˈɜːld", "English: Hello world"),
        ("bɔ̃ʒˈuʁ lə mˈɔ̃d", "French: Bonjour le monde"),
        ("prʲɪvʲˈet mʲˈir", "Russian: Привет мир"),
        ("ˈɡuːtən tˈaːk", "German: Guten Tag"),
        ("ˈola mˈundo", "Spanish: Hola mundo"),
    ]

    for ipa, desc in test_cases:
        converted = convert_phonemes(ipa, verbose=True)
        count = count_phonemes(converted)
        print(f"{desc}")
        print(f"  Input:  {ipa}")
        print(f"  Output: {converted}")
        print(f"  Count:  {count}")
        print()
