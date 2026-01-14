# chinese_g2p.py
"""
中文 G2P 转换器

使用 pypinyin 将中文转换为拼音，再映射到 Kokoro 兼容的 IPA 音素
"""

import re
from typing import List, Dict, Tuple

try:
    from pypinyin import pinyin, Style
    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False
    print("Warning: pypinyin 未安装，中文 G2P 不可用")


# 声母映射：拼音声母 → IPA
INITIAL_MAP = {
    'b': 'p',
    'p': 'pʰ',
    'm': 'm',
    'f': 'f',
    'd': 't',
    't': 'tʰ',
    'n': 'n',
    'l': 'l',
    'g': 'k',
    'k': 'kʰ',
    'h': 'x',
    'j': 'tɕ',
    'q': 'tɕʰ',
    'x': 'ɕ',
    'zh': 'tʂ',
    'ch': 'tʂʰ',
    'sh': 'ʂ',
    'r': 'ɻ',
    'z': 'ts',
    'c': 'tsʰ',
    's': 's',
    'y': 'j',
    'w': 'w',
}

# 韵母映射：拼音韵母 → IPA
FINAL_MAP = {
    # 单韵母
    'a': 'ɑ',
    'o': 'o',
    'e': 'ɤ',
    'i': 'i',
    'u': 'u',
    'v': 'y',  # ü
    'ü': 'y',

    # 复韵母
    'ai': 'aɪ',
    'ei': 'eɪ',
    'ao': 'aʊ',
    'ou': 'oʊ',

    # 鼻韵母
    'an': 'an',
    'en': 'ən',
    'ang': 'ɑŋ',
    'eng': 'əŋ',
    'ong': 'ʊŋ',

    # i 开头的韵母
    'ia': 'jɑ',
    'ie': 'je',
    'iao': 'jaʊ',
    'iu': 'joʊ',
    'ian': 'jɛn',
    'in': 'in',
    'iang': 'jɑŋ',
    'ing': 'iŋ',
    'iong': 'jʊŋ',

    # u 开头的韵母
    'ua': 'wɑ',
    'uo': 'wo',
    'uai': 'waɪ',
    'ui': 'weɪ',
    'uan': 'wan',
    'un': 'wən',
    'uang': 'wɑŋ',
    'ueng': 'wəŋ',

    # ü 开头的韵母
    've': 'ɥe',
    'ue': 'ɥe',
    'van': 'ɥɛn',
    'uan_j': 'ɥɛn',  # j/q/x 后的 uan
    'vn': 'yn',
    'un_j': 'yn',    # j/q/x 后的 un

    # 特殊韵母
    'er': 'əɻ',
    'ir': 'ɻ̩',  # zhi, chi, shi, ri 的韵母
    'iii': 'ɿ',  # zi, ci, si 的韵母
}

# 整体认读音节
WHOLE_SYLLABLES = {
    'zhi': 'tʂɻ̩',
    'chi': 'tʂʰɻ̩',
    'shi': 'ʂɻ̩',
    'ri': 'ɻɻ̩',
    'zi': 'tsɿ',
    'ci': 'tsʰɿ',
    'si': 'sɿ',
    'yi': 'i',
    'wu': 'u',
    'yu': 'y',
    'ye': 'je',
    'yue': 'ɥe',
    'yuan': 'ɥɛn',
    'yin': 'in',
    'yun': 'yn',
    'ying': 'iŋ',
}

# 声调标记（可选，Kokoro 可能不需要）
TONE_MARKS = {
    1: '˥',   # 阴平 55
    2: '˧˥',  # 阳平 35
    3: '˨˩˦', # 上声 214
    4: '˥˩',  # 去声 51
    5: '',    # 轻声
}


class ChineseG2P:
    """中文 G2P 转换器"""

    def __init__(self, include_tone: bool = False):
        """
        Args:
            include_tone: 是否包含声调标记
        """
        if not HAS_PYPINYIN:
            raise ImportError("请安装 pypinyin: pip install pypinyin")

        self.include_tone = include_tone

    def _parse_pinyin(self, py: str) -> Tuple[str, str, int]:
        """
        解析拼音为声母、韵母、声调

        Args:
            py: 带声调数字的拼音，如 'zhong1'

        Returns:
            (initial, final, tone)
        """
        # 提取声调数字
        tone = 5  # 默认轻声
        if py and py[-1].isdigit():
            tone = int(py[-1])
            py = py[:-1]

        if not py:
            return '', '', tone

        # 检查整体认读音节
        if py in WHOLE_SYLLABLES:
            return '', py, tone

        # 提取声母（双字母声母优先）
        initial = ''
        for init in ['zh', 'ch', 'sh']:
            if py.startswith(init):
                initial = init
                py = py[len(init):]
                break
        else:
            if py[0] in INITIAL_MAP and py[0] not in 'aoe':
                initial = py[0]
                py = py[1:]

        final = py
        return initial, final, tone

    def _pinyin_to_ipa(self, py: str) -> str:
        """
        将单个拼音转换为 IPA

        Args:
            py: 带声调的拼音

        Returns:
            IPA 字符串
        """
        initial, final, tone = self._parse_pinyin(py.lower())

        # 整体认读音节
        if not initial and final in WHOLE_SYLLABLES:
            ipa = WHOLE_SYLLABLES[final]
        else:
            # 声母
            ipa_initial = INITIAL_MAP.get(initial, '')

            # 处理 j/q/x 后的特殊韵母
            if initial in ['j', 'q', 'x']:
                if final == 'u':
                    final = 'v'
                elif final == 'uan':
                    final = 'van'
                elif final == 'un':
                    final = 'vn'

            # 韵母
            ipa_final = FINAL_MAP.get(final, final)

            ipa = ipa_initial + ipa_final

        # 添加声调
        if self.include_tone and tone in TONE_MARKS:
            ipa += TONE_MARKS[tone]

        return ipa

    def convert(self, text: str) -> Tuple[str, List[Dict]]:
        """
        将中文文本转换为 IPA 音素

        Args:
            text: 中文文本

        Returns:
            (phonemes, tokens) - 音素字符串和分词列表
        """
        tokens = []
        all_phonemes = []

        # 使用正则分离中文字符和标点
        pattern = r'([\u4e00-\u9fff]+|[^\u4e00-\u9fff]+)'
        segments = re.findall(pattern, text)

        for segment in segments:
            if re.match(r'[\u4e00-\u9fff]', segment):
                # 中文字符
                py_list = pinyin(segment, style=Style.TONE3, neutral_tone_with_five=True)

                for i, (char, py) in enumerate(zip(segment, py_list)):
                    py_str = py[0] if py else ''
                    ipa = self._pinyin_to_ipa(py_str)

                    tokens.append({
                        'text': char,
                        'phonemes': ipa,
                        'phoneme_count': len(ipa),
                        'is_word': True
                    })

                    if ipa:
                        all_phonemes.append(ipa)
            else:
                # 非中文字符（标点、空格等）
                for char in segment:
                    if char in '，。！？、；：""''':
                        # 中文标点映射
                        punct_map = {
                            '，': ',', '。': '.', '！': '!', '？': '?',
                            '、': ',', '；': ';', '：': ':',
                            '"': '"', '"': '"', ''': "'", ''': "'"
                        }
                        mapped = punct_map.get(char, '')
                        if mapped:
                            tokens.append({
                                'text': char,
                                'phonemes': mapped,
                                'phoneme_count': 1,
                                'is_word': False
                            })
                            all_phonemes.append(mapped)
                    elif char.strip():
                        # 其他可见字符
                        tokens.append({
                            'text': char,
                            'phonemes': char,
                            'phoneme_count': 1,
                            'is_word': False
                        })

        return ' '.join(all_phonemes), tokens


def test_chinese_g2p():
    """测试中文 G2P"""
    g2p = ChineseG2P(include_tone=False)

    test_texts = [
        "你好",
        "今天天气真不错",
        "我爱中国",
        "春眠不觉晓，处处闻啼鸟。",
    ]

    print("中文 G2P 测试")
    print("=" * 50)

    for text in test_texts:
        phonemes, tokens = g2p.convert(text)
        print(f"\n文本: {text}")
        print(f"音素: {phonemes}")
        print("分词:")
        for t in tokens:
            if t['is_word']:
                print(f"  '{t['text']}' → {t['phonemes']}")


if __name__ == '__main__':
    test_chinese_g2p()
