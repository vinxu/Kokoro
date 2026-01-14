# multilingual_g2p.py
"""
多语言 G2P (Grapheme-to-Phoneme) 转换器

使用 espeak-ng 作为后端，支持 100+ 种语言的文本到音素转换
"""

import re
import subprocess
from typing import Tuple, List, Dict, Optional
from phoneme_mapping import convert_phonemes, count_phonemes


class MultilingualG2P:
    """
    多语言 G2P 转换器

    使用 espeak-ng 命令行工具进行音素转换，
    支持分词并返回每个单词的音素信息
    """

    # Kokoro 语言代码 → espeak-ng 语言代码
    LANG_MAP = {
        # Kokoro 原生支持的语言
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
        'ca': 'ca',        # Catalan
        'eu': 'eu',        # Basque
        'gl': 'gl',        # Galician

        # 扩展支持 - 西里尔语系
        'ru': 'ru',        # Russian
        'uk': 'uk',        # Ukrainian
        'be': 'be',        # Belarusian
        'bg': 'bg',        # Bulgarian
        'sr': 'sr',        # Serbian
        'mk': 'mk',        # Macedonian

        # 其他语言
        'el': 'el',        # Greek
        'tr': 'tr',        # Turkish
        'ar': 'ar',        # Arabic
        'he': 'he',        # Hebrew
        'ko': 'ko',        # Korean
        'vi': 'vi',        # Vietnamese
        'th': 'th',        # Thai
        'id': 'id',        # Indonesian
        'ms': 'ms',        # Malay
    }

    # 语言的分词模式
    # CJK 语言需要按字符分词，其他按空格/标点
    CJK_LANGS = {'j', 'z', 'ko'}

    def __init__(self, espeak_path: str = 'espeak-ng'):
        """
        Args:
            espeak_path: espeak-ng 可执行文件路径
        """
        self.espeak_path = espeak_path
        self._verify_espeak()

    def _verify_espeak(self):
        """验证 espeak-ng 是否可用"""
        try:
            result = subprocess.run(
                [self.espeak_path, '--version'],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"espeak-ng 返回错误: {result.stderr}")
        except FileNotFoundError:
            raise RuntimeError(
                f"未找到 espeak-ng，请先安装:\n"
                f"  macOS: brew install espeak-ng\n"
                f"  Ubuntu: sudo apt-get install espeak-ng\n"
                f"  Windows: 从 GitHub releases 下载"
            )

    def _espeak_g2p(self, text: str, lang: str) -> str:
        """
        调用 espeak-ng 获取 IPA 音素

        Args:
            text: 输入文本
            lang: espeak-ng 语言代码

        Returns:
            IPA 音素字符串
        """
        try:
            result = subprocess.run(
                [
                    self.espeak_path,
                    '-v', lang,
                    '-q',           # 安静模式，不播放声音
                    '--ipa=2',      # 输出 IPA (格式 2: Unicode)
                    text
                ],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                print(f"Warning: espeak-ng error for '{text}': {result.stderr}")
                return ''
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            print(f"Warning: espeak-ng timeout for '{text}'")
            return ''
        except Exception as e:
            print(f"Warning: espeak-ng failed for '{text}': {e}")
            return ''

    def _tokenize(self, text: str, lang: str) -> List[Dict]:
        """
        分词，保留单词和标点

        Args:
            text: 输入文本
            lang: Kokoro 语言代码

        Returns:
            分词列表，每个元素包含 text 和 is_word
        """
        # CJK 语言按字符分词
        if lang in self.CJK_LANGS:
            return self._tokenize_cjk(text)

        # 其他语言按空格和标点分词
        # 匹配：单词（含连字符）或单个标点
        # \u0400-\u04FF: 西里尔字母
        # \u0370-\u03FF: 希腊字母
        # \u0590-\u05FF: 希伯来字母
        # \u0600-\u06FF: 阿拉伯字母
        pattern = r"[\w\u0400-\u04FF\u0370-\u03FF\u0590-\u05FF\u0600-\u06FF]+(?:[-'][\w\u0400-\u04FF]+)*|[^\w\s]"

        matches = re.findall(pattern, text, re.UNICODE)

        tokens = []
        for match in matches:
            is_word = bool(re.match(
                r"[\w\u0400-\u04FF\u0370-\u03FF\u0590-\u05FF\u0600-\u06FF]",
                match,
                re.UNICODE
            ))
            tokens.append({
                'text': match,
                'is_word': is_word
            })

        return tokens

    def _tokenize_cjk(self, text: str) -> List[Dict]:
        """
        CJK 语言分词（按字符）

        Args:
            text: 输入文本

        Returns:
            分词列表
        """
        tokens = []

        # CJK 字符范围
        cjk_pattern = r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]'

        i = 0
        while i < len(text):
            char = text[i]

            if re.match(cjk_pattern, char):
                # CJK 字符，单独作为一个 token
                tokens.append({'text': char, 'is_word': True})
                i += 1
            elif char.isspace():
                # 跳过空格
                i += 1
            elif re.match(r'[^\w\s]', char):
                # 标点符号
                tokens.append({'text': char, 'is_word': False})
                i += 1
            else:
                # 其他字符（如英文），收集连续的
                j = i
                while j < len(text) and re.match(r'\w', text[j]):
                    j += 1
                if j > i:
                    tokens.append({'text': text[i:j], 'is_word': True})
                    i = j
                else:
                    i += 1

        return tokens

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
        # 获取 espeak 语言代码
        espeak_lang = self.LANG_MAP.get(lang, 'en-us')

        # 分词
        words = self._tokenize(text, lang)

        tokens = []
        all_phonemes = []

        for word_info in words:
            word = word_info['text']

            if word_info['is_word']:
                # 获取单词的音素
                try:
                    word_ipa = self._espeak_g2p(word, espeak_lang)
                    # 转换为 Kokoro 格式
                    converted = convert_phonemes(word_ipa)
                    phoneme_count = count_phonemes(converted)
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

                if converted:
                    all_phonemes.append(converted)
            else:
                # 标点符号
                punct_phoneme = word if word in '.!?,' else ''
                tokens.append({
                    'text': word,
                    'phonemes': punct_phoneme,
                    'phoneme_count': 1 if punct_phoneme else 0,
                    'is_word': False
                })
                if punct_phoneme:
                    all_phonemes.append(punct_phoneme)

        # 合并所有音素（用空格分隔单词）
        full_phonemes = ' '.join(all_phonemes)

        return full_phonemes, tokens

    def g2p_batch(self, texts: List[str], lang: str = 'a') -> List[Tuple[str, List[Dict]]]:
        """
        批量 G2P 转换

        Args:
            texts: 文本列表
            lang: 语言代码

        Returns:
            结果列表，每个元素为 (phonemes, tokens)
        """
        return [self.g2p(text, lang) for text in texts]

    def get_supported_languages(self) -> List[str]:
        """返回支持的语言代码列表"""
        return list(self.LANG_MAP.keys())

    def get_language_name(self, lang: str) -> str:
        """获取语言名称"""
        names = {
            'a': 'American English',
            'b': 'British English',
            'e': 'Spanish',
            'f': 'French',
            'h': 'Hindi',
            'i': 'Italian',
            'j': 'Japanese',
            'p': 'Portuguese (Brazil)',
            'z': 'Chinese',
            'de': 'German',
            'ru': 'Russian',
            'pl': 'Polish',
            'nl': 'Dutch',
            'ko': 'Korean',
            'tr': 'Turkish',
            'ar': 'Arabic',
        }
        return names.get(lang, lang)


if __name__ == '__main__':
    # 测试
    g2p = MultilingualG2P()

    test_cases = [
        ("Hello, world!", "a"),
        ("Bonjour le monde!", "f"),
        ("Hola mundo", "e"),
        ("Guten Tag", "de"),
        ("Привет мир", "ru"),
        ("Ciao mondo", "i"),
    ]

    print("Multilingual G2P Test\n" + "=" * 50)

    for text, lang in test_cases:
        phonemes, tokens = g2p.g2p(text, lang)
        lang_name = g2p.get_language_name(lang)

        print(f"\n{lang_name}: {text}")
        print(f"  Phonemes: {phonemes}")
        print(f"  Tokens:")
        for t in tokens:
            if t['is_word']:
                print(f"    '{t['text']}' -> '{t['phonemes']}' ({t['phoneme_count']} phonemes)")
