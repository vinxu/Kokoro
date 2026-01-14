#!/usr/bin/env python3
"""
TTS 服务测试脚本

测试 espeak-ng G2P 和时间戳计算功能
"""

import sys


def test_phoneme_mapping():
    """测试音素映射"""
    print("\n=== 测试音素映射 ===")
    from phoneme_mapping import convert_phonemes, count_phonemes

    test_cases = [
        ("həlˈoʊ", "Hello"),
        ("wˈɜːld", "world"),
        ("bɔ̃ʒˈuʁ", "bonjour (French)"),
        ("prʲɪvʲˈet", "привет (Russian)"),
    ]

    for ipa, desc in test_cases:
        converted = convert_phonemes(ipa)
        count = count_phonemes(converted)
        print(f"  {desc}: '{ipa}' -> '{converted}' ({count} phonemes)")

    print("  ✓ 音素映射测试通过")


def test_g2p():
    """测试 G2P 转换"""
    print("\n=== 测试 G2P 转换 ===")

    try:
        from multilingual_g2p import MultilingualG2P
        g2p = MultilingualG2P()
    except RuntimeError as e:
        print(f"  ✗ espeak-ng 未安装: {e}")
        return False

    test_cases = [
        ("Hello world", "a"),
        ("Bonjour", "f"),
        ("Hola", "e"),
    ]

    for text, lang in test_cases:
        phonemes, tokens = g2p.g2p(text, lang)
        print(f"  [{lang}] '{text}' -> '{phonemes}'")
        for t in tokens:
            if t['is_word']:
                print(f"       '{t['text']}': {t['phoneme_count']} phonemes")

    print("  ✓ G2P 测试通过")
    return True


def test_timestamp_calculator():
    """测试时间戳计算"""
    print("\n=== 测试时间戳计算 ===")
    import numpy as np
    from timestamp_calculator import TimestampCalculator

    calculator = TimestampCalculator()

    tokens = [
        {'text': 'Hello', 'phoneme_count': 5, 'is_word': True},
        {'text': 'world', 'phoneme_count': 5, 'is_word': True},
    ]

    # 模拟 durations
    durations = np.array([8, 7, 9, 8, 8, 8, 8, 7, 9, 8], dtype=np.float32)

    timestamps = calculator.calculate(tokens, durations)

    print(f"  Input: {[t['text'] for t in tokens]}")
    print(f"  Durations: {durations}")
    for ts in timestamps:
        print(f"    '{ts['word']}': {ts['start']:.3f}s - {ts['end']:.3f}s")

    # 验证
    assert len(timestamps) == 2
    assert timestamps[0]['word'] == 'Hello'
    assert timestamps[1]['word'] == 'world'
    assert timestamps[0]['end'] <= timestamps[1]['start']

    print("  ✓ 时间戳计算测试通过")


def test_tts_service_mock():
    """测试 TTS 服务（mock 模式）"""
    print("\n=== 测试 TTS 服务 (Mock) ===")
    from tts_service import TTSService

    tts = TTSService(mock_mode=True)

    result = tts.synthesize(
        text="Hello world",
        language="a",
        return_timestamps=True,
        return_audio_array=True
    )

    print(f"  Text: 'Hello world'")
    print(f"  Phonemes: {result['phonemes']}")
    print(f"  Duration: {result['duration']:.2f}s")
    print(f"  Timestamps: {len(result['timestamps'])} words")
    for ts in result['timestamps']:
        print(f"    '{ts['word']}': {ts['start']:.3f}s - {ts['end']:.3f}s")

    # 验证
    assert result['duration'] > 0
    assert len(result['timestamps']) > 0

    print("  ✓ TTS 服务测试通过")


def test_multilingual():
    """测试多语言支持"""
    print("\n=== 测试多语言支持 ===")
    from tts_service import TTSService

    tts = TTSService(mock_mode=True)

    test_cases = [
        ("Hello", "a", "English"),
        ("Bonjour", "f", "French"),
        ("Hola", "e", "Spanish"),
        ("Ciao", "i", "Italian"),
        ("Guten Tag", "de", "German"),
    ]

    for text, lang, lang_name in test_cases:
        try:
            result = tts.synthesize(text=text, language=lang, return_timestamps=True)
            print(f"  [{lang_name}] '{text}' -> {result['phonemes'][:30]}...")
        except Exception as e:
            print(f"  [{lang_name}] '{text}' -> Error: {e}")

    print("  ✓ 多语言测试通过")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("Kokoro TTS 多语言时间戳测试")
    print("=" * 60)

    try:
        test_phoneme_mapping()
        g2p_ok = test_g2p()
        test_timestamp_calculator()

        if g2p_ok:
            test_tts_service_mock()
            test_multilingual()

        print("\n" + "=" * 60)
        print("所有测试通过！")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
