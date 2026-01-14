#!/usr/bin/env python3
"""
Kokoro TTS 多语言时间戳演示

为每种语言生成音频和时间戳
"""

import os
import json
import numpy as np
from scipy.io import wavfile
from tts_service import TTSService

# 多语言演示文本（每段约30秒）
DEMO_TEXTS = {
    'a': {
        'name': 'English (American)',
        'text': '''The sun was setting over the mountains, painting the sky in shades of orange and pink.
Sarah walked along the quiet path, listening to the birds singing their evening songs.
She thought about her day at work, the meetings, the coffee breaks, and the conversations with colleagues.
Life in the city was busy, but these moments of peace made everything worthwhile.
Tomorrow would bring new challenges and opportunities, but for now, she simply enjoyed the beauty of nature around her.''',
    },
    'b': {
        'name': 'English (British)',
        'text': '''The old bookshop stood at the corner of the street, its windows filled with treasures from centuries past.
Inside, dust particles danced in the afternoon sunlight, creating a magical atmosphere.
The owner, an elderly gentleman with silver hair, carefully arranged the leather-bound volumes on the shelves.
Each book had its own story, not just within its pages, but in how it came to rest in this little shop.
Visitors often spent hours browsing, discovering forgotten authors and rediscovering classics they had loved in their youth.''',
    },
    'f': {
        'name': 'French',
        'text': '''Paris s'éveillait doucement sous les premiers rayons du soleil. Les cafés ouvraient leurs portes,
diffusant l'arôme du café fraîchement préparé dans les rues pavées. Marie marchait le long de la Seine,
admirant les reflets dorés sur l'eau. Elle pensait à son nouveau projet artistique, une série de peintures
inspirées par la vie quotidienne de la ville. Les passants pressés, les amoureux sur les ponts,
les musiciens dans le métro, tout cela formait une symphonie visuelle qu'elle voulait capturer sur sa toile.''',
    },
    'e': {
        'name': 'Spanish',
        'text': '''El mercado de la ciudad estaba lleno de colores y sonidos. Los vendedores gritaban sus ofertas
mientras los clientes examinaban las frutas frescas y las verduras del campo. Carmen caminaba entre los puestos,
buscando los ingredientes perfectos para la cena familiar del domingo. Su abuela le había enseñado las recetas
tradicionales, transmitidas de generación en generación. Compró tomates rojos y brillantes, aceitunas negras,
y un poco de jamón serrano. Esta noche, toda la familia se reuniría alrededor de la mesa para compartir historias y risas.''',
    },
    'de': {
        'name': 'German',
        'text': '''Der Schwarzwald erstreckte sich majestätisch vor ihren Augen. Die hohen Tannen ragten in den blauen Himmel,
und ein kleiner Bach plätscherte fröhlich über die Steine. Familie Müller machte ihren traditionellen Sonntagsausflug.
Die Kinder rannten voraus, sammelten bunte Blätter und beobachteten die Eichhörnchen. Vater Hans erzählte Geschichten
aus seiner Kindheit, als er selbst diese Wälder erkundet hatte. Mutter Anna hatte einen Picknickkorb mit selbstgebackenem
Kuchen und heißem Tee vorbereitet. Es waren diese einfachen Momente, die das Leben so kostbar machten.''',
    },
    'i': {
        'name': 'Italian',
        'text': '''La piazza del paese si animava ogni sera al tramonto. I vecchi sedevano sulle panchine,
giocando a carte e discutendo di politica. I bambini correvano tra le fontane, le loro risate echeggiavano
tra i palazzi antichi. Marco aprì le porte del suo ristorante, preparandosi per un'altra serata di lavoro.
Il profumo della pasta fresca e del sugo di pomodoro si diffondeva nell'aria. I turisti cominciavano ad arrivare,
attratti dalla promessa di autentica cucina italiana. Per Marco, ogni piatto era un'opera d'arte,
una celebrazione delle tradizioni della sua terra.''',
    },
    'ru': {
        'name': 'Russian',
        'text': '''Москва просыпалась под звуки утреннего города. Метро наполнялось людьми, спешащими на работу.
Анна стояла у окна своей квартиры, наблюдая за пробуждением улиц внизу. Она думала о предстоящем дне,
о встречах с друзьями и о новой книге, которую начала читать вчера вечером. Зимнее солнце робко
пробивалось сквозь облака, освещая купола древних церквей. Город хранил столько историй,
столько воспоминаний о прошлом и надежд на будущее. Каждый день здесь был новым приключением.''',
    },
    'pl': {
        'name': 'Polish',
        'text': '''Kraków budził się do życia w ciepły letni poranek. Rynek Główny powoli zapełniał się turystami
i mieszkańcami miasta. Zapach świeżych obwarzanków unosił się w powietrzu, mieszając się z aromatem kawy
z pobliskich kawiarni. Piotr siedział na ławce, obserwując gołębie i słuchając hejnału z wieży Mariackiej.
Studiował historię sztuki i każdy zakątek tego miasta był dla niego żywą lekcją przeszłości.
Gotyckie kościoły, renesansowe kamienice, barokowe pałace - wszystko to tworzyło niepowtarzalną atmosferę miejsca,
które kochał całym sercem.''',
    },
    'nl': {
        'name': 'Dutch',
        'text': '''De windmolens draaiden langzaam in de ochtendwind. De tulpenvelden strekten zich uit tot aan de horizon,
een zee van rode, gele en paarse bloemen. Jan fietste langs de grachten naar zijn werk in het centrum van Amsterdam.
Hij passeerde de smalle huizen met hun karakteristieke gevels, de kleine bruggen over het water,
en de gezellige cafés die net hun deuren openden. Het voorjaar was zijn favoriete seizoen, wanneer de stad
tot leven kwam na de lange winter. Overal zag je mensen op terrassen zitten, genietend van de eerste warme zonnestralen.''',
    },
    'tr': {
        'name': 'Turkish',
        'text': '''İstanbul'un tarihi sokaklarında yürümek, zamanda yolculuk yapmak gibiydi. Ayasofya'nın kubbeleri
gökyüzüne yükselirken, Sultanahmet Meydanı turistlerle dolup taşıyordu. Ahmet, babasının küçük halı dükkanında
çalışıyordu, tıpkı dedesi ve onun dedesi gibi. Her halının bir hikayesi vardı, Anadolu'nun dağlarından gelen
yün ipliklerle dokunmuş. Müşteriler gelip giderken, o çay ikram ediyor ve geleneksel desenlerin anlamlarını
anlatıyordu. Akşam ezanı okunduğunda, şehir bir an için duraksıyor, sonra yeniden hareketleniyordu.''',
    },
}


def run_demo(output_dir: str = './demo_output', voice: str = 'af_heart'):
    """
    运行多语言演示

    Args:
        output_dir: 输出目录
        voice: 使用的语音
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 初始化 TTS 服务
    print("=" * 60)
    print("Kokoro TTS 多语言时间戳演示")
    print("=" * 60)
    print()

    tts = TTSService(model_dir='./kokoro-model')

    results = []

    for lang_code, info in DEMO_TEXTS.items():
        lang_name = info['name']
        text = info['text']

        print(f"\n{'─' * 60}")
        print(f"[{lang_name}] ({lang_code})")
        print(f"Text: {text}")
        print()

        try:
            # 合成语音（使用长文本方法，自动分句）
            result = tts.synthesize_long(
                text=text,
                language=lang_code,
                voice=voice,
                return_timestamps=True,
                return_audio_array=True
            )

            # 打印结果
            print(f"Phonemes: {result['phonemes']}")
            print(f"Duration: {result['duration']:.2f}s")
            print(f"\nTimestamps:")
            for ts in result['timestamps']:
                duration = ts['end'] - ts['start']
                print(f"  {ts['start']:5.2f}s - {ts['end']:5.2f}s ({duration:.2f}s): {ts['word']}")

            # 保存音频
            audio_file = os.path.join(output_dir, f"{lang_code}_{lang_name.split()[0].lower()}.wav")
            audio = result['audio']
            audio_int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
            wavfile.write(audio_file, 24000, audio_int16)
            print(f"\nAudio saved: {audio_file}")

            # 记录结果
            results.append({
                'language_code': lang_code,
                'language_name': lang_name,
                'text': text,
                'phonemes': result['phonemes'],
                'duration': result['duration'],
                'timestamps': result['timestamps'],
                'audio_file': audio_file
            })

        except Exception as e:
            print(f"Error: {e}")
            results.append({
                'language_code': lang_code,
                'language_name': lang_name,
                'text': text,
                'error': str(e)
            })

    # 保存汇总 JSON
    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"演示完成！")
    print(f"输出目录: {output_dir}")
    print(f"汇总文件: {summary_file}")
    print(f"{'=' * 60}")

    # 打印汇总表格
    print(f"\n汇总:")
    print(f"{'语言':<20} {'时长':>8} {'单词数':>8} {'状态':>10}")
    print("-" * 50)
    for r in results:
        if 'error' in r:
            print(f"{r['language_name']:<20} {'—':>8} {'—':>8} {'❌ 失败':>10}")
        else:
            word_count = len(r['timestamps'])
            print(f"{r['language_name']:<20} {r['duration']:>6.2f}s {word_count:>8} {'✅ 成功':>10}")


if __name__ == '__main__':
    run_demo()
