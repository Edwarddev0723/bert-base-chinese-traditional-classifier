from datasets import load_dataset, Dataset
import pandas as pd
import itertools
import opencc
import random

MAX_LENGTH = 512


def sample_streaming(dataset_name: str, split: str, num_samples: int) -> pd.DataFrame:
    """
    從指定資料集以 streaming 方式抽樣 num_samples 筆資料，返回含 'text' 欄位的 DataFrame。
    """
    stream = load_dataset(dataset_name, split=split, streaming=True)
    samples = list(itertools.islice(stream, num_samples))
    df = pd.DataFrame(samples)
    col = 'text' if 'text' in df.columns else df.columns[0]
    return df[[col]].rename(columns={col: 'text'})


def sample_unique_streaming(
    dataset_name: str,
    split: str,
    num_samples: int,
    exclude_texts: set[str]
) -> pd.DataFrame:
    """
    從指定資料集 streaming 抽樣，過濾掉 exclude_texts 中的文本，直到收集到 num_samples。
    """
    stream = load_dataset(dataset_name, split=split, streaming=True)
    collected = []
    for item in stream:
        text = item.get('text', list(item.values())[0])
        if text not in exclude_texts:
            collected.append(text)
            if len(collected) >= num_samples:
                break
    return pd.DataFrame({'text': collected})


def random_hybrid_segments_distributed(
    trad_text: str,
    segment_num: int = 3,
    converter: opencc.OpenCC = opencc.OpenCC('t2s.json')
) -> str:
    """
    將繁體文本按 segment_num 段隨機替換為簡體，平均分佈簡化字符。
    """
    trad_chars = list(trad_text)
    simp_chars = list(converter.convert(trad_text))
    N = len(trad_chars)
    if N < 2 or segment_num < 1:
        return trad_text
    span = max(1, N // segment_num)
    hybrid = trad_chars.copy()
    for seg in range(segment_num):
        start = seg * span
        end = min(start + span // 2, N)
        for i in range(start, end):
            hybrid[i] = simp_chars[i]
    return ''.join(hybrid)


def split_long_text(
    text: str,
    label: int,
    max_length: int = MAX_LENGTH
) -> list[dict]:
    """
    將長文本按 max_length 分割為多個樣本，返回包含 'text' 與 'label' 的 dict list。
    """
    if len(text) <= max_length:
        return [{'text': text, 'label': label}]
    return [
        {'text': text[i:i+max_length], 'label': label}
        for i in range(0, len(text), max_length)
    ]


def prepare_dataset(
    trad_samples: int = 55000,
    simp_samples: int = 85000,
    mix_unique_samples: int = 40000,
    segment_num: int = 3,
    min_length: int = 10
) -> Dataset:
    """
    整合繁體、簡體與混合樣本，文本分割，並轉為 Hugging Face Dataset。
    """
    # 1. 抽樣繁體與簡體
    df_trad = sample_streaming('voidful/fineweb-zhtw', 'train', trad_samples)
    df_trad['label'] = 1
    df_simp = sample_streaming('opencsg/chinese-fineweb-edu', 'train', simp_samples)
    df_simp['label'] = 0

    # 2. 收集已用文本
    used = set(df_trad['text']).union(df_simp['text'])

    # 3. 抽樣不重複混合文本
    df_mixed = sample_unique_streaming(
        'voidful/fineweb-zhtw', 'train', mix_unique_samples, used
    )
    df_mixed['text'] = df_mixed['text'].apply(
        lambda x: random_hybrid_segments_distributed(x, segment_num)
    )
    df_mixed['label'] = 2

    # 4. 合併並分割長文本
    df_total = pd.concat([df_trad, df_simp, df_mixed], ignore_index=True)
    records = []
    for _, row in df_total.iterrows():
        records.extend(
            split_long_text(row['text'], int(row['label']))
        )
    df_proc = pd.DataFrame(records)

    # 5. 篩除短文本與打亂
    df_proc = df_proc[df_proc['text'].str.len() >= min_length]
    df_proc = df_proc.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # 6. 轉為 HF Dataset
    return Dataset.from_pandas(df_proc[['text', 'label']])
