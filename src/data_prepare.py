# data_prepare.py
"""
資料集下載、處理、切分等相關函式
"""
import pandas as pd
import itertools
import opencc
import os
from datasets import load_dataset, Dataset

MAX_LENGTH = 512

def sample_streaming(dataset_name, split, num_samples):
    stream = load_dataset(dataset_name, split=split, streaming=True)
    sampled = list(itertools.islice(stream, num_samples))
    df = pd.DataFrame(sampled)
    col = "text" if "text" in df.columns else df.columns[0]
    df = df[[col]].rename(columns={col: "text"})
    return df

def sample_unique_streaming(dataset_name, split, num_samples, exclude_texts):
    stream = load_dataset(dataset_name, split=split, streaming=True)
    result = []
    for sample in stream:
        t = sample["text"] if "text" in sample else list(sample.values())[0]
        if t not in exclude_texts:
            result.append(t)
            if len(result) >= num_samples:
                break
    return pd.DataFrame({"text": result})

def random_hybrid_segments_distributed(trad_text, segment_num=3):
    # 將 opencc 配置名稱改為 't2s'，避免找不到 t2s.json
    converter = opencc.OpenCC('t2s')
    trad_chars = list(trad_text)
    simp_chars = list(converter.convert(trad_text))
    N = len(trad_chars)
    hybrid = trad_chars.copy()
    if N < 2 or segment_num < 1:
        return trad_text
    span = N // segment_num
    for seg in range(segment_num):
        start = seg * span
        end = min(start + max(1, span // 2), N)
        for i in range(start, end):
            hybrid[i] = simp_chars[i]
    return "".join(hybrid)

def split_long_text(text, label, max_length=MAX_LENGTH):
    if len(text) <= max_length:
        return [{"text": text, "label": label}]
    return [{"text": text[i:i+max_length], "label": label}
            for i in range(0, len(text), max_length)]

def prepare_dataset(cache_path="../data/processed/train.parquet"):
    # 0. 檢查是否已有快取檔案
    if os.path.exists(cache_path):
        print(f"偵測到快取檔案 {cache_path}，直接載入...")
        df_total = pd.read_parquet(cache_path)
        dataset = Dataset.from_pandas(df_total[["text", "label"]])
        return dataset, df_total
    # 1. 分別抽樣
    df_trad = sample_streaming("voidful/fineweb-zhtw", "train", 55000)
    df_trad["label"] = 1
    df_simp = sample_streaming("opencsg/chinese-fineweb-edu", "train", 85000)
    df_simp["label"] = 0
    # 2. 標記已有 text
    used_texts = set(df_trad["text"]).union(set(df_simp["text"]))
    # 3. 重新 streaming 不重複的混合資料
    df_mixed_raw = sample_unique_streaming(
        "voidful/fineweb-zhtw", "train", 40000, used_texts
    )
    df_mixed_raw["text"] = df_mixed_raw["text"].apply(lambda x: random_hybrid_segments_distributed(x, segment_num=3))
    df_mixed_raw["label"] = 2
    # 4. 合併
    df_total = pd.concat([df_trad, df_simp, df_mixed_raw], ignore_index=True)
    # 5. 長文本切片
    records = []
    for _, row in df_total.iterrows():
        records.extend(split_long_text(row["text"], row["label"], MAX_LENGTH))
    df_total = pd.DataFrame(records)
    # 6. 刪除過短樣本
    df_total = df_total[df_total["text"].str.len() >= 10].reset_index(drop=True)
    # 7. 亂數排序
    df_total = df_total.sample(frac=1.0, random_state=42).reset_index(drop=True)
    # 8. 轉 Hugging Face Dataset
    dataset = Dataset.from_pandas(df_total[["text", "label"]])
    # 9. 長度分布監控
    lengths = df_total["text"].str.len()
    print(f"文本長度統計: \n{lengths.describe()}")
    overlong_ratio = (lengths >= MAX_LENGTH).mean()
    print(f">={MAX_LENGTH}字的樣本比例: {overlong_ratio:.2%}")
    if overlong_ratio > 0.3:
        print("警告：超長文本比例仍然過高，建議再檢查資料清理！")
    # 10. 儲存快取檔案
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_total.to_parquet(cache_path, index=False)
    print(f"已將處理後資料儲存至 {cache_path}")
    return dataset, df_total

if __name__ == "__main__":
    dataset, df_total = prepare_dataset()
    print(dataset)
    print(df_total.head())
