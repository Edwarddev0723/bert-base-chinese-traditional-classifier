# filename: data_prep.py
# pip install datasets opencc-python-reimplemented zhon transformers tqdm psutil

import os, re, random, hashlib, multiprocessing as mp, psutil
from functools import lru_cache
from typing import List, Dict
from datasets import load_dataset, Dataset, disable_caching, concatenate_datasets
from transformers import AutoTokenizer
from tqdm.auto import tqdm

# ---------- GLOBAL CONFIG ----------
SEED            = 42
MAX_TOKENS      = 256
BUFFER_SIZE     = 50_000
BATCH_SIZE      = 5_000
HF_ENDPOINT     = "https://hf-mirror.com"

os.environ["HF_ENDPOINT"] = HF_ENDPOINT
rng = random.Random(SEED)
disable_caching()
tok = AutoTokenizer.from_pretrained("ckiplab/bert-base-chinese")

# ---------- PARALLEL ----------
def get_optimal_num_proc() -> int:
    """動態決定並行數量（CPU / RAM 雙門檻）"""
    cpu = mp.cpu_count()
    mem = psutil.virtual_memory().total / 2**30
    return min(8, cpu) if mem >= 16 else min(4, max(2, cpu // 2))

NUM_PROC = get_optimal_num_proc()

# ---------- SENTENCE SPLITTER ----------
_SENT_BOUND = re.compile(r"([。！？；…]+)")

def sent_split(text: str) -> List[str]:
    """將段落切成保留標點的句子清單"""
    parts, buff = [], []
    for seg in _SENT_BOUND.split(text):
        if seg:
            buff.append(seg)
            if _SENT_BOUND.fullmatch(seg):
                parts.append("".join(buff).strip())
                buff = []
    if buff:
        parts.append("".join(buff).strip())
    return [s for s in parts if s]

def chunk_by_tokens(sentences: List[str], max_tokens: int = MAX_TOKENS) -> List[str]:
    """依 tokenizer token 數將句子累加成區塊"""
    chunks, cur, cur_tok = [], [], 0
    for s in sentences:
        s_tokens = tok(s, add_special_tokens=False)["input_ids"]
        if cur_tok + len(s_tokens) > max_tokens and cur:
            chunks.append("".join(cur))
            cur, cur_tok = [], 0
        cur.append(s)
        cur_tok += len(s_tokens)
    if cur:
        chunks.append("".join(cur))
    return chunks or ["".join(sentences)]

# ---------- OPENCC ----------
@lru_cache(maxsize=None)
def _get_cc(mode: str):
    from opencc import OpenCC
    tables = {"simp": "s2twp.json", "norm": "t2tw.json"}
    return OpenCC(tables[mode])

def to_trad(texts: List[str], mode: str) -> List[str]:
    """簡→繁＋用字正規化"""
    if mode == "simp2trad":
        texts = [_get_cc("simp").convert(t) for t in texts]
    return [_get_cc("norm").convert(t).replace("\u3000", "").strip() for t in texts]

# ---------- DATA SAMPLING ----------
def stream_take(ds_name: str, split: str, take_k: int):
    """串流隨機抽樣，避免一次載入巨大資料集"""
    return list(
        load_dataset(ds_name, split=split, streaming=True)
        .shuffle(seed=SEED, buffer_size=BUFFER_SIZE)
        .take(take_k)
    )

# ---------- SOURCE PIPELINE ----------
def prep_source(cfg: Dict) -> Dataset:
    """對單一資料來源完成轉繁體、斷句、chunk、標籤"""
    raw = stream_take(cfg["name"], cfg["split"], cfg["rows"])
    ds = Dataset.from_list(raw)

    ds = ds.map(
        lambda b: {"text": to_trad(b[cfg["field"]], cfg["mode"])},
        batched=True, batch_size=BATCH_SIZE, num_proc=NUM_PROC
    )

    def sent_chunk(batch):
        out = []
        for txt in batch["text"]:
            chunks = chunk_by_tokens(sent_split(txt))
            out.extend(chunks)
        return {"text": out}

    ds = ds.map(sent_chunk, batched=True, batch_size=512, num_proc=NUM_PROC)
    ds = ds.filter(lambda ex: len(ex["text"]) >= 10)
    ds = ds.add_column("label", [cfg["label"]] * len(ds))
    return ds

# ---------- MERGE & DEDUP ----------
def dedup_dataset(ds: Dataset) -> Dataset:
    def add_hash(batch):
        return {"hash": [hashlib.md5(t[:128].encode()).hexdigest() for t in batch["text"]]}

    def mark_unique(batch, seen=set()):
        flags = []
        for h in batch["hash"]:
            flags.append(h not in seen)
            seen.add(h)
        return {"keep": flags}

    ds = (ds.map(add_hash, batched=True, num_proc=NUM_PROC, batch_size=BATCH_SIZE)
            .map(mark_unique, batched=True, num_proc=1, batch_size=BATCH_SIZE)
            .filter(lambda ex: ex["keep"])
            .remove_columns(["hash", "keep"])
            .shuffle(seed=SEED))
    return ds

# ---------- MAIN ENTRY ----------
def prepare_dataset(
    target_rows: int = 150_000,
    ratio_per_src: float = 0.5,
    save_dir: str = "data_cache",
    parquet_prefix: str = "fineweb_cci3_mix",
):
    """一次完成多來源整理並輸出 parquet"""

    sources_cfg = [
        dict(name="voidful/fineweb-zhtw", split="train",
             rows=int(target_rows * ratio_per_src), mode="norm",
             label=1, field="text"),
        dict(name="BAAI/CCI3-Data", split="train",
             rows=int(target_rows * ratio_per_src), mode="simp2trad",
             label=0, field="content"),
    ]

    ds_list = [prep_source(cfg) for cfg in sources_cfg]
    full_ds = dedup_dataset(concatenate_datasets(ds_list))

    os.makedirs(save_dir, exist_ok=True)
    out_path = f"{save_dir}/{parquet_prefix}_{len(full_ds)}.parquet"
    full_ds.to_parquet(out_path, compression="zstd")
    print(f"Saved → {out_path} | total rows = {len(full_ds):,}")
    return out_path

from data_prep import prepare_dataset

if __name__ == "__main__":
    prepare_dataset(
        target_rows=150_000,   # 需要幾筆資料
        ratio_per_src=0.5,     # 兩資料源各半
        save_dir="data_cache", # 輸出資料夾
    )
