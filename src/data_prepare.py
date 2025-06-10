# filename: data_prep.py
# pip install datasets opencc-python-reimplemented zhon transformers tqdm psutil

import os, re, random, hashlib, multiprocessing as mp, psutil
from functools import lru_cache, partial
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

tqdm_cfg = dict(
    ncols=80,
    colour="green",
    unit="ex",
    unit_scale=True,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
)

# ---------- PARALLEL ----------
def get_optimal_num_proc() -> int:
    cpu = mp.cpu_count()
    mem = psutil.virtual_memory().total / 2**30
    return min(8, cpu) if mem >= 16 else min(4, max(2, cpu // 2))

NUM_PROC = get_optimal_num_proc()

# ---------- SENTENCE SPLITTER ----------
_SENT_BOUND = re.compile(r"([。！？；…]+)")

def sent_split(text: str) -> List[str]:
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
    if mode == "simp2trad":
        texts = [_get_cc("simp").convert(t) for t in texts]
    return [_get_cc("norm").convert(t).replace("\u3000", "").strip() for t in texts]

# ---------- DATA SAMPLING ----------
def stream_take(ds_name: str, split: str, take_k: int):
    iterator = (
        load_dataset(ds_name, split=split, streaming=True)
        .shuffle(seed=SEED, buffer_size=BUFFER_SIZE)
    )
    out = []
    with tqdm(total=take_k, desc=f"stream {ds_name}", **tqdm_cfg) as pbar:
        for ex in iterator:
            out.append(ex)
            pbar.update()
            if len(out) >= take_k:
                break
    return out

# ---------- SOURCE PIPELINE ----------
def prep_source(cfg: Dict) -> Dataset:
    desc = cfg["name"].split("/")[-1]
    raw = stream_take(cfg["name"], cfg["split"], cfg["rows"])
    ds  = Dataset.from_list(raw)

    ds = ds.map(
        lambda b: {"text": to_trad(b[cfg["field"]], cfg["mode"])},
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=NUM_PROC,
        with_tqdm=True,
        desc=f"toTrad  {desc}",
    )

    def sent_chunk(batch):
        out = []
        for txt in batch["text"]:
            out.extend(chunk_by_tokens(sent_split(txt)))
        return {"text": out}

    ds = ds.map(
        sent_chunk,
        batched=True,
        batch_size=512,
        num_proc=NUM_PROC,
        with_tqdm=True,
        desc=f"chunk    {desc}",
    )

    ds = ds.filter(lambda ex: len(ex["text"]) >= 10, with_tqdm=True, desc=f"filter   {desc}")
    ds = ds.add_column("label", [cfg["label"]] * len(ds))
    return ds

# ---------- MERGE & DEDUP ----------
def dedup_dataset(ds: Dataset) -> Dataset:
    total = len(ds)
    pbar = tqdm(total=total, desc="dedup", **tqdm_cfg)

    def add_hash(batch):
        return {"hash": [hashlib.md5(t[:128].encode()).hexdigest() for t in batch["text"]]}

    def mark_unique(batch, seen=set()):
        flags = []
        for h in batch["hash"]:
            flags.append(h not in seen)
            seen.add(h)
            pbar.update()
        return {"keep": flags}

    ds = (
        ds.map(add_hash, batched=True, num_proc=NUM_PROC, batch_size=BATCH_SIZE, with_tqdm=False)
          .map(mark_unique, batched=True, num_proc=1, batch_size=BATCH_SIZE, with_tqdm=False)
          .filter(lambda ex: ex["keep"], with_tqdm=True, desc="filter keep")
          .remove_columns(["hash", "keep"])
          .shuffle(seed=SEED)
    )
    pbar.close()
    return ds

# ---------- MAIN ENTRY ----------
def prepare_dataset(
    target_rows: int = 150_000,
    ratio_per_src: float = 0.5,
    save_dir: str = "data_cache",
    parquet_prefix: str = "fineweb_cci3_mix",
):
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

if __name__ == "__main__":
    prepare_dataset(
        target_rows=150_000,
        ratio_per_src=0.5,
        save_dir="data_cache",
    )
