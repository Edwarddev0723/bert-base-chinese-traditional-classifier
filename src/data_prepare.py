# pip install datasets opencc-python-reimplemented zhon transformers tqdm psutil

import os, re, random, hashlib, multiprocessing as mp, psutil, warnings
from datasets import load_dataset, Dataset, disable_caching, concatenate_datasets
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from functools import lru_cache

# ---------- CONFIG ----------
SEED, MAX_TOK, BUF, BATCH = 42, 256, 50_000, 5_000
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
tok = AutoTokenizer.from_pretrained("ckiplab/bert-base-chinese")
disable_caching(); random.seed(SEED)
warnings.filterwarnings("ignore", message="Token indices sequence length")

NUM_PROC = min(8, mp.cpu_count()) if psutil.virtual_memory().total/2**30 >= 16 else 4
_SENT_BOUND = re.compile(r"([。！？；…]+)")

# ---------- OpenCC ----------
@lru_cache(maxsize=None)
def _cc(table):  # lazy share across workers
    from opencc import OpenCC
    return OpenCC(table)

def to_trad(texts, mode):
    if mode == "simp2trad":
        texts = [_cc("s2t.json").convert(t) for t in texts]  # 只改字形
    return [_cc("t2tw.json").convert(t).replace("\u3000", "").strip() for t in texts]

# ---------- Sentence utils ----------
def sent_split(text):
    parts, buf = [], []
    for seg in _SENT_BOUND.split(text):
        if seg:
            buf.append(seg)
            if _SENT_BOUND.fullmatch(seg):
                parts.append("".join(buf).strip()); buf=[]
    if buf: parts.append("".join(buf).strip())
    return [s for s in parts if s]

def chunk_by_tokens(sentences, limit=MAX_TOK):
    chunks, cur, n = [], [], 0
    for s in sentences:
        l = len(tok(s, add_special_tokens=False)["input_ids"])
        if n + l > limit and cur:
            chunks.append("".join(cur)); cur, n = [], 0
        cur.append(s); n += l
    return chunks + ["".join(cur)] if cur else chunks

# ---------- Source streaming ----------
def stream_take(name, split, rows, key):
    ds = load_dataset(name, split=split, streaming=True)\
        .shuffle(seed=SEED, buffer_size=BUF)
    out, it = [], iter(ds)
    with tqdm(total=rows, desc=f"stream {name.split('/')[-1]}", ncols=80) as p:
        while len(out) < rows:
            ex = next(it)
            if ex.get(key): out.append({"text": ex[key]}); p.update()
    return out

# ---------- Pipeline per source ----------
def prep(name, rows, mode, label, split, key):
    raw = stream_take(name, split, rows, key)
    ds  = Dataset.from_list(raw)

    # 1. 簡→繁字形 + 異體正規化
    ds  = ds.map(lambda b: {"text": to_trad(b["text"], mode)},
                 batched=True, batch_size=BATCH, num_proc=NUM_PROC)

    # 2. 句切 + token 分塊 → 回傳「清單的清單」
    ds  = ds.map(lambda b: {"chunks": [chunk_by_tokens(sent_split(t)) for t in b["text"]]},
                 batched=True, batch_size=512, num_proc=NUM_PROC)

    # 3. explode：把 chunks 展平成多行；每行配對同一 label
    ds  = ds.flatten_indices()            # 先把嵌套展開
    ds  = ds.map(lambda b: {"text": b["chunks"]})\
             .remove_columns("chunks")    # 改欄名並刪舊欄

    ds  = ds.filter(lambda ex: len(ex["text"]) >= 10)

    # 4. 現在 rows 與 text 1:1，可安全加 label
    ds  = ds.add_column("label", [label] * len(ds))
    return ds

# ---------- Dedup ----------
def dedup(ds):
    def add(batch):
        return {"key": [f"{hashlib.md5(t[:128].encode()).hexdigest()}_{lab}"
                        for t, lab in zip(batch["text"], batch["label"])]}
    seen=set()
    def keep(batch):
        flags=[]
        for k in batch["key"]:
            flags.append(k not in seen); seen.add(k)
        return {"keep": flags}
    ds = ds.map(add, batched=True, batch_size=BATCH, num_proc=NUM_PROC)\
           .map(keep, batched=True, num_proc=1)\
           .filter(lambda ex: ex["keep"]).remove_columns(["key","keep"])
    return ds.shuffle(seed=SEED)

# ---------- Main ----------
def build_parquet(rows=150_000, out_dir="data_cache"):
    each = rows//2
    srcs = [
        ("voidful/fineweb-zhtw", "train", "norm", 1, "text"),
        ("BAAI/CCI4.0-M2-Base-v1", "train", "simp2trad", 0, "content"),
    ]
    parts = [prep(n, each, m, lab, sp, k) for n,sp,m,lab,k in srcs]
    full  = dedup(concatenate_datasets(parts))
    os.makedirs(out_dir, exist_ok=True)
    path  = f"{out_dir}/fineweb_cci3_mix_{len(full)}.parquet"
    full.to_parquet(path, compression="zstd")
    print("Saved ->", path)
    return path

if __name__ == "__main__":
    build_parquet()
