#!/usr/bin/env python
"""
Tokenization & train/val split utilities – *CLI‑ready*
-----------------------------------------------------
Adds a proper command‑line interface on top of the original helpers so
that you can invoke it directly, e.g.:

```bash
python tokenizer_util.py \
  --input fineweb.parquet \
  --out_dir data_cache_v2 \
  --model ckiplab/bert-base-chinese \
  --max_len 256 --stride 128 --test_size 0.2
```

The encoded `datasets.DatasetDict` is written via `save_to_disk()` to
`out_dir`.  Raw train/val DataFrames are optionally saved as Parquet for
quick inspection.
"""
from __future__ import annotations

import argparse, os, multiprocessing as mp
from pathlib import Path
from functools import lru_cache
from typing import Tuple, Union

import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Global defaults – can be overridden via CLI flags
# ---------------------------------------------------------------------------
MAX_SEQ_LEN  = 256
STRIDE       = 128
TEST_SIZE    = 0.2
RANDOM_STATE = 42
NUM_PROC     = mp.cpu_count() or 1

# ---------------------------------------------------------------------------
# Tokenizer bootstrap – cached to avoid re‑instantiation cost
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def get_tokenizer(model_name: str = "ckiplab/bert-base-chinese") -> PreTrainedTokenizerBase:
    """Lazily load and cache a HF tokenizer."""
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)

# ---------------------------------------------------------------------------
# Core tokenization for a single example (handles overflow)
# ---------------------------------------------------------------------------

def _chunk_tokenize(text: str, label: int, tokenizer: PreTrainedTokenizerBase,
                    max_len: int = MAX_SEQ_LEN, stride: int = STRIDE):
    """Tokenize *one* text, returning lists of `input_ids`, `attention_mask`,
    and duplicated `label` entries – handling long docs via overflow chunks."""
    if len(text) <= max_len:
        out = tokenizer(text, truncation=True, max_length=max_len,
                         return_attention_mask=True, padding=False,
                         return_token_type_ids=False)
        return [out["input_ids"]], [out["attention_mask"]], [label]

    # long text – sliding window
    out = tokenizer(text, truncation=True, max_length=max_len, stride=stride,
                    return_overflowing_tokens=True, return_attention_mask=True,
                    padding=False, return_token_type_ids=False)
    n_chunks = len(out["input_ids"])
    return out["input_ids"], out["attention_mask"], [label] * n_chunks

# ---------------------------------------------------------------------------
# Vectorised map fn for datasets.map (batched=False for clarity)
# ---------------------------------------------------------------------------

def _tokenize_example(example, tokenizer: PreTrainedTokenizerBase, max_len: int, stride: int):
    ids, masks, labs = _chunk_tokenize(example["text"], example["label"], tokenizer, max_len, stride)
    return {
        "input_ids": ids,
        "attention_mask": masks,
        "labels": labs,
    }

# ---------------------------------------------------------------------------
# Public helper – split, tokenize, return encoded DatasetDict + raw DataFrames
# ---------------------------------------------------------------------------

def split_and_tokenize(dataset: Union[Dataset, pd.DataFrame],
                        tokenizer: PreTrainedTokenizerBase | None = None,
                        test_size: float = TEST_SIZE,
                        random_state: int = RANDOM_STATE,
                        max_len: int = MAX_SEQ_LEN,
                        stride: int = STRIDE,
                        keep_raw: bool = True
                       ) -> Tuple[DatasetDict, pd.DataFrame | None, pd.DataFrame | None]:
    """Stratified split → tokenization → DatasetDict."""
    tok = tokenizer or get_tokenizer()

    # to pandas then stratified split
    if isinstance(dataset, Dataset):
        df = dataset.to_pandas()
    else:
        df = dataset.copy()

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )

    raw_ds = DatasetDict({
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
    })

    # tokenize with multiprocessing
    encode = lambda ex: _tokenize_example(ex, tok, max_len, stride)
    encoded = raw_ds.map(
        encode,
        num_proc=NUM_PROC,
        remove_columns=[c for c in raw_ds["train"].column_names if c not in ("text", "label")],
        batched=False,
    )

    return (encoded, train_df, val_df) if keep_raw else (encoded, None, None)

# ---------------------------------------------------------------------------
# CLI entry point – build & save encoded dataset
# ---------------------------------------------------------------------------

def _load_dataset(path: Path) -> Dataset:
    ext = path.suffix.lower()
    if ext == ".parquet":
        df = pd.read_parquet(path)
        return Dataset.from_pandas(df)
    if ext in {".csv", ".tsv"}:
        df = pd.read_csv(path)
        return Dataset.from_pandas(df)
    if path.is_dir():
        # assume HF saved dataset (load_from_disk)
        return load_from_disk(str(path))
    raise ValueError(f"Unsupported input format: {path}")


def main():
    p = argparse.ArgumentParser(description="Tokenize & split any text dataset")
    p.add_argument("--input", "-i", required=True, type=Path,
                   help="Path to source dataset (Parquet/CSV or saved HF dataset)")
    p.add_argument("--out_dir", "-o", required=True, type=Path,
                   help="Directory to save the encoded DatasetDict")
    p.add_argument("--model", default="ckiplab/bert-base-chinese",
                   help="HF model name for tokenizer")
    p.add_argument("--max_len", type=int, default=MAX_SEQ_LEN)
    p.add_argument("--stride", type=int, default=STRIDE)
    p.add_argument("--test_size", type=float, default=TEST_SIZE)
    p.add_argument("--no_raw", action="store_true",
                   help="Do *not* save raw train/val Parquet copies")

    args = p.parse_args()

    ds = _load_dataset(args.input)
    tokenizer = get_tokenizer(args.model)

    encoded, train_df, val_df = split_and_tokenize(
        ds, tokenizer=tokenizer, test_size=args.test_size,
        max_len=args.max_len, stride=args.stride, keep_raw=not args.no_raw,
    )

    # make sure output dir exists & save
    args.out_dir.mkdir(parents=True, exist_ok=True)
    encoded.save_to_disk(str(args.out_dir))

    if not args.no_raw:
        train_df.to_parquet(args.out_dir / "train_raw.parquet", index=False)
        val_df.to_parquet(args.out_dir / "val_raw.parquet", index=False)

    print(f"✓ Encoded dataset saved to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
