#!/usr/bin/env python
"""
BERT fine-tune CLI – zero-crash across any 🤗 Transformers
========================================================
訓練一個中文序列分類模型，此模型基於一個已經 tokenized 的
`DatasetDict` (透過 `save_to_disk` 儲存)。

主要設計
----------
* **簽名感知 (Signature-aware) 的 TrainingArguments** – 每個關鍵字參數都會
    根據您 *已安裝* 的 Transformers 版本進行過濾。
* **自動對齊策略** – 自動協調 `evaluation_strategy` (`eval_strategy`),
    `save_strategy`, 和 `load_best_model_at_end` 以避免排程器不匹配的錯誤。
* **支援快速子集抽樣** – 可使用 `--rows N` 或 `start-end` 格式。
* **自動轉換標籤** – 將列表形式的標籤（如 `[1]`）自動轉換為純量（`1`）。
* **可選的 Weights & Biases 日誌** (`--no_wandb`)。

範例
-------
```bash
python train.py -d data_cache_v2 -o ckpt --rows 5000 \
  --project myproj --run debug-5k
```
"""
from __future__ import annotations

import argparse
import logging
import math
import re
from inspect import signature
from pathlib import Path
from typing import Tuple

import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

try:
    import wandb
except ImportError:
    wandb = None

# 設定日誌記錄器
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# TrainingArguments 建構器 – 版本兼容且策略一致
# ---------------------------------------------------------------------------

def build_training_args(
    *,
    out_dir: Path,
    steps_per_epoch: int,
    warmup_steps: int,
    batch_size: int,
    grad_accum: int,
    epochs: int,
    use_wandb: bool,
) -> TrainingArguments:
    """
    建立一個策略一致且對不同 transformers 版本兼容的 TrainingArguments 物件。
    """
    base_args = {
        "output_dir": str(out_dir),
        "overwrite_output_dir": True,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size * 2,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": 2e-5,
        "warmup_steps": warmup_steps,
        "weight_decay": 0.01,
        "logging_strategy": "steps",
        "logging_steps": max(1, steps_per_epoch // 10),
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "fp16": True,
        "seed": 42,
        "dataloader_num_workers": 4,
        "report_to": ["wandb"] if use_wandb else [],
        "lr_scheduler_type": "linear",
    }

    sig = signature(TrainingArguments).parameters
    if "evaluation_strategy" in sig:
        base_args["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in sig:
        base_args["eval_strategy"] = "epoch"

    supported_args = {k: v for k, v in base_args.items() if k in sig}
    
    logging.info(f"使用的評估策略參數: {'evaluation_strategy' if 'evaluation_strategy' in supported_args else 'eval_strategy'}")

    return TrainingArguments(**supported_args)


# ---------------------------------------------------------------------------
# 主執行流程類別
# ---------------------------------------------------------------------------

class BertTrainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.use_wandb = not args.no_wandb and (wandb is not None)

    def run(self):
        """執行完整的訓練流程"""
        self.init_wandb()
        ds_dict = self.prepare_dataset()

        tokenizer = AutoTokenizer.from_pretrained(self.args.model, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.args.model,
            num_labels=self.args.num_labels,
            hidden_dropout_prob=0.3,
            torch_dtype="auto",
            gradient_checkpointing=True,
        )
        
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        num_train_rows = len(ds_dict["train"])
        steps_per_epoch = math.ceil(num_train_rows / (self.args.batch_size * self.args.grad_accum))
        total_train_steps = steps_per_epoch * self.args.epochs
        warmup_steps = int(0.1 * total_train_steps)

        training_args = build_training_args(
            out_dir=self.args.output_dir,
            steps_per_epoch=steps_per_epoch,
            warmup_steps=warmup_steps,
            batch_size=self.args.batch_size,
            grad_accum=self.args.grad_accum,
            epochs=self.args.epochs,
            use_wandb=self.use_wandb,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds_dict["train"],
            eval_dataset=ds_dict["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        
        logging.info("開始模型訓練...")
        trainer.train()
        logging.info("模型訓練完成。")

        if self.use_wandb:
            wandb.finish()

    def init_wandb(self):
        if self.use_wandb:
            logging.info(f"初始化 Weights & Biases: project='{self.args.project}', run='{self.args.run}'")
            wandb.init(project=self.args.project, name=self.args.run, save_code=True)

    def prepare_dataset(self) -> DatasetDict:
        """載入、修正、抽樣和驗證資料集"""
        logging.info(f"從 {self.args.dataset_dir} 載入資料集...")
        ds_dict = load_from_disk(str(self.args.dataset_dir))

        # --- 新增的修正 (關鍵) ---
        # 檢查並修正可能存在的過度巢狀化特徵 (例如 [[...]] -> [...])
        def unnest_features(example):
            for key in ['input_ids', 'attention_mask', 'token_type_ids']:
                if key in example and isinstance(example[key], list) and len(example[key]) > 0 and isinstance(example[key][0], list):
                    example[key] = example[key][0]
            return example

        for split in ds_dict.keys():
            # 檢查第一個樣本來判斷是否需要修正
            first_item = ds_dict[split][0]
            if 'input_ids' in first_item and isinstance(first_item['input_ids'], list) and len(first_item['input_ids']) > 0 and isinstance(first_item['input_ids'][0], list):
                 logging.info(f"偵測到 '{split}' 資料集中有巢狀特徵，正在進行即時修正...")
                 ds_dict[split] = ds_dict[split].map(unnest_features, num_proc=4)

        if self.args.rows:
            ds_dict["train"] = self._subset(ds_dict["train"], self.args.rows)
            logging.info(f"已抽樣訓練集，使用前 {len(ds_dict['train']):,} 筆資料。")

        for split in ("train", "validation"):
            if split in ds_dict:
                ds_dict[split] = self._squeeze_label(ds_dict[split], split_name=split)

        self._sanity_check(ds_dict, self.args.num_labels)
        return ds_dict

    def _parse_rows(self, spec: str) -> Tuple[int, int]:
        if re.fullmatch(r"\d+", spec):
            return 0, int(spec)
        m = re.fullmatch(r"(\d+)-(\d+)", spec)
        if not m:
            raise ValueError("--rows 需要是 N 或 start-end 格式 (例如 5000 或 1000-14999)")
        a, b = map(int, m.groups())
        if b < a:
            raise ValueError("--rows 的結束值不能小於起始值")
        return a, b + 1

    def _subset(self, ds: Dataset, spec: str | None) -> Dataset:
        if spec is None:
            return ds
        s, e = self._parse_rows(spec)
        return ds.select(range(s, min(e, len(ds))))

    def _squeeze_label(self, ds: Dataset, split_name: str) -> Dataset:
        if ds and isinstance(ds[0]["labels"], list):
            def fn(ex):
                uniq_labels = set(ex["labels"])
                if len(uniq_labels) != 1:
                    raise ValueError(f"發現多標籤範例: {ex['labels']}")
                ex["labels"] = ex["labels"][0]
                return ex
            logging.info(f"正在轉換 '{split_name}' 的標籤格式...")
            ds = ds.map(fn, num_proc=4)
        return ds

    def _sanity_check(self, dd: DatasetDict, num_labels: int):
        required_cols = {"input_ids", "attention_mask", "labels"}
        for split, dset in dd.items():
            if missing_cols := required_cols - set(dset.column_names):
                raise ValueError(f"資料集 '{split}' 缺少必要欄位: {missing_cols}")
            
            labels = pd.Series(dset["labels"])
            if not labels.isin(range(num_labels)).all():
                raise ValueError(f"資料集 '{split}' 的標籤必須在 0 到 {num_labels-1} 之間")
            
            dist_str = labels.value_counts().sort_index().to_dict()
            logging.info(f"{split}: {len(dset):,} 筆, 標籤分佈={dist_str}")


def main():
    p = argparse.ArgumentParser(
        description="一個能兼容不同版本 Transformers 的 BERT 微調腳本。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-d", "--dataset_dir", required=True, type=Path, help="已儲存的 DatasetDict 資料夾路徑")
    p.add_argument("-o", "--output_dir", required=True, type=Path, help="模型檢查點和輸出的儲存路徑")
    p.add_argument("--rows", type=str, help="抽樣訓練集，可為數字 N 或 'start-end' 區間")
    p.add_argument("--num_labels", type=int, default=3, help="分類任務的標籤數量")
    p.add_argument("--batch_size", type=int, default=32, help="每個設備的訓練批次大小")
    p.add_argument("--grad_accum", type=int, default=2, help="梯度累積步數")
    p.add_argument("--epochs", type=int, default=3, help="總訓練輪數")
    p.add_argument("--project", default="bert-project", help="Weights & Biases 的專案名稱")
    p.add_argument("--run", default="run-1", help="Weights & Biases 的運行名稱")
    p.add_argument("--no_wandb", action="store_true", help="停用 Weights & Biases 日誌記錄")
    p.add_argument("--model", default="ckiplab/bert-base-chinese", help="要使用的預訓練模型名稱")
    
    args = p.parse_args()
    
    trainer = BertTrainer(args)
    trainer.run()


if __name__ == "__main__":
    main()
