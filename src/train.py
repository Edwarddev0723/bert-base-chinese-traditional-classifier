# filename: bert_plateau_trainer.py
# pip install transformers wandb datasets accelerate

import math
import wandb
from typing import Dict, Any
from datasets import DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    SchedulerType
)

# --------------------------------------------------
# 基本組件
# --------------------------------------------------
def init_wandb(project: str, run_name: str) -> None:
    """啟動 Weights & Biases 追蹤"""
    wandb.init(project=project, name=run_name, save_code=True)


def get_tokenizer(model_name: str = "ckiplab/bert-base-chinese",
                  cache_dir: str = "./hf_cache"):
    """回傳 CKIP tokenizer（fast 版本）"""
    return AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        use_fast=True
    )


def get_model(num_labels: int = 3,
              model_name: str = "ckiplab/bert-base-chinese"):
    """建立分類模型（含 gradient checkpointing 以省顯存）"""
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        hidden_dropout_prob=0.3,
        torch_dtype="auto",
        gradient_checkpointing=True
    )


def calc_training_steps(train_size: int,
                        batch_size: int,
                        grad_accum: int,
                        epochs: int) -> Dict[str, int]:
    """計算訓練相關步數"""
    steps_per_epoch = math.ceil(train_size / (batch_size * grad_accum))
    total_steps     = steps_per_epoch * epochs
    warmup_steps    = int(0.1 * total_steps)
    return {
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps
    }


def build_training_args(output_dir: str,
                        steps_cfg: Dict[str, int],
                        batch_size: int,
                        grad_accum: int,
                        epochs: int) -> TrainingArguments:
    """建立 ReduceLROnPlateau 版本 TrainingArguments"""
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,

        # —— 調度器設定 —— #
        lr_scheduler_type=SchedulerType.REDUCE_ON_PLATEAU,
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=1,
        lr_scheduler_threshold=1e-4,
        lr_scheduler_threshold_mode="rel",
        lr_scheduler_min_lr=1e-7,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # —— 基本訓練 —— #
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-5,
        warmup_steps=steps_cfg["warmup_steps"],
        weight_decay=0.01,

        # —— 紀錄與儲存 —— #
        logging_strategy="steps",
        logging_steps=max(1, steps_cfg["steps_per_epoch"] // 10),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,

        fp16=True,
        seed=42,
        dataloader_num_workers=4,
        report_to="wandb",
        disable_tqdm=False
    )


# --------------------------------------------------
# 高階封裝
# --------------------------------------------------
def train_with_plateau(dataset_dict: DatasetDict,
                       project: str = "my-bert-chinese-project",
                       run_name: str = "bert-plateau-run1",
                       output_dir: str = "./model_ckpt",
                       num_labels: int = 3,
                       batch_size: int = 32,
                       grad_accum: int = 2,
                       epochs: int = 3) -> Trainer:
    """
    一行呼叫即可完成：
      1. W&B 初始
      2. tokenizer / model 建立
      3. TrainingArguments（ReduceLROnPlateau）
      4. Trainer  → start training
    回傳已訓練完成的 Trainer 物件
    """
    # 1. W&B
    init_wandb(project, run_name)

    # 2. 模型與 tokenizer
    tokenizer = get_tokenizer()
    model     = get_model(num_labels=num_labels)

    # 3. 步數計算
    steps_cfg = calc_training_steps(
        train_size=len(dataset_dict["train"]),
        batch_size=batch_size,
        grad_accum=grad_accum,
        epochs=epochs
    )

    # 4. TrainingArguments
    training_args = build_training_args(
        output_dir=output_dir,
        steps_cfg=steps_cfg,
        batch_size=batch_size,
        grad_accum=grad_accum,
        epochs=epochs
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        tokenizer=tokenizer
    )

    trainer.train()
    wandb.finish()
    return trainer


# --------------------------------------------------
# CLI 測試
# --------------------------------------------------
if __name__ == "__main__":
    from datasets import load_from_disk
    # 假設事先存好的 arrow 目錄
    ds_dict = load_from_disk("data_cache/arrow_dataset")
    train_with_plateau(ds_dict)
