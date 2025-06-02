# train.py
"""
模型訓練主程式與 Trainer 設定
"""
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers.integrations import WandbCallback
import numpy as np
from tokenizer_util import get_tokenizer, split_and_tokenize
from data_prepare import prepare_dataset

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def main():
    dataset, _ = prepare_dataset()
    tokenizer = get_tokenizer()
    encoded_dataset, df_train, df_val = split_and_tokenize(dataset, tokenizer)
    labels_set = set(encoded_dataset["train"]["label"])
    print("所有 label:", labels_set)
    assert labels_set.issubset({0, 1, 2}), "labels 有超出 0/1/2 的數值"
    print("有無 NaN:", np.isnan(encoded_dataset["train"]["label"]).any())
    encoded_dataset = encoded_dataset.map(lambda x: {"label": int(x["label"])})
    model = AutoModelForSequenceClassification.from_pretrained(
        "ckiplab/bert-base-chinese",
        num_labels=3,
        hidden_dropout_prob=0.3,  # 適度防過擬合
    )
    import torch
    # 順序偵測 mps > cuda > cpu
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用 MPS 裝置")
        fp16 = False  # Apple Silicon 不支援 fp16
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("使用 CUDA GPU 裝置")
        fp16 = True
    else:
        device = torch.device("cpu")
        print("使用 CPU 裝置")
        fp16 = False

    model.to(device)
    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir="./model_ckpt",
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=2e-5,
        lr_scheduler_type="linear",
        weight_decay=0.1,
        warmup_ratio=0.06,
        save_strategy="steps",         # 改為 "steps"
        save_steps=1000,               # 每1000步存一次
        logging_strategy="steps",
        logging_steps=200,
        eval_strategy="steps",         # 和 save 保持一致
        eval_steps=1000,               # 每1000步驗證一次（和 save_steps 一樣）
        save_total_limit=2,
        fp16=fp16,                     # ⚠️ Apple Silicon 不支援 fp16
        seed=42,
        report_to=["wandb"] if WANDB_AVAILABLE else "none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        disable_tqdm=False,
        max_grad_norm=1.0,
    )
    if WANDB_AVAILABLE:
        # 直接於程式內設定 wandb 參數
        wandb.init(
            project="bert-zh-tw-classifier",
            name="bert-zh-tw-run",
            config={
                "batch_size": 16,
                "epochs": 3,
                "learning_rate": 2e-5,
                "model": "ckiplab/bert-base-chinese",
                "max_length": 512,
            }
        )
        print("已啟用 Weights & Biases 日誌紀錄。")
    else:
        print("未安裝 wandb，將不啟用 Weights & Biases 日誌。")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        callbacks=[WandbCallback()] if WANDB_AVAILABLE else None,
    )
    trainer.train()
    model.save_pretrained("./bert-zh-tw-classifier")
    tokenizer.save_pretrained("./bert-zh-tw-classifier")
    trainer.save_model("./model_ckpt")
    tokenizer.save_pretrained("./model_ckpt")
    print("訓練完成，模型已儲存。")
    return trainer, encoded_dataset

if __name__ == "__main__":
    main()
